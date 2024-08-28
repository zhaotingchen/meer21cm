import numpy as np
from astropy import constants, units
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import healpy as hp
from astropy.io import fits
from hiimtool.basic_util import check_unit_equiv, jy_to_kelvin
from astropy.cosmology import Planck18
import inspect
import sys
from powerbox import PowerBox

f_21 = 1420405751.7667  # in Hz
A_10 = 2.85 * 1e-15 / units.s
lamb_21 = (constants.c / f_21 * units.s).to("m")


def center_to_edges(arr):
    """
    Extend a linear spaced monotonic array
    so that the original array is the middle point of the output array.
    """
    result = arr.copy()
    dx = np.diff(arr)
    result = np.append(result[:-1] - dx / 2, result[-2:] + dx[-2:] / 2)
    return result


def coeff_hi_density_to_temp(z=0, cosmo=Planck18):
    r"""
    The conversion coefficient :math:`C_{\rm HI}` so that

    .. math::
        \bar{T}_{\rm HI} = C_{\rm HI} \rho_{\rm HI}

    Parameters
    ----------
    z: float, defulat 0.0.
        The redshift

    cosmo: cosmology, default Planck18.
        The cosmology used

    Returns
    -------
    C_HI: quantity.
        The coefficient
    """
    C_HI = (
        3
        * A_10
        * constants.h
        * constants.c**3
        * (1 + z) ** 2
        / 32
        / np.pi
        / (constants.m_e + constants.m_p)
        / constants.k_B
        / (f_21 * units.Hz) ** 2
        / cosmo.H(z)
    ).to(units.K / units.M_sun * units.Mpc**3)
    return C_HI


def omega_hi_to_average_temp(omega_hi, z=0, cosmo=Planck18):
    """
    The average HI brightness temperature from given HI density.

    Parameters
    ----------
    omega_hi:float
        The HI density relative to the z=0 critical density.

    z: float, defulat 0.0.
        The redshift

    cosmo: cosmology, default Planck18.
        The cosmology used

    Returns
    -------
    t_bar: float.
        The average HI temperature in Kelvin
    """
    c_hi = coeff_hi_density_to_temp(z=z, cosmo=cosmo)
    t_bar = (c_hi * cosmo.critical_density0 * omega_hi).to("K").value
    return t_bar


def freq_to_redshift(freq):
    """
    Convert frequency of 21cm emission to redshift

    Parameters
    ----------
    freq: float

    Returns
    -------
    redshift: float
    """
    return f_21 / freq - 1


def redshift_to_freq(redshift):
    """
    Convert redshift to frequency

    Parameters
    ----------
    redshift: float

    Returns
    -------
    freq: float
    """
    return f_21 / (1 + redshift)


def get_ang_between_coord(ra1, dec1, ra2, dec2, unit="deg"):
    """
    Calculate the angle between two points on the sphere.

    Parameters
    ----------
        ra1: float array
            The RA of the first point
        dec1: float array
            The Dec of the first point
        ra2: float array
            The RA of the second point
        dec2: float array
            The Dec of the second point
        unit: str, default 'deg'.
    Returns
    -------
        result: float array.
            The angle in the specified unit.

    """
    vec1 = hp.ang2vec(ra1, dec1, lonlat=True)
    vec2 = hp.ang2vec(ra2, dec2, lonlat=True)
    result = (np.arccos((vec1 * vec2).sum(axis=-1)) * units.rad).to(unit).value
    return result.T


def generate_colored_noise(x_size, x_len, power_k_func, seed=None):
    """
    Generate random 1D gaussian fluctuations following a specific spectrum. This is similar to ``colorednoise`` package for generating colored noise. It is simply wrapping the :class:`powerbox.PowerBox` under the hood.
    Note that the Fourier convention used should be consistent with :py:mod:`np.fft`.

    Parameters
    ----------
        x_size: int
            The number of sampling.
        x_len: float
            The **total length** of the sampling.
        power_k_func: func
            The power spectrum of the random noise in Fourier space.
        seed: int, default None
            The seed number for random generator for sampling. If None, a random seed is used.

    Returns
    -------
        rand_arr: float array.
            The random noise.
    """

    pb = PowerBox(
        x_size,
        power_k_func,
        dim=1,
        boxlength=x_len,
        a=0.0,
        b=2 * np.pi,
        seed=None,
    )
    rand_arr = pb.delta_x()
    return rand_arr


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def read_healpix_fits(file):
    hp_map = hp.read_map(file)
    hp_nside = hp.get_nside(hp_map)
    with fits.open(file) as hdul:
        header = hdul[1].header
        map_unit = units.Unit(header["TUNIT1"])
        map_freq = units.Quantity(header["FREQ"]).to("Hz").value
    return hp_map, hp_nside, map_unit, map_freq


def get_wcs_coor(wcs, xx, yy, ang_unit="deg"):
    """
    Retrieve RA and Dec coordinates of pixels in the WCS.

    Parameters
    ----------
        wproj: :class:`astropy.wcs.WCS` object.
            The two-dimensional wcs object for the map.
        xx: array
            The first indx of the pixel.
        yy: array
            The second indx of the pixel.

    Returns
    -------
        ra: array.
            The RA of the pixels.
        dec: array.
            The Dec of the pixels.
    """
    assert wcs.naxis == 2, "input wcs must be 2-dimensional."
    coor = wcs.pixel_to_world(xx, yy)
    ra = getattr(coor.ra, ang_unit)
    dec = getattr(coor.dec, ang_unit)
    return ra, dec


def pcaclean(
    signal,
    N_fg,
    weights=None,
    return_analysis=False,
    mean_centre=False,
    los_axis=-1,
    return_A=False,
    mean_centre_weights=None,
):
    r"""
    Performs PCA cleaning of the map data. If ``mean_centre`` is set to ``True``,
    then the input signal is first mean-centered so that

    .. math::
        \vec{d}^{\rm mc} = \vec{d} - \langle  \vec{d} \rangle,

    where the mean of the data vector at a specific channel i is

    .. math::
        \langle \vec{d} \rangle_i = \frac{\sum_a w_{ia} d_{ia}}{\sum_a w_{ia}},

    where a loops over each sampling in channel i and :math:`w` is the weight of each element.

    A covariance matrix is then calculated on the mean-centered data

    .. math::
        C_{ij} = \frac{\sum_{a,b }w_{ia} d^{\rm mc}_{ia}
             w_{jb} d^{\rm mc}_{jb}}{\sum_{a,b }w_{ia}w_{jb}},

    over which the eigenvalue decomposition is performed.
    See Section 4.3. of MeerKLASS Collaboration 2024 [1] for more details.

    Note that, in the rigorous definition, the data vector should be zero-meaned,
    and the weights used to calculate the mean and the covariance should be the same.
    However, in practice, many people don't remove the mean of the data.
    Some use one type of weights (often just binary masks) for mean calculation, and then
    use different weights for covariance calculation. While it is not encouraged, that flexibility
    is allowed in the function, by setting a different weight ``mean_centre_weights``.

    Parameters
    ----------
        signal: array
            The input signal to be cleaned.
        N_fg: int.
            Number of modes to be removed.
        weights: array, default None
            The weights of each element in the signal.
            Default will set uniform weights for each element.
        return_analysis: bool, default False
            If True, instead of residual maps the function will return eigenanalysis quantities.
        mean_centre: bool, default False
            Whether to mean-center the input data vector
        los_axis: int, default -1.
            Which axis is the line-of-sight, i.e. spectral axis.
        return_A: bool, default False.
            Whether to return the mixing matrix A.
        mean_centre_weights: array, default None.
            The weights of each element for mean calculation.
            Default follows the ``weights`` argument.

    Returns
    -------
        residual: array.
            The residual after PCA cleaning.
        A: array, if ``return_A=True``.
            The mixing matrix.
        covariance: array, if ``return_analysis=True``.
            The covariance matrix of the input signal.
        eignumb: array, if ``return_analysis=True``.
            The number indexing of the eigenvalues starting from 1.
        eigenval: array, if ``return_analysis=True``.
            The eigenvalues of the covariance matrix.
        V: array, if ``return_analysis=True``.
            The eigenvectors of the covariance matrix.

    References
    ----------
    .. [1] MeerKLASS collab, "MeerKLASS L-band deep-field intensity maps: entering the HI dominated regime", https://arxiv.org/abs/2407.21626.
    """
    assert len(signal.shape) == 3, "map must be 3D."
    if los_axis < 0:
        # change -1 to 2
        los_axis = 3 + los_axis
    # make sure los is the fist axis
    axes = [0, 1, 2]
    axes.remove(los_axis)
    axes = [
        los_axis,
    ] + axes
    # transpose map data
    signal = np.transpose(signal, axes=axes)
    nz, nx, ny = signal.shape
    signal = signal.reshape((nz, -1))
    if weights is not None:
        weights = np.transpose(weights, axes=axes)
        weights = weights.reshape((nz, -1))
    else:
        weights = np.ones_like(signal)

    if mean_centre_weights is not None:
        mean_centre_weights = np.transpose(mean_centre_weights, axes=axes)
        mean_centre_weights = mean_centre_weights.reshape((nz, -1))
    if mean_centre:
        if mean_centre_weights is None:
            signal = (
                signal
                - np.sum(signal * weights, 1)[:, None] / np.sum(weights, 1)[:, None]
            )
        else:
            signal = (
                signal
                - np.sum(signal * mean_centre_weights, 1)[:, None]
                / np.sum(mean_centre_weights, 1)[:, None]
            )
    ### Covariance calculation:
    covariance = (
        np.einsum("ia,ja->ij", signal * weights, signal * weights)
    ) / np.einsum("ia,ja->ij", weights, weights)
    V = np.linalg.eigh(covariance)[1][
        :, ::-1
    ]  # Eigenvectors from covariance matrix with most dominant first
    if return_analysis:
        eigenval = np.linalg.eigh(covariance)[0]
        eignumb = np.linspace(1, len(eigenval), len(eigenval))
        eigenval = eigenval[::-1]  # Put largest eigenvals first
        return covariance, eignumb, eigenval, V
    A = V[:, :N_fg]  # Mixing matrix, first N_fg most dominant modes of eigenvectors
    S = np.dot(
        A.T, signal
    )  # not including weights in mode subtraction (as per Yi-Chao's approach)
    residual = signal - np.dot(A, S)
    residual = np.reshape(residual, (nz, nx, ny))
    residual = np.transpose(residual, axes=np.argsort(axes))
    if return_A:
        return residual, A
    return residual


def radec_to_indx(ra_arr, dec_arr, wproj, to_int=True, ang_unit="deg"):
    coor = SkyCoord(ra_arr, dec_arr, unit=ang_unit)
    indx_1, indx_2 = wproj.world_to_pixel(coor)
    if to_int:
        indx_1 = np.round(indx_1).astype("int")
        indx_2 = np.round(indx_2).astype("int")
    return indx_1, indx_2


def convert_hpmap_in_jy_to_temp(hp_map, freq):
    nside = hp.get_nside(hp_map)
    hp_map = jy_to_kelvin(hp_map, hp.nside2resol(nside), freq)
    return hp_map


def healpix_to_wcs(hp_map, xx, yy, wcs):
    """
    Project the healpix map onto a 2-D wcs grid.
    Map has to be in temperature unit.
    """
    nside = hp.get_nside(hp_map)
    ra_map, dec_map = get_wcs_coor(wcs, xx, yy)
    pix_indx = hp.ang2pix(nside, ra_map, dec_map, lonlat=True)
    output_map = hp_map[pix_indx]
    return output_map


def rebin_spectrum(spectrum, rebin_width=3, mode="avg"):
    assert spectrum.size % 2 == 1
    assert rebin_width % 2 == 1
    rebin_pad = spectrum.size % rebin_width
    if rebin_pad % 2 == 1:
        rebin_pad += rebin_width
    rebin_pad = rebin_pad // 2
    spectrum_rebin = (
        spectrum[rebin_pad:-rebin_pad].reshape((-1, rebin_width)).mean(axis=-1)
    )
    if mode == "sum":
        spectrum_rebin *= rebin_width
    return spectrum_rebin


def hod_obuljen18(
    logmh,
    m0h=9.52,
    mminh=11.27,
    alpha=0.44,
    cosmo=Planck18,
    input_has_h=True,
    output_has_h=False,
):
    r"""
    HI-halo mass relation reported in Obuljen et al. (2018) [1].
    The default settings assume the mass is in M_sun/h, and returns mass **without** h unit.
    Note that, if you want input without h unit, you must change the ``m0h`` and ``mminh`` manually as well.
    The HI-halo mass relation follows

    .. math::
        M_{\rm HI} (M_h) = M_0 (M_h/M_{\rm min})^\alpha \, {\rm exp}[-M_{\rm min}/M_h],



    Parameters
    ----------
        logmh: float.
            input halo mass in log10
        m0h: optional, default 9.52.
            The :math:`M_0` parameter in the HOD in log10
        mminh: optional, default 11.27.
            The :math:`M_{min}` parameter in the HOD in log10
        alpha: optional, default 0.44.
            The :math:`\alpha` parameter in the HOD.
        cosmo: optional, default Planck18.
            The default cosmology, used only for h unit.
        input_has_h: optional, default True.
            Whether the input mass has h unit.
        output_has_h: optional, default False.
            Whether the output mass has h unit.

    Returns
    -------
        The HI mass for the input halo mass.

    References
    ----------
    .. [1] Obuljen, A. et al., "The H I content of dark matter haloes at z ≈ 0 from ALFALFA ", https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.5124O.
    """
    input_h_unit = 1.0 if input_has_h else cosmo.h
    output_h_unit = 1.0 if output_has_h else 1 / cosmo.h
    marr = 10**logmh * input_h_unit  # in Msun/h
    himass = 10 ** (m0h) * (marr / 10**mminh) ** alpha * np.exp(-(10**mminh) / marr)
    return himass * output_h_unit
