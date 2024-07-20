import numpy as np
from astropy import constants, units
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import healpy as hp
from astropy.io import fits
from hiimtool.basic_util import check_unit_equiv, jy_to_kelvin, f_21
from astropy.cosmology import Planck18
import inspect
import sys

python_ver = sys.version_info[0] + sys.version_info[1] / 10
if python_ver >= 3.9:
    from powerbox import PowerBox


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
    rand_arr = None
    if python_ver >= 3.9:
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


def get_wcs_coor(wcs, xx, yy):
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
    ra = coor.ra.deg
    dec = coor.dec.deg
    return ra, dec


def pcaclean(
    signal,
    N_fg,
    weights=None,
    return_analysis=False,
    mean_centre=False,
    los_axis=-1,
    return_A=False,
):
    """
    Performs PCA cleaning of the map data.
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
    if mean_centre:
        signal = (
            signal - np.sum(signal * weights, 1)[:, None] / np.sum(weights, 1)[:, None]
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
    Residual = signal - np.dot(A, S)
    Residual = np.reshape(Residual, (nz, nx, ny))
    Residual = np.transpose(Residual, axes=np.argsort(axes))
    if return_A:
        return Residual, A
    return Residual


def radec_to_indx(ra_arr, dec_arr, wproj, to_int=True):
    coor = SkyCoord(ra_arr, dec_arr, unit="deg")
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


def find_rotation_matrix(vec):
    """
    find the rotation matrix to rotate the input vector to (0,0,1).
    """
    theta_rot = np.arctan2(vec[1], vec[0])
    rot_mat_1 = np.array(
        [
            [np.cos(-theta_rot), -np.sin(-theta_rot), 0],
            [np.sin(-theta_rot), np.cos(-theta_rot), 0],
            [0, 0, 1],
        ]
    )
    inter_vec = rot_mat_1 @ vec
    phi_rot = -np.arctan2(inter_vec[0], inter_vec[2])
    rot_mat_2 = np.array(
        [
            [np.cos(-phi_rot), 0, -np.sin(-phi_rot)],
            [0, 1, 0],
            [np.sin(-phi_rot), 0, np.cos(-phi_rot)],
        ]
    )
    return rot_mat_2 @ rot_mat_1


def minimum_enclosing_box_of_lightcone(ra_arr, dec_arr, freq, cosmo=Planck18):
    """
    not really minimum but should be okay.
    """
    ra_temp = ra_arr.copy()
    ra_temp[ra_temp > 180] -= 360
    ra_mean = ra_temp.mean()
    dec_mean = dec_arr.mean()
    mean_vec = hp.ang2vec(ra_mean, dec_mean, lonlat=True)
    rot_mat = find_rotation_matrix(mean_vec)
    z_arr = f_21 / freq - 1
    vec_arr = hp.ang2vec(ra_arr, dec_arr, lonlat=True)
    # rotate so that centre of field is the line-of-sight [0,0,1]
    vec_arr = np.einsum("ab,ib->ia", rot_mat, vec_arr)
    comov_dist_arr = cosmo.comoving_distance(z_arr)
    pos_arr = vec_arr[:, None, :] * comov_dist_arr[None, :, None]
    pos_arr = pos_arr.reshape((-1, 3))
    x_min, y_min, z_min = pos_arr.min(axis=0)
    x_max, y_max, z_max = pos_arr.max(axis=0)
    inv_rot = np.linalg.inv(rot_mat)
    return (x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min, inv_rot)


def hod_obuljen18(logmh, m0h=9.52, mminh=11.27, alpha=0.44, cosmo=Planck18):
    marr = 10**logmh  # in Msun/h
    himass = 10 ** (m0h) * (marr / 10**mminh) ** alpha * np.exp(-(10**mminh) / marr)
    return himass / cosmo.h
