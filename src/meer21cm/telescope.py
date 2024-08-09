import numpy as np
from .util import get_wcs_coor, get_ang_between_coord, freq_to_redshift
from astropy import units, constants
from astropy.wcs.utils import proj_plane_pixel_area
from scipy.signal import convolve
from astropy.cosmology import Planck18


def weighted_convolution(
    signal,
    kernel,
    weights,
    kernel_renorm=True,
    los_axis=-1,
):
    r"""
    Perform weighted convolution of signal. The weighted convolution of the signal is defined as

    .. math::
        \tilde{s} = [(s \cdot w) * b]/[w * b],

    where :math:`s` is the signal, :math:`w` is the weight,
    :math:`b` is the convolution kernel and :math:`w` denotes convolution.

    The convolution also creates new weights for the output signal so that

    .. math::
        \tilde{w} = [w * b]^2 / [w * b^2]

    Parameters
    ----------
        signal: float.
            The input signal to be convolved
        kernel: float.
            The convolution kernel
        weights: float.
            The weights for the signal
        kernel_renorm: boolean, default True.
            Whether to renormalise the kernel so that the sum of the kernel is one.
            Should be set to ``True`` for temperature and ``False`` for flux density.
        los_axis: int, default -1.
            which axis is the los.


    Returns
    -------
        conv_signal: float.
            The convolved signal.
        conv_weights: float.
            The convolved weights.
    """
    if los_axis < 0:
        los_axis += 3
    # make sure los is the last axis
    axes = [0, 1, 2]
    axes.remove(los_axis)
    axes = axes + [
        los_axis,
    ]
    signal = np.transpose(signal * weights, axes=axes)
    kernel = np.transpose(kernel, axes=axes)
    weights = np.transpose(weights, axes=axes)
    if kernel_renorm:
        kernel /= kernel.sum(axis=(0, 1))[None, None, :]
    kernel_square = kernel**2
    kernel_square /= kernel_square.sum(axis=(0, 1))[None, None, :]
    conv_signal = np.zeros_like(signal)
    conv_variance = np.zeros_like(signal)
    for ch_i in range(signal.shape[-1]):
        weight_renorm = convolve(
            weights[:, :, ch_i],
            kernel[:, :, ch_i],
            mode="same",
        )
        weight_renorm[weight_renorm == 0] = np.inf
        conv_signal[:, :, ch_i] = (
            convolve(
                signal[:, :, ch_i],
                kernel[:, :, ch_i],
                mode="same",
            )
            / weight_renorm
        )
        conv_variance[:, :, ch_i] = (
            convolve(
                weights[:, :, ch_i],
                kernel_square[:, :, ch_i],
                mode="same",
            )
            / weight_renorm**2
        )
    conv_variance[conv_variance == 0] = np.inf
    conv_weights = 1 / conv_variance
    conv_weights = np.transpose(conv_weights, axes=axes)
    conv_signal = np.transpose(conv_signal, axes=axes)
    return conv_signal, conv_weights


def gaussian_beam(sigma):
    r"""
    Returns a Gaussian beam function

    .. math::
        B(\theta) = {\rm exp}[-\frac{\theta^2}{2\sigma^2}]

    when the beam width :math:`\sigma` is specified.

    Parameters
    ----------
        sigma: float.
            The width of the gaussian beam profile.
    Returns
    -------
        beam_func: function.
            The beam function.
    """
    return lambda x: np.exp(-(x**2) / 2 / sigma**2)


def isotropic_beam_profile(xdim, ydim, wproj, beam_func, ang_unit=units.deg):
    """
    Generate an isotropic image of the beam for given ``wproj`` and ``beam_func``. The image can later be used to convolve or deconvolve with intensity maps.

    Parameters
    ----------
        xdim: int.
            The number of pixels in the first axis.
        ydim: int.
            The number of pixels in the second axis.
        wproj: :class:`astropy.wcs.WCS` object.
            The two-dimensional wcs object for the map.
        beam_func: function.
            The beam function.
        ang_unit: str or :class:`astropy.units.Unit`.
            The unit of input values for ``beam_func``.
    Returns
    -------
        beam_image: float array.
            The image of the beam.
    """
    xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim), indexing="ij")
    ra, dec = get_wcs_coor(wproj, xx, yy)
    ra_cen, dec_cen = get_wcs_coor(wproj, xdim // 2, ydim // 2)
    ang_dist = (
        (get_ang_between_coord(ra, dec, ra_cen, dec_cen) * units.deg).to(ang_unit).value
    )
    return beam_func(ang_dist)


def dish_beam_sigma(dish_diameter, nu, gamma=1.0, ang_unit=units.deg):
    r"""
    Calculate the beam size of a dish telescope assuming

    .. math::
        \theta_{\rm FWHM} = \gamma \frac{\lambda}{D},

    where :math:`\theta_{\rm FWHM}` is the FWHM of the beam,
    :math:`\gamma` is the aperture efficiency,
    :math:`\lambda` is the observing wavelength,
    and D is the dish diameter.

    The sigma of the Gaussian beam is then
    :math:`\sigma = \theta_{\rm FWHM}/ 2\sqrt{2 {\rm ln}2}`.

    Parameters
    ----------
        dish_diameter: float.
            The diameter of the dish in metre.
        nu: float.
            The observing frequency in Hz.
        gamma: float, default 1.0.
            The aperture efficiency.
        ang_unit: str or :class:`astropy.units.Unit`, default ``deg``.
            The unit of the output.
    Returns
    -------
        beam_sigma: float.
            The sigma of the beam.
    """
    beam_fwhm = (
        constants.c / (nu * units.Hz * dish_diameter * units.m) * units.rad
    ).to(ang_unit).value * gamma
    beam_sigma = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    return beam_sigma


def cmb_temperature(nu, tcmb0=Planck18.Tcmb0.value):
    """
    Calculate the background CMB temperature at given frequencies.

    Parameters
    ----------
        nu: float.
            The observing frequency in Hz.
        tcmb0: float, default ``Planck18.Tcmb0.value``.
            The background CMB temperature at z=0.
    Returns
    -------
        tcmb: float.
            The CMB temperature at given frequencies in Kelvin.
    """
    redshift = freq_to_redshift(nu)
    return tcmb0 * (1 + redshift)


def receiver_temperature_meerkat(nu):
    """
    The receiver temperature of MeerKAT.

    Parameters
    ----------
        nu: float.
            The observing frequency in Hz.
    Returns
    -------
        Trx: float.
            The receiver temperature at given frequencies in Kelvin.
    """
    Trx = 7.5 + 10 * (nu / 1e9 - 0.75) ** 2
    return Trx


def galaxy_temperature(nu, tgal_408MHz=25, sp_indx=-2.75):
    """
    The temperature template of the Milky Way.

    Parameters
    ----------
        nu: float.
            The observing frequency in Hz.
        tgal_408MHz: float.
            The average galaxy temperature at 408MHz in Kelvin.
        sp_indx: float.
            The spectral index to extrapolate it to input frequencies.
    Returns
    -------
        Tgal: float.
            The galaxy temperature at given frequencies in Kelvin.
    """
    Tgal = tgal_408MHz * (nu / 408 / 1e6) ** sp_indx
    return Tgal
