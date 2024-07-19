import numpy as np
from scipy.signal import convolve
from .util import get_wcs_coor, get_ang_between_coord
from astropy import units, constants
from astropy.wcs.utils import proj_plane_pixel_area


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


def dish_beam_sigma(dish_diameter, nu, gamma=1.0, unit=units.rad):
    """
    Calculate the beam size of a dish telescope assuming

    .. math::
        \theta_{\rm FWHM} = \gamma \frac{\lambda}{D}

    , where :math:`\theta_{\rm FWHM}` is the FWHM of the beam,
    :math:`\gamma` is the aperture efficiency,
    :math:`\lambda` is the observing wavelength,
    and D is the dish diameter.

    The sigma of the Gaussian beam is then
    :math:`\sigma = \theta_{\rm FWHM}/ 2\sqrt{2 {\rm ln}2}`
    """
    beam_fwhm = (
        constants.c / (nu * units.Hz * dish_diameter * units.m) * units.rad
    ).to(unit).value * gamma
    beam_sigma = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    return beam_sigma
