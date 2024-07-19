from meer21cm.telescope import *
import numpy as np
from astropy import constants, units


def test_gaussian_beam():
    gaussian_func = gaussian_beam(1)
    x = np.linspace(-100, 100, 1001)
    assert np.allclose(gaussian_func(x), np.exp(-(x**2) / 2))


def test_isotropic_beam_profile(test_wproj, test_W):
    xdim, ydim, _ = test_W.shape
    beam_func = gaussian_beam(0.5)
    beam_image = isotropic_beam_profile(xdim, ydim, test_wproj, beam_func)
    assert np.max(beam_image) == 1.0
    pix_resol = np.sqrt(proj_plane_pixel_area(test_wproj))
    beam_pix = np.exp(-(pix_resol**2) / 2 / 0.5**2)
    assert np.abs(beam_image[xdim // 2, ydim // 2 - 1] - beam_pix) < 0.01
    assert np.abs(beam_image[xdim // 2, ydim // 2 + 1] - beam_pix) < 0.01
    assert np.abs(beam_image[xdim // 2 - 1, ydim // 2] - beam_pix) < 0.01
    assert np.abs(beam_image[xdim // 2 + 1, ydim // 2] - beam_pix) < 0.01


def test_dish_beam_sigma():
    nu = 1e9  # Hz
    dish_size = 10  # m
    beam_sigma = dish_beam_sigma(dish_size, nu)
    fwhm = (constants.c / (nu * units.Hz * dish_size * units.m)).to("").value
    assert np.allclose(beam_sigma * 2 * np.sqrt(2 * np.log(2)), fwhm)
