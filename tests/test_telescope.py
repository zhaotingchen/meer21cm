from meer21cm.telescope import *
import numpy as np
from astropy import constants, units
from astropy.cosmology import Planck18
from meer21cm.util import f_21
from meer21cm import Specification


def test_cos_beam():
    # input fhmw/2 should give 0.5
    sigma = 0.5
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    assert np.abs(cos_beam(sigma)(fwhm / 2) - 0.5) < 1e-3


def test_beam(test_wproj):
    sp = Specification()
    beam_image = kat_beam(
        sp.nu,
        sp.wproj,
        sp.num_pix_x,
        sp.num_pix_y,
    )
    sigma_beam_ch = np.sqrt(beam_image.sum(axis=(0, 1)) / 2 / np.pi) * 0.3
    pars = np.polyfit(sp.nu / 1e6, sigma_beam_ch, 2)
    assert np.abs(pars[0] / pars[1]) < 1e-3
    assert np.abs(pars[0] / pars[2]) < 1e-3
    assert pars[1] < 0


def test_weighted_convolution(test_wproj, test_W):
    xdim, ydim, _ = test_W.shape
    test_image = np.random.normal(size=test_W.shape)
    test_image *= test_W
    beam_func = gaussian_beam(0.00000000001)
    beam_image = isotropic_beam_profile(xdim, ydim, test_wproj, beam_func)
    beam_cube = np.zeros_like(test_W)
    beam_cube += beam_image[:, :, None]
    conv_map, conv_mask = weighted_convolution(test_image, beam_cube, test_W)
    conv_map *= test_W
    conv_mask *= test_W
    assert np.allclose(conv_map, test_image)
    assert np.allclose(test_W, conv_mask)


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
    fwhm = (
        (constants.c / (nu * units.Hz * dish_size * units.m)).to("").value * 180 / np.pi
    )
    assert np.allclose(beam_sigma * 2 * np.sqrt(2 * np.log(2)), fwhm)


def test_cmb_temperature():
    zarr = np.linspace(0, 2, 101)
    nuarr = f_21 / (1 + zarr)
    temp1 = Planck18.Tcmb(zarr).value
    temp2 = cmb_temperature(nuarr)
    assert np.allclose(temp1, temp2)


def test_receiver_temperature_meerkat():
    nu = 0.75 * 1e9
    assert receiver_temperature_meerkat(nu) == 7.5
    nu = 0.85 * 1e9
    assert receiver_temperature_meerkat(nu) == 7.6


def test_galaxy_temperature():
    nu = np.linspace(0.5, 1, 11) * 1e9
    tgal = galaxy_temperature(nu, tgal_408MHz=0)
    assert np.allclose(tgal, np.zeros_like(tgal))
    tgal = galaxy_temperature(nu, sp_indx=-1)
    assert np.allclose(tgal, 25 * (408 * 1e6 / nu))
