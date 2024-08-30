from meer21cm import Specification
from astropy.cosmology import Planck18, Planck15
import numpy as np
from astropy import units, constants
import pytest
from astropy.wcs.utils import proj_plane_pixel_area
from meer21cm.util import freq_to_redshift, center_to_edges


def test_cosmo():
    spec = Specification()
    assert spec.h == Planck18.h


def test_update_pars():
    spec = Specification()
    # test string input
    spec.cosmo = "Planck15"
    assert spec.cosmo is Planck15
    assert spec.h == Planck15.h
    # test direct input
    spec.cosmo = Planck15


def test_defaults(test_nu, test_W):
    spec = Specification()
    assert np.allclose(spec.nu, test_nu)
    assert np.allclose(spec.map_has_sampling, np.ones(test_W.shape))
    assert np.allclose(spec.z_ch, freq_to_redshift(test_nu))
    assert np.allclose(spec.z, freq_to_redshift(test_nu).mean())
    x_res = 0.3 * np.pi / 180 * Planck18.comoving_distance(spec.z).value
    assert np.allclose(spec.pix_resol_in_mpc, x_res)
    los = Planck18.comoving_distance(spec.z_ch).value
    z_res = (los[0] - los[-1]) / len(spec.nu)
    assert np.allclose(z_res, spec.los_resol_in_mpc)


def test_unit_conversion():
    spec = Specification(map_unit=units.mK)
    assert spec.map_unit_type == "T"
    spec = Specification(map_unit=units.Jy)
    assert spec.map_unit_type == "F"
    with pytest.raises(Exception):
        spec = Specification(map_unit=units.m)


def test_velocity(test_nu, test_wproj):
    spec = Specification()
    assert np.allclose(spec.dvdf_ch, (constants.c / test_nu).to("km/s").value)
    assert np.allclose(
        spec.vel_resol_ch,
        (constants.c / test_nu).to("km/s").value * np.diff(test_nu).mean(),
    )
    assert np.allclose(spec.vel_resol, spec.vel_resol_ch.mean())
    assert np.allclose(spec.dvdf, spec.dvdf_ch.mean())
    assert np.allclose(spec.freq_resol, np.diff(test_nu).mean())
    assert np.allclose(spec.pixel_area, proj_plane_pixel_area(test_wproj))
    assert np.allclose(spec.pix_resol, np.sqrt(spec.pixel_area))


def test_read_fits(test_fits):
    sp = Specification()
    # should be None
    sp.read_from_fits()
    sp.read_gal_cat()
    # set map file
    sp.map_file = test_fits
    sp.num_pix_x = 1
    sp.num_pix_y = 1
    sp.read_from_fits()
    assert np.allclose(sp.data.shape, (133, 73, 2))
    assert np.allclose(sp.counts.shape, (133, 73, 2))
    assert np.allclose(sp.ra_map.shape, (133, 73))
    assert np.allclose(sp.dec_map.shape, (133, 73))

    assert np.allclose(sp.map_has_sampling.shape, (133, 73, 2))
    assert sp.num_pix_x == 133
    assert sp.num_pix_y == 73
    assert len(sp.nu) == 2
    sp.W_HI
    assert np.allclose(sp.w_HI, sp.counts)
    sp.weighting = "uniform"
    sp.read_from_fits()
    assert np.allclose(sp.w_HI, sp.counts > 0)


def test_gal_readin(test_gal_fits):
    sp = Specification()
    sp.gal_file = test_gal_fits
    sp.read_gal_cat()
    nu_edges = center_to_edges(sp.nu)
    assert np.mean(sp.freq_gal >= nu_edges[sp.ch_id_gal]) == 1
    assert np.mean(sp.freq_gal <= nu_edges[sp.ch_id_gal + 1]) == 1
    assert len(sp.ra_gal) == len(sp.z_gal)
    assert len(sp.dec_gal) == len(sp.z_gal)
