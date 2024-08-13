from meer21cm import Specification
from astropy.cosmology import Planck18, Planck15
import numpy as np
from astropy import units, constants
import pytest
from astropy.wcs.utils import proj_plane_pixel_area
from meer21cm.util import freq_to_redshift


def test_cosmo():
    spec = Specification()
    assert spec.h == Planck18.h


def test_update_pars():
    spec = Specification()
    spec.cosmo = "Planck15"
    assert spec.cosmo is Planck15
    assert spec.h == Planck15.h


def test_defaults(test_nu, test_W):
    spec = Specification()
    assert np.allclose(spec.nu, test_nu)
    assert np.allclose(spec.map_has_sampling, np.ones(test_W.shape))
    assert np.allclose(spec.z_ch, freq_to_redshift(test_nu))
    assert np.allclose(spec.z, freq_to_redshift(test_nu).mean())


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
