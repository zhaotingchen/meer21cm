from meer21cm import CosmologyCalculator, Specification
from meer21cm.cosmology import CosmologyParameters
from astropy.cosmology import Planck18, Planck15
import numpy as np
import camb
from meer21cm.util import f_21
import pytest


def test_set_background():
    pars = CosmologyParameters()
    pars.get_derived_Ode()
    pars.set_astropy_cosmo()
    assert np.allclose(pars.cosmo.Onu0, Planck18.Onu0)
    assert np.allclose(pars.cosmo.Ogamma0, Planck18.Ogamma0)
    assert np.abs(pars.cosmo.Ok0) < 1e-5
    # test w0wa
    pars = CosmologyParameters(
        w0=-0.85,
        wa=0.1,
    )
    pars.get_derived_Ode()
    pars.set_astropy_cosmo()
    assert np.allclose(pars.cosmo.Onu0, Planck18.Onu0)
    assert np.allclose(pars.cosmo.Ogamma0, Planck18.Ogamma0)
    assert np.abs(pars.cosmo.Ok0) < 1e-5


@pytest.mark.parametrize("ps_type, accuracy", [("linear", 0.01), ("nonlinear", 0.05)])
def test_compare_matter_power(ps_type, accuracy):
    pars = CosmologyParameters(ps_type=ps_type)
    pkcamb = pars.get_matter_power_spectrum_camb()
    pkbacco = pars.get_matter_power_spectrum_bacco()
    # accuracy not great for nonlinear
    assert np.allclose(np.abs(pkcamb / pkbacco - 1) < accuracy, True)
    # test a different cosmology
    pars = CosmologyParameters(w0=-0.85, wa=-0.1, ps_type=ps_type)
    pkcamb = pars.get_matter_power_spectrum_camb()
    pkbacco = pars.get_matter_power_spectrum_bacco()
    # accuracy not great for nonlinear
    assert np.allclose(np.abs(pkcamb / pkbacco - 1) < accuracy, True)


def test_cosmo():
    coscal = CosmologyCalculator()
    assert coscal.h == Planck18.h
    # test input sigma8 resulting in consistent As with Planck18
    coscal.camb_pars
    assert np.abs(np.log(1e10 * coscal.As) - 3.047) < 1e-3
    # only test invoking, the function itself is tested in util
    t1, ohi1 = coscal.average_hi_temp, coscal.omegahi
    # test update omegahi, scales correctly
    coscal.omegahi = 5.5e-4
    np.allclose(coscal.average_hi_temp / t1, (coscal.omegahi / ohi1))
    # test another cosmology is included in util


def test_update_pars():
    coscal = CosmologyCalculator()
    coscal.camb_pars
    As = coscal.As
    coscal.cosmo = "Planck15"
    coscal.cosmo
    coscal.camb_pars
    assert coscal.h == Planck15.h
    # As has been updated
    assert coscal.As != As


@pytest.mark.parametrize("test_nonlinear", [("both"), ("none")])
def test_matter_power(test_nonlinear):
    coscal = CosmologyCalculator(nonlinear=test_nonlinear)
    pars = coscal.camb_pars
    pars.set_matter_power(
        redshifts=np.unique(np.array([0.0, coscal.z])).tolist(), kmax=2.0
    )
    pars.NonLinear = getattr(camb.model, "NonLinear_" + coscal.nonlinear)
    results = camb.get_results(pars)
    k_test, z, pk_test = results.get_matter_power_spectrum()
    k_test = k_test * coscal.h
    pk_test = pk_test / coscal.h**3
    ksel = (k_test > coscal.kmin) * (k_test < coscal.kmax)
    k_test = k_test[ksel]
    pk_test = pk_test[-1, ksel]
    pk_interp = coscal.matter_power_spectrum_fnc(k_test)
    assert np.abs((pk_test - pk_interp) / pk_test).max() < 5e-3


def test_inheritance():
    sp = Specification(cosmo=Planck15)
    coscal = CosmologyCalculator(**sp.__dict__)
    assert coscal.cosmo is Planck15


def test_cache():
    coscal = CosmologyCalculator()
    test1 = coscal.matter_power_spectrum_fnc(1)
    coscal.nu = [f_21, f_21]
    coscal.nu
    test2 = coscal.matter_power_spectrum_fnc(1)
    assert test1 != test2
    coscal.cosmo = "WMAP1"
    test3 = coscal.matter_power_spectrum_fnc(1)
    assert test3 != test2
