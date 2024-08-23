from meer21cm import CosmologyCalculator
from astropy.cosmology import Planck18, Planck15
import numpy as np
import camb


def test_cosmo():
    coscal = CosmologyCalculator()
    assert coscal.h == Planck18.h
    # test input sigma8 resulting in consistent As with Planck18
    coscal.camb_pars
    assert np.abs(np.log(1e10 * coscal.As) - 3.047) < 1e-3
    coscal.average_hi_temp


def test_update_pars():
    coscal = CosmologyCalculator()
    coscal.camb_pars
    As = coscal.As
    coscal.cosmo = "Planck15"
    coscal.cosmo
    coscal.camb_pars
    assert coscal.h == Planck15.h
    assert coscal.As != As


def test_matter_power():
    coscal = CosmologyCalculator(nonlinear="both")
    pars = coscal.camb_pars
    pars.set_matter_power(
        redshifts=np.unique(np.array([0.0, coscal.z])).tolist(), kmax=2.0
    )
    pars.NonLinear = getattr(camb.model, "NonLinear_" + coscal.nonlinear)
    results = camb.get_results(pars)
    k_test, z, pk_test = results.get_nonlinear_matter_power_spectrum(
        hubble_units=False, k_hunit=False
    )
    ksel = (k_test > coscal.kmin) * (k_test < coscal.kmax)
    k_test = k_test[ksel]
    pk_test = pk_test[-1, ksel]
    coscal.get_matter_power_spectrum()
    pk_interp = coscal.matter_power_spectrum_fnc(k_test)
    assert np.abs((pk_test - pk_interp) / pk_test).max() < 5e-3
