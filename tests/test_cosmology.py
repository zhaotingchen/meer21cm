from meer21cm import CosmologyCalculator
from astropy.cosmology import Planck18, Planck15
import numpy as np


def test_cosmo():
    coscal = CosmologyCalculator()
    assert coscal.h == Planck18.h
    # test input sigma8 resulting in consistent As with Planck18
    coscal.camb_pars
    assert np.abs(np.log(1e10 * coscal.As) - 3.047) < 1e-3


def test_update_pars():
    coscal = CosmologyCalculator()
    coscal.camb_pars
    As = coscal.As
    coscal.cosmo = "Planck15"
    coscal.cosmo
    coscal.camb_pars
    assert coscal.h == Planck15.h
    assert coscal.As != As
