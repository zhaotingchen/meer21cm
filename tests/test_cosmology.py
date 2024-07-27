from meer21cm.cosmology import CosmologyCalculator
from astropy.cosmology import Planck18
import numpy as np


def test_cosmo():
    coscal = CosmologyCalculator()
    assert coscal.h == Planck18.h
    # test input sigma8 resulting in consistent As with Planck18
    coscal.camb_pars
    assert np.abs(np.log(1e10 * coscal.As) - 3.047) < 1e-3
