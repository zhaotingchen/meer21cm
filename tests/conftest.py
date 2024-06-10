import pytest
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

@pytest.fixture
def test_wcs():
    map_file = 'data/test_fits.fits'
    wcs = WCS(map_file)
    return wcs

@pytest.fixture
def test_wproj():
    map_file = 'data/test_fits.fits'
    wcs = WCS(map_file)
    wproj = wcs.dropaxis(-1)
    return wproj

@pytest.fixture
def test_W():
    map_file = 'data/test_W.npy'
    return np.load(map_file)[:,:,0][:,:,None]

@pytest.fixture
def test_nu():
    file = 'data/test_nu.npy'
    return np.load(file)

@pytest.fixture
def test_GAMA_range():
    raminGAMA,ramaxGAMA = 339,351
    decminGAMA,decmaxGAMA = -35,-30
    return (raminGAMA,ramaxGAMA),(decminGAMA,decmaxGAMA)