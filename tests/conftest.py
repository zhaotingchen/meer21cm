import pytest
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os
import meerstack

data_dir = meerstack.__file__.rsplit("/", 1)[0] + "/data/"


@pytest.fixture
def test_haslam_map():
    map_file = data_dir + "haslam408_dsds_Remazeilles2014.fits"
    return map_file


@pytest.fixture
def test_gsm_1ghz_jy():
    map_file = data_dir + "gdsm16_1ghz_jy.fits"
    return map_file


@pytest.fixture
def test_gsm_1ghz():
    map_file = data_dir + "gdsm16_1ghz.fits"
    return map_file


@pytest.fixture
def test_wcs():
    map_file = data_dir + "test_fits.fits"
    wcs = WCS(map_file)
    return wcs


@pytest.fixture
def test_wproj():
    map_file = data_dir + "test_fits.fits"
    wcs = WCS(map_file)
    wproj = wcs.dropaxis(-1)
    return wproj


@pytest.fixture
def test_W():
    map_file = data_dir + "test_W.npy"
    return np.load(map_file)[:, :, 0][:, :, None]


@pytest.fixture
def test_nu():
    file = data_dir + "test_nu.npy"
    return np.load(file)


@pytest.fixture
def test_GAMA_range():
    raminGAMA, ramaxGAMA = 339, 351
    decminGAMA, decmaxGAMA = -35, -30
    return (raminGAMA, ramaxGAMA), (decminGAMA, decmaxGAMA)
