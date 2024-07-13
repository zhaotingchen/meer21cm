import numpy as np
import pytest
from astropy.cosmology import Planck18
from hiimtool.basic_util import himf_pars_jones18, centre_to_edges, f_21
from meer21cm.util import *
import sys

python_ver = sys.version_info


def test_generate_colored_noise():
    if python_ver < (3, 9):
        return 1
    rand_arr = [
        generate_colored_noise(100, 100, lambda k: np.ones_like(k)) for i in range(1000)
    ]
    rand_arr = np.array(rand_arr)
    assert np.allclose(rand_arr.mean(), 0.0)
    assert np.abs(rand_arr.std() - 1.0) < 0.2


def test_get_default_args():
    def test_func(x, arg1=1):
        return 1

    defaults = get_default_args(test_func)
    assert len(defaults) == 1
    for k, v in defaults.items():
        assert k == "arg1"
        assert v == 1


def test_get_wcs_coor(test_wproj, test_wcs):
    with pytest.raises(Exception) as e_info:
        get_wcs_coor(test_wcs, np.arange(10), np.arange(10))
    get_wcs_coor(test_wproj, np.arange(10), np.arange(10))


def test_PCAclean():
    test_arr = np.random.normal(size=(10))
    with pytest.raises(Exception) as e_info:
        PCAclean(test_arr, 1, returnAnalysis=True)
    test_arr = np.random.normal(size=(200, 200, 10))
    C, eignumb, eigenval, V = PCAclean(test_arr, 1, returnAnalysis=True)
    assert (np.abs(C - np.eye((test_arr.shape[-1]))) < 0.1).mean() == 1
    test_arr = np.random.normal(size=(10, 200, 200))
    C, eignumb, eigenval, V = PCAclean(
        test_arr,
        1,
        returnAnalysis=True,
        los_axis=0,
        w=np.ones_like(test_arr),
        W=np.ones_like(test_arr),
        MeanCentre=True,
    )
    assert C.shape == (10, 10)
    assert V.shape == (10, 10)
    assert np.allclose(eignumb, np.linspace(1, 10, 10))
    assert np.std(eigenval) < 0.1
    res_arr = PCAclean(
        test_arr,
        1,
        returnAnalysis=False,
        los_axis=0,
        w=np.ones_like(test_arr),
        W=np.ones_like(test_arr),
        MeanCentre=True,
    )
    assert res_arr.shape == test_arr.shape
    assert np.abs((res_arr).mean()) < 1e-3
    res_arr, A_mat = PCAclean(
        test_arr,
        1,
        returnAnalysis=False,
        los_axis=0,
        w=np.ones_like(test_arr),
        W=np.ones_like(test_arr),
        MeanCentre=True,
        return_A=True,
    )
    res_arr, A_mat = PCAclean(
        test_arr,
        1,
        returnAnalysis=False,
        los_axis=0,
        MeanCentre=True,
        return_A=True,
    )


def test_radec_to_indx(test_wproj):
    indx_i, indx_j = radec_to_indx(0, -30, test_wproj, to_int=True)
    indx_1, indx_2 = radec_to_indx(0, -30, test_wproj, to_int=False)
    assert np.round(indx_1) == indx_i
    assert np.round(indx_2) == indx_j


def test_rebin_spectrum():
    test_spectrum = np.zeros(503)
    test_spectrum[503 // 2] = 1.0
    test_rebin = rebin_spectrum(test_spectrum, rebin_width=3)
    assert test_rebin.sum() == 1 / 3
    assert test_rebin.size == 503 // 3
    test_rebin = rebin_spectrum(test_spectrum, rebin_width=3, mode="sum")
    assert test_rebin.sum() == 1


def test_find_rotation_matrix():
    vec = np.random.randn(3)
    vec /= np.sqrt(np.sum(vec**2))
    rot_mat = find_rotation_matrix(vec)
    assert np.allclose(rot_mat @ vec, [0, 0, 1])


def test_minimum_enclosing_box_of_lightcone():
    ra_rand = np.random.uniform(0, 359.9, 1)[0]
    dec_rand = np.random.uniform(-90, 89.9, 1)[0]
    ra_test = np.array((ra_rand, ra_rand + 0.1))
    dec_test = np.array((dec_rand, dec_rand + 0.1))
    freq_test = np.array((f_21 / (1 + 0.5001), f_21 / (1 + 0.5)))
    (xmin, ymin, zmin, xlen, ylen, zlen, rot_mat) = minimum_enclosing_box_of_lightcone(
        ra_test, dec_test, freq_test
    )
    cov_dist = Planck18.comoving_distance(0.5)
    cov_scale = Planck18.comoving_distance(0.5001) - Planck18.comoving_distance(0.5)
    assert ((xlen - cov_dist * 0.1 * np.pi / 180) / xlen).to("").value < 0.01
    assert ((ylen - cov_dist * 0.1 * np.pi / 180) / ylen).to("").value < 0.01
    assert ((cov_scale - zlen) / cov_scale).to("").value < 0.01
    test_vec = rot_mat @ np.array(
        [(xmin + xlen / 2).value, (ymin + ylen / 2).value, (zmin + zlen / 2).value]
    )
    test_vec /= np.sqrt(np.sum(test_vec**2))
    ra_cen, dec_cen = hp.vec2ang(test_vec, lonlat=True)
    assert np.abs(ra_rand - ra_cen) < 0.1
    assert np.abs(dec_rand - dec_cen) < 0.1


def test_hod_obuljen18():
    mass_1 = np.log10(hod_obuljen18(11.27))
    mass_2 = np.log10(10**9.52 * np.exp(-1) / Planck18.h)
    assert np.allclose(mass_1, mass_2)
