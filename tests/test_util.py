import numpy as np
import pytest
from astropy.cosmology import Planck18
from hiimtool.basic_util import himf_pars_jones18, centre_to_edges, f_21
from meer21cm.util import *
import sys

python_ver = sys.version_info


def test_freq_redshift():
    assert freq_to_redshift(f_21) == 0.0
    assert freq_to_redshift(f_21 / 2) == 1.0
    assert redshift_to_freq(0.0) == f_21
    assert redshift_to_freq(1.0) == f_21 / 2


def test_get_ang_between_coord():
    ra1 = np.zeros(11)
    ra2 = np.array([0])
    dec1 = np.linspace(-30, -40, 11)
    dec2 = np.array([80])
    ang = get_ang_between_coord(ra1, dec1, ra2, dec2)
    assert np.allclose(ang.ravel(), dec2 - dec1)


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


def test_pcaclean():
    test_arr = np.random.normal(size=(10))
    with pytest.raises(Exception) as e_info:
        pcaclean(test_arr, 1, return_analysis=True)
    test_arr = np.random.normal(size=(200, 200, 10))
    C, eignumb, eigenval, V = pcaclean(test_arr, 1, return_analysis=True)
    assert (np.abs(C - np.eye((test_arr.shape[-1]))) < 0.1).mean() == 1
    test_arr = np.random.normal(size=(10, 200, 200))
    # test renorm
    C, eignumb, eigenval, V = pcaclean(
        test_arr,
        1,
        weights=2 * np.ones_like(test_arr),
        return_analysis=True,
        los_axis=0,
        mean_centre=True,
    )
    assert C.shape == (10, 10)
    assert V.shape == (10, 10)
    assert (np.abs(C - np.eye((test_arr.shape[0]))) < 0.1).mean() == 1
    assert np.allclose(eignumb, np.linspace(1, 10, 10))
    assert np.std(eigenval) < 0.1
    res_arr = pcaclean(
        test_arr,
        1,
        return_analysis=False,
        los_axis=0,
        weights=2 * np.ones_like(test_arr),
        mean_centre=True,
    )
    assert res_arr.shape == test_arr.shape
    assert np.abs((res_arr).mean()) < 1e-3
    res_arr, A_mat = pcaclean(
        test_arr,
        1,
        return_analysis=False,
        los_axis=0,
        mean_centre=True,
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
    vec_rand = hp.ang2vec(ra_rand, dec_rand, lonlat=True)
    rot_mat_rand = find_rotation_matrix(vec_rand)
    ang_div = 0.005 * np.pi / 180
    z_2 = np.cos(ang_div)
    vec_arr = np.zeros((4, 3))
    sign_i = [-1, 0, 1, 0]
    sign_j = [0, -1, 0, 1]
    for indx in range(4):
        vec_arr[indx] = np.linalg.inv(rot_mat_rand) @ np.array(
            [
                sign_i[indx] * np.sqrt(1 - z_2**2),
                sign_i[indx] * np.sqrt(1 - z_2**2),
                z_2,
            ]
        )
    ra_test, dec_test = hp.vec2ang(vec_arr, lonlat=True)
    freq_test = np.array((f_21 / (1 + 0.5001), f_21 / (1 + 0.5)))
    (
        xmin,
        ymin,
        zmin,
        xlen,
        ylen,
        zlen,
        rot_mat,
        pos_arr,
    ) = minimum_enclosing_box_of_lightcone(
        ra_test, dec_test, freq_test, return_coord=True
    )
    cov_dist = Planck18.comoving_distance(0.5).value
    cov_scale = (
        Planck18.comoving_distance(0.5001) - Planck18.comoving_distance(0.5)
    ).value
    assert np.abs((xlen - cov_dist * 0.01 * np.pi / 180) / xlen) < 1e-3
    assert np.abs((ylen - cov_dist * 0.01 * np.pi / 180) / ylen) < 1e-3
    assert np.abs((cov_scale - zlen) / cov_scale) < 1e-3
    test_vec = rot_mat @ np.array(
        [(xmin + xlen / 2), (ymin + ylen / 2), (zmin + zlen / 2)]
    )
    test_vec /= np.sqrt(np.sum(test_vec**2))
    ra_cen, dec_cen = hp.vec2ang(test_vec, lonlat=True)
    assert np.abs(ra_rand - ra_cen) < 0.05
    assert np.abs(dec_rand - dec_cen) < 0.05
    ra_test = np.ones((4, 2)) * ra_test[:, None]
    dec_test = np.ones((4, 2)) * dec_test[:, None]
    freq_test = np.ones((4, 2)) * freq_test[None, :]

    (
        xmin,
        ymin,
        zmin,
        xlen,
        ylen,
        zlen,
        rot_mat,
        pos_arr,
    ) = minimum_enclosing_box_of_lightcone(
        ra_test, dec_test, freq_test, tile=False, return_coord=True
    )
    assert np.abs((xlen - cov_dist * 0.01 * np.pi / 180) / xlen) < 1e-3
    assert np.abs((ylen - cov_dist * 0.01 * np.pi / 180) / ylen) < 1e-3
    assert np.abs((cov_scale - zlen) / cov_scale) < 1e-3
    test_vec = rot_mat @ np.array(
        [(xmin + xlen / 2), (ymin + ylen / 2), (zmin + zlen / 2)]
    )
    test_vec /= np.sqrt(np.sum(test_vec**2))
    ra_cen, dec_cen = hp.vec2ang(test_vec, lonlat=True)
    assert np.abs(ra_rand - ra_cen) < 0.05
    assert np.abs(dec_rand - dec_cen) < 0.05


def test_hod_obuljen18():
    mass_1 = np.log10(hod_obuljen18(11.27))
    mass_2 = np.log10(10**9.52 * np.exp(-1) / Planck18.h)
    assert np.allclose(mass_1, mass_2)
