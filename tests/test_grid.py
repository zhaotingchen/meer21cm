import matplotlib.pyplot as plt
from meer21cm.grid import *
import pytest


@pytest.mark.parametrize("window", list(allowed_window_scheme))
def test_uniform_grids(window):
    box_len = np.array([10, 10, 10])
    ndim_rg = np.array([10, 10, 10])
    pos_arr = np.zeros((10, 10, 10, 3))
    pos_arr[:, :, :, 0] += np.arange(10)[:, None, None] + 0.5
    pos_arr[:, :, :, 1] += np.arange(10)[None, :, None] + 0.5
    pos_arr[:, :, :, 2] += np.arange(10)[None, None, :] + 0.5
    test_map, test_weights, test_counts = project_particle_to_regular_grid(
        pos_arr,
        box_len,
        ndim_rg,
        compensate=True,
        window=window,
    )
    assert np.allclose(test_map, np.ones_like(test_map))
    assert np.allclose(test_weights, np.ones_like(test_map))
    assert np.allclose(test_counts, np.ones_like(test_map))
    test_map, test_weights, test_counts = project_particle_to_regular_grid(
        pos_arr,
        box_len,
        ndim_rg,
        compensate=True,
        window=window,
        particle_value=np.ones(10**3),
        particle_weights=np.random.uniform(1, 2, size=10**3),
    )
    assert np.allclose(test_map, np.ones_like(test_map))
    assert np.allclose(test_counts, np.ones_like(test_map))


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
