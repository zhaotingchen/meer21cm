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
