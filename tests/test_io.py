import pytest
import numpy as np
from meer21cm.io import *


def test_cal_freq(test_nu):
    assert np.diff(test_nu).mean() * 1e6 == meerkat_4k_delta_nu
    assert cal_freq(0) == meerkat_L_band_nu_min
    assert cal_freq(4096) == meerkat_L_band_nu_max


def test_filter_incomplete_los():
    input_array = np.ones((3, 3, 4))
    has_sampling = np.ones((3, 3, 4))
    has_sampling[0, 0, 0] = 0
    has_sampling[0, 0, 2] = 0
    has_sampling[1, 1, 1] = 0
    (output_array, sampling, weights, counts,) = filter_incomplete_los(
        input_array,
        has_sampling,
        has_sampling,
        has_sampling,
    )
    assert output_array[0, 0].mean() == 0.0
    assert output_array[1, 1].mean() == 0.0
    assert output_array.sum() == 28
    # swap los axis
    input_array = np.ones((3, 4, 3))
    has_sampling = np.ones((3, 4, 3))
    has_sampling[0, 0, 0] = 0
    has_sampling[0, 2, 0] = 0
    has_sampling[1, 1, 1] = 0
    (output_array, sampling, weights, counts,) = filter_incomplete_los(
        input_array,
        has_sampling,
        has_sampling,
        has_sampling,
        los_axis=1,
    )
    assert output_array[0, :, 0].mean() == 0.0
    assert output_array[1, :, 1].mean() == 0.0
    assert output_array.sum() == 28


def test_read_map(test_fits, test_W):
    map_data, counts, map_has_sampling, ra, dec, nu, wproj = read_map(test_fits)
    assert np.allclose(map_data, np.zeros_like(test_W[:, :, :2]))
    assert np.allclose(counts, np.zeros_like(map_data))
    assert np.allclose(map_has_sampling, np.zeros_like(map_data))
    assert np.allclose(nu, cal_freq(np.array([1, 2])))
    map_data, counts, map_has_sampling, ra, dec, nu, wproj = read_map(
        test_fits,
        counts_file=test_fits,
    )
    assert np.isnan(counts).mean() == 1.0
