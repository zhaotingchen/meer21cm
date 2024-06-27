import numpy as np
import pytest
from meerstack.mock import gen_random_gal_pos, run_poisson_mock
from astropy.cosmology import Planck18
from hiimtool.basic_util import himf_pars_jones18, centre_to_edges, f_21
from unittest.mock import patch
import matplotlib.pyplot as plt
from meerstack.stack import weight_source_peaks, stack
from meerstack.util import radec_to_indx
from astropy import constants, units


def test_weight_source_peaks(test_wproj, test_W, test_nu):
    map_in = np.zeros((test_W.shape[0], test_W.shape[1], len(test_nu)))
    ra_in = np.array([350.0])
    dec_in = np.array([-25.0])
    z_in = np.array([0.45])
    gal_freq = f_21 / (1 + z_in) / 1e6
    indx_1, indx_2 = radec_to_indx(ra_in, dec_in, test_wproj)
    # which channel each source centre belongs to
    gal_which_ch = np.argmin(np.abs(gal_freq[None, :] - test_nu[:, None]), axis=0)
    map_gal_indx, map_gal_weight = weight_source_peaks(
        map_in,
        test_wproj,
        ra_in,
        dec_in,
        z_in,
        test_nu,
        ignore_double_counting=True,
    )
    assert np.allclose(map_gal_indx, -1 * np.ones_like(map_in))
    # test no vel
    map_gal_indx, map_gal_weight = weight_source_peaks(
        map_in,
        test_wproj,
        ra_in,
        dec_in,
        z_in,
        test_nu,
        no_vel=True,
        ignore_double_counting=False,
    )
    assert map_gal_indx[indx_1, indx_2, gal_which_ch] == 0
    # test with velocity
    map_gal_indx, map_gal_weight = weight_source_peaks(
        map_in,
        test_wproj,
        ra_in,
        dec_in,
        z_in,
        test_nu,
        no_vel=False,
        ignore_double_counting=False,
    )
    assert map_gal_indx[indx_1, indx_2, gal_which_ch] == 0
    test_indx = np.where(map_gal_indx != -1)
    assert np.unique(test_indx[0]) == indx_1
    assert np.unique(test_indx[1]) == indx_2
    num_of_pixel_per_gal = np.sum(map_gal_indx != -1)

    # test with beam
    map_gal_indx, map_gal_weight = weight_source_peaks(
        map_in,
        test_wproj,
        ra_in,
        dec_in,
        z_in,
        test_nu,
        no_vel=False,
        no_sel_weight=False,
        gal_sel_indx=[0, 1],
        ignore_double_counting=False,
        velocity_profile="step",
        sigma_beam_in=np.ones(len(test_nu)) * 0.3,
    )
    assert map_gal_indx[indx_1, indx_2, gal_which_ch] == 0
    assert map_gal_indx[indx_1 - 1, indx_2, gal_which_ch] == 0
    assert map_gal_indx[indx_1, indx_2 - 1, gal_which_ch] == 0
    assert map_gal_indx[indx_1 + 1, indx_2, gal_which_ch] == 0
    assert map_gal_indx[indx_1, indx_2 + 1, gal_which_ch] == 0

    # test two galaxies right next to each other
    ra_in = np.array([350.0, 350.0])
    dec_in = np.array([-27.0, -27.0])
    z_in = np.array([0.45, 0.4505])
    gal_freq = f_21 / (1 + z_in) / 1e6
    indx_1, indx_2 = radec_to_indx(ra_in, dec_in, test_wproj)
    map_gal_indx, map_gal_weight = weight_source_peaks(
        map_in,
        test_wproj,
        ra_in,
        dec_in,
        z_in,
        test_nu,
        no_vel=False,
        ignore_double_counting=False,
        velocity_profile="step",
    )
    assert map_gal_weight.sum() < 2
    assert np.sum(map_gal_indx != -1) < 2 * num_of_pixel_per_gal
    map_gal_indx, map_gal_weight = weight_source_peaks(
        map_in,
        test_wproj,
        ra_in,
        dec_in,
        z_in,
        test_nu,
        no_vel=False,
        no_sel_weight=True,
        ignore_double_counting=False,
        velocity_profile="step",
    )
    assert np.allclose(map_gal_weight.sum(), 2)


def test_stack(test_wproj, test_W, test_nu):
    map_in = np.zeros((test_W.shape[0], test_W.shape[1], len(test_nu)))
    ra_in = np.array([350.0])
    dec_in = np.array([-25.0])
    z_in = np.array([0.45])
    gal_freq = f_21 / (1 + z_in) / 1e6
    indx_1, indx_2 = radec_to_indx(ra_in, dec_in, test_wproj)
    # which channel each source centre belongs to
    gal_which_ch = np.argmin(np.abs(gal_freq[None, :] - test_nu[:, None]), axis=0)
    map_in[indx_1, indx_2, gal_which_ch] = 1.0
    (
        stack_3D_map,
        stack_3D_weight,
        x_edges,
        ang_edges,
        map_gal_indx,
        map_gal_weight,
    ) = stack(
        map_in,
        test_wproj,
        ra_in,
        dec_in,
        z_in,
        test_nu,
        W_map_in=None,
        w_map_in=None,
        velocity_width_halfmax=250,
        velocity_profile="gaussian",
        sigma_beam_in=None,
        no_vel=False,
        internal_step=2000,
        verbose=False,
        ignore_double_counting=False,
        project_mat=None,
        gal_sel_indx=None,
        no_sel_weight=False,
        stack_angular_num_nearby_pix=10,
        # x_unit=units.km / units.s,
        return_indx_and_weight=True,
    )
    ang_mid_point = stack_3D_map.shape[0] // 2
    freq_mid_point = stack_3D_map.shape[2] // 2
    assert stack_3D_map[ang_mid_point, ang_mid_point, freq_mid_point] == 1.0
