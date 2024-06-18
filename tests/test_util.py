import numpy as np
import pytest
from astropy.cosmology import Planck18
from hiimtool.basic_util import himf_pars_jones18, centre_to_edges, f_21
from meerstack.util import get_wcs_coor, PCAclean, radec_to_indx


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
