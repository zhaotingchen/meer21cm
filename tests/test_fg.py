import numpy as np
import pytest
from meer21cm.fg import ForegroundSimulation as FgSim
from astropy.cosmology import Planck18
from hiimtool.basic_util import himf_pars_jones18, centre_to_edges, f_21
from astropy import units, constants
import healpy as hp


def test_sync_read(test_haslam_map, test_gsm_1ghz):
    fgsim = FgSim(hp_nside=512)
    fgsim.sync_map
    fgsim = FgSim(hp_nside=512, sync_map_file=test_haslam_map)
    fgsim.sync_map
    fgsim.sync_spindx
    fgsim = FgSim(hp_nside=512, sync_map_file=test_haslam_map, sync_indx_file=-2.7)
    fgsim.sync_map
    fgsim.sync_spindx
    fgsim = FgSim(hp_nside=512, sync_map_file=test_haslam_map, sync_indx_file=-2.7)
    fgsim.sync_map
    fgsim.sync_spindx
    fgsim = FgSim(
        hp_nside=512,
        sync_map_file=test_haslam_map,
        sync_indx_file=[test_gsm_1ghz, test_haslam_map],
    )
    fgsim.sync_map
    fgsim.sync_spindx
    with pytest.raises(Exception):
        fgsim = FgSim(
            hp_nside=512, sync_map_file=test_haslam_map, sync_indx_file=[1, 2]
        )
        fgsim.sync_map
        fgsim.sync_spindx


def test_unit_conversion(test_gsm_1ghz_jy, test_gsm_1ghz):
    fgsim = FgSim(hp_nside=512, map_unit=units.K)
    assert fgsim.map_unit_type == "T"
    fgsim = FgSim(hp_nside=512, map_unit=units.Jy)
    assert fgsim.map_unit_type == "F"
    with pytest.raises(Exception):
        fgsim = FgSim(hp_nside=512, map_unit=units.m)
    fgsim = FgSim(hp_nside=512, sync_map_file=test_gsm_1ghz_jy, verbose=True)
    map1 = fgsim.sync_map
    fgsim = FgSim(hp_nside=512, sync_map_file=test_gsm_1ghz, verbose=True)
    map2 = fgsim.sync_map
    assert np.allclose(map1, map2)


def test_fg_cube(
    test_wproj,
    test_W,
    test_haslam_map,
    test_gsm_1ghz,
):
    fgsim = FgSim(
        hp_nside=128,
        sync_map_file=test_haslam_map,
        sync_indx_file=[test_gsm_1ghz, test_haslam_map],
    )
    out_map = fgsim.fg_cube(9e8).ravel()
    assert hp.get_nside(out_map) == 128
    fgsim.wproj = test_wproj
    fgsim.num_pix_x = test_W.shape[0]
    fgsim.num_pix_y = test_W.shape[1]

    out_map = fgsim.fg_cube(
        9e8,
    )
    assert out_map.shape[0] == test_W.shape[0]
    assert out_map.shape[1] == test_W.shape[1]
    fgsim.sigma_beam_ch = [1]
    out_map = fgsim.fg_cube(
        9e8,
    )
    fgsim.sync_cube
    fgsim.nu = 9e8
    fgsim.cache = True
    fgsim.sync_cube
