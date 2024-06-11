import numpy as np
import pytest
from meerstack.fg import ForegroundSimulation as FgSim
from astropy.cosmology import Planck18
from hiimtool.basic_util import himf_pars_jones18,centre_to_edges,f_21
from astropy import units,constants

def test_sync_read():
    fgsim = FgSim(hp_nside=512)
    fgsim.sync_map
    
def test_unit_conversion():
    fgsim = FgSim(hp_nside=512,map_unit=units.K)
    assert fgsim.map_unit_type=='T'
    fgsim = FgSim(hp_nside=512,map_unit=units.Jy)
    assert fgsim.map_unit_type=='F'
    with pytest.raises(Exception):
        fgsim = FgSim(hp_nside=512,map_unit=units.m)
    