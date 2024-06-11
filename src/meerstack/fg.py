import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import meerstack.util as stack_util
from meerstack.util import check_unit_equiv
from astropy import constants,units
from astropy.io import fits
try:
    from functools import cached_property
except:
    from backports.cached_property import cached_property

default_data_dir = stack_util.__file__.rsplit('/',1)[0]+'/data/'

class ForegroundSimulation():
    def __init__(
        self, 
        hp_nside,
        verbose=False,
        map_unit=units.K,
        base_freq=408*1e6,
    ):
        self.hp_nside = hp_nside
        self.verbose = verbose
        self.map_unit = map_unit
        if not check_unit_equiv(map_unit,units.K):
            if not check_unit_equiv(map_unit,units.Jy):
                raise(
                    ValueError,
                    'map unit has be to either temperature or flux density.'
                )
            else:
                self.map_unit_type = 'F'  
        else:
            self.map_unit_type = 'T'
        self.base_freq = base_freq
        self.sync_nside = None
        self.sync_unit = None
        self.sync_freq = None
    @cached_property
    def sync_map(self,file=None):
        if file is None:
            file = default_data_dir+'haslam408_dsds_Remazeilles2014.fits'
        sync_map = hp.read_map(file)
        self.sync_nside = hp.get_nside(sync_map)
        with fits.open(file) as hdul:
            header = hdul[1].header
            self.sync_unit = units.Unit(header['TUNIT1'])
            self.sync_freq = units.Quantity(header['FREQ']).to('Hz').value
        return sync_map
        