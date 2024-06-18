import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import meerstack.util as stack_util
from meerstack.util import read_healpix_fits
from meerstack.util import convert_hpmap_in_jy_to_temp
from meerstack.util import healpix_to_wcs

from astropy import constants, units
from astropy.io import fits

try:
    from functools import cached_property
except:
    from backports.cached_property import cached_property
from collections.abc import Iterable
from hiimtool.basic_util import check_unit_equiv, jy_to_kelvin

default_data_dir = stack_util.__file__.rsplit("/", 1)[0] + "/data/"


class ForegroundSimulation:
    def __init__(
        self,
        hp_nside,
        verbose=False,
        map_unit=units.K,
        sync_map_file=None,
        sync_indx_file=None,
        sync_uni_indx=-2.55,
        do_point_souce=False,
    ):
        self.hp_nside = hp_nside
        self.verbose = verbose
        self.map_unit = map_unit
        if not check_unit_equiv(map_unit, units.K):
            if not check_unit_equiv(map_unit, units.Jy):
                raise (
                    ValueError,
                    "map unit has be to either temperature or flux density.",
                )
            else:
                self.map_unit_type = "F"
        else:
            self.map_unit_type = "T"
        self.sync_map_file = sync_map_file
        self.sync_indx_file = sync_indx_file
        self.sync_nside = None
        self.sync_unit = None
        self.sync_freq = None
        self.sync_uni_indx = sync_uni_indx
        self.do_point_souce = do_point_souce

    @cached_property
    def sync_map(self):
        file = self.sync_map_file
        if file is None:
            file = default_data_dir + "haslam408_dsds_Remazeilles2014.fits"
        sync_map, self.sync_nside, self.sync_unit, self.sync_freq = read_healpix_fits(
            file
        )
        if check_unit_equiv(self.sync_unit, units.Jy):
            if self.verbose:
                print("convert " + file + " to temperature")
            sync_map = (sync_map * self.sync_unit / 1 / units.Jy).to("").value
            sync_map = convert_hpmap_in_jy_to_temp(sync_map, self.sync_freq)
        return sync_map

    @cached_property
    def sync_spindx(self):
        sync_indx_file = self.sync_indx_file
        if sync_indx_file is None:
            use_map = False
            uni_indx = self.sync_uni_indx
        elif isinstance(sync_indx_file, float):
            uni_indx = sync_indx_file
            use_map = False
        elif isinstance(sync_indx_file, Iterable):
            if isinstance(sync_indx_file[0], str) and isinstance(
                sync_indx_file[1], str
            ):
                map1 = sync_indx_file[0]
                map2 = sync_indx_file[1]
                use_map = True
        else:
            raise TypeError("cannot recognise sync_indx_file")
        if use_map:
            map1_val, map1_nside, map1_unit, map1_freq = read_healpix_fits(map1)
            map2_val, map2_nside, map2_unit, map2_freq = read_healpix_fits(map2)
            if map1_nside != self.sync_nside:
                map1_val = hp.ud_grade(map1_val, self.sync_nside)
            if map2_nside != self.sync_nside:
                map2_val = hp.ud_grade(map2_val, self.sync_nside)
            unit_conv = (1 * map1_unit / map2_unit).to("").value
            sp_indx = np.log(unit_conv * map1_val / map2_val) / np.log(
                map1_freq / map2_freq
            )
        else:
            sp_indx = np.ones(hp.nside2npix(self.sync_nside)) * uni_indx
        return sp_indx

    def fg_cube(
        self,
        freq,
        source_type="sync",
        wproj=None,
        xdim=None,
        ydim=None,
        sigma_beam_ch=None,
    ):
        if source_type == "sync":
            in_map = self.sync_map
            spindx = self.sync_spindx
            base_freq = self.sync_freq
        freq = np.array([freq]).ravel()
        if hp.get_nside(in_map) != self.hp_nside:
            in_map = hp.ud_grade(in_map, self.hp_nside)
            spindx = hp.ud_grade(spindx, self.hp_nside)
        in_map = in_map[:, None]
        spindx = spindx[:, None]
        freq = freq[None, :]
        out_map = in_map * (freq / base_freq) ** spindx
        if sigma_beam_ch is not None:
            for ch_id in range(out_map.shape[-1]):
                out_map[:, ch_id] = hp.smoothing(
                    out_map[:, ch_id], sigma=sigma_beam_ch[ch_id] * np.pi / 180
                )
        if wproj is not None:
            xx, yy = np.meshgrid(
                np.arange(xdim),
                np.arange(ydim),
                indexing="ij",
            )
            out_map_proj = np.zeros(xx.shape + (out_map.shape[-1],))
            for ch_id in range(out_map.shape[-1]):
                out_map_proj[:, :, ch_id] = healpix_to_wcs(
                    out_map[:, ch_id], xx, yy, wproj
                )
            out_map = out_map_proj
        return out_map
