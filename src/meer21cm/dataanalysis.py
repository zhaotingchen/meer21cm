import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from meer21cm.util import (
    check_unit_equiv,
    get_wcs_coor,
    radec_to_indx,
    freq_to_redshift,
)
from astropy import constants, units
import astropy
from meer21cm.io import (
    cal_freq,
    meerklass_L_deep_nu_min,
    meerklass_L_deep_nu_max,
    read_map,
)
from astropy.wcs.utils import proj_plane_pixel_area
import meer21cm

default_data_dir = meer21cm.__file__.rsplit("/", 1)[0] + "/data/"


class Specification:
    def __init__(
        self,
        nu=None,
        wproj=None,
        num_pix_x=133,
        num_pix_y=73,
        cosmo="Planck18",
        map_has_sampling=None,
        sigma_beam_ch=None,
        beam_unit=units.deg,
        map_unit=units.K,
        map_file=None,
        counts_file=None,
        los_axis=-1,
        nu_min=-np.inf,
        nu_max=np.inf,
        **kwparams,
    ):
        if "_cosmo" in kwparams.keys() and cosmo == "Planck18":
            self._cosmo = kwparams["_cosmo"]
        else:
            self._cosmo = cosmo
        self.map_file = map_file
        self.counts_file = counts_file
        self.los_axis = los_axis
        self.nu_min = nu_min
        self.nu_max = nu_max
        if nu is None:
            nu = cal_freq(
                np.arange(4096) + 1,
            )
            nu_sel = (nu > meerklass_L_deep_nu_min) * (nu < meerklass_L_deep_nu_max)
            nu = nu[nu_sel]
        self.nu = nu
        if wproj is None:
            map_file = default_data_dir + "test_fits.fits"
            wcs = WCS(map_file)
            wproj = wcs.dropaxis(-1)
        self.wproj = wproj
        self.num_pix_x = num_pix_x
        self.num_pix_y = num_pix_y
        self.sigma_beam_ch = sigma_beam_ch
        self.beam_unit = beam_unit
        if map_has_sampling is None:
            map_has_sampling = np.ones((num_pix_x, num_pix_y, len(nu)), dtype="bool")
        self._map_has_sampling = map_has_sampling
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
        xx, yy = np.meshgrid(np.arange(num_pix_x), np.arange(num_pix_y), indexing="ij")
        # the coordinates of each pixel in the map
        self._ra_map, self._dec_map = get_wcs_coor(wproj, xx, yy)
        self.__dict__.update(kwparams)
        self.cosmo = self._cosmo

    @property
    def z_ch(self):
        """
        The redshift of each frequency channel
        """
        return freq_to_redshift(self.nu)

    @property
    def z(self):
        """
        The effective centre redshift of the frequency range
        """
        return self.z_ch.mean()

    @property
    def cosmo(self):
        return self._cosmo

    @cosmo.setter
    def cosmo(self, value):
        cosmo = value
        if isinstance(value, str):
            cosmo = getattr(astropy.cosmology, value)
        self._cosmo = cosmo
        self.ns = cosmo.meta["n"]
        self.sigma8 = cosmo.meta["sigma8"]
        self.tau = cosmo.meta["tau"]
        self.Oc0 = cosmo.meta["Oc0"]
        # there is probably a more elegant way of doing this, but I dont know how
        # maybe just inheriting astropy cosmology class?
        for key in cosmo.__dir__():
            if key[0] != "_":
                self.__dict__.update({key: getattr(cosmo, key)})

    @property
    def dvdf_ch(self):
        """
        velocity resolution per unit frequency in each channel, in km/s/Hz
        """
        return (constants.c / self.nu).to("km/s").value

    @property
    def vel_resol_ch(self):
        """
        velocity resolution of each channel in km/s
        """
        return self.dvdf_ch * self.freq_resol

    @property
    def dvdf(self):
        """
        velocity resolution per unit frequency on average, in km/s/Hz
        """
        return self.dvdf_ch.mean()

    @property
    def vel_resol(self):
        """
        velocity resolution on average in km/s
        """
        return self.vel_resol_ch.mean()

    @property
    def freq_resol(self):
        """
        frequency resolution in Hz
        """
        return np.diff(self.nu).mean()

    @property
    def pixel_area(self):
        """
        angular area of the map pixel in deg^2
        """
        return proj_plane_pixel_area(self.wproj)

    @property
    def pix_resol(self):
        """
        angular resolution of the map pixel in deg
        """
        return np.sqrt(self.pixel_area)

    @property
    def data(self):
        """
        The map data
        """
        return self._data

    @property
    def counts(self):
        """
        The number of hits per pixel for the map data
        """
        return self._counts

    @property
    def map_has_sampling(self):
        """
        A binary window for whether a pixel has been sampled
        """
        return self._map_has_sampling

    @property
    def ra_map(self):
        """
        A binary window for whether a pixel has been sampled
        """
        return self._ra_map

    @property
    def dec_map(self):
        """
        A binary window for whether a pixel has been sampled
        """
        return self._dec_map

    @property
    def W_HI(self):
        """
        alias for the binary map sampling.
        """
        return self._map_has_sampling

    @property
    def w_HI(self):
        """
        alias for the weights per pixel.
        """
        return self._counts

    def read_from_fits(self):
        if self.map_file is None:
            print("no map_file specified")
            return None
        (
            self._data,
            self._counts,
            self._map_has_sampling,
            self._ra_map,
            self._dec_map,
            self.nu,
            self.wproj,
        ) = read_map(
            self.map_file,
            counts_file=self.counts_file,
            nu_min=self.nu_min,
            nu_max=self.nu_max,
            los_axis=self.los_axis,
        )
        self.num_pix_x, self.num_pix_y = self._ra_map.shape
