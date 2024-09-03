import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from meer21cm.util import (
    check_unit_equiv,
    get_wcs_coor,
    radec_to_indx,
    freq_to_redshift,
    f_21,
    center_to_edges,
    find_ch_id,
    tagging,
    find_property_with_tags,
)
from astropy import constants, units
import astropy
from meer21cm.io import (
    cal_freq,
    meerklass_L_deep_nu_min,
    meerklass_L_deep_nu_max,
    read_map,
    filter_incomplete_los,
)
from astropy.wcs.utils import proj_plane_pixel_area
from itertools import chain
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
        filter_map_los=True,
        gal_file=None,
        weighting="counts",
        ra_range=(-np.inf, np.inf),
        dec_range=(-400, 400),
        data=None,
        weights_map_pixel=None,
        counts=None,
        **kwparams,
    ):
        self.dependency_dict = find_property_with_tags(self)
        funcs = list(chain.from_iterable(list(self.dependency_dict.values())))
        for func_i in np.unique(np.array(funcs)):
            setattr(self, func_i + "_dep_attr", [])
        for dep_attr, inp_func in self.dependency_dict.items():
            for func in inp_func:
                old_dict = getattr(self, func + "_dep_attr")
                setattr(
                    self,
                    func + "_dep_attr",
                    old_dict
                    + [
                        "_" + dep_attr,
                    ],
                )

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
        self._nu = np.array(nu)
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
        self.map_has_sampling = map_has_sampling
        self.map_unit = map_unit
        self.map_unit_type
        xx, yy = np.meshgrid(np.arange(num_pix_x), np.arange(num_pix_y), indexing="ij")
        # the coordinates of each pixel in the map
        self._ra_map, self._dec_map = get_wcs_coor(wproj, xx, yy)
        self.__dict__.update(kwparams)
        self.cosmo = self._cosmo
        self.filter_map_los = filter_map_los
        self.gal_file = gal_file
        self.weighting = weighting
        self.ra_range = ra_range
        self.dec_range = dec_range
        self._sigma_beam_ch_in_mpc = None
        self.data = data
        self.weights_map_pixel = weights_map_pixel
        self.counts = counts

    @property
    def map_unit_type(self):
        map_unit = self.map_unit
        if not check_unit_equiv(map_unit, units.K):
            if not check_unit_equiv(map_unit, units.Jy):
                raise (
                    ValueError,
                    "map unit has be to either temperature or flux density.",
                )
            else:
                map_unit_type = "F"
        else:
            map_unit_type = "T"
        return map_unit_type

    def clean_cache(self, attr):
        """
        set the input attributes to None
        """
        for att in attr:
            if att in self.__dict__.keys():
                setattr(self, att, None)

    @property
    def beam_unit(self):
        """
        The unit of input beam size parameter sigma
        """
        return self._beam_unit

    @beam_unit.setter
    def beam_unit(self, value):
        self._beam_unit = value
        if "beam_dep_attr" in dir(self):
            self.clean_cache(self.beam_dep_attr)

    @property
    def sigma_beam_ch(self):
        """
        The input beam size parameter sigma for each channel
        """
        return self._sigma_beam_ch

    @sigma_beam_ch.setter
    def sigma_beam_ch(self, value):
        self._sigma_beam_ch = value
        if "beam_dep_attr" in dir(self):
            self.clean_cache(self.beam_dep_attr)

    @property
    @tagging("cosmo", "beam")
    def sigma_beam_ch_in_mpc(self):
        """
        The input beam size parameter in Mpc in each channel
        """
        if self._sigma_beam_ch_in_mpc is None and self.sigma_beam_ch is not None:
            self._sigma_beam_ch_in_mpc = (
                self.comoving_distance(self.z_ch).to("Mpc").value
                * (self.sigma_beam_ch * self.beam_unit).to("rad").value
            )
        return self._sigma_beam_ch_in_mpc

    @property
    def sigma_beam_in_mpc(self):
        """
        The channel averaged beam size in Mpc
        """
        if self.sigma_beam_ch_in_mpc is None:
            return None
        return self.sigma_beam_ch_in_mpc.mean()

    @property
    def nu(self):
        """
        The input frequencies of the survey
        """
        return self._nu

    @nu.setter
    def nu(self, value):
        self._nu = np.array(value)
        if "nu_dep_attr" in dir(self):
            self.clean_cache(self.nu_dep_attr)

    # nu dependent, but it calculates on the fly
    # so no need for tags
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
        # cosmology changed, clear cache
        self.clean_cache(self.cosmo_dep_attr)

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
    def pix_resol_in_mpc(self):
        """
        angular resolution of the map pixel in Mpc
        """
        return (
            np.sqrt(self.pixel_area)
            * np.pi
            / 180
            * self.comoving_distance(self.z).to("Mpc").value
        )

    @property
    def los_resol_in_mpc(self):
        """
        effective frequency resolution in Mpc
        """
        comov_dist = self.comoving_distance(self.z_ch).value
        los_resol_in_mpc = (comov_dist.max() - comov_dist.min()) / len(self.nu)
        return los_resol_in_mpc

    @property
    def data(self):
        """
        The map data
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def counts(self):
        """
        The number of hits per pixel for the map data
        """
        return self._counts

    @counts.setter
    def counts(self, value):
        self._counts = value

    @property
    def map_has_sampling(self):
        """
        A binary window for whether a pixel has been sampled
        """
        return self._map_has_sampling

    @map_has_sampling.setter
    def map_has_sampling(self, value):
        self._map_has_sampling = value

    W_HI = map_has_sampling

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
    def weights_map_pixel(self):
        """
        the weights per map pixel.
        """
        return self._weights_map_pixel

    @weights_map_pixel.setter
    def weights_map_pixel(self, value):
        self._weights_map_pixel = value

    w_HI = weights_map_pixel

    @property
    def ra_gal(self):
        """
        RA coordinates of galaxy catalogue for cross-correlation
        """
        return self._ra_gal

    @property
    def dec_gal(self):
        """
        Dec coordinates of galaxy catalogue for cross-correlation
        """
        return self._dec_gal

    @property
    def z_gal(self):
        """
        Redshifts of galaxy catalogue for cross-correlation
        """
        return self._z_gal

    @property
    def freq_gal(self):
        """
        The 21cm line frequency for each galaxy in Hz.
        """
        return f_21 / (1 + self.z_gal)

    @property
    def ch_id_gal(self):
        """
        The channel id (0-indexed) of each galaxy in the catalogue
        for cross-correlation.
        Galaxies out of the frequency range will be given len(self.nu) as indices.
        """
        return find_ch_id(self.freq_gal, self.nu)

    def read_gal_cat(self):
        """
        Read in a galaxy catalogue for cross-correlation
        """
        if self.gal_file is None:
            print("no gal_file specified")
            return None
        hdu = fits.open(self.gal_file)
        hdr = hdu[1].header
        ra_g = hdu[1].data["RA"]  # Right ascension (J2000) [deg]
        dec_g = hdu[1].data["DEC"]  # Declination (J2000) [deg]
        z_g = hdu[1].data["Z"]  # Spectroscopic redshift, -1 for none attempted

        # select only galaxies that fall into range
        z_edges = center_to_edges(self.z_ch)
        zmin, zmax = self.z_ch.min(), self.z_ch.max()
        z_Lband = (z_g > zmin) & (z_g < zmax)
        ra_g = ra_g[z_Lband]
        dec_g = dec_g[z_Lband]
        z_g = z_g[z_Lband]

        # select only ra_range and dec_range
        self._ra_gal = ra_g
        self._dec_gal = dec_g
        self._z_gal = z_g
        self.trim_gal_to_range()

    def read_from_fits(self):
        if self.map_file is None:
            print("no map_file specified")
            return None
        (
            self.data,
            self.counts,
            self.map_has_sampling,
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
        if self.filter_map_los:
            (self.data, self.map_has_sampling, _, self.counts,) = filter_incomplete_los(
                self.data,
                self.map_has_sampling,
                self.counts,
                self.counts,
            )
        self.trim_map_to_range()

        if self.weighting.lower()[:5] == "count":
            self.weights_map_pixel = self.counts
        elif self.weighting.lower()[:7] == "uniform":
            self.weights_map_pixel = (self.counts > 0).astype("float")

    def trim_map_to_range(self):
        ra_temp = self.ra_map.copy()
        ra_temp[ra_temp > 180] -= 360
        ra_range = np.array(self.ra_range)
        ra_range[ra_range > 180] -= 360
        map_sel = (
            (ra_temp > ra_range[0])
            * (ra_temp < ra_range[1])
            * (self.dec_map > self.dec_range[0])
            * (self.dec_map < self.dec_range[1])
        )[:, :, None]
        self.data = self.data * map_sel
        self.counts = self.counts * map_sel
        self.map_has_sampling = self.map_has_sampling * map_sel

    def trim_gal_to_range(self):
        ra_temp = self.ra_gal.copy()
        ra_temp[ra_temp > 180] -= 360
        ra_range = np.array(self.ra_range)
        ra_range[ra_range > 180] -= 360
        gal_sel = (
            (ra_temp > ra_range[0])
            * (ra_temp < ra_range[1])
            * (self.dec_gal > self.dec_range[0])
            * (self.dec_gal < self.dec_range[1])
        )
        self._ra_gal = self.ra_gal[gal_sel]
        self._dec_gal = self.dec_gal[gal_sel]
        self._z_gal = self.z_gal[gal_sel]
