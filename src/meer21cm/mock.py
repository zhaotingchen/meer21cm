import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
from scipy.interpolate import interp1d

matplotlib.rcParams["figure.figsize"] = (18, 9)
from astropy.cosmology import Planck18
from numpy.random import default_rng
from hiimtool.basic_util import busy_function_simple, find_indx_for_subarr
from hiimtool.basic_util import (
    himf,
    himf_pars_jones18,
    cal_himf,
    sample_from_dist,
    tully_fisher,
)
from astropy import constants, units
from hiimtool.basic_util import centre_to_edges
from astropy.wcs.utils import proj_plane_pixel_area
from scipy.ndimage import gaussian_filter
from .stack import stack
from .util import (
    check_unit_equiv,
    get_wcs_coor,
    radec_to_indx,
    get_default_args,
    hod_obuljen18,
    freq_to_redshift,
    lamb_21,
    f_21,
)
from .plot import plot_map
from .grid import (
    find_rotation_matrix,
    minimum_enclosing_box_of_lightcone,
)
import healpy as hp
from meer21cm.power import PowerSpectrum

from powerbox import LogNormalPowerBox
from halomod import TracerHaloModel as THM
from powerbox import dft


# 20 lines missing before adding in
class MockSimulation(PowerSpectrum):
    def __init__(
        self,
        density="poisson",
        relative_resol_to_pix=0.5,
        target_relative_to_num_g=2.5,
        kaiser_rsd=False,
        seed=None,
        **params,
    ):
        super().__init__(**params)
        self.density = density.lower()
        self.relative_resol_to_pix = relative_resol_to_pix
        self.kaiser_rsd = kaiser_rsd
        if seed is None:
            seed = np.random.randint(0, 2**32)
        self.seed = seed
        self.rng = default_rng(self.seed)
        init_attr = [
            "_x_start",
            "_y_start",
            "_z_start",
            "_x_len",
            "_y_len",
            "_z_len",
            "_rot_mat_sky_to_box",
            "_pix_coor_in_cartesian",
            "_box_len",
            "_box_resol",
            "_box_ndim",
            "_mock_matter_field_r",
            "_mock_matter_field",
            "_mock_tracer_position",
            "_mock_tracer_field_1",
            "_mock_tracer_field_2",
        ]
        for attr in init_attr:
            setattr(self, attr, None)
        self.target_relative_to_num_g = target_relative_to_num_g
        self.upgrade_sampling_from_gridding = True

    @property
    def mock_matter_field(self):
        """
        The simulated dark matter density field.
        """
        return self._mock_matter_field

    def get_mock_matter_field(self):
        self._mock_matter_field = self.get_mock_field(bias=1)

    def get_mock_field(self, bias):
        if self.box_ndim is None:
            self.get_enclosing_box()
        pb = LogNormalPowerBox(
            N=self.box_ndim,
            dim=3,
            pk=self.matter_power_spectrum_fnc,
            boxlength=self.box_len,
            seed=self.seed,
        )
        if self._mock_matter_field_r is None:
            self._mock_matter_field_r = pb.delta_x()
        delta_k = dft.fft(
            self._mock_matter_field_r,
            L=pb.boxlength,
            a=pb.fourier_a,
            b=pb.fourier_b,
            backend=pb.fftbackend,
        )[0]
        if self.kaiser_rsd:
            mumode = np.nan_to_num(pb.kvec[-1][None, None, :] / pb.k())
            delta_k *= bias + mumode**2 * self.f_growth
        else:
            delta_k *= bias
        self._mock_field = dft.ifft(
            delta_k,
            L=pb.boxlength,
            a=pb.fourier_a,
            b=pb.fourier_b,
            backend=pb.fftbackend,
        )[0].real
        return self._mock_field

    @property
    def mock_tracer_field_1(self):
        """
        The simulated tracer field 1.
        """
        mean_amp = self.mean_amp_1
        if isinstance(mean_amp, str):
            mean_amp = getattr(self, mean_amp)
        return self._mock_tracer_field_1 * mean_amp

    @property
    def mock_tracer_field_2(self):
        """
        The simulated tracer field 2.
        """
        mean_amp = self.mean_amp_2
        if isinstance(mean_amp, str):
            mean_amp = getattr(self, mean_amp)
        return self._mock_tracer_field_2 * mean_amp

    def get_mock_tracer_field(self):
        self._mock_tracer_field_1 = self.get_mock_field(bias=self.tracer_bias_1)
        if self.tracer_bias_2 is not None:
            self._mock_tracer_field_2 = self.get_mock_field(bias=self.tracer_bias_2)

    @property
    def mock_tracer_position(self):
        """
        The simulated tracer positions.
        """
        return self._mock_tracer_position


class HISimulation:
    def __init__(
        self,
        nu,
        wproj,
        num_g=1,
        num_pix_x=1,
        num_pix_y=1,
        density="poisson",
        himf_pars=himf_pars_jones18(Planck18.h / 0.7),
        auto_mmin=None,
        **sim_parameters,
    ):
        self.cache = False
        self.nu = nu
        self.freq_resol = np.diff(nu).mean()
        self.z_ch = freq_to_redshift(nu)
        self.wproj = wproj
        self.num_g_tot = num_g
        self.num_g = num_g
        self.density = density.lower()
        self.himf_pars = himf_pars
        self.auto_mmin = auto_mmin
        if (
            self.density != "poisson"
            and self.density != "lognormal"
            and self.density != "custom"
        ):
            raise ValueError("density has to be poisson, lognormal or custom")
        func_list = (generate_hi_flux, flux_to_sky_map)
        if self.density == "poisson":
            func_list += (gen_random_gal_pos,)
        elif self.density == "lognormal":
            func_list += (gen_clustering_gal_pos,)
        for func in func_list:
            defaults = get_default_args(func)
            self.__dict__.update(defaults)
        self.__dict__.update(sim_parameters)
        if self.density == "poisson" and "dndz" not in self.__dict__.keys():
            self.dndz = lambda x: np.ones_like(x)
        if "W_HI" in self.__dict__.keys():
            if len(self.W_HI.shape) == 3:
                W_HI = np.mean(self.W_HI, axis=-1) == 1
                self.W_HI = W_HI[:, :, None] * np.ones_like(nu)[None, None, :]
        else:
            self.W_HI = np.ones((num_pix_x, num_pix_y, len(nu)))
        # in deg^2
        self.pix_area = proj_plane_pixel_area(wproj)
        self.pix_resol = np.sqrt(self.pix_area)
        self.zmin = np.min(self.z_ch)
        self.zmax = np.max(self.z_ch)
        self.ra_range = np.array(self.ra_range)
        self.ra_range[self.ra_range > 180] -= 360
        dvdf = (constants.c / nu).to("km/s").value.mean()
        self.vel_resol
        self.dvdf
        xx, yy = np.meshgrid(np.arange(num_pix_x), np.arange(num_pix_y), indexing="ij")
        # the coordinates of each pixel in the map
        self.ra_map, self.dec_map = get_wcs_coor(wproj, xx, yy)

    @property
    def dvdf_ch(self):
        """
        in km/s/Hz
        """
        return (constants.c / self.nu).to("km/s").value

    @property
    def vel_resol_ch(self):
        """
        in km/s
        """
        return self.dvdf_ch * self.freq_resol

    @property
    def dvdf(self):
        """
        in km/s/Hz
        """
        return self.dvdf_ch.mean()

    @property
    def vel_resol(self):
        """
        in km/s
        """
        return self.vel_resol_ch.mean()

    def sim_gal_pos_poisson(self):
        ra_g_mock, dec_g_mock, inside_range = gen_random_gal_pos(
            self.wproj,
            self.W_HI[:, :, 0],
            self.num_g,
            ra_range=self.ra_range,
            dec_range=self.dec_range,
            seed=self.seed,
        )
        # update number of galaxies to include outside range
        self.num_g_tot = ra_g_mock.size
        z_g_mock = sample_from_dist(
            self.dndz, self.zmin, self.zmax, size=self.num_g_tot, seed=self.seed
        )
        return ra_g_mock, dec_g_mock, z_g_mock, inside_range

    def sim_gal_pos_lognormal(self):
        (
            ra_g_mock,
            dec_g_mock,
            z_g_mock,
            inside_range,
            mmin_halo,
        ) = gen_clustering_gal_pos(
            self.nu,
            self.cosmo,
            self.wproj,
            self.num_g,
            self.W_HI,
            relative_resol_to_pix=self.relative_resol_to_pix,
            ra_range=self.ra_range,
            dec_range=self.dec_range,
            seed=self.seed,
            target_relative_to_num_g=self.target_relative_to_num_g,
            kaiser_rsd=self.kaiser_rsd,
        )
        self.num_g_tot = ra_g_mock.size
        self.mmin_halo = mmin_halo
        if self.auto_mmin is not None:
            self.mmin = self.get_mmin(mmin_halo)
        return ra_g_mock, dec_g_mock, z_g_mock, inside_range

    def get_gal_pos(self, cache=None):
        if cache is None:
            cache = self.cache
        ra_g_mock, dec_g_mock, z_g_mock, inside_range = getattr(
            self, "sim_gal_pos_" + self.density
        )()
        indx_1_g, indx_2_g = radec_to_indx(ra_g_mock, dec_g_mock, self.wproj)

        if cache:
            self.ra_g_mock = ra_g_mock
            self.dec_g_mock = dec_g_mock
            self.inside_range = inside_range
            self.indx_1_g = indx_1_g
            self.indx_2_g = indx_2_g
            self.z_g_mock = z_g_mock
        return ra_g_mock, dec_g_mock, z_g_mock, inside_range, indx_1_g, indx_2_g

    def get_mmin(self, mmin_halo):
        if self.auto_mmin is None:
            return None
        return np.log10(self.auto_mmin(mmin_halo))

    def get_hifluxdensity_ch(self, cache=None):
        if cache is None:
            cache = self.cache
        if "ra_g_mock" not in self.__dict__.keys():
            ra_g_mock, dec_g_mock, z_g_mock, _, _, _ = self.get_gal_pos(cache=cache)
        else:
            ra_g_mock = self.ra_g_mock
            dec_g_mock = self.dec_g_mock
            z_g_mock = self.z_g_mock
        hifluxd_ch, himass_g = generate_hi_flux(
            self.nu,
            ra_g_mock,
            dec_g_mock,
            z_g_mock,
            self.wproj,
            himf_pars=self.himf_pars,
            verbose=self.verbose,
            seed=self.seed,
            mmin=self.mmin,
            mmax=self.mmax,
            no_vel=self.no_vel,
            tf_slope=self.tf_slope,
            tf_zero=self.tf_zero,
            cosmo=self.cosmo,
            internal_step=self.internal_step,
        )
        if cache:
            self.himass_g = himass_g
            self.hifluxd_ch = hifluxd_ch
        return hifluxd_ch, himass_g

    def gal_ch_id(self, z_g=None, cache=None):
        if cache is None:
            cache = self.cache
        if z_g is None:
            if "z_g_mock" not in self.__dict__.keys():
                _, _, z_g, _, _, _ = self.get_gal_pos(cache=cache)
            else:
                z_g = self.z_g_mock
        gal_freq = f_21 / (1 + z_g)
        # which channel the galaxies belong to
        gal_which_ch = np.argmin(np.abs(gal_freq[None, :] - self.nu[:, None]), axis=0)
        if cache:
            self.gal_which_ch = gal_which_ch
        return gal_which_ch

    def get_hi_map(self, cache=None):
        if cache is None:
            cache = self.cache
        if "hifluxd_ch" not in self.__dict__.keys():
            hifluxd_ch, _ = self.get_hifluxdensity_ch(cache=cache)
        else:
            hifluxd_ch = self.hifluxd_ch
        if "ra_g_mock" not in self.__dict__.keys():
            ra_g_mock, dec_g_mock, _, _, _, _ = self.get_gal_pos(cache=cache)
        else:
            ra_g_mock = self.ra_g_mock
            dec_g_mock = self.dec_g_mock
        if "gal_ch_id" not in self.__dict__.keys():
            gal_which_ch = self.gal_ch_id(cache=cache)
        else:
            gal_which_ch = self.gal_which_ch
        himap_g = flux_to_sky_map(
            hifluxd_ch,
            ra_g_mock,
            dec_g_mock,
            gal_which_ch,
            self.ra_map,
            self.dec_map,
            self.nu,
            self.wproj,
            self.W_HI,
            fast_ang_pos=self.fast_ang_pos,
            hp_map_extend=self.hp_map_extend,
            map_unit=self.map_unit,
            sigma_beam_ch=self.sigma_beam_ch,
        )
        if cache:
            self.hi_map = himap_g
        return himap_g


def gen_clustering_gal_pos(
    nu,
    cosmo,
    wproj,
    num_g,
    W_HI,
    relative_resol_to_pix=0.5,
    ra_range=(-np.inf, np.inf),
    dec_range=(-400, 400),
    seed=None,
    target_relative_to_num_g=2.5,
    kaiser_rsd=False,
):
    if kaiser_rsd and seed is None:
        raise ValueError("seed must be set for simulating RSD.")
    ra_range = np.array(ra_range)
    ra_range[ra_range > 180] -= 360
    num_pix_x = W_HI.shape[0]
    num_pix_y = W_HI.shape[1]
    pix_area = proj_plane_pixel_area(wproj)
    pix_resol = np.sqrt(pix_area)
    tot_area = pix_area * (W_HI[:, :, 0].sum())
    range_area = (ra_range[1] - ra_range[0]) * (dec_range[1] - dec_range[0])
    if range_area < tot_area:
        fraction_area = tot_area / range_area
    else:
        fraction_area = 1
    xx, yy = np.meshgrid(np.arange(num_pix_x), np.arange(num_pix_y), indexing="ij")
    ra_map, dec_map = get_wcs_coor(wproj, xx, yy)
    z_ch = freq_to_redshift(nu)
    ra_pix = ra_map[W_HI[:, :, 0] > 0]
    dec_pix = dec_map[W_HI[:, :, 0] > 0]
    # obtain cuboid dimensions
    (
        x_start,
        y_start,
        z_start,
        L_x,
        L_y,
        L_z,
        rot_back,
    ) = minimum_enclosing_box_of_lightcone(ra_pix, dec_pix, nu, cosmo=cosmo)
    target_resol = (
        relative_resol_to_pix
        * pix_resol
        * np.pi
        / 180
        * cosmo.comoving_distance(z_ch).value.min()
    )
    L_x = (np.floor(L_x // target_resol) + 1) * target_resol
    L_y = (np.floor(L_y // target_resol) + 1) * target_resol
    L_z = (np.floor(L_z // target_resol) + 1) * target_resol
    L_box = np.array([L_x, L_y, L_z])
    N_box = (L_box / target_resol).astype("int")
    # powerbox bug: only even N works
    N_box[N_box % 2 == 1] += 1
    hm = THM(z=z_ch.mean(), cosmo_model=cosmo, hc_spectrum="nonlinear")
    cumu_halo_density = hm.ngtm * cosmo.h**3
    n_halo = cumu_halo_density * np.prod(L_box)
    target_halo_num = target_relative_to_num_g * num_g * fraction_area
    mmin_halo = np.log10(hm.m[np.where(n_halo < target_halo_num)[0][0] - 1])
    nbar_halo = cumu_halo_density[np.where(n_halo < target_halo_num)[0][0] - 1]
    power_hh = hm.power_hh(hm.k, mmin=mmin_halo) / cosmo.h**3
    power_hh_func = interp1d(np.log10(hm.k * cosmo.h), np.log10(power_hh))
    pk_func = lambda k: 10 ** (power_hh_func(np.log10(k)))
    pb = LogNormalPowerBox(
        N=N_box,
        dim=3,
        pk=pk_func,
        boxlength=L_box,
        seed=seed,
    )
    halo_pos = pb.create_discrete_sample(
        nbar=nbar_halo,
        randomise_in_cell=True,
        store_pos=True,
        min_at_zero=True,
    )
    if kaiser_rsd:
        pmm_func = lambda k: hm.power_auto_matter_fnc(k / cosmo.h) / cosmo.h**3
        pb2 = LogNormalPowerBox(
            N=N_box,
            dim=3,
            pk=pmm_func,
            boxlength=L_box,
            seed=seed,
        )
        delta_m = pb2.delta_x()
        delta_k = dft.fft(
            delta_m,
            L=pb.boxlength,
            a=pb.fourier_a,
            b=pb.fourier_b,
            backend=pb.fftbackend,
        )[0]
        # rotation of a vector is not deterministic, any direction would do
        velocity_k_z = np.nan_to_num(
            -1j
            * (1 / (1 + hm.z))
            * cosmo.H(hm.z).to("km s^-1 Mpc^-1").value
            * hm.growth_factor
            * pb.kvec[2][None, None, :]
            / pb.k() ** 2
            * delta_k
        )
        velocity_z = dft.ifft(
            velocity_k_z,
            L=pb.boxlength,
            a=pb.fourier_a,
            b=pb.fourier_b,
            backend=pb.fftbackend,
        )[0].real
        x_edges = centre_to_edges(pb.x[0] - pb.x[0][0])
        y_edges = centre_to_edges(pb.x[1] - pb.x[1][0])
        z_edges = centre_to_edges(pb.x[2] - pb.x[2][0])
        indx_x = np.digitize(halo_pos[:, 0], x_edges) - 1
        indx_y = np.digitize(halo_pos[:, 1], y_edges) - 1
        indx_z = np.digitize(halo_pos[:, 2], z_edges) - 1
        indx_x[indx_x == (len(x_edges) - 1)] = -1
        indx_y[indx_y == (len(y_edges) - 1)] = -1
        indx_z[indx_z == (len(z_edges) - 1)] = -1
        velocity_halo_para = velocity_z[indx_x, indx_y, indx_z]

    halo_pos[:, 0] += x_start
    halo_pos[:, 1] += y_start
    halo_pos[:, 2] += z_start
    halo_pos_on_sky = np.einsum("ij,aj->ai", rot_back, halo_pos)
    halo_comov_dist = np.sqrt(np.sum(halo_pos_on_sky**2, axis=-1))
    z_interp = np.linspace(z_ch.min() - 0.2, z_ch.max() + 0.2, 1001)
    comov_interp = cosmo.comoving_distance(z_interp).value
    inv_comov_func = interp1d(comov_interp, z_interp)
    halo_z = inv_comov_func(halo_comov_dist)
    z_g_mock = halo_z.copy()
    if kaiser_rsd:
        halo_comov_dist += (
            (1 + z_g_mock)
            * velocity_halo_para
            / cosmo.comoving_distance(z_g_mock).value
        )
        z_g_mock = inv_comov_func(halo_comov_dist)
    ra_g_mock, dec_g_mock = hp.vec2ang(
        halo_pos_on_sky / halo_comov_dist[:, None], lonlat=True
    )
    indx_in_band = (halo_z >= z_ch.min()) * (halo_z <= z_ch.max())
    ra_g_mock = ra_g_mock[indx_in_band]
    dec_g_mock = dec_g_mock[indx_in_band]
    z_g_mock = z_g_mock[indx_in_band]
    indx_1_g, indx_2_g = radec_to_indx(ra_g_mock, dec_g_mock, wproj)
    indx_in_wproj = (
        (indx_1_g >= 0)
        * (indx_2_g >= 0)
        * (indx_1_g < W_HI.shape[0])
        * (indx_2_g < W_HI.shape[1])
    )
    ra_g_mock = ra_g_mock[indx_in_wproj]
    dec_g_mock = dec_g_mock[indx_in_wproj]
    z_g_mock = z_g_mock[indx_in_wproj]
    indx_1_g = indx_1_g[indx_in_wproj]
    indx_2_g = indx_2_g[indx_in_wproj]
    indx_in_W = W_HI[indx_1_g, indx_2_g, 0] > 0
    ra_g_mock = ra_g_mock[indx_in_W]
    dec_g_mock = dec_g_mock[indx_in_W]
    z_g_mock = z_g_mock[indx_in_W]
    ra_g_temp = ra_g_mock.copy()
    ra_g_temp[ra_g_temp > 180] -= 360
    inside_range = (
        (ra_g_temp > ra_range[0])
        * (ra_g_temp < ra_range[1])
        * (dec_g_mock > dec_range[0])
        * (dec_g_mock < dec_range[1])
    )
    return ra_g_mock, dec_g_mock, z_g_mock, inside_range, mmin_halo


def gen_random_gal_pos(
    wproj,
    W_HI,
    num_g,
    ra_range=(-np.inf, np.inf),
    dec_range=(-400, 400),
    seed=None,
):
    """
    Generate random galaxy positions assuming poisson distribution (uniform sampling in angular space).
    The region over which galaxies are generated is controlled by the mask `W_HI`.
    On the other hand, the number of galaxies `num_g` are matched to the number inside the range
    specified by `ra_range` and `dec_range`.

    Parameters
    ----------
        wproj: :class:`astropy.wcs.WCS` object.
            The two-dimensional wcs object for the map.
        W_HI: numpy array.
            The two-dimensional binary mask for the map.
        num_g: float.
            Number of random galaxies inside the range.
        ra_range: list of two floats, default (-inf,inf).
            The RA range in which a required number `num_g` of galaxies is generated.
        dec_range: list of two floats, default (-400,400).
            The Dec range in which a required number `num_g` of galaxies is generated.
        seed: int, default None.
            The seed number for random generator for sampling. If None, a random seed is used.

    Returns
    -------
        ra_g_mock: float array.
            The RA of the mock galaxies in degree.
        dec_g_mock: float array.
            The Dec of the mock galaxies in degree.
        inside_range: boolean array.
            Whether the galaxies are inside the specified range of RA and Dec.
    """
    rng = np.random.default_rng(seed)
    ra_range = np.array(ra_range)
    ra_range[ra_range > 180] -= 360
    # randomly select positions until enough are unmasked
    num_gal_in_range = 0
    indx_1, indx_2 = np.where(W_HI.astype("bool"))
    indx_1_g = np.array([])
    indx_2_g = np.array([])
    while num_gal_in_range < num_g:
        indx_1_rand = rng.uniform(indx_1.min(), indx_1.max(), size=num_g)
        indx_2_rand = rng.uniform(indx_2.min(), indx_2.max(), size=num_g)
        coor_rand = wproj.pixel_to_world(indx_1_rand, indx_2_rand)
        ra_rand = coor_rand.ra.degree
        dec_rand = coor_rand.dec.degree
        ra_rand[ra_rand > 180] -= 360
        sel_rand_indx = (
            W_HI[
                np.round(indx_1_rand).astype("int"), np.round(indx_2_rand).astype("int")
            ]
        ).astype("bool")
        indx_1_g = np.append(indx_1_g, indx_1_rand[sel_rand_indx])
        indx_2_g = np.append(indx_2_g, indx_2_rand[sel_rand_indx])
        inside_range = (
            sel_rand_indx
            * (ra_rand > ra_range[0])
            * (ra_rand < ra_range[1])
            * (dec_rand > dec_range[0])
            * (dec_rand < dec_range[1])
        )
        num_gal_in_range += inside_range.sum()
    # convert to angular coor
    coor_g = wproj.pixel_to_world(indx_1_g, indx_2_g)
    ra_g_mock = coor_g.ra.degree
    dec_g_mock = coor_g.dec.degree
    # indx_1_g = np.round(indx_1_g).astype('int')
    # indx_2_g = np.round(indx_2_g).astype('int')
    ra_g_temp = ra_g_mock.copy()
    ra_g_temp[ra_g_temp > 180] -= 360
    inside_range = (
        (ra_g_temp > ra_range[0])
        * (ra_g_temp < ra_range[1])
        * (dec_g_mock > dec_range[0])
        * (dec_g_mock < dec_range[1])
    )
    # only need num_g sources inside the range
    # this if is for avoiding num_g exceeding the maximum index
    if inside_range.mean() == 1:
        indx_stop = num_g
    else:
        if np.sum(inside_range) == num_g:
            indx_stop = np.where(inside_range)[0][-1] + 1
        else:
            indx_stop = np.where(inside_range)[0][num_g]
    ra_g_mock = ra_g_mock[:indx_stop]
    dec_g_mock = dec_g_mock[:indx_stop]
    inside_range = inside_range[:indx_stop]
    return ra_g_mock, dec_g_mock, inside_range


def generate_hi_flux(
    nu,
    ra_g_mock,
    dec_g_mock,
    z_g_mock,
    wproj,
    himf_pars,
    verbose=False,
    seed=None,
    mmin=11.7,
    mmax=14.0,
    no_vel=True,
    tf_slope=None,
    tf_zero=None,
    cosmo=Planck18,
    internal_step=1001,
    **kwargs,
):
    # in km/s/freq
    dvdf = (constants.c / nu).to("km/s").value.mean()
    # in km/s
    vel_resol = dvdf * np.diff(nu).mean()
    rng = default_rng(seed=seed)
    num_g = ra_g_mock.size
    # samples from himf distribution
    himass_g = sample_from_dist(
        lambda x: himf(10**x, himf_pars[0], 10 ** himf_pars[1], himf_pars[2]),
        mmin,
        mmax,
        size=num_g,
        seed=seed,
    )
    # get velocity for the sources
    if no_vel:
        num_ch_vel = 0
    else:
        hivel_g = tully_fisher(10**himass_g / 1.4, tf_slope, tf_zero, inv=True)
        incli_g = np.abs(np.sin(rng.uniform(0, 2 * np.pi, size=num_g)))
        hiwidth_g = incli_g * hivel_g
        num_ch_vel = (int(hiwidth_g.max() / vel_resol)) // 2 + 2

    comov_dist_g = cosmo.comoving_distance(z_g_mock).value
    lumi_dist_g = (1 + z_g_mock) * comov_dist_g
    # convert to flux. from 1705.04210
    # in Jy km s-1
    hiintflux_g = 10**himass_g * (1 + z_g_mock) / 2.356 / 1e5 / (lumi_dist_g) ** 2
    # random busy functions
    busy_c = 10 ** (rng.uniform(-3, -2, size=num_g))
    busy_b = 10 ** (rng.uniform(-2, 0, size=num_g))
    # zero is the centre of source along los
    vel_ch_arr = np.linspace(-num_ch_vel, num_ch_vel, 2 * num_ch_vel + 1) * vel_resol
    if no_vel:
        hiprofile_g = hiintflux_g[None, :]
    else:
        vel_int_arr = np.linspace(-hiwidth_g.max(), hiwidth_g.max(), num=internal_step)
        hiprofile_g = busy_function_simple(
            vel_int_arr[:, None],
            1,
            busy_b,
            (busy_c / hiwidth_g)[None, :] * 2,
            hiwidth_g[None, :] / 2,
        )
    # the integral over velocity should give the flux
    hiprofile_g = (
        hiprofile_g / (np.sum(hiprofile_g, axis=0))[None, :] * hiintflux_g[None, :]
    )

    gal_freq = f_21 / (1 + z_g_mock)
    # which channel the galaxies belong to
    gal_which_ch = np.argmin(np.abs(gal_freq[None, :] - nu[:, None]), axis=0)
    # obtain the emission line profile for each galaxy
    if no_vel:
        hifluxd_ch = hiprofile_g
    else:
        hicumflux_g = np.cumsum(hiprofile_g, axis=0)
        vel_start_pos = (nu[gal_which_ch] - gal_freq) * dvdf
        vel_gal_arr = vel_ch_arr[:, None] - vel_start_pos[None, :]
        vel_indx = np.argmin(
            np.abs(vel_gal_arr[:, :, None] - vel_int_arr[None, None, :]).reshape(
                (-1, len(vel_int_arr))
            ),
            axis=1,
        )
        vel_indx = vel_indx.reshape(vel_gal_arr.shape)
        hifluxd_ch = np.zeros(vel_indx.shape)
        for i in range(num_g):
            hifluxd_ch[:, i] = hicumflux_g[:, i][vel_indx[:, i]]
        hifluxd_ch = np.diff(hifluxd_ch, axis=0)
        hifluxd_ch = np.concatenate((np.zeros(num_g)[None, :], hifluxd_ch), axis=0)
    hifluxd_ch /= vel_resol
    return hifluxd_ch, himass_g


def flux_to_sky_map(
    sourceflux_ch,
    ra_source,
    dec_source,
    source_which_ch,
    ra_map,
    dec_map,
    nu,
    wproj,
    W_map,
    fast_ang_pos=True,
    hp_map_extend=2.0,
    map_unit=units.Jy,
    sigma_beam_ch=None,
):
    indx_1_g, indx_2_g = radec_to_indx(ra_source, dec_source, wproj)
    num_ch_vel = (len(sourceflux_ch) - 1) // 2
    # in deg^2
    pix_area = proj_plane_pixel_area(wproj)
    pix_resol = np.sqrt(pix_area)
    # if using healpix
    if not fast_ang_pos:
        # calculate the resolution
        for i in range(10):
            nside = 2**i
            hp_resol = hp.nside2resol(nside, arcmin=True) / 60
            if hp_resol < (pix_resol / 2):
                break
        ipix_test = np.unique(hp.ang2pix(nside, ra_map, dec_map, lonlat=True))
        ra_temp = ra_map.copy()
        ra_temp[ra_temp > 180] -= 360
        # obtaining the sub-area of the healpix map needed
        vertices = [
            hp.ang2vec(
                ra_temp.min() - hp_map_extend,
                dec_map.min() - hp_map_extend,
                lonlat=True,
            ),
            hp.ang2vec(
                ra_temp.min() - hp_map_extend,
                dec_map.max() + hp_map_extend,
                lonlat=True,
            ),
            hp.ang2vec(
                ra_temp.max() + hp_map_extend,
                dec_map.max() + hp_map_extend,
                lonlat=True,
            ),
            hp.ang2vec(
                ra_temp.max() - hp_map_extend,
                dec_map.min() - hp_map_extend,
                lonlat=True,
            ),
        ]
        ipix_map = hp.query_polygon(nside, vertices)
        error_message = "Healpix map does not include all map pixels. "
        error_message += "Try increasing hp_map_extend"
        assert (
            np.isin(
                ipix_test,
                ipix_map,
            ).mean()
            == 1
        ), error_message
        ipix_g_mock = hp.ang2pix(nside, ra_source, dec_source, lonlat=True)
        error_message = "Healpix map does not include all mock galaxies. "
        error_message += "Try increasing hp_map_extend"
        assert (
            np.isin(
                ipix_g_mock,
                ipix_map,
            ).mean()
            == 1
        ), error_message
        sky_map = np.zeros((len(ipix_map), len(nu)))
        indx_gal_to_map = find_indx_for_subarr(ipix_g_mock, ipix_map)
        sky_map = np.concatenate(
            (sky_map[:, 0][:, None], sky_map, sky_map[:, 0][:, None]), axis=-1
        )
    else:
        sky_map = np.zeros(ra_map.shape + nu.shape)
        # add a filler channel for both start and end of frequency
        sky_map = np.concatenate(
            (sky_map[:, :, 0][:, :, None], sky_map, sky_map[:, :, 0][:, :, None]),
            axis=-1,
        )

    for i, indx_diff in enumerate(
        np.linspace(-num_ch_vel, num_ch_vel, 2 * num_ch_vel + 1).astype("int")
    ):
        # note the start filler thus +1
        indx_z = source_which_ch + 1 + indx_diff
        # throw away edge bits
        indx_z[indx_z < 0] = 0
        indx_z[indx_z >= sky_map.shape[-1]] = sky_map.shape[-1] - 1
        if fast_ang_pos:
            sky_map[indx_1_g, indx_2_g, indx_z] += sourceflux_ch[i]
        else:
            sky_map[indx_gal_to_map, indx_z] += sourceflux_ch[i]

    if fast_ang_pos:
        sky_map = sky_map[:, :, 1:-1]
        if sigma_beam_ch is not None:
            for ch_id in range(sky_map.shape[-1]):
                sky_map[:, :, ch_id] = gaussian_filter(
                    sky_map[:, :, ch_id], sigma_beam_ch[ch_id] / np.sqrt(pix_area)
                )
    else:
        sky_map = sky_map[:, 1:-1]
        ra_hp, dec_hp = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
        indx_1_hp, indx_2_hp = radec_to_indx(ra_hp, dec_hp, wproj)
        indx_1_hp[indx_1_hp < 0] = -1
        indx_2_hp[indx_2_hp < 0] = -1
        indx_1_hp[indx_1_hp >= W_map.shape[0]] = -1
        indx_2_hp[indx_2_hp >= W_map.shape[1]] = -1
        himap_ch = np.zeros(
            np.array([W_map.shape[0], W_map.shape[1], len(nu)]) + [1, 1, 0]
        )
        for ch_id in range(sky_map.shape[-1]):
            hpmap_ch_i = np.zeros(hp.nside2npix(nside))
            hpmap_ch_i[ipix_map] += sky_map[:, ch_id]
            if sigma_beam_ch is not None:
                hpmap_ch_i = hp.smoothing(
                    hpmap_ch_i, sigma=sigma_beam_ch[ch_id] * np.pi / 180
                )
            himap_temp = np.zeros_like(himap_ch[:, :, 0])
            himap_temp[
                indx_1_hp[hpmap_ch_i != 0], indx_2_hp[hpmap_ch_i != 0]
            ] = hpmap_ch_i[hpmap_ch_i != 0]
            himap_ch[:, :, ch_id] = himap_temp.copy()
        himap_ch = himap_ch[:-1, :-1]
        sky_map = himap_ch

    # convert to temp if needed
    if check_unit_equiv(map_unit, units.K):
        z_ch = freq_to_redshift(nu)
        sky_map = (
            (
                sky_map
                * units.Jy
                / (2 * constants.k_B / (lamb_21 * (1 + (z_ch))) ** 2)
                / (pix_area * np.pi**2 / 180**2)
            )
            .to(map_unit)
            .value
        )
    elif check_unit_equiv(map_unit, units.Jy):
        sky_map = (sky_map * units.Jy).to(map_unit).value
    else:
        raise ValueError("map unit must be either temperature or flux density")
    return sky_map


def run_poisson_mock(
    nu,
    num_g,
    himf_pars,
    wproj,
    base_map=None,
    verbose=False,
    seed=None,
    mmin=11.7,
    mmax=14.0,
    no_vel=True,
    W_HI=None,
    w_HI=None,
    tf_slope=None,
    tf_zero=None,
    cosmo=Planck18,
    map_unit=units.Jy,
    sigma_beam_ch=None,
    mycmap="bwr",
    do_stack=True,
    dndz=lambda x: np.ones_like(x),
    zmin=None,
    zmax=None,
    internal_step=1001,
    x_dim=None,
    y_dim=None,
    stack_angular_num_nearby_pix=10,
    x_unit=units.km / units.s,
    ignore_double_counting=False,
    return_indx_and_weight=False,
    ra_range=(-np.inf, np.inf),
    dec_range=(-400, 400),
    velocity_width_halfmax=50,
    fix_ra_dec=None,
    fix_z=None,
    fast_ang_pos=True,
    hp_map_extend=2.0,
    **kwargs,
):
    def mock_stack(verbose):
        stack_result = stack(
            himap_g,
            wproj,
            ra_g_mock[inside_range],
            dec_g_mock[inside_range],
            z_g_mock[inside_range],
            nu,
            W_map_in=W_HI,
            w_map_in=w_HI,
            no_vel=no_vel,
            sigma_beam_in=sigma_beam_ch,
            velocity_width_halfmax=velocity_width_halfmax,
            stack_angular_num_nearby_pix=stack_angular_num_nearby_pix,
            ignore_double_counting=ignore_double_counting,
            x_unit=x_unit,
            verbose=verbose,
            return_indx_and_weight=return_indx_and_weight,
        )
        return stack_result

    if base_map is not None:
        num_pix_x = base_map.shape[0]
        num_pix_y = base_map.shape[1]
    elif x_dim is not None and y_dim is not None:
        num_pix_x = x_dim
        num_pix_y = y_dim
    else:
        raise ValueError("either base_map or (x_dim,y_dim) is needed")

    hisim = HISimulation(
        nu,
        wproj,
        num_g,
        num_pix_x,
        num_pix_y,
        density="poisson",
        himf_pars=himf_pars,
        base_map=base_map,
        verbose=verbose,
        seed=seed,
        mmin=mmin,
        mmax=mmax,
        no_vel=no_vel,
        W_HI=W_HI,
        w_HI=w_HI,
        tf_slope=tf_slope,
        tf_zero=tf_zero,
        cosmo=cosmo,
        map_unit=map_unit,
        sigma_beam_ch=sigma_beam_ch,
        mycmap=mycmap,
        do_stack=do_stack,
        dndz=dndz,
        zmin=zmin,
        zmax=zmax,
        internal_step=internal_step,
        x_dim=x_dim,
        y_dim=y_dim,
        stack_angular_num_nearby_pix=stack_angular_num_nearby_pix,
        x_unit=x_unit,
        ignore_double_counting=ignore_double_counting,
        return_indx_and_weight=return_indx_and_weight,
        ra_range=ra_range,
        dec_range=dec_range,
        velocity_width_halfmax=velocity_width_halfmax,
        fast_ang_pos=fast_ang_pos,
        hp_map_extend=hp_map_extend,
    )
    # the coordinates of each pixel in the map
    ra, dec = hisim.ra_map, hisim.dec_map
    # in deg^2
    pix_area = hisim.pix_area
    pix_resol = hisim.pix_resol

    if zmin is not None:
        hisim.zmin = zmin
    if zmax is not None:
        hisim.zmax = zmax
    if W_HI is None:
        W_HI = np.ones(ra.shape + nu.shape)
    if base_map is None:
        base_map = np.zeros(ra.shape + nu.shape)

    if len(W_HI.shape) == 2:
        W_HI = W_HI[:, :, None]
    elif len(W_HI.shape) != 3:
        raise ValueError("W mask has to be 2d or 3d")

    if W_HI[:, :, 0].sum() == 0:
        raise ValueError("all pixels are masked by W_HI")

    if not np.allclose(W_HI.mean(axis=-1), W_HI[:, :, 0]):
        raise ValueError("mask has to be the same for every channel")
    # ra_g_mock, dec_g_mock, inside_range = gen_random_gal_pos(
    #    wproj, W_HI[:, :, 0], num_g, ra_range=ra_range, dec_range=dec_range, seed=seed
    # )
    ra_g_mock, dec_g_mock, rand_z, inside_range, indx_1_g, indx_2_g = hisim.get_gal_pos(
        cache=True
    )
    # update num_g to include sources outside the range
    num_g = hisim.num_g_tot
    # indx_1_g, indx_2_g = radec_to_indx(ra_g_mock, dec_g_mock, wproj)
    if fix_ra_dec is not None:
        hisim.ra_g_mock[inside_range] = fix_ra_dec[0]
        hisim.dec_g_mock[inside_range] = fix_ra_dec[1]
        fix_indx = radec_to_indx(fix_ra_dec[0], fix_ra_dec[1], wproj)
        hisim.indx_1_g[inside_range] = fix_indx[0]
        hisim.indx_2_g[inside_range] = fix_indx[1]

    if fix_z is not None:
        hisim.z_g_mock[inside_range] = fix_z
        rand_z = hisim.z_g_mock
    if verbose:
        plt.hist(rand_z, bins=20, density=True)
        plt.title("redshift distribution")
        plt.xlabel("z")
        plt.ylabel("dn/dz")
        # plt.show()
        # plt.close()

        plt.figure()
        plt.subplot(projection=wproj)
        ax = plt.gca()
        ax.invert_xaxis()
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_major_formatter("d")
        contours = plt.contour(W_HI[:, :, 0].T, levels=[0.5], colors="black")
        if inside_range.mean() < 1:
            plt.scatter(
                ra_g_mock[(1 - inside_range).astype("bool")],
                dec_g_mock[(1 - inside_range).astype("bool")],
                transform=ax.get_transform("world"),
                s=1,
                label="galaxy outside the range but in the map",
                color="tab:blue",
            )
        plt.scatter(
            ra_g_mock[inside_range],
            dec_g_mock[inside_range],
            transform=ax.get_transform("world"),
            s=1,
            label="galaxy positions",
            color="tab:red",
        )
        lon = ax.coords[0]
        lat = ax.coords[1]
        ax.coords.grid(True, color="black", ls="solid")
        plt.xlabel("R.A [deg]", fontsize=18)
        plt.ylabel("Dec. [deg]", fontsize=18)
        plt.legend()

    vel_resol = hisim.vel_resol

    z_g_mock = rand_z
    hifluxd_ch, himass_g = hisim.get_hifluxdensity_ch(cache=True)
    num_ch_vel = len(hifluxd_ch - 1) // 2
    gal_freq = f_21 / (1 + rand_z)
    # which channel the galaxies belong to
    gal_which_ch = hisim.gal_ch_id()

    himap_g = hisim.get_hi_map()

    if verbose:
        plot_map(
            himap_g,
            wproj,
            W=W_HI,
            title="mock HI signal",
            ZeroCentre=False,
            cbar_label=f"{map_unit:latex}",
        )
        plt.show()
    # overlay mock on the original map
    himap_g += base_map
    if not do_stack:
        stack_result = ()
    else:
        # run stacking
        stack_result = mock_stack(verbose)
    return (
        himap_g,
        ra_g_mock,
        dec_g_mock,
        z_g_mock,
        indx_1_g,
        indx_2_g,
        gal_which_ch,
        hifluxd_ch,
        inside_range,
    ) + stack_result


def run_lognormal_mock(
    nu,
    num_g,
    himf_pars,
    wproj,
    base_map=None,
    verbose=False,
    seed=None,
    mmin_from_hod=hod_obuljen18,
    mmin=11.0,
    mmax=14.0,
    W_HI=None,
    w_HI=None,
    no_vel=True,
    tf_slope=None,
    tf_zero=None,
    cosmo=Planck18,
    map_unit=units.Jy,
    sigma_beam_ch=None,
    mycmap="bwr",
    zmin=None,
    zmax=None,
    internal_step=1001,
    x_dim=None,
    y_dim=None,
    ra_range=(-np.inf, np.inf),
    dec_range=(-400, 400),
    fix_ra_dec=None,
    fix_z=None,
    fast_ang_pos=True,
    hp_map_extend=2.0,
    velocity_width_halfmax=50,
    stack_angular_num_nearby_pix=10,
    x_unit=units.km / units.s,
    ignore_double_counting=False,
    return_indx_and_weight=False,
    do_stack=False,
    kaiser_rsd=False,
):
    def mock_stack(verbose):
        stack_result = stack(
            himap_g,
            wproj,
            ra_g_mock[inside_range],
            dec_g_mock[inside_range],
            z_g_mock[inside_range],
            nu,
            W_map_in=W_HI,
            w_map_in=w_HI,
            no_vel=no_vel,
            sigma_beam_in=sigma_beam_ch,
            velocity_width_halfmax=velocity_width_halfmax,
            stack_angular_num_nearby_pix=stack_angular_num_nearby_pix,
            ignore_double_counting=ignore_double_counting,
            x_unit=x_unit,
            verbose=verbose,
            return_indx_and_weight=return_indx_and_weight,
        )
        return stack_result

    if base_map is not None:
        num_pix_x = base_map.shape[0]
        num_pix_y = base_map.shape[1]
    elif x_dim is not None and y_dim is not None:
        num_pix_x = x_dim
        num_pix_y = y_dim
    else:
        raise ValueError("either base_map or (x_dim,y_dim) is needed")
    if W_HI is None:
        W_HI = np.ones((num_pix_x, num_pix_y, len(nu)))
    if len(W_HI.shape) == 2:
        W_HI = W_HI[:, :, None]
    elif len(W_HI.shape) != 3:
        raise ValueError("W mask has to be 2d or 3d")

    if W_HI[:, :, 0].sum() == 0:
        raise ValueError("all pixels are masked by W_HI")
    hisim = HISimulation(
        nu,
        wproj,
        num_g,
        num_pix_x,
        num_pix_y,
        density="lognormal",
        himf_pars=himf_pars,
        base_map=base_map,
        verbose=verbose,
        seed=seed,
        mmin=mmin,
        mmax=mmax,
        no_vel=no_vel,
        W_HI=W_HI,
        w_HI=w_HI,
        tf_slope=tf_slope,
        tf_zero=tf_zero,
        cosmo=cosmo,
        map_unit=map_unit,
        sigma_beam_ch=sigma_beam_ch,
        mycmap=mycmap,
        zmin=zmin,
        zmax=zmax,
        internal_step=internal_step,
        x_dim=x_dim,
        y_dim=y_dim,
        ra_range=ra_range,
        dec_range=dec_range,
        fast_ang_pos=fast_ang_pos,
        hp_map_extend=hp_map_extend,
        kaiser_rsd=kaiser_rsd,
    )
    # the coordinates of each pixel in the map
    ra, dec = hisim.ra_map, hisim.dec_map
    if base_map is None:
        base_map = np.zeros(ra.shape + nu.shape)
    (
        ra_g_mock,
        dec_g_mock,
        z_g_mock,
        inside_range,
        indx_1_g,
        indx_2_g,
    ) = hisim.get_gal_pos(cache=True)
    mmin_halo = hisim.mmin_halo
    gal_which_ch = hisim.gal_ch_id()
    hisim.mmin = np.log10(mmin_from_hod(mmin_halo))
    hifluxd_ch, himass_g = hisim.get_hifluxdensity_ch(cache=True)
    inside_indx = np.where(inside_range)[0]
    mass_indx = np.argsort(himass_g[inside_range])[-num_g:]
    inside_range = np.zeros_like(inside_range)
    inside_range[inside_indx[mass_indx]] = True
    hisim.inside_range = inside_range
    if fix_ra_dec is not None:
        ra_g_mock[inside_range] = fix_ra_dec[0]
        dec_g_mock[inside_range] = fix_ra_dec[1]
        fix_indx = radec_to_indx(fix_ra_dec[0], fix_ra_dec[1], wproj)
        indx_1_g[inside_range] = fix_indx[0]
        indx_2_g[inside_range] = fix_indx[1]
        hisim.ra_g_mock[inside_range] = fix_ra_dec[0]
        hisim.dec_g_mock[inside_range] = fix_ra_dec[1]
        hisim.indx_1_g[inside_range] = fix_indx[0]
        hisim.indx_2_g[inside_range] = fix_indx[1]

    if fix_z is not None:
        z_g_mock[inside_range] = fix_z
        hisim.z_g_mock[inside_range] = fix_z

    if verbose:
        plt.hist(z_g_mock, bins=20, density=True)
        plt.title("redshift distribution")
        plt.xlabel("z")
        plt.ylabel("dn/dz")
        # plt.show()
        # plt.close()

        plt.figure()
        plt.subplot(projection=wproj)
        ax = plt.gca()
        ax.invert_xaxis()
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_major_formatter("d")
        contours = plt.contour(W_HI[:, :, 0].T, levels=[0.5], colors="black")
        if inside_range.mean() < 1:
            plt.scatter(
                ra_g_mock[(1 - inside_range).astype("bool")],
                dec_g_mock[(1 - inside_range).astype("bool")],
                transform=ax.get_transform("world"),
                s=1,
                label="galaxy outside the range but in the map",
                color="tab:blue",
            )
        plt.scatter(
            ra_g_mock[inside_range],
            dec_g_mock[inside_range],
            transform=ax.get_transform("world"),
            s=1,
            label="galaxy positions",
            color="tab:red",
        )
        lon = ax.coords[0]
        lat = ax.coords[1]
        ax.coords.grid(True, color="black", ls="solid")
        plt.xlabel("R.A [deg]", fontsize=18)
        plt.ylabel("Dec. [deg]", fontsize=18)
        plt.legend()
    himap_g = hisim.get_hi_map(cache=True)
    if verbose:
        plot_map(
            himap_g,
            wproj,
            W=W_HI,
            title="mock HI signal",
            ZeroCentre=False,
            cbar_label=f"{map_unit:latex}",
        )
        plt.show()
    # overlay mock on the original map
    himap_g += base_map
    if not do_stack:
        stack_result = ()
    else:
        # run stacking
        stack_result = mock_stack(verbose)
    return (
        himap_g,
        ra_g_mock,
        dec_g_mock,
        z_g_mock,
        indx_1_g,
        indx_2_g,
        gal_which_ch,
        hifluxd_ch,
        inside_range,
    ) + stack_result
