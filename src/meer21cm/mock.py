import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
from scipy.interpolate import interp1d

matplotlib.rcParams["figure.figsize"] = (18, 9)
from astropy.cosmology import Planck18
from numpy.random import default_rng
from astropy import constants, units
from astropy.wcs.utils import proj_plane_pixel_area
from scipy.ndimage import gaussian_filter
from .util import (
    check_unit_equiv,
    get_wcs_coor,
    radec_to_indx,
    get_default_args,
    hod_obuljen18,
    freq_to_redshift,
    lamb_21,
    f_21,
    tagging,
    busy_function_simple,
    find_indx_for_subarr,
    himf,
    himf_pars_jones18,
    cal_himf,
    sample_from_dist,
    center_to_edges,
    tully_fisher,
    redshift_to_freq,
    random_sample_indx,
    find_ch_id,
    mass_intflux_coeff,
    Obuljen18,
    create_udres_wproj,
    sample_map_from_highres,
    angle_in_range,
    get_nd_slicer,
)
from .plot import plot_map
from .grid import (
    find_rotation_matrix,
    minimum_enclosing_box_of_lightcone,
)
import healpy as hp
from meer21cm.power import PowerSpectrum, Specification
from meer21cm.telescope import weighted_convolution
from halomod import TracerHaloModel as THM
import warnings


# 20 lines missing before adding in
class MockSimulation(PowerSpectrum):
    def __init__(
        self,
        density="lognormal",
        relative_resol_to_pix=0.5,
        target_relative_to_num_g=1.5,
        num_discrete_source=100,
        discrete_base_field=2,
        strict_num_source=True,
        auto_relative=False,
        highres_sim=None,
        parallel_plane=True,
        **params,
    ):
        super().__init__(**params)
        self.density = density.lower()
        self.relative_resol_to_pix = relative_resol_to_pix
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
            "_mock_tracer_position_in_box",
            "_mock_tracer_position_in_radecz",
            "_mock_tracer_field_1",
            "_mock_tracer_field_2",
            "_mock_tracer_field_1_r",
            "_mock_tracer_field_2_r",
            "_mock_kaiser_field_k",
        ]
        for attr in init_attr:
            setattr(self, attr, None)
        self.target_relative_to_num_g = target_relative_to_num_g
        self.num_discrete_source = num_discrete_source
        self.discrete_base_field = discrete_base_field
        self.strict_num_source = strict_num_source
        self.auto_relative = auto_relative
        self.highres_sim = highres_sim
        self.parallel_plane = parallel_plane

    @property
    def highres_sim(self):
        """
        If not None, the mock field will first be gridded to a high resolution
        sky map, convolved with the beam and then gridded to the resolution
        specified by ``self.wproj``. The ratio of the angular resolution between
        the high-res map and the target map is specified by this ``highres_sim``.
        """
        return self._highres_sim

    @highres_sim.setter
    def highres_sim(self, value):
        self._highres_sim = value

    @property
    def strict_num_source(self):
        """
        Whether the number of galaxies simulated in the galaxy catalogue,
        i.e. the size of ``self.ra_gal.size``, strictly equals to
        ``self.num_discrete_source``. Note that due to Poisson sampling and
        the fact that survey volume is smaller than the enclosing box,
        the number of mock tracers will be larger than the input
        ``self.num_discrete_source``. The exact number is determined by
        ``self.num_discrete_source * self.target_relative_to_num_g``.

        The mock tracers inside the specified ``self.W_HI`` range will be counted
        and propagate into the galaxy catalogue. If ``strict_num_source`` is true,
        and if number of tracers inside the range is larger than
        ``self.num_discrete_source``, a random sampling is performed to trim off
        the excess number of sources. Doing this may result in small mismatch between
        the input and output power spectrum.
        """
        return self._strict_num_source

    @strict_num_source.setter
    def strict_num_source(self, value):
        self._strict_num_source = value
        self.clean_cache(self.discrete_dep_attr)

    @property
    def parallel_plane(self):
        """
        Whether the mock field is generated in the parallel-plane limit.
        """
        return self._parallel_plane

    @parallel_plane.setter
    def parallel_plane(self, value):
        self._parallel_plane = value
        self.clean_cache(self.rsd_dep_attr)

    @property
    def auto_relative(self):
        """
        Whether ``target_relative_to_num_g`` is automatically calculated.
        If True, it will be set to the ratio between the volume of the
        enclosing box and the survey volume.
        """
        return self._auto_relative

    @auto_relative.setter
    def auto_relative(self, value):
        self._auto_relative = value
        self.clean_cache(self.discrete_dep_attr)

    @property
    def target_relative_to_num_g(self):
        """
        The target number of discrete tracers to simulate
        at the field level comparing to the desired number
        ``num_discrete_source``. Needs to be larger than 1
        because some tracers are trimmed off when projecting
        the field to the sky map.

        Note that the random sampling does not exactly return
        ``target_relative_to_num_g * num_discrete_source`` number
        of tracers but a very close value.
        """
        return self._target_relative_to_num_g

    @target_relative_to_num_g.setter
    def target_relative_to_num_g(self, value):
        self._target_relative_to_num_g = value
        self.clean_cache(self.discrete_dep_attr)

    @property
    def num_discrete_source(self):
        """
        The total number of discrete tracer sources.
        """
        return self._num_discrete_source

    @num_discrete_source.setter
    def num_discrete_source(self, value):
        self._num_discrete_source = value
        self.clean_cache(self.discrete_dep_attr)

    @property
    def discrete_base_field(self):
        """
        Which tracer (1 or 2) field to sample the
        discrete tracer positions.
        """
        return self._discrete_base_field

    @discrete_base_field.setter
    def discrete_base_field(self, value):
        if isinstance(value, str):
            value = int(value)
        if not value in [1, 2]:
            raise ValueError("discrete_base_field must be 1 or 2")
        self._discrete_base_field = value
        self.clean_cache(self.discrete_dep_attr)

    @property
    @tagging("cosmo", "nu", "mock", "box")
    def mock_matter_field_r(self):
        """
        The simulated dark matter density field in real space.
        """
        if self._mock_matter_field_r is None:
            self.get_mock_matter_field_r()
        return self._mock_matter_field_r

    @property
    @tagging("cosmo", "nu", "mock", "box", "rsd")
    def mock_matter_field(self):
        """
        The simulated dark matter density field in redshift space.
        If ``self.kaiser_rsd`` is False, it is simply set to the real space field.
        """
        if self._mock_matter_field is None:
            self.get_mock_matter_field()
        return self._mock_matter_field

    def get_mock_matter_field_r(self):
        self._mock_matter_field_r = self.get_mock_field_r(bias=1)

    def get_mock_field_r(self, bias=1):
        """
        Generate a mock field in real space that follows the input matter power
        spectrum ``self.matter_power_spectrum_fnc`` * bias**2.
        """
        if self.box_ndim is None:
            self.get_enclosing_box()
        self.propagate_field_k_to_model()
        if self.density == "lognormal":
            backend = generate_lognormal_field
        elif self.density == "gaussian":
            backend = generate_gaussian_field
        else:
            raise ValueError(
                f"density must be 'lognormal' or 'gaussian', got {self.density}"
            )
        power_array = self.matter_power_spectrum_fnc(self.kmode) * bias**2
        delta_x = backend(self.box_ndim, self.box_len, power_array, self.seed)
        return delta_x

    @property
    @tagging("cosmo", "nu", "mock", "box", "rsd")
    def mock_kaiser_field_k(self):
        """
        The Kaiser rsd effect correction for the mock matter field in k-space.
        """
        if self._mock_kaiser_field_k is None:
            self.get_mock_kaiser_field_k()
        return self._mock_kaiser_field_k

    def get_mock_kaiser_field_k(self):
        r"""
        Generate the Kaiser rsd effect for the mock matter field in k-space.

        In the parrallel-plane limit, the Kaiser rsd effect is given by:

        .. math::

            \delta_{\rm rsd} = f \mu^2 \delta_k

        where :math:`f` is the growth rate, :math:`\mu` is the cosine of the angle
        between the wave vector and the line of sight, and :math:`\delta_k` is the Fourier
        transform of the real-space matter density field.

        This function returns :math:`\delta_{\rm rsd}` in k-space so that for any mock tracer field,
        the Kaiser effect can be applied by adding :math:`\delta_{\rm rsd}` to the real-space tracer field in k-space.
        """
        mock_field = self.mock_matter_field_r
        delta_k = np.fft.fftn(mock_field, norm="forward")
        slicer = get_nd_slicer()
        with np.errstate(divide="ignore", invalid="ignore"):
            kvecoverk2 = np.array(
                [self.k_vec[i][slicer[i]] / self.kmode**2 for i in range(3)]
            )
        kvecoverk2[:, self.kmode == 0] = 0
        u_k = -1j * kvecoverk2 * delta_k[None]
        u_r = np.array([np.fft.ifftn(u_k[i], norm="forward") for i in range(3)]).real
        if self.parallel_plane:
            box_coord_l = np.zeros((3,) + tuple(self.box_ndim))
            box_coord_l[-1] = 1.0
        else:
            box_coord = np.meshgrid(*self.x_vec, indexing="ij")
            box_coord = np.array(box_coord) + self.box_origin[:, None, None, None]
            box_coord_l = box_coord / np.sqrt((box_coord**2).sum(axis=0))[None]
        u_in_xhat = (u_r * box_coord_l).sum(axis=0)
        y_k = np.array(
            [np.fft.fftn(u_in_xhat * box_coord_l[i], norm="forward") for i in range(3)]
        )
        y_k_dot_k = np.array(
            [(y_k[i] * self.k_vec[i][slicer[i]]) for i in range(3)]
        ).sum(axis=0)
        delta_rsd_k = 1j * self.f_growth * y_k_dot_k
        self._mock_kaiser_field_k = delta_rsd_k
        return delta_rsd_k

    def get_mock_matter_field(self):
        self._mock_matter_field = self.get_mock_field_in_redshift_space(
            delta_x=self.mock_matter_field_r, sigma_v=0
        )

    def get_mock_field_in_redshift_space(self, delta_x, sigma_v=0):
        if self.kaiser_rsd:
            delta_k = np.fft.fftn(delta_x, norm="forward")
            delta_k += self.mock_kaiser_field_k
            mumode = self.mumode
            fog = np.fft.fftshift(
                self.fog_term(sigma_v, kmode=self.kmode, mumode=mumode)
            )
            delta_k *= fog
            delta_x = np.fft.ifftn(delta_k, norm="forward").real
        return delta_x

    @property
    @tagging("cosmo", "nu", "mock", "box", "tracer_1", "rsd")
    def mock_tracer_field_1(self):
        """
        The simulated tracer field 1 in redshift space.
        """
        if self._mock_tracer_field_1 is None:
            self.get_mock_tracer_field(1)
        mean_amp = self.mean_amp_1
        if isinstance(mean_amp, str):
            mean_amp = getattr(self, mean_amp)
        return self._mock_tracer_field_1 * mean_amp

    @property
    @tagging("cosmo", "nu", "mock", "box", "tracer_2", "rsd")
    def mock_tracer_field_2(self):
        """
        The simulated tracer field 2 in redshift space.
        """
        if self._mock_tracer_field_2 is None:
            self.get_mock_tracer_field(2)
        mean_amp = self.mean_amp_2
        if isinstance(mean_amp, str):
            mean_amp = getattr(self, mean_amp)
        return self._mock_tracer_field_2 * mean_amp

    @property
    @tagging("cosmo", "nu", "mock", "box", "tracer_1")
    def mock_tracer_field_1_r(self):
        """
        The simulated tracer field 1 in real space.
        """
        if self._mock_tracer_field_1_r is None:
            self.get_mock_tracer_field_r(1)
        mean_amp = self.mean_amp_1
        if isinstance(mean_amp, str):
            mean_amp = getattr(self, mean_amp)
        return self._mock_tracer_field_1_r * mean_amp

    @property
    @tagging("cosmo", "nu", "mock", "box", "tracer_2")
    def mock_tracer_field_2_r(self):
        """
        The simulated tracer field 2 in real space.
        """
        mean_amp = self.mean_amp_2
        if self._mock_tracer_field_2_r is None:
            self.get_mock_tracer_field_r(2)
        if isinstance(mean_amp, str):
            mean_amp = getattr(self, mean_amp)
        return self._mock_tracer_field_2_r * mean_amp

    def get_mock_tracer_field_r(self, tracer_i):
        delta_x = self.get_mock_field_r(bias=getattr(self, f"tracer_bias_{tracer_i}"))
        setattr(self, f"_mock_tracer_field_{tracer_i}_r", delta_x)
        return delta_x

    def get_mock_tracer_field(self, tracer_i):
        delta_x = self.get_mock_field_in_redshift_space(
            delta_x=getattr(self, f"mock_tracer_field_{tracer_i}_r"),
            sigma_v=getattr(self, f"sigma_v_{tracer_i}"),
        )
        setattr(self, f"_mock_tracer_field_{tracer_i}", delta_x)
        return delta_x

    @property
    @tagging("cosmo", "nu", "mock", "box", "tracer_1", "tracer_2", "discrete", "rsd")
    def mock_tracer_position_in_box(self):
        """
        The simulated tracer positions in the box.
        """
        if self._mock_tracer_position_in_box is None:
            self.get_mock_tracer_position_in_box(self.discrete_base_field)
        return self._mock_tracer_position_in_box

    def get_mock_tracer_position_in_box(self, tracer_i):
        """
        Function to retrieve the tracer positions. Modified from ``powerbox``.
        """
        rng = default_rng(self.seed)
        # note that `_mock...` does not have mean amplitude so this is what
        # we should be using instead of `mock...`, this is just to invoke
        # the simulation
        getattr(self, "mock_tracer_field_" + str(tracer_i))
        # now actually getting the underlying overdensity
        pos_value = getattr(self, "_mock_tracer_field_" + str(tracer_i)) + 1
        if self.auto_relative:
            print("automatically reset target_relative_to_num_g")
            self.target_relative_to_num_g = np.prod(self.box_len) / self.survey_volume
        num_g = self.target_relative_to_num_g * self.num_discrete_source
        pos_value /= pos_value.sum() / num_g
        # taken from powerbox
        n_per_cell = rng.poisson(pos_value)
        args = self.x_vec
        X = np.meshgrid(*args, indexing="ij")
        tracer_positions = np.array([x.flatten() for x in X]).T
        tracer_positions = tracer_positions.repeat(n_per_cell.flatten(), axis=0)
        tracer_positions += (
            rng.uniform(-0.5, 0.5, size=(np.sum(n_per_cell), len(self.box_ndim)))
            * self.box_resol[None, :]
        )
        tracer_which_cell = np.arange(pos_value.size)
        tracer_which_cell = np.repeat(tracer_which_cell, n_per_cell.flatten())
        self._mock_tracer_position_in_box = tracer_positions
        self._mock_tracer_which_cell = tracer_which_cell

    @property
    @tagging("cosmo", "nu", "mock", "box", "tracer_1", "tracer_2", "discrete", "rsd")
    def mock_tracer_position_in_radecz(self):
        """
        The simulated tracer positions projected onto the grid. The tracers outside
        the binary selection window ``map_has_sampling`` or outside the frequency range
        can be trimmed off. Furthermore, excess tracers are also excluded. The selection
        function is stored at ``inside_range``.

        Returns a list in the order of (ra,dec,z,inside_range).
        """
        if self._mock_tracer_position_in_radecz is None:
            self.get_mock_tracer_position_in_radecz()
        return self._mock_tracer_position_in_radecz

    @property
    def ra_mock_tracer(self):
        """
        The RA coordinate of mock galaxies on the sky
        """
        return self.mock_tracer_position_in_radecz[0]

    @property
    def dec_mock_tracer(self):
        """
        The Dec coordinate of mock galaxies on the sky
        """
        return self.mock_tracer_position_in_radecz[1]

    @property
    def z_mock_tracer(self):
        """
        The redshift of mock galaxies
        """
        return self.mock_tracer_position_in_radecz[2]

    @property
    def mock_inside_range(self):
        """
        Whether the mock galaxies are inside the survey area and frequency range
        """
        return self.mock_tracer_position_in_radecz[3]

    def get_mock_tracer_position_in_radecz(self):
        (
            ra_mock_tracer,
            dec_mock_tracer,
            z_mock_tracer,
            tracer_comov_dist,
        ) = self.ra_dec_z_for_coord_in_box(self.mock_tracer_position_in_box)
        freq_tracer = redshift_to_freq(z_mock_tracer)
        tracer_ch_id = find_ch_id(freq_tracer, self.nu)
        # num_ch id is for tracer outside the frequency range
        z_sel = tracer_ch_id < len(self.nu)
        # ra_temp = ra_mock_tracer.copy()
        # ra_temp[ra_temp > 180] -= 360
        # ra_range = np.array(self.ra_range)
        # ra_range[ra_range > 180] -= 360
        radec_sel = (
            angle_in_range(ra_mock_tracer, self.ra_range[0], self.ra_range[1])
            * (dec_mock_tracer > self.dec_range[0])
            * (dec_mock_tracer < self.dec_range[1])
        )
        inside_range = z_sel * radec_sel
        # there may be an excess
        num_tracer_in = inside_range.sum()
        if num_tracer_in < self.num_discrete_source:
            warnings.warn(
                "Not enough tracers inside the ra, dec, z range. "
                + "Try increasing target_relative_to_num_g."
            )
        elif self.strict_num_source:
            inside_indx = np.where(inside_range)[0]
            rand_indx = random_sample_indx(
                len(inside_indx), self.num_discrete_source, seed=self.seed
            )
            inside_range = np.zeros_like(inside_range)
            inside_range[inside_indx[rand_indx]] = True
        mock_inside_range = inside_range
        self._mock_tracer_position_in_radecz = (
            ra_mock_tracer,
            dec_mock_tracer,
            z_mock_tracer,
            mock_inside_range,
        )
        self._mock_tracer_comov_dist = tracer_comov_dist

    def propagate_mock_tracer_to_gal_cat(self, trim=True):
        """
        Propagate the mock tracer positions to the galaxy data catalogue.

        If trim, only specified ``num_discrete_source`` number of tracers
        inside the ra-dec-z range will be propagated
        """
        ra, dec, z, inside_range = self.mock_tracer_position_in_radecz
        inside_range = inside_range.copy()
        # if False, inside_range is all True
        inside_range = (inside_range + 1 - trim) > 0
        self._ra_gal = ra[inside_range]
        self._dec_gal = dec[inside_range]
        self._z_gal = z[inside_range]

    def propagate_mock_field_to_data(
        self, field, beam=True, highres_sim=None, average=True
    ):
        """
        Grid the mock tracer field onto the sky map and
        """
        if highres_sim is None:
            highres_sim = self.highres_sim
        if self.sigma_beam_ch is None:
            beam = False
        if highres_sim is None:
            wproj_hires = self.wproj
            num_pix_x = self.num_pix_x
            num_pix_y = self.num_pix_y
        else:
            wproj_hires = create_udres_wproj(self.wproj, highres_sim)
            num_pix_x = self.num_pix_x * highres_sim
            num_pix_y = self.num_pix_y * highres_sim
        map_highres, _ = self.grid_field_to_sky_map(
            field,
            average=average,
            mask=False,
            wproj=wproj_hires,
            num_pix_x=num_pix_x,
            num_pix_y=num_pix_y,
        )
        # if highres_sim is None and not beam:
        #    return map_highres
        if beam:
            beam_image = self.get_beam_image(
                wproj_hires, num_pix_x, num_pix_y, cache=False
            )
            map_highres, _ = weighted_convolution(
                map_highres, beam_image, np.ones_like(map_highres)
            )
        if highres_sim is None:
            return map_highres
        spec = Specification(
            wproj=wproj_hires,
            num_pix_x=num_pix_x,
            num_pix_y=num_pix_y,
        )
        ra_map = spec.ra_map
        dec_map = spec.dec_map
        map_highres = sample_map_from_highres(
            map_highres,
            ra_map,
            dec_map,
            self.wproj,
            self.num_pix_x,
            self.num_pix_y,
        )
        return map_highres


class HIGalaxySimulation(MockSimulation):
    def __init__(
        self,
        no_vel=True,
        tf_slope=None,
        tf_zero=None,
        halo_model=None,
        hi_mass_from="hod",
        himf_pars=None,
        num_ch_ext_on_each_side=5,
        **params,
    ):
        super().__init__(**params)
        # has to have tracer 2 for sim
        if self.tracer_bias_2 is None:
            self.tracer_bias_2 = 1.0
        self.no_vel = no_vel
        self.tf_slope = tf_slope
        self.tf_zero = tf_zero
        if halo_model is None:
            halo_model = THM(
                cosmo_model=self.cosmo,
                z=self.z,
                hod_model=Obuljen18,
            )
        self.halo_model = halo_model
        self.hi_mass_from = hi_mass_from.lower()
        if himf_pars is None:
            himf_pars = himf_pars_jones18(self.h / 0.7)
        self.himf_pars = himf_pars
        self.num_ch_ext_on_each_side = num_ch_ext_on_each_side
        init_attr = [
            "_halo_mass_mock_tracer",
            "_hi_mass_mock_tracer",
            "_hi_profile_mock_tracer",
        ]
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def hi_mass_from(self):
        """
        Methods for calculating HI mass.
        Can either be 'hod' or 'himf'.
        """
        return self._hi_mass_from

    @hi_mass_from.setter
    def hi_mass_from(self, value):
        self._hi_mass_from = value
        if "himass_dep_attr" in dir(self):
            self.clean_cache(self.himass_dep_attr)

    @property
    def no_vel(self):
        """
        If True, HI sources will have no velocity width.
        """
        return self._no_vel

    @no_vel.setter
    def no_vel(self, value):
        self._no_vel = value
        if "hivel_dep_attr" in dir(self):
            self.clean_cache(self.hivel_dep_attr)

    @property
    def tf_slope(self):
        """
        Slope of Tully-Fisher relation.
        See :func:`meer21cm.util.tully_fisher`.
        """
        return self._tf_slope

    @tf_slope.setter
    def tf_slope(self, value):
        self._tf_slope = value
        if "hivel_dep_attr" in dir(self):
            self.clean_cache(self.hivel_dep_attr)

    @property
    def tf_zero(self):
        """
        The intercept of the T-F relation. See :func:`meer21cm.util.tully_fisher`.
        """
        return self._tf_zero

    @tf_zero.setter
    def tf_zero(self, value):
        self._tf_zero = value
        if "hivel_dep_attr" in dir(self):
            self.clean_cache(self.hivel_dep_attr)

    @property
    def halo_model(self):
        """
        A :class:`halomod.TracerHaloModel` object for storing the halo model.
        """
        return self._halo_model

    @halo_model.setter
    def halo_model(self, value):
        self._halo_model = value
        if "hm_dep_attr" in dir(self):
            self.clean_cache(self.hm_dep_attr)

    @property
    @tagging(
        "hm", "cosmo", "nu", "mock", "box", "tracer_1", "tracer_2", "discrete", "rsd"
    )
    def halo_mass_mock_tracer(self):
        """
        The halo mass of the mock tracers in log10 M_sun/h.
        """
        if self._halo_mass_mock_tracer is None:
            self.get_halo_mass_mock_tracer()
        return self._halo_mass_mock_tracer

    def get_halo_mass_mock_tracer(self):
        # propagate mock catalogue to galaxy catalogue
        self.propagate_mock_tracer_to_gal_cat()
        num_g_tot_in_mockrange = self.ra_gal.size
        num_g_tot_in_map = self.ra_mock_tracer.size
        hm = self.halo_model
        dlog10m = hm.dlog10m
        m_arr_inv = hm.m[::-1]
        # in (Mpc/h)^-3
        n_halo = np.cumsum((hm.dndlog10m * dlog10m)[::-1])
        # in Mpc^-3
        n_halo *= self.h**3
        dV_pix = self.pix_resol_in_mpc**2 * self.los_resol_in_mpc
        m_indx_mock = np.where(
            (n_halo * dV_pix * self.W_HI.sum()) < num_g_tot_in_mockrange
        )[0].max()
        logm_halo_min = np.log10(m_arr_inv)[m_indx_mock]
        dndlog10_fnc = interp1d(np.log10(hm.m), hm.dndlog10m)
        self._halo_mass_mock_tracer = sample_from_dist(
            dndlog10_fnc,
            logm_halo_min,
            np.log10(hm.m.max()),
            size=num_g_tot_in_map,
            seed=self.seed,
        )

    @property
    @tagging(
        "hm",
        "cosmo",
        "nu",
        "mock",
        "box",
        "tracer_1",
        "tracer_2",
        "discrete",
        "rsd",
        "himass",
    )
    def hi_mass_mock_tracer(self):
        """
        The HI mass of the mock tracers in log10 M_sun.
        """
        if self._hi_mass_mock_tracer is None:
            getattr(self, "get_hi_mass_mock_tracer" + "_" + self.hi_mass_from)()
        return self._hi_mass_mock_tracer

    def get_hi_mass_mock_tracer_hod(self):
        himass_g = (
            self.halo_model.hod.total_occupation(10**self.halo_mass_mock_tracer)
            / self.h
        )  # no h
        self._hi_mass_mock_tracer = np.log10(himass_g)

    @property
    @tagging(
        "hm",
        "cosmo",
        "nu",
        "mock",
        "box",
        "tracer_1",
        "tracer_2",
        "discrete",
        "rsd",
        "himass",
        "hivel",
    )
    def hi_profile_mock_tracer(self):
        """
        The emission line profiles of the mock tracers in Jansky
        """
        if self._hi_profile_mock_tracer is None:
            self.get_hi_profile_mock_tracer()
        return self._hi_profile_mock_tracer

    def get_hi_profile_mock_tracer(self):
        hifluxd_ch = hi_mass_to_flux_profile(
            self.hi_mass_mock_tracer,
            self.z_mock_tracer,
            self.nu,
            self.tf_slope,
            self.tf_zero,
            cosmo=self.cosmo,
            seed=self.seed,
            no_vel=self.no_vel,
            num_ch_ext_on_each_side=self.num_ch_ext_on_each_side,
        )
        self._hi_profile_mock_tracer = hifluxd_ch

    def propagate_hi_profile_to_map(
        self, return_highres=False, beam=True, beam_image=None
    ):
        """
        Project the ``hi_profile_mock_tracer`` onto sky maps and convolve with the beam (if ``beam``).
        If ``return_highres``, the returned map is the higher resolution map
        specified by ``highres_sim``. If not, the map will be downsampled to the original resolution.
        """
        if self.sigma_beam_ch is None and beam_image is None:
            beam = False
        highres = self.highres_sim
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y
        hifluxd_ch = self.hi_profile_mock_tracer
        if highres is None:
            # no need for downsampling
            return_highres = True
            wproj = self.wproj
        else:
            wproj = create_udres_wproj(self.wproj, highres)
            num_pix_x *= highres
            num_pix_y *= highres

        indx_0, indx_1 = radec_to_indx(
            self.ra_mock_tracer, self.dec_mock_tracer, wproj, to_int=False
        )
        nu_ext = self.nu.copy()
        for i in range(self.num_ch_ext_on_each_side * 2):
            nu_ext = center_to_edges(nu_ext)
        indx_z = find_ch_id(redshift_to_freq(self.z_mock_tracer), nu_ext)
        num_ch_vel = hifluxd_ch.shape[0] // 2
        hi_map_ext_in_jy = np.zeros((num_pix_x, num_pix_y, len(nu_ext)))
        for i, indx_diff in enumerate(
            np.linspace(-num_ch_vel, num_ch_vel, 2 * num_ch_vel + 1).astype("int")
        ):
            hiflux_i = hifluxd_ch[i]
            indx_2 = indx_z + indx_diff
            indx_bins = [
                np.arange(hi_map_ext_in_jy.shape[i] + 1) - 0.5 for i in range(3)
            ]
            map_i, _ = np.histogramdd(
                np.array([indx_0, indx_1, indx_2]).T,
                bins=indx_bins,
                weights=hiflux_i,
            )
            hi_map_ext_in_jy += map_i
        # remove the excess channels used for taking into account of galaxies
        # whose centres are outside the frequency range but the profile tails are inside
        hi_map_in_jy = hi_map_ext_in_jy[
            :, :, self.num_ch_ext_on_each_side : -self.num_ch_ext_on_each_side
        ]
        if beam:
            if beam_image is None:
                beam_image = self.get_beam_image(
                    wproj, num_pix_x, num_pix_y, cache=False
                )
            hi_map_in_jy, _ = weighted_convolution(
                hi_map_in_jy, beam_image, np.ones_like(hi_map_in_jy)
            )
        if return_highres:
            return hi_map_in_jy
        spec = Specification(
            wproj=wproj,
            num_pix_x=num_pix_x,
            num_pix_y=num_pix_y,
        )
        ra_map = spec.ra_map
        dec_map = spec.dec_map
        hi_map_in_jy = sample_map_from_highres(
            hi_map_in_jy,
            ra_map,
            dec_map,
            self.wproj,
            self.num_pix_x,
            self.num_pix_y,
            average=False,
        )
        return hi_map_in_jy


def hi_mass_to_flux_profile(
    loghimass,
    z_g,
    nu,
    tf_slope=None,
    tf_zero=None,
    cosmo=Planck18,
    seed=None,
    num_ch_ext_on_each_side=5,
    internal_step=1001,
    no_vel=True,
):
    r"""
    Convert HI mass to emission line profile.

    The relation between mass and flux (flux density integrated) of a HI source can be written as [1]

    .. math::
        M_{\rm HI} = \frac{16\pi m_H}{3 h f_{21} A_{10}}\, D_L^2\, S,

    where :math:`M_{\rm HI}` is the HI mass, :math:`m_H` is the mass of neutral hydrogen atom,
    :math:`h` is the Planck constant, :math:`f_{21}` is the rest frequency of 21cm line,
    :math:`A_{10}` is the spontaneous emission rate of 21cm line,
    :math:`D_L` is the luminosity distance of the source, and
    :math:`S` is the flux.

    If ``no_vel`` is set to ``False``, the w50 parameters of the HI sources are calculated using
    a Tully-Fisher relation. Random inclinations and busy function parameters are assigned to the sources.
    The busy functions are then used as the emission profiles. The profiles are gridded into frequency channels.
    The gridded profile, ``hifluxd_ch`` has the shape of (ch_offset,num_source), with the zeroth axis corresponding
    to the channel offset, and varies from (-N,-N+1,...,0,...,N-1,N).

    Parameters
    ----------
    loghimass: array
        The HI mass of sources in log10 solmar mass (**no h**).
    z_g: array
        The redshifts of the sources
    nu: array
        The frequency channels in Hz.
    tf_slope: float, default None.
        The slope of the T-F relation. See :func:`meer21cm.util.tully_fisher`.
    tf_zero: float, default None.
        The intercept of the T-F relation. See :func:`meer21cm.util.tully_fisher`.
    cosmo: optional, default Planck18.
        The cosmology used.
    seed: optional, default None.
        The seed number for rng.
    num_ch_ext_on_each_side: optional, default 5.
        Internal parameter, no need to change.
    internal_step: optional, default 1001.
        Internal parameter, decrease for less accuracy.
        Can be lower when velocity resolution is low.
    no_vel: optional, default True.
        If True, source will have no emission line profile and just a delta peak.

    Returns
    -------
    hifluxd_ch: array
        The flux density of each source in Jy.

    References
    ----------
    .. [1] Meyer et al., "Tracing HI Beyond the Local Universe", https://arxiv.org/abs/1705.04210
    """
    rng = np.random.default_rng(seed=seed)
    # internally allowing some extension so galaxies
    # outside the frequency range can be accurately calculated
    # so the tail of profile can be correctly distributed into the range
    nu_ext = nu.copy()
    for i in range(num_ch_ext_on_each_side * 2):
        nu_ext = center_to_edges(nu_ext)
    num_g = loghimass.size
    lumi_dist_g = cosmo.luminosity_distance(z_g).to("Mpc").value
    # in Jy Hz
    hiintflux_g = 10**loghimass / mass_intflux_coeff / lumi_dist_g**2
    freq_resol = np.diff(nu).mean()
    # no velocity, just one channel
    if no_vel:
        hifluxd_ch = hiintflux_g / freq_resol
        return hifluxd_ch[None, :]
    # get w_50
    hivel_g = tully_fisher(10**loghimass / 1.4, tf_slope, tf_zero, inv=True)
    incli_g = np.abs(np.sin(rng.uniform(0, 2 * np.pi, size=num_g)))
    # get width
    hiwidth_g = incli_g * hivel_g
    # in km/s/freq
    dvdf = (constants.c / nu).to("km/s").value.mean()
    vel_resol = dvdf * np.diff(nu).mean()
    # get the number of channels needed to plot the profile
    num_ch_vel = (int(hiwidth_g.max() / vel_resol)) // 2 + 2
    # in terms of frequency offset
    freq_ch_arr = np.linspace(-num_ch_vel, num_ch_vel, 2 * num_ch_vel + 1) * freq_resol
    # busy function parameters
    busy_c = 10 ** (rng.uniform(-3, -2, size=num_g))
    busy_b = 10 ** (rng.uniform(-2, 0, size=num_g))
    # fine resolution velocity offset points for calculating the profile
    vel_int_arr = np.linspace(-hiwidth_g.max(), hiwidth_g.max(), num=internal_step)
    hiprofile_g = busy_function_simple(
        vel_int_arr[:, None],
        1,
        busy_b,
        (busy_c / hiwidth_g)[None, :] * 2,
        hiwidth_g[None, :] / 2,
    )
    # the sum over profile should give the integrated flux
    hiprofile_g = (
        hiprofile_g / (np.sum(hiprofile_g, axis=0))[None, :] * hiintflux_g[None, :]
    )
    gal_freq = redshift_to_freq(z_g)
    gal_which_ch = find_ch_id(gal_freq, nu_ext)
    # the fine resolution point in Hz
    freq_int_arr = (
        vel_int_arr[None, :] * gal_freq[:, None] / constants.c.to("km/s").value
    )
    hicumflux_g = np.cumsum(hiprofile_g, axis=0)
    # central freq of a galaxy relative to the frequency channel
    freq_start_pos = np.zeros_like(gal_freq)
    inrange_sel = gal_which_ch < nu_ext.size
    freq_start_pos[inrange_sel] = (
        nu_ext[gal_which_ch[inrange_sel]] - gal_freq[inrange_sel] - freq_resol / 2
    )
    freq_gal_arr = freq_ch_arr[:, None] - freq_start_pos[None, :]
    freq_indx = np.argmin(
        np.abs(freq_gal_arr[:, :, None] - freq_int_arr[None, :, :]).reshape(
            (-1, freq_int_arr.shape[-1])
        ),
        axis=1,
    )
    freq_indx = freq_indx.reshape(freq_gal_arr.shape)
    hifluxd_ch = np.zeros(freq_indx.shape)
    for i in range(num_g):
        hifluxd_ch[:, i] = hicumflux_g[:, i][freq_indx[:, i]]
    hifluxd_ch = np.diff(hifluxd_ch, axis=0)
    hifluxd_ch = np.concatenate((np.zeros(num_g)[None, :], hifluxd_ch), axis=0)
    # from Jy Hz to Jy
    hifluxd_ch /= freq_resol
    return hifluxd_ch


def generate_gaussian_field(
    box_ndim, box_len, power_spectrum, seed, ps_has_volume=True
):
    r"""
    Generate a Gaussian field with the given power spectrum.
    If ``ps_has_volume`` is ``True``, the power spectrum is assumed to have the volume unit,
    usually in the unit of :math:`(Mpc)^{-3}`.

    The length unit of the box is arbitrary, as long as the unit in ``box_len`` and
    ``power_spectrum`` are consistent.

    Parameters
    ----------
    box_ndim: array
        The number of grid points in each dimension.
    box_len: array
        The length of the box in each dimension.
    power_spectrum: array
        The input power spectrum.
    seed: int
        The seed for the random number generator.
    ps_has_volume: bool, default True
        If ``True``, the power spectrum is assumed to have the volume unit.

    Returns
    -------
    delta_x: array
        The generated Gaussian field.
    """
    rng = np.random.default_rng(seed)
    noise_real = rng.normal(0, 1, box_ndim)
    noise_k = np.fft.fftn(noise_real)
    ps = power_spectrum.copy()
    if ps_has_volume:
        ps = ps / np.prod(box_len) * np.prod(box_ndim)
    scaling = np.sqrt(ps)
    delta_k = noise_k * scaling

    # Inverse FFT to get real-valued Gaussian field
    delta_x = np.fft.ifftn(delta_k).real
    return delta_x


def generate_lognormal_field(
    box_ndim, box_len, power_spectrum, seed, ps_has_volume=True
):
    r"""
    Generate a lognormal field with the given power spectrum.
    If ``ps_has_volume`` is ``True``, the power spectrum is assumed to have the volume unit,
    usually in the unit of :math:`(Mpc)^{-3}`.

    The length unit of the box is arbitrary, as long as the unit in ``box_len`` and
    ``power_spectrum`` are consistent.

    Parameters
    ----------
    box_ndim: array
        The number of grid points in each dimension.
    box_len: array
        The length of the box in each dimension.
    power_spectrum: array
        The input power spectrum.
    seed: int
        The seed for the random number generator.
    ps_has_volume: bool, default True
        If ``True``, the power spectrum is assumed to have the volume unit.

    Returns
    -------
    delta_x: array
        The generated lognormal field.
    """
    ps = power_spectrum.copy()
    if ps_has_volume:
        ps = ps / np.prod(box_len) * np.prod(box_ndim)
    # Compute correlation function ξ_δ(r) via inverse FFT
    xi_delta = np.fft.ifftn(ps).real
    # Compute Gaussian correlation function ξ_G(r) = ln(1 + ξ_δ(r))
    xi_G = np.log(1 + xi_delta + 1e-10)  # Avoid log(0)
    # Compute Gaussian power spectrum Delta_G(k)
    Delta_G = np.abs(np.fft.fftn(xi_G))
    delta_x_g = generate_gaussian_field(
        box_ndim, box_len, Delta_G, seed, ps_has_volume=False
    )
    # Apply lognormal transformation
    sigma_sq = np.sum(Delta_G) / np.prod(box_ndim)
    delta_x = np.exp(delta_x_g - sigma_sq / 2.0) - 1.0
    return delta_x


def generate_colored_noise(x_size, x_len, power_spectrum, seed=None):
    """
    Generate random 1D gaussian fluctuations following a specific spectrum.
    This is similar to ``colorednoise`` package for generating colored noise.
    It is simply wrapping the ``generate_gaussian_field`` function under the hood.
    Note that the Fourier convention used should be consistent with :py:mod:`np.fft`,
    and the power spectrum is dimensionless.

    Parameters
    ----------
        x_size: int
            The number of sampling.
        x_len: float
            The **total length** of the sampling.
        power_spectrum: array
            The power spectrum of the random noise in Fourier space.
        seed: int, default None
            The seed number for random generator for sampling. If None, a random seed is used.

    Returns
    -------
        rand_arr: float array.
            The random noise.
    """

    rand_arr = generate_gaussian_field(
        x_size, x_len, power_spectrum, seed, ps_has_volume=False
    )
    rand_arr -= rand_arr.mean()
    return rand_arr
