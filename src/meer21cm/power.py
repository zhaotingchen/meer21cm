"""
This module handles computation of power spectrum from gridded fields.
"""
import numpy as np
from meer21cm.cosmology import CosmologyCalculator


class ModelPowerSpectrum(CosmologyCalculator):
    def __init__(
        self,
        kmode=None,
        mumode=None,
        tracer_bias_1=1.0,
        sigma_v_1=0.0,
        tracer_bias_2=None,
        sigma_v_2=0.0,
        matter_only_rsd=False,
        include_beam=[True, False],
        fog_profile="lorentz",
        cross_coeff=1.0,
        weights_1=None,
        weights_2=None,
        mean_amp_1=1.0,
        mean_amp_2=1.0,
        sampling_resol=None,
        include_sampling=[True, False],
        **params,
    ):
        super().__init__(**params)
        self.tracer_bias_1 = tracer_bias_1
        self.sigma_v_1 = sigma_v_1
        self.tracer_bias_2 = tracer_bias_2
        self.sigma_v_2 = sigma_v_2
        self.matter_only_rsd = matter_only_rsd
        self.kmode = kmode
        self.mumode = mumode
        if kmode is None:
            self.kmode = np.geomspace(self.kmin, self.kmax, 100)
        if mumode is None:
            self.mumode = np.zeros_like(self.kmode)
        self.include_beam = include_beam
        self.has_beam = True
        if self.sigma_beam_ch is None and (np.array(self.include_beam).sum() > 0):
            print("no input beam found, setting include_beam to False")
            self.include_beam = [False, False]
            self.has_beam = False
        self.fog_profile = fog_profile
        self.cross_coeff = cross_coeff
        self._auto_power_matter_model = None
        self._auto_power_tracer_1_model = None
        self._auto_power_tracer_2_model = None
        self._cross_power_tracer_model = None
        self.weights_1 = weights_1
        self.weights_2 = weights_2
        self.mean_amp_1 = mean_amp_1
        self.mean_amp_2 = mean_amp_2

        self.sampling_resol = sampling_resol
        self.has_resol = True
        if self.sampling_resol is None:
            self.has_resol = False
        self.include_sampling = include_sampling

    def fog_lorentz(self, sigma_v):
        """
        sqrt(1/(1+(sigma_v k_parallel /H_0)^2))
        """
        H_0 = self.H0.to("km s^-1 Mpc^-1").value
        k_parallel = self.kmode * self.mumode
        fog = np.sqrt(1 / (1 + (sigma_v * k_parallel / H_0) ** 2))
        return fog

    def fog_term(self, sigma_v):
        return getattr(self, "fog_" + self.fog_profile)(sigma_v)

    @property
    def auto_power_matter_model(self):
        return self._auto_power_matter_model

    @property
    def auto_power_tracer_1_model(self):
        return self._auto_power_tracer_1_model

    @property
    def auto_power_tracer_2_model(self):
        return self._auto_power_tracer_2_model

    @property
    def cross_power_tracer_model(self):
        return self._cross_power_tracer_model

    def step_sampling(self):
        return 1.0

    def beam_attenuation(self):
        if not self.has_beam:
            return 1.0
        # in the future for asymmetric beam this way
        # of writing may be probelmatic
        k_perp = self.kmode * np.sqrt(1 - self.mumode**2)
        sigma_beam_rad = (self.sigma_beam_ch.mean() * self.beam_unit).to("rad").value
        sigma_beam_mpc = sigma_beam_rad * self.comoving_distance(self.z).to("Mpc").value
        B_beam = gaussian_beam_attenuation(k_perp, sigma_beam_mpc)
        return B_beam

    def cal_rsd_power(
        self,
        power_in_real_space,
        beta1,
        sigmav_1,
        beta2=None,
        sigmav_2=None,
        r=1.0,
    ):
        if beta2 is None:
            beta2 = beta1
        if sigmav_2 is None:
            sigmav_2 = sigmav_1
        power_in_redshift_space = (
            power_in_real_space
            * (
                r
                + (beta1 + beta2) * self.mumode**2
                + beta1 * beta2 * self.mumode**4
            )
            * self.fog_term(sigmav_1)
            * self.fog_term(sigmav_2)
        )
        return power_in_redshift_space

    def get_model_power(self):
        B_beam = self.beam_attenuation()
        B_sampling = self.step_sampling()
        tracer_beam_indx = np.array(self.include_beam).astype("int")
        tracer_samp_indx = np.array(self.include_sampling).astype("int")

        if self.matter_power_spectrum_fnc is None:
            self.get_matter_power_spectrum()
        pk3d_mm_r = self.matter_power_spectrum_fnc(self.kmode)
        beta_m = self.f_growth
        self._auto_power_matter_model = self.cal_rsd_power(
            pk3d_mm_r,
            beta_m,
            0.0,
        )
        pk3d_tt_r = self.tracer_bias_1**2 * pk3d_mm_r
        if self.matter_only_rsd:
            beta_1 = self.f_growth
        else:
            beta_1 = self.f_growth / self.tracer_bias_1
        self._auto_power_tracer_1_model = self.cal_rsd_power(
            pk3d_tt_r,
            beta_1,
            self.sigma_v_1,
        )
        self._auto_power_tracer_1_model *= B_beam ** (tracer_beam_indx[0] * 2)
        self._auto_power_tracer_1_model *= B_sampling ** (tracer_samp_indx[0] * 2)
        self._auto_power_tracer_1_model[self.kmode == 0] = 0.0
        self._auto_power_tracer_1_model = get_modelpk_conv(
            self._auto_power_tracer_1_model,
            weights1_in_real=self.weights_1,
        )
        if isinstance(self.mean_amp_1, str):
            self.mean_amp_1 = getattr(self, self.mean_amp_1)
        self._auto_power_tracer_1_model *= self.mean_amp_1**2
        if self.tracer_bias_2 is not None:
            if self.matter_only_rsd:
                beta_2 = self.f_growth
            else:
                beta_2 = self.f_growth / self.tracer_bias_2
            pk3d_tt_r = self.tracer_bias_2**2 * pk3d_mm_r
            self._auto_power_tracer_2_model = self.cal_rsd_power(
                pk3d_tt_r,
                beta_2,
                self.sigma_v_2,
            )
            self._auto_power_tracer_2_model[self.kmode == 0] = 0.0
            self._auto_power_tracer_2_model *= B_beam ** (tracer_beam_indx[1] * 2)
            self._auto_power_tracer_2_model *= B_sampling ** (tracer_samp_indx[1] * 2)

            self._auto_power_tracer_2_model = get_modelpk_conv(
                self._auto_power_tracer_2_model,
                weights1_in_real=self.weights_2,
            )
            if isinstance(self.mean_amp_2, str):
                self.mean_amp_2 = getattr(self, self.mean_amp_2)
            self._auto_power_tracer_2_model *= self.mean_amp_2**2
            # cross power
            pk3d_tt_r = self.tracer_bias_1 * self.tracer_bias_2 * pk3d_mm_r
            self._cross_power_tracer_model = self.cal_rsd_power(
                pk3d_tt_r,
                beta1=beta_1,
                sigmav_1=self.sigma_v_1,
                beta2=beta_2,
                sigmav_2=self.sigma_v_2,
                r=self.cross_coeff,
            )
            self._cross_power_tracer_model[self.kmode == 0] = 0.0
            self._cross_power_tracer_model *= B_beam ** (
                tracer_beam_indx[0] + tracer_beam_indx[1]
            )
            self._cross_power_tracer_model *= B_sampling ** (
                tracer_samp_indx[0] + tracer_samp_indx[1]
            )
            self._cross_power_tracer_model[self.kmode == 0] = 0.0
            self._cross_power_tracer_model = get_modelpk_conv(
                self._cross_power_tracer_model,
                weights1_in_real=self.weights_1,
                weights2=self.weights_2,
            )
            self._cross_power_tracer_model *= self.mean_amp_2 * self.mean_amp_1


class FieldPowerSpectrum:
    def __init__(
        self,
        field_1,
        box_len,
        weights_1=None,
        mean_center_1=False,
        unitless_1=False,
        remove_sn_1=False,
        field_2=None,
        weights_2=None,
        mean_center_2=False,
        unitless_2=False,
        remove_sn_2=False,
        corrtype=None,
        mean_amp_1=1.0,
        mean_amp_2=1.0,
    ):
        self.field_1 = field_1
        self.weights_1 = weights_1
        self.field_2 = field_2
        self.weights_2 = weights_2
        self.box_len = np.array(box_len)
        self.box_ndim = np.array(field_1.shape)
        self.box_resol = self.box_len / self.box_ndim
        self.mean_center_1 = mean_center_1
        self.unitless_1 = unitless_1
        self.mean_center_2 = mean_center_2
        self.unitless_2 = unitless_2
        self.remove_sn_1 = remove_sn_1
        self.remove_sn_2 = remove_sn_2
        if field_2 is not None:
            error_message = "field_1 and field_2 must have same dimensions"
            assert np.allclose(field_2.shape, field_1.shape), error_message
        self._fourier_field_1 = None
        self._fourier_field_2 = None
        self.mean_amp_1 = mean_amp_1
        self.mean_amp_2 = mean_amp_2

    def set_corr_type(self, corr_type, tracer_indx):
        """
        A utility function to help decide whether a tracer field
        needs to be mean centred, renormalised by its mean, and shot noise removed.
        Currently only two types are supported, "Gal" and "HI" (case-insensitive).
        If the tracer is galaxy (number counts),
        the auto power spectrum is mean centred, renormalised, and then
        shot noise removed. If HI, none of the above will be performed.

        Parameters
        ----------
        corr_type: str
            The tracer type.
        tracer_indx: int
            Either 1 or 2.
        """
        if corr_type[:3].lower() == "gal":
            mean_center = True
            unitless = True
            remove_sn = True
        elif corr_type[:2].lower() == "hi":
            mean_center = False
            unitless = False
            remove_sn = False
        else:
            raise ValueError("unknown corr_type")
        if not tracer_indx in [1, 2]:
            raise ValueError("tracer_indx should be either 1 or 2")
        setattr(self, "mean_center_" + str(tracer_indx), mean_center)
        setattr(self, "unitless_" + str(tracer_indx), unitless)
        setattr(self, "remove_sn_" + str(tracer_indx), remove_sn)

    @property
    def k_vec(self):
        return get_k_vector(
            self.box_ndim,
            self.box_resol,
        )

    @property
    def k_perp(self):
        return get_vec_mode(self.k_vec[:-1])

    @property
    def k_para(self):
        return self.k_vec[-1]

    @property
    def k_mode(self):
        return get_vec_mode(self.k_vec)

    @property
    def fourier_field_1(self):
        return self._fourier_field_1

    def get_fourier_field_1(self):
        result = get_fourier_density(
            self.field_1,
            weights=self.weights_1,
            mean_center=self.mean_center_1,
            unitless=self.unitless_1,
        )
        self._fourier_field_1 = result

    @property
    def fourier_field_2(self):
        return self._fourier_field_2

    def get_fourier_field_2(self):
        if self.field_2 is None:
            return None
        result = get_fourier_density(
            self.field_2,
            weights=self.weights_2,
            mean_center=self.mean_center_2,
            unitless=self.unitless_2,
        )
        self._fourier_field_2 = result

    @property
    def auto_power_3d_1(self):
        if self._fourier_field_1 is None:
            self.get_fourier_field_1()
        power_spectrum = get_power_spectrum(
            self.fourier_field_1,
            self.box_len,
            weights=self.weights_1,
        )
        if self.remove_sn_1:
            field = get_renormed_field(
                self.field_1,
                weights=self.weights_1,
                mean_center=self.mean_center_1,
                unitless=self.unitless_1,
            )
            power_spectrum -= get_shot_noise(
                self.field_1,
                self.box_len,
                weights=self.weights_1,
            )
        return power_spectrum

    @property
    def auto_power_3d_2(self):
        if self.field_2 is None:
            return None
        if self._fourier_field_2 is None:
            self.get_fourier_field_2()
        power_spectrum = get_power_spectrum(
            self.fourier_field_2,
            self.box_len,
            weights=self.weights_2,
        )
        if self.remove_sn_2:
            power_spectrum -= get_shot_noise(
                self.field_2,
                self.box_len,
                weights=self.weights_2,
            )
        return power_spectrum

    @property
    def cross_power_3d(self):
        if self.field_2 is None:
            return None
        if self._fourier_field_1 is None:
            self.get_fourier_field_1()
        if self._fourier_field_2 is None:
            self.get_fourier_field_2()
        weights_2 = self.weights_2
        # if none, the default for get_power_spectrum is
        # to use weights_1, here we want separate weights_2
        if weights_2 is None:
            weights_2 = np.ones(self.fourier_field_2.shape)
        power_spectrum = get_power_spectrum(
            self.fourier_field_1,
            self.box_len,
            weights=self.weights_1,
            field_2=self.fourier_field_2,
            weights_2=weights_2,
        )
        return power_spectrum


def get_renormed_field(
    real_field,
    weights=None,
    mean_center=False,
    unitless=False,
):
    """
    Mean center the field and renormalise it by dividing the mean.
    """
    field = np.array(real_field)
    if weights is None:
        weights = np.ones_like(field)
    weights = np.array(weights)
    if mean_center or unitless:
        field_mean = np.sum(weights * field) / np.sum(weights)
    else:
        return real_field
    if mean_center:
        field -= field_mean
    if unitless:
        field /= field_mean
    return field


def get_fourier_density(
    real_field,
    weights=None,
    mean_center=False,
    unitless=False,
    norm="forward",
):
    """
    Perform Fourier transform of a density field in real space. Note that
    this is deliberately written in a way that is not dimension specific.
    It can be used to calculate power spectrum of arbitrary dimension.

    Note that, the field is multiplied by the weights
    and then Fourier-transformed, and is **not weight normalised**.
    """
    field = get_renormed_field(
        real_field,
        weights=weights,
        mean_center=mean_center,
        unitless=unitless,
    )
    if weights is None:
        weights = np.ones_like(field)
    weights = np.array(weights)
    fourier_field = np.fft.fftn(field * weights, norm=norm)
    return fourier_field


def get_k_vector(box_ndim, box_resol):
    """
    Get the wavenumber vector along each direction
    for a given box.
    """
    kvecarr = tuple(
        2
        * np.pi
        * np.fft.fftfreq(
            box_ndim[i],
            d=box_resol[i],
        )
        for i in range(len(box_ndim))
    )
    return kvecarr


def get_vec_mode(vecarr):
    """
    Calculate the mode of the n-dimensional vectors on the grids
    """
    result = np.sqrt(
        np.sum(
            (np.meshgrid(*([(vec) ** 2 for vec in vecarr]), indexing="ij")),
            0,
        )
    )
    return result


def get_shot_noise(
    real_field,
    box_len,
    weights=None,
):
    box_len = np.array(box_len)
    box_volume = np.prod(box_len)
    if weights is None:
        weights = np.ones(real_field.shape)
    weights = np.array(weights)
    shot_noise = (
        box_volume
        * np.sum((weights * real_field) ** 2)
        / np.sum(weights * real_field) ** 2
    )
    return shot_noise


def get_modelpk_conv(psmod, weights1_in_real=None, weights2=None, renorm=True):
    """
    Convolve a model power spectrum with real-space weights.
    """
    if weights1_in_real is None:
        return psmod
    weights_fourier = np.fft.fftn(weights1_in_real)
    if weights2 is None:
        weights_fourier = np.abs(weights_fourier) ** 2
    else:
        weights_fourier *= np.conj(np.fft.fftn(weights2))
    weights_fourier = np.real(weights_fourier)
    power_conv = (
        np.fft.ifftn(np.fft.fftn(psmod) * np.fft.fftn(weights_fourier))
        / weights1_in_real.size**2
    )
    if renorm:
        weights_renorm = power_weights_renorm(weights1_in_real, weights2=weights2)
        power_conv *= weights_renorm
    return power_conv.real


def power_weights_renorm(weights1_in_real, weights2=None):
    """
    Calculate the renormalization coefficient based on the weights
    on the density field when calculating power spectrum

    Parameters
    ----------
        weights1_in_real: array.
            The weights of the density field in real space.
            Must be in the shape of the regular grid field.
        weights2: array, None.
            If cross-correlation, the weights for the second field.

    Returns
    -------
        weights_norm: float.
           The renormalization coefficient.
    """
    if weights2 is None:
        weights2 = weights1_in_real
    weights_norm = weights1_in_real.size / np.sum(weights1_in_real * weights2)
    return weights_norm


def get_power_spectrum(
    fourier_field,
    box_len,
    weights=None,
    field_2=None,
    weights_2=None,
):
    box_len = np.array(box_len)
    if field_2 is None:
        field_2 = fourier_field
    fourier_field = np.array(fourier_field)
    field_2 = np.array(field_2)
    if weights is None:
        weights = np.ones(fourier_field.shape)
    weights_norm = power_weights_renorm(weights, weights_2)
    power = np.real(fourier_field * np.conj(field_2)) * weights_norm
    box_volume = np.prod(box_len)
    return power * box_volume


def get_gaussian_noise_floor(
    sigma_n,
    box_ndim,
    box_volume=1.0,
    counts=None,
):
    box_ndim = np.array(box_ndim)
    if counts is None:
        counts = np.ones(box_ndim.tolist())
    counts = np.array(counts)
    box_std = sigma_n / np.sqrt(counts)
    fourier_var = np.sum(box_std**2) / np.prod(box_ndim) ** 2
    return fourier_var * box_volume


def bin_3d_to_1d(
    ps3d,
    kfield,
    k1dedges,
    weights=None,
    error=False,
):
    r"""
    Bin a 3d distribution, e.g. power spectrum :math:`P_{3D}(\vec{k})`, into 1D average.

    Note that, the distribution is unraveled to a 1D array, so essentially an array of any
    dimension would do, as long as ``ps3d``, ``kfield``, and ``weights`` have the same size.

    The mean of the 1D average is calculated as

    .. math::
        \hat{P}_{\rm 1D}^{i} = \big(\sum_j P_{\rm 3D}^{ j} w_{ j} \big)/\big(\sum_j w_{ j}\big),

    where j loops over all the modess that fall into the :math:`i^{\rm th}` bin
    and :math:`w_{ j}` is the weights.

    If ``error`` is set to ``True``, a sampling error is calculated and returned so that

    .. math::
        (\Delta P_{\rm 1D}^{\rm i})^2 = \big(\sum_j (P_{\rm 3D}^{\rm j}-\hat{P}_{\rm 1D}^{\rm i})^2 w_{\rm j}^2 \big) \Big/ \big(\sum_j w_{\rm j}\big)^2.



    Parameters
    ----------
        ps3d: array.
            The 3D distribution to be binned
        weights2: array, None.
            If cross-correlation, the weights for the second field.

    Returns
    -------
        weights_norm: float.
           The renormalization coefficient.
    """
    ps3d = np.ravel(ps3d)
    kfield = np.ravel(kfield)
    if weights is None:
        weights = np.ones_like(ps3d)
    weights = np.array(weights).ravel()

    indx = (kfield[:, None] >= k1dedges[None, :-1]) * (
        kfield[:, None] < k1dedges[None, 1:]
    )
    ps1d = np.sum(ps3d[:, None] * indx * weights[:, None], 0) / np.sum(
        indx * weights[:, None], 0
    )
    k1deff = np.sum(kfield[:, None] * indx * weights[:, None], 0) / np.sum(
        indx * weights[:, None], 0
    )
    if error is True:
        ps1derr = np.sqrt(
            np.sum(
                (ps3d[:, None] - ps1d[None, :]) ** 2 * (indx * weights[:, None]) ** 2, 0
            )
            / np.sum((indx * weights[:, None]), 0) ** 2
        )
    nmodes = np.sum(indx * (weights[:, None] > 0), 0)

    if error is True:
        return ps1d, ps1derr, k1deff, nmodes
    else:
        return ps1d, k1deff, nmodes


def bin_3d_to_cy(
    ps3d,
    kperp_i,
    kperpedges,
    weights=None,
):
    ps3d = np.array(ps3d)
    kperpedges = np.array(kperpedges)
    kperp_i = np.array(kperp_i).ravel()
    ps3d = ps3d.reshape((len(kperp_i), -1))
    if weights is None:
        weights = np.ones_like(ps3d)
    weights = np.array(weights)
    indx = (kperp_i[:, None] >= kperpedges[None, :-1]) * (
        kperp_i[:, None] < kperpedges[None, 1:]
    )
    weights = indx[:, None, :] * weights[:, :, None]
    pscy = np.sum(ps3d[:, :, None] * weights, 0) / np.sum(weights, 0)
    return pscy


def get_independent_fourier_modes(box_dim):
    r"""
    Return a boolean array on whether the k-mode is independent.
    For real-valued signal, a specific k-mode :math:`\vec{k}` and it's opposite
    :math:`-\vec{k}` are conjugate to each other. This functions finds all the
    pairs and only assign one of them with ``True``.

    The indexing of the output array is consistent with the ``np.fft.fftfreq``
    convention.

    Parameters
    ----------
    box_dim: array.
        The shape of the signal.

    Returns
    -------
    unique: boolean array.
        Whether the k-mode is indendent.

    """
    kvec = get_k_vector(box_dim, np.ones(len(box_dim)))
    kvecmin = [(np.abs(kvec[i])[kvec[i] != 0]).min() for i in range(len(box_dim))]
    kvec = [kvec[i] / kvecmin[i] for i in range(len(box_dim))]
    kvecmax = [(np.abs(kvec[i])).max() for i in range(len(box_dim))]
    kvecmax = np.max(kvecmax)
    base = 2 * kvecmax + 1
    kvec = [kvec[i] * (base**i) for i in range(len(box_dim))]
    k_indx = np.sum(
        (np.meshgrid(*([(vec) for vec in kvec]), indexing="ij")),
        0,
    )
    _, indx = np.unique(np.abs(k_indx), return_index=True)
    unique = np.zeros(np.prod(box_dim))
    unique[indx] += 1
    unique = unique.reshape(box_dim) > 0
    return unique


def gaussian_beam_attenuation(k_perp, beam_sigma_in_mpc):
    """
    The beam attenuation term to be multiplied to model power
    spectrum assuming a Gaussian beam.

    Parameter
    ---------
    k_perp: float.
        The transverse k-scale in Mpc^-1
    beam_sigma_in_mpc: float.
        The sigma of the Gaussian beam in Mpc.
    """
    return np.exp(-(k_perp**2) * beam_sigma_in_mpc**2 / 2)


def step_window_attenuation(k_dir, step_size_in_mpc):
    """
    The beam attenuation term to be multiplied to model power
    spectrum assuming a Gaussian beam.

    Parameter
    ---------
    k_perp: float.
        The transverse k-scale in Mpc^-1
    beam_sigma_in_mpc: float.
        The sigma of the Gaussian beam in Mpc.
    """
    # note np.sinc is sin(pi x)/(pi x)
    return np.sinc(k_dir * step_size_in_mpc / np.pi / 2)


class PowerSpectrum(FieldPowerSpectrum, ModelPowerSpectrum):
    def __init__(
        self,
        field_1=None,
        box_len=None,
        weights_1=None,
        mean_center_1=False,
        unitless_1=False,
        remove_sn_1=False,
        field_2=None,
        weights_2=None,
        mean_center_2=False,
        unitless_2=False,
        remove_sn_2=False,
        corrtype=None,
        k1dbins=None,
        kmode=None,
        mumode=None,
        tracer_bias_1=1.0,
        sigma_v_1=0.0,
        tracer_bias_2=None,
        sigma_v_2=0.0,
        matter_only_rsd=False,
        include_beam=[True, False],
        fog_profile="lorentz",
        cross_coeff=1.0,
        model_k_from_field=False,
        mean_amp_1=1.0,
        mean_amp_2=1.0,
        sampling_resol=None,
        include_sampling=[True, False],
        **params,
    ):
        if field_1 is None:
            field_1 = np.ones([10, 10, 10])
        if box_len is None:
            box_len = np.array([10, 10, 10])
        FieldPowerSpectrum.__init__(
            self,
            field_1,
            box_len,
            weights_1=weights_1,
            mean_center_1=mean_center_1,
            unitless_1=unitless_1,
            remove_sn_1=remove_sn_1,
            field_2=field_2,
            weights_2=weights_2,
            mean_center_2=mean_center_2,
            unitless_2=unitless_2,
            remove_sn_2=remove_sn_2,
            corrtype=corrtype,
        )
        if model_k_from_field:
            # use field kmode to propagate into model
            kmode = self.k_mode
            mumode = self.k_para
            slice_indx = (None,) * (len(field_1.shape) - 1)
            slice_indx += (slice(None, None, None),)
            mumode = self.k_para[slice_indx] / kmode
        ModelPowerSpectrum.__init__(
            self,
            kmode=kmode,
            mumode=mumode,
            tracer_bias_1=tracer_bias_1,
            sigma_v_1=sigma_v_1,
            tracer_bias_2=tracer_bias_2,
            sigma_v_2=sigma_v_2,
            matter_only_rsd=matter_only_rsd,
            include_beam=include_beam,
            fog_profile=fog_profile,
            cross_coeff=cross_coeff,
            weights_1=weights_1,
            weights_2=weights_2,
            mean_amp_1=mean_amp_1,
            mean_amp_2=mean_amp_2,
            sampling_resol=sampling_resol,
            include_sampling=include_sampling,
            **params,
        )
        self.k1dbins = k1dbins

    def get_1d_power(
        self,
        power3d,
        k1dbins=None,
        k1dweights=None,
        filter_dependent_k=True,
    ):
        if k1dbins is None:
            k1dbins = self.k1dbins
        if k1dweights is None:
            k1dweights = np.ones_like(self.field_1)
        if isinstance(power3d, str):
            power3d = getattr(self, power3d)
        if filter_dependent_k:
            indep_modes = get_independent_fourier_modes(self.box_ndim)
        power1d, k1deff, nmodes = bin_3d_to_1d(
            power3d,
            self.k_mode,
            k1dbins,
            weights=indep_modes * k1dweights,
        )
        return power1d, k1deff, nmodes

    def step_sampling(self):
        if not self.has_resol:
            return 1.0
        k_x = self.k_vec[0][:, None, None]
        k_y = self.k_vec[1][None, :, None]
        k_para = self.k_mode * self.mumode
        sampling_resol = self.sampling_resol
        B_sampling = np.nan_to_num(
            step_window_attenuation(k_x, sampling_resol[0])
            * step_window_attenuation(k_y, sampling_resol[1])
            * step_window_attenuation(k_para, sampling_resol[2])
        )
        return B_sampling
