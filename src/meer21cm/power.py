"""
This power handles computation of power spectrum from gridded fields.
"""
import numpy as np


class PowerSpectrum:
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
        fullpk=False,
        k1dbins=None,
    ):
        self.fullpk = fullpk
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
        result = get_fourier_density(
            self.field_1,
            weights=self.weights_1,
            mean_center=self.mean_center_1,
            unitless=self.unitless_1,
        )
        return result

    @property
    def fourier_field_2(self):
        if self.field_2 is None:
            return None
        result = get_fourier_density(
            self.field_2,
            weights=self.weights_2,
            mean_center=self.mean_center_2,
            unitless=self.unitless_2,
        )
        return result

    @property
    def auto_power_3d_1(self):
        power_spectrum = get_power_spectrum(
            self.fourier_field_1,
            self.box_len,
            weights=self.weights_1,
        )
        if self.remove_sn_1:
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
        power_spectrum = get_power_spectrum(
            self.fourier_field_1,
            self.box_len,
            weights=self.weights_1,
            field_2=self.fourier_field_2,
            weights_2=self.weights_2,
        )
        return power_spectrum


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
    field = np.array(real_field)
    if weights is None:
        weights = np.ones_like(field)
    weights = np.array(weights)
    if mean_center or unitless:
        field_mean = np.sum(weights * real_field) / np.sum(weights)
    if mean_center:
        field -= field_mean
    if unitless:
        field /= field_mean
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
    if weights_2 is None:
        weights_2 = np.ones(fourier_field.shape)
    weights = np.array(weights)
    weights_2 = np.array(weights_2)
    weights_norm = fourier_field.size / np.sum(weights * weights_2)
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
    ps3d = np.ravel(ps3d)
    kfield = np.ravel(kfield)
    if weights is None:
        weights = np.ones_like(ps3d)
    weights = np.array(weights)

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
        nmodes = np.sum(indx * weights[:, None], 0)

    if error is True:
        return ps1d, ps1derr, nmodes, k1deff
    else:
        return ps1d, k1deff


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
    base = int(np.log10(kvecmax)) + 1
    kvec = [kvec[i] * 10 ** (base * i) for i in range(len(box_dim))]
    k_indx = np.sum(
        (np.meshgrid(*([(vec) for vec in kvec]), indexing="ij")),
        0,
    )
    _, indx = np.unique(np.abs(k_indx), return_index=True)
    unique = np.zeros(np.prod(box_dim))
    unique[indx] += 1
    unique = unique.reshape(box_dim) > 0
    return unique
