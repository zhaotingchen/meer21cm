import numpy as np


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
    if weights is None:
        weights = np.ones_like(real_field)
    if mean_center or unitless:
        field_mean = np.sum(weights * real_field) / np.sum(weights)
    if mean_center:
        real_field -= field_mean
    if unitless:
        real_field /= field_mean
    fourier_field = np.fft.fftn(real_field * weights, norm=norm)
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
            np.meshgrid(*([vec**2 for vec in vecarr]), indexing="ij"),
            axis=0,
        )
    )
    return result


def get_power_spectrum(
    fourier_field,
    box_len,
    weights=None,
    field_2=None,
    weights_2=None,
):
    if field_2 is None:
        field_2 = fourier_field
    if weights is None:
        weights = np.ones(fourier_field.shape)
    if weights_2 is None:
        weights_2 = np.ones(fourier_field.shape)
    weights_norm = fourier_field.size / np.sum(weights * weights_2)
    power = np.real(fourier_field * np.conj(field_2)) * weights_norm
    box_volume = np.prod(box_len)
    return power * box_volume
