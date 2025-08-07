import numpy as np
from meer21cm.util import pcaclean, dft_matrix, inv_dft_matrix


def fft_matrix(mat, norm="backward"):
    r"""
    Perform the Fourier transform of a matrix.

    .. math::
        \tilde{M} = \mathcal{F} M \mathcal{F}^{-1}

    where :math:`\mathcal{F}` is the Fourier transform matrix.
    See also :func:`meer21cm.util.dft_matrix`.

    Parameters
    ----------
        mat: np.ndarray
            The matrix to be transformed.
        norm: str, default "backward"
            The normalization of the Fourier transform.

    Returns
    -------
        mat_fft: np.ndarray
            The Fourier transformed matrix.
    """

    return (
        dft_matrix(mat.shape[0], norm=norm)
        @ mat
        @ inv_dft_matrix(mat.shape[0], norm=norm)
    )


def analytic_transfer_function(clean_mat_1, clean_mat_2=None):
    r"""
    Calculate the analytic transfer function of a clean matrix.
    See Chen 2025 [1] for derivations.

    For a foreground cleaning matrix :math:`R_{ab}`,
    the residual data vector for power spectrum estimation is :math:`r_{ija} = \sum_b R_{ab} m_{ijb}`,
    where i,j are the pixel indices, and a,b are the frequency indices.

    Under the flat sky approximation, the signal loss, as well as the mode-mixing,
    is along the line-of-sight (k_para) direction.
    The unnormalised window function matrix is

    .. math::
        H_{ab} = |\tilde{R}^1_{ab} (\tilde{R}^2_{ab})^*|_{\rm Re}

    The corresponding signal loss is :math:`\sum_b H_{ab}`, and the analytical
    transfer function is the inverse of the signal loss.

    After normalisation, the window function matrix is

    .. math::
        W = {\rm diag}\Big(\sum_b H_{ab}\Big)^{-1} H

    Parameters
    ----------
        clean_mat_1: np.ndarray
            The clean matrix that applies to the data vector.
        clean_mat_2: np.ndarray, optional
            The clean matrix that applies to the second data vector for cross-correlation.
            If not provided, it is assumed to be the same as clean_mat_1 for auto-correlation.

    Returns
    -------
        transfer_func: np.ndarray
            The analytical transfer function.
        Wab: np.ndarray
            The normalised window function matrix.

    References
    ----------
    .. [1] Chen, Z.,, "A quadratic estimator view of the transfer function correction in intensity mapping surveys", https://ui.adsabs.harvard.edu/abs/2025MNRAS.542L...1C/abstract.
    """
    assert (clean_mat_1.ndim == 2) and (clean_mat_1.shape[0] == clean_mat_1.shape[1])
    if clean_mat_2 is None:
        clean_mat_2 = clean_mat_1
    assert np.allclose(clean_mat_1.shape, clean_mat_2.shape)
    num_k = clean_mat_1.shape[0] // 2 + 1
    R_mat_fourier_1 = fft_matrix(clean_mat_1)
    R_mat_fourier_2 = fft_matrix(clean_mat_2)
    Hab = (np.conj(R_mat_fourier_1) * R_mat_fourier_2).real
    Hab = Hab[:num_k, :num_k]
    signal_loss = Hab.sum(1)
    renorm_mat = np.diag(1 / signal_loss)
    Wab = renorm_mat @ Hab
    return 1 / signal_loss, Wab
