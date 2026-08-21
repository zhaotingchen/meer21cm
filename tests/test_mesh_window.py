"""
Unit tests for the exact mesh-level (FFT) window.

``build_mesh_window_matrix`` replaces the smooth (Hankel/Wigner) continuous
layer of the window formalism with the estimator's own mesh response, so the
windowed multipoles are exact for **any** LOS — including the varying
``n̂(x)`` of a true lightcone observer, where the discrete-μ projector is
out of validity (the 1/d expansion diverges).

Checks:

1. the matrix equals the direct full-grid convolution model
   (piecewise theory on the k_in shells) — implementation accuracy;
2. a far observer reduces the mesh matrix to the discrete-μ model
   ``(2ℓ+1)L_ℓ(k̂·n̂_ref)|w̃|²`` (the Y_ℓm machinery vs an independent
   Legendre binning), and the monopole is n̂-independent while P2/P4 are not;
3. end-to-end: the **true-observer** Yamamoto estimator of a windowed
   Gaussian field agrees with the mesh model within the seed scatter
   (median |Δ/σ| ≤ 2 for P0, P2 and P4) — the accuracy the mesh window is
   built for.
"""

import numpy as np
import pytest

from meer21cm.estimator import FieldPowerSpectrum
from meer21cm.mock import generate_gaussian_field
from meer21cm.power_ops import bin_3d_to_1d, power_weights_renorm
from meer21cm.window import build_mesh_window_matrix
from meer21cm.spherical import get_real_Ylm, unit_khat_from_k_vec
from meer21cm.util import get_nd_slicer

ELLS = (0, 2, 4)
BOX_LEN = (200.0, 200.0, 200.0)
BOX_NDIM = (48, 48, 48)
N_K_IN = 40
N_SEED = 16


def _theory_grid(k_mode, k0=0.05, amp=1.0e4, index=1.5):
    """Isotropic power-law theory on the rFFT grid."""
    k = np.asarray(k_mode, dtype=float)
    p = np.zeros_like(k)
    p[k > 0] = amp * (k[k > 0] / k0) ** (-index)
    return p


def _mask(box_ndim, box_len, sigma_perp=60.0, sigma_para=40.0):
    """Anisotropic ellipsoidal Gaussian selection (deterministic)."""
    ndim = np.asarray(box_ndim, int)
    resol = np.asarray(box_len, float) / ndim
    x = (np.arange(ndim[0]) + 0.5) * resol[0] - box_len[0] / 2
    y = (np.arange(ndim[1]) + 0.5) * resol[1] - box_len[1] / 2
    z = (np.arange(ndim[2]) + 0.5) * resol[2] - box_len[2] / 2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.exp(
        -0.5
        * (
            xx**2 / sigma_perp**2
            + yy**2 / sigma_perp**2
            + zz**2 / sigma_para**2
        )
    )


def _configure_bins(fps, k_hi=0.18):
    k_nyq = np.asarray(fps.k_nyquist, dtype=float)
    slicer = get_nd_slicer()
    w = np.ones_like(fps.k_mode, dtype=float)
    for i in range(3):
        w = w * (np.abs(fps.k_vec[i])[slicer[i]] <= 0.5 * k_nyq[i])
    w[0, 0, 0] = 0.0
    fps.k1dweights = w
    # start above the box fundamental (2π/L = 0.0314) so every bin has modes
    fps.k1dbins = np.linspace(0.035, k_hi, 10)
    return fps


def _make_fps(seed, weights, los_observer, n_seed=None):
    """A windowed Gaussian realisation with the given observer."""
    k_mode = np.asarray(
        FieldPowerSpectrum(
            np.zeros(BOX_NDIM, dtype="f4"),
            BOX_LEN,
            _skip_specification=True,
        ).k_mode,
        dtype=float,
    )
    if n_seed is None:
        n_seed = seed
    field = generate_gaussian_field(
        BOX_NDIM, BOX_LEN, _theory_grid(k_mode), int(n_seed)
    )
    fps = FieldPowerSpectrum(
        field,
        BOX_LEN,
        weights_1=weights,
        los="endpoint",
        los_observer=los_observer,
        _skip_specification=True,
    )
    return _configure_bins(fps)


def _bin(fps, p3d):
    """Estimator-style |k|-shell binning with the fps k1dweights."""
    p1d, _, _ = bin_3d_to_1d(
        p3d[None],
        np.asarray(fps.k_mode),
        fps.k1dbins,
        vectorize=True,
        weights=fps.k1dweights,
    )
    return np.asarray(p1d, float)[0]


def _hermitian_z(c_rfft, shape):
    nz = c_rfft.shape[2]
    out = np.zeros(shape, dtype=np.result_type(c_rfft, complex))
    out[..., :nz] = c_rfft
    out[..., nz:] = np.conj(np.flip(c_rfft[..., 1 : shape[2] - nz + 1], axis=-1))
    return out


def _direct_mesh_model(fps, weights, k_in, theory_nodes):
    """Reference: bin[4πR Re Σ_m Y_ℓm(k̂) (FFT[wY_ℓm(n̂)]FFT[w]* ⊛ t·P_pw)].

    ``P_pw`` is the piecewise-constant theory on the k_in Voronoi shells, so
    the matrix apply (columns = unit-shell responses) must equal this
    exactly.
    """
    R = float(power_weights_renorm(weights, weights))
    n_grid = int(np.prod(np.asarray(weights).shape))
    w_tilde = np.fft.rfftn(weights, norm="forward")
    shape = np.asarray(weights).shape
    nz = w_tilde.shape[2]
    k_mode = np.asarray(fps.k_mode, dtype=float).ravel()
    edges = np.concatenate(([0.0], 0.5 * (k_in[:-1] + k_in[1:]), [np.inf]))
    p_pw = np.zeros(w_tilde.shape, dtype=float)
    for j in range(len(k_in)):
        sel = (k_mode >= edges[j]) & (k_mode < edges[j + 1])
        p_pw.ravel()[sel] = theory_nodes[j]
    p_full = np.zeros(shape)
    p_full[..., :nz] = p_pw
    p_full[..., nz:] = np.flip(p_pw[..., 1 : shape[2] - nz + 1], axis=-1)
    xi_p = np.fft.ifftn(p_full)
    khat = unit_khat_from_k_vec(fps.k_vec)
    xhat = fps.los_xhat
    out = {}
    for ell in ELLS:
        cube = np.zeros(w_tilde.shape, dtype=complex)
        for m in range(-ell, ell + 1):
            ylm = get_real_Ylm(ell, m)
            c_rfft = np.fft.rfftn(weights * ylm(*xhat), norm="forward") * np.conj(
                w_tilde
            )
            conv = (
                np.fft.fftn(np.fft.ifftn(_hermitian_z(c_rfft, shape)) * xi_p) * n_grid
            )
            cube = cube + ylm(*khat) * conv[..., :nz]
        out[int(ell)] = _bin(fps, 4.0 * np.pi * R * np.real(cube))
    return out


def _far_observer():
    return np.array([-BOX_LEN[0] / 2, -BOX_LEN[1] / 2, 1.0e7], dtype=float)


def _true_observer():
    return np.array([-BOX_LEN[0] / 2, -BOX_LEN[1] / 2, 150.0], dtype=float)


# ---------------------------------------------------------------------------
# 1. Implementation accuracy: matrix == direct convolution
# ---------------------------------------------------------------------------


def test_mesh_window_matrix_matches_direct_convolution():
    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)
    mat = build_mesh_window_matrix(
        fps, k_in, ells=ELLS, weights=weights, mode_scale=None
    )
    model = mat.apply({0: theory_nodes})
    direct = _direct_mesh_model(fps, weights, k_in, theory_nodes)
    for ell in ELLS:
        assert np.allclose(
            model[ell], direct[ell], rtol=1e-8, atol=1e-8 * np.max(np.abs(direct[ell]))
        ), f"ell={ell}: matrix != direct convolution"
    assert mat.ells_in == (0,)


# ---------------------------------------------------------------------------
# 2. Far-observer limit: reduces to the discrete-μ model; n̂-independence
# ---------------------------------------------------------------------------


def test_mesh_window_far_observer_matches_discrete_mu():
    """Far observer: the mesh cube is (2ℓ+1)L_ℓ(k̂·n̂_ref)|w̃|², so the
    binned result must equal the discrete-μ binning of the scalar window."""
    from scipy.special import eval_legendre

    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _far_observer())
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)
    mat = build_mesh_window_matrix(
        fps, k_in, ells=ELLS, weights=weights, mode_scale=None
    )
    model = mat.apply({0: theory_nodes})

    # discrete-μ reference: R bin[(2ℓ+1)L_ℓ(μ_n) (|w̃|² ⊛ P_pw)]
    R = float(power_weights_renorm(weights, weights))
    n_grid = int(np.prod(np.asarray(weights).shape))
    w_tilde = np.fft.rfftn(weights, norm="forward")
    a_full = np.abs(w_tilde) ** 2
    shape = np.asarray(weights).shape
    nz = a_full.shape[2]
    k_mode = np.asarray(fps.k_mode, dtype=float).ravel()
    edges = np.concatenate(([0.0], 0.5 * (k_in[:-1] + k_in[1:]), [np.inf]))
    p_pw = np.zeros(a_full.shape, dtype=float)
    for j in range(len(k_in)):
        sel = (k_mode >= edges[j]) & (k_mode < edges[j + 1])
        p_pw.ravel()[sel] = theory_nodes[j]
    p_full = np.zeros(shape)
    p_full[..., :nz] = p_pw
    p_full[..., nz:] = np.flip(p_pw[..., 1 : shape[2] - nz + 1], axis=-1)
    conv = (
        np.fft.fftn(np.fft.ifftn(_hermitian_z(a_full, shape)) * np.fft.ifftn(p_full))
        * n_grid
    )
    conv = R * np.real(conv)[..., :nz]
    mu = np.asarray(fps.mu_mode, dtype=float)
    ref = {}
    for ell in ELLS:
        f = (2 * ell + 1) * eval_legendre(ell, mu)
        ref[int(ell)] = _bin(fps, conv * f)
    for ell in ELLS:
        assert np.allclose(
            model[ell], ref[ell], rtol=1e-6, atol=1e-6 * np.max(np.abs(ref[ell]))
        ), f"ell={ell}: far-observer mesh != discrete-μ"


def test_mesh_window_monopole_is_los_independent():
    weights = _mask(BOX_NDIM, BOX_LEN)
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)

    fps_true = _make_fps(0, weights, _true_observer())
    m_true = build_mesh_window_matrix(
        fps_true, k_in, ells=ELLS, weights=weights, mode_scale=None
    ).apply({0: theory_nodes})
    fps_far = _make_fps(0, weights, _far_observer())
    m_far = build_mesh_window_matrix(
        fps_far, k_in, ells=ELLS, weights=weights, mode_scale=None
    ).apply({0: theory_nodes})

    # the monopole is n̂-independent (|F0|² does not involve n̂) ...
    assert np.allclose(
        m_true[0], m_far[0], rtol=1e-10, atol=1e-10 * np.max(np.abs(m_far[0]))
    )
    # ... while the quadrupole/hexadecapole carry the wide-angle structure
    assert not np.allclose(m_true[2], m_far[2], rtol=1e-3)
    assert not np.allclose(m_true[4], m_far[4], rtol=1e-3)


# ---------------------------------------------------------------------------
# 3. End-to-end: true-observer Yamamoto vs the mesh model, within the noise
# ---------------------------------------------------------------------------


def test_mesh_window_true_observer_p2p4_within_noise():
    """The mesh window must predict the true-observer Yamamoto multipoles of
    a windowed isotropic field within the seed scatter (median |Δ/σ| ≤ 2)."""
    weights = _mask(BOX_NDIM, BOX_LEN)
    fps0 = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)
    mat = build_mesh_window_matrix(
        fps0, k_in, ells=ELLS, weights=weights, mode_scale=None
    )
    model = mat.apply({0: theory_nodes})

    data = {ell: [] for ell in ELLS}
    for seed in range(N_SEED):
        fps = _make_fps(seed, weights, _true_observer(), n_seed=seed)
        meas = fps.measure_multipoles(
            which="auto_1",
            k1dbins=fps.k1dbins,
            ells=ELLS,
            k1dweights=fps.k1dweights,
        )
        for ell in ELLS:
            data[ell].append(np.asarray(meas.P_ell[ell], dtype=float))

    for ell in ELLS:
        stack = np.stack(data[ell], axis=0)
        mean = np.nanmean(stack, axis=0)
        sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(N_SEED)
        dsig = (mean - model[ell]) / np.where(sem > 0, sem, np.nan)
        finite = np.isfinite(dsig)
        assert finite.any()
        median = float(np.nanmedian(np.abs(dsig[finite])))
        assert median <= 2.0, (
            f"ell={ell}: median |data-model|/sigma = {median:.2f} > 2 "
            f"(model under-predicts the true-observer multipoles)"
        )
        # signal sanity: the quadrupole/hexadecapole are non-trivial here
        assert np.median(np.abs(mean[2])) > 0.05 * np.median(np.abs(mean[0]))
