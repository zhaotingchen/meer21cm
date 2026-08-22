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
from meer21cm.window import (
    build_mesh_window_matrix,
    _bin_lag_phases,
    _extend_hermitian_z,
    _map_cell_stencils,
    _mode_index_grids,
    _pair_lag_scalars,
    map_sampling_shot_diagonal,
)
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


def _direct_mesh_model(fps, weights, k_in, theory_nodes):
    """Reference: bin[4πR Re Σ_m Y_ℓm(k̂) (FFT[wY_ℓm(n̂)]FFT[w]* ⊛ t·P_pw)].

    ``P_pw`` is the piecewise-constant theory on the k_in Voronoi shells, so
    the matrix apply (columns = unit-shell responses) must equal this
    exactly.  Uses full ``np.fft.fftn`` (not an rFFT Hermitian extension) so
    the reference is independent of ``_extend_hermitian_z``.
    """
    R = float(power_weights_renorm(weights, weights))
    n_grid = int(np.prod(np.asarray(weights).shape))
    w = np.asarray(weights, dtype=float)
    shape = w.shape
    w_full = np.fft.fftn(w, norm="forward")
    k_mode = np.asarray(fps.k_mode, dtype=float).ravel()
    edges = np.concatenate(([0.0], 0.5 * (k_in[:-1] + k_in[1:]), [np.inf]))
    # piecewise theory on the rFFT grid, then Hermitian-extend via full FFT
    # of an even real array built on the full grid
    nz = shape[2] // 2 + 1
    p_pw = np.zeros((shape[0], shape[1], nz), dtype=float)
    for j in range(len(k_in)):
        sel = (k_mode >= edges[j]) & (k_mode < edges[j + 1])
        p_pw.ravel()[sel] = theory_nodes[j]
    p_full = np.zeros(shape)
    p_full[..., :nz] = p_pw
    # isotropic |k|-shell: even under k -> -k, so z-flip alone is exact
    p_full[..., nz:] = np.flip(p_pw[..., 1 : shape[2] - nz + 1], axis=-1)
    xi_p = np.fft.ifftn(p_full)
    khat = unit_khat_from_k_vec(fps.k_vec)
    xhat = fps.los_xhat
    out = {}
    for ell in ELLS:
        cube = np.zeros((shape[0], shape[1], nz), dtype=complex)
        for m in range(-ell, ell + 1):
            ylm = get_real_Ylm(ell, m)
            c_full = np.fft.fftn(w * ylm(*xhat), norm="forward") * np.conj(w_full)
            conv = np.fft.fftn(np.fft.ifftn(c_full) * xi_p) * n_grid
            cube = cube + ylm(*khat) * conv[..., :nz]
        out[int(ell)] = _bin(fps, 4.0 * np.pi * R * np.real(cube))
    return out


def _far_observer():
    return np.array([-BOX_LEN[0] / 2, -BOX_LEN[1] / 2, 1.0e7], dtype=float)


def _true_observer():
    return np.array([-BOX_LEN[0] / 2, -BOX_LEN[1] / 2, 150.0], dtype=float)


def _asymmetric_mask(box_ndim, box_len):
    """Selection that is NOT centro-symmetric under (x,y) -> (-x,-y)."""
    base = _mask(box_ndim, box_len, sigma_perp=50.0, sigma_para=35.0)
    ndim = np.asarray(box_ndim, int)
    resol = np.asarray(box_len, float) / ndim
    x = (np.arange(ndim[0]) + 0.5) * resol[0] - box_len[0] / 2
    y = (np.arange(ndim[1]) + 0.5) * resol[1] - box_len[1] / 2
    z = (np.arange(ndim[2]) + 0.5) * resol[2] - box_len[2] / 2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    # offset Gaussian blob + an x-wedge
    blob = np.exp(
        -0.5
        * (
            (xx - 40.0) ** 2 / 25.0**2
            + (yy + 30.0) ** 2 / 20.0**2
            + zz**2 / 30.0**2
        )
    )
    wedge = np.clip(0.5 + 0.5 * np.tanh(xx / 15.0), 0.05, 1.0)
    return base * (0.4 + 0.6 * blob) * wedge


# ---------------------------------------------------------------------------
# 0. Hermitian extension correctness
# ---------------------------------------------------------------------------


def test_extend_hermitian_z_matches_fftn():
    """_extend_hermitian_z(rfftn(a), a.shape) must equal fftn(a)."""
    rng = np.random.default_rng(0)
    for shape in [(16, 16, 16), (15, 17, 18), (32, 24, 20)]:
        a = rng.standard_normal(shape)
        a[:3, 5:8, :] *= 3.0
        a[:, :2, :4] += 1.0
        ref = np.fft.fftn(a)
        got = _extend_hermitian_z(np.fft.rfftn(a), shape)
        assert np.allclose(got, ref, rtol=1e-12, atol=1e-12 * np.max(np.abs(ref))), (
            f"shape={shape}: max rel err "
            f"{np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)):.3e}"
        )


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


def test_mesh_window_matrix_matches_direct_convolution_asymmetric_window():
    """Same as above, but with a non-centro-symmetric window and an off-z
    observer — exercises the (x,y)->(-x,-y) part of the Hermitian extension.
    """
    weights = _asymmetric_mask(BOX_NDIM, BOX_LEN)
    observer = np.array([-80.0, 60.0, 120.0], dtype=float)
    fps = _make_fps(0, weights, observer)
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
        ), f"ell={ell}: asymmetric matrix != direct convolution"


def test_mesh_window_out_mode_scale_identity_matches_default():
    """out_mode_scale=1 and deconvolve_mas=False must match the default API."""
    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)
    rng = np.random.default_rng(1)
    ms = 0.6 + 0.4 * rng.random(fps.k_mode.shape)
    ref = build_mesh_window_matrix(fps, k_in, ells=ELLS, weights=weights, mode_scale=ms)
    got = build_mesh_window_matrix(
        fps,
        k_in,
        ells=ELLS,
        weights=weights,
        mode_scale=ms,
        out_mode_scale=np.ones_like(ms),
        deconvolve_mas=False,
    )
    assert np.allclose(
        got.matrix, ref.matrix, rtol=1e-12, atol=1e-12 * np.max(np.abs(ref.matrix))
    )


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
    # |w̃|² from full FFT so the reference does not use _extend_hermitian_z
    R = float(power_weights_renorm(weights, weights))
    n_grid = int(np.prod(np.asarray(weights).shape))
    w = np.asarray(weights, dtype=float)
    shape = w.shape
    w_full = np.fft.fftn(w, norm="forward")
    a_full = np.abs(w_full) ** 2
    nz = shape[2] // 2 + 1
    k_mode = np.asarray(fps.k_mode, dtype=float).ravel()
    edges = np.concatenate(([0.0], 0.5 * (k_in[:-1] + k_in[1:]), [np.inf]))
    p_pw = np.zeros((shape[0], shape[1], nz), dtype=float)
    for j in range(len(k_in)):
        sel = (k_mode >= edges[j]) & (k_mode < edges[j + 1])
        p_pw.ravel()[sel] = theory_nodes[j]
    p_full = np.zeros(shape)
    p_full[..., :nz] = p_pw
    p_full[..., nz:] = np.flip(p_pw[..., 1 : shape[2] - nz + 1], axis=-1)
    conv = np.fft.fftn(np.fft.ifftn(a_full) * np.fft.ifftn(p_full)) * n_grid
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


# ---------------------------------------------------------------------------
# 4. Map-sampling shot diagonal (exact per-cell b=b' term)
# ---------------------------------------------------------------------------


def _shot_fps(seed=0):
    """A FieldPowerSpectrum with the map-cell geometry attached (what the
    lightcone chain provides: pix_coor_in_box + grid_scheme)."""
    fps = _make_fps(seed, _mask(BOX_NDIM, BOX_LEN), _true_observer())
    rng = np.random.default_rng(11)
    n_cell = 600
    fps.grid_scheme = "cic"
    fps.pix_coor_in_box = rng.uniform([0.0, 0.0, 0.0], list(BOX_LEN), size=(n_cell, 3))
    return fps


def test_shot_diagonal_matches_direct_per_cell():
    """The per-lag stencil reconstruction of the b=b' diagonal equals the
    direct per-cell sum Σ_b m_b² |W_b(k)|² (machine precision)."""
    fps = _shot_fps()
    rng = np.random.default_rng(5)
    n_cell = fps.pix_coor_in_box.shape[0]
    m2 = rng.uniform(0.0, 2.0, n_cell)

    w, idx3 = _map_cell_stencils(fps)
    lags, B = _pair_lag_scalars(w, idx3, m2)
    valid = np.all(idx3 >= 0, axis=2)
    box_ndim = np.asarray(BOX_NDIM, dtype=int)
    Nx, Ny, Nz = box_ndim
    modes = [(3, 5, 2), (10, 7, 6), (23, 3, 9), (11, 11, 10)]

    for (i, j, k) in modes:
        # direct: |Σ_j w_j e^{-2πi idx.n/N}|² per cell, summed with m²
        ph = np.exp(
            -2.0j
            * np.pi
            * (idx3[:, :, 0] * i / Nx + idx3[:, :, 1] * j / Ny + idx3[:, :, 2] * k / Nz)
        )
        Wb = np.where(valid, w * ph, 0.0).sum(axis=1)
        direct = float(np.sum(m2 * np.abs(Wb) ** 2))
        # lag reconstruction: Σ_d B_d e^{-2πi d.n/N}
        ph_lag = np.exp(
            -2.0j
            * np.pi
            * (lags[:, 0] * i / Nx + lags[:, 1] * j / Ny + lags[:, 2] * k / Nz)
        )
        lag = float(np.sum(B * ph_lag).real)
        assert np.isclose(
            lag, direct, rtol=1e-12, atol=1e-12 * direct
        ), f"mode {(i, j, k)}: lag {lag:.6g} != direct {direct:.6g}"


def test_shot_offset_matches_helper():
    """The mesh matrix with map_m2: the P0 rows subtract the model's own
    diagonal per column and apply() adds the data's diagonal as an offset."""
    fps = _shot_fps()
    weights = _mask(BOX_NDIM, BOX_LEN)
    rng = np.random.default_rng(7)
    n_cell = fps.pix_coor_in_box.shape[0]
    m2 = rng.uniform(0.0, 2.0, n_cell)
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)

    mat_plain = build_mesh_window_matrix(
        fps, k_in, ells=ELLS, weights=weights, mode_scale=None
    )
    mat_shot = build_mesh_window_matrix(
        fps, k_in, ells=ELLS, weights=weights, mode_scale=None, map_m2=m2
    )
    # the correction is monopole-only: P2/P4 rows unchanged
    assert np.allclose(mat_shot.matrix[12:], mat_plain.matrix[12:])
    # P0 rows: plain − cols; apply() adds the offset
    shot = map_sampling_shot_diagonal(
        fps, weights=weights, mode_scale=None, map_m2=m2, k_in=k_in
    )
    n_out = len(fps.k1dbins) - 1
    diff = (mat_shot.matrix[0:n_out] - mat_plain.matrix[0:n_out]) @ theory_nodes
    expect = -(shot["cols"].T @ theory_nodes)
    assert np.allclose(diff, expect, rtol=1e-10, atol=1e-10 * np.max(np.abs(expect)))
    # apply: plain − cols@theory + offset == the corrected model
    model_shot = mat_shot.apply({0: theory_nodes})
    model_plain = mat_plain.apply({0: theory_nodes})
    assert np.allclose(
        model_shot[0],
        model_plain[0] + shot["offset"][0] - shot["cols"].T @ theory_nodes,
        rtol=1e-10,
        atol=1e-10 * np.max(np.abs(model_plain[0])),
    )
    # and P2/P4 are unchanged by the offset (monopole-only)
    for ell in (2, 4):
        assert np.allclose(model_shot[ell], model_plain[ell])
