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


def _direct_mesh_model(fps, weights, k_in, theory_nodes, out_mode_scale=None):
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
        p3d = 4.0 * np.pi * R * np.real(cube)
        if out_mode_scale is not None:
            p3d = p3d * np.asarray(out_mode_scale, dtype=float)
        out[int(ell)] = _bin(fps, p3d)
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

    for i, j, k in modes:
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


def test_ngp_raw_cell_comb_and_mas_out_api():
    """Library helpers assemble the preferred MAS-out mesh recipe."""
    from meer21cm.window import build_mesh_window_mas_out, ngp_raw_cell_comb

    fps = _shot_fps()
    comb = ngp_raw_cell_comb(fps)
    assert comb.shape == tuple(BOX_NDIM)
    assert float(comb.sum()) == pytest.approx(float(fps.pix_coor_in_box.shape[0]))

    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    weights = _mask(BOX_NDIM, BOX_LEN)
    mat = build_mesh_window_mas_out(
        fps, k_in, renorm_weights=weights, ells=ELLS, raw_comb=weights
    )
    assert mat.matrix.shape[0] == len(ELLS) * (len(fps.k1dbins) - 1)


def test_map_sampling_mode_scale_level0():
    """Level-0 map sampling matches mean-sinc |S|^2."""
    from meer21cm.multipole_ops import map_sampling_mode_scale
    from meer21cm.power_ops import step_window_attenuation
    from meer21cm.util import get_nd_slicer

    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _far_observer())
    fps.pix_resol_in_mpc = 5.0
    fps.los_resol_in_mpc = 4.0
    s = map_sampling_mode_scale(fps, z_resolved=False)
    slicer = get_nd_slicer()
    kx = np.asarray(fps.k_vec[0][slicer[0]], dtype=float)
    ky = np.asarray(fps.k_vec[1][slicer[1]], dtype=float)
    kz = np.asarray(fps.k_vec[2][slicer[2]], dtype=float)
    expect = (
        step_window_attenuation(kx, 5.0, p=2)
        * step_window_attenuation(ky, 5.0, p=2)
        * step_window_attenuation(kz, 4.0, p=2)
    )
    assert np.allclose(s, expect)


# ---------------------------------------------------------------------------
# 5. Extended-q path (Band 2 / Band 3) — additive, defaults unchanged
# ---------------------------------------------------------------------------


def test_cell_sampling_kernel_matches_radial_sinc():
    """Aligned cells + q ∥ n̂ recover sinc(q Δ_∥ / 2)."""
    from meer21cm.multipole_ops import cell_sampling_kernel
    from meer21cm.power_ops import step_window_attenuation

    n_cell = 32
    nhat = np.zeros((n_cell, 3))
    nhat[:, 2] = 1.0
    dperp = np.full(n_cell, 5.0)
    dpar = np.full(n_cell, 4.0)
    q_abs = 0.02
    s = cell_sampling_kernel((0.0, 0.0, 1.0), q_abs, nhat, dperp, dpar)
    expect = np.sqrt(step_window_attenuation(q_abs, 4.0, p=2))
    assert np.allclose(s, expect)
    # low-q |S|² vs level-0 radial factor
    t0 = step_window_attenuation(q_abs, 4.0, p=2)
    assert abs(float(np.mean(s**2)) / t0 - 1.0) < 1e-3


def test_cell_sampling_kernel_lowq_matches_level0_mean():
    """Angle-averaged |S_b|² tracks the survey-mean sinc at low q."""
    from meer21cm.multipole_ops import (
        cell_sampling_kernel_mu_rms,
        map_sampling_mode_scale,
    )

    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _far_observer())
    fps.pix_resol_in_mpc = 5.0
    fps.los_resol_in_mpc = 4.0
    s0 = map_sampling_mode_scale(fps, z_resolved=False)
    k = np.asarray(fps.k_mode, dtype=float)
    low = (k > 0.035) & (k < 0.055)
    t_mean = float(np.mean(s0[low]))
    # isotropic cells with the same mean widths
    n_cell = 64
    dperp = np.full(n_cell, 5.0)
    dpar = np.full(n_cell, 4.0)
    q_abs = 0.045
    rms = cell_sampling_kernel_mu_rms(q_abs, dperp, dpar, nmu=16)
    assert abs(float(np.mean(rms**2)) / t_mean - 1.0) < 2e-3


# ---------------------------------------------------------------------------
# 6. Beam transfer at the output mode
# ---------------------------------------------------------------------------


def test_beam_out_mode_scale_level0_matches_gaussian_attenuation():
    """Level 0 is the legacy Gaussian B(k_perp)^2 on the box-frame k_perp."""
    from meer21cm.multipole_ops import beam_out_mode_scale
    from meer21cm.power_ops import gaussian_beam_attenuation
    from meer21cm.util import get_nd_slicer

    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _far_observer())
    sigma = 12.0
    got = beam_out_mode_scale(fps, level=0, sigma_beam_in_mpc=sigma)
    slicer = get_nd_slicer()
    kx = np.asarray(fps.k_vec[0][slicer[0]], dtype=float)
    ky = np.asarray(fps.k_vec[1][slicer[1]], dtype=float)
    kz = np.asarray(fps.k_vec[2][slicer[2]], dtype=float)
    expect = (
        gaussian_beam_attenuation(np.sqrt(kx**2 + ky**2 + 0.0 * kz), sigma) ** 2
    )
    assert np.allclose(got, expect)


def test_beam_ylm_decomposition_matches_cell_average():
    """Addition-theorem <B> equals the brute-force cell mean on a small set."""
    from meer21cm.multipole_ops import mean_gaussian_beam_on_modes
    from meer21cm.spherical import get_real_Ylm

    rng = np.random.default_rng(4)
    nhat = rng.normal(size=(48, 3))
    nhat = nhat / np.linalg.norm(nhat, axis=1)[:, None]
    sigma_ch = np.array([8.0, 10.0, 14.0])
    k_hat = np.array([0.3, 0.4, np.sqrt(1.0 - 0.3**2 - 0.4**2)])
    k_abs = 0.07
    mu = nhat @ k_hat
    brute = 0.0
    for sig in sigma_ch:
        brute += float(np.mean(np.exp(-0.5 * (k_abs * sig) ** 2 * (1.0 - mu**2))))
    brute /= len(sigma_ch)
    # single-mode arrays with the same broadcasting as the rFFT helpers
    k_abs_a = np.array([k_abs])
    khat_a = (np.array([k_hat[0]]), np.array([k_hat[1]]), np.array([k_hat[2]]))
    got = mean_gaussian_beam_on_modes(
        k_abs_a, khat_a, nhat, sigma_ch, coherent=True, ell_max=16, nmu=64
    )
    assert abs(float(got[0]) / brute - 1.0) < 2e-4
    del get_real_Ylm


def test_beam_level1_reduces_to_level0_for_narrow_footprint():
    """Constant sigma and n̂ ≈ ẑ recovers the box-frame k_perp Gaussian."""
    from meer21cm.multipole_ops import (
        beam_out_mode_scale,
        mean_gaussian_beam_on_modes,
    )
    from meer21cm.power_ops import gaussian_beam_attenuation
    from meer21cm.spherical import unit_khat_from_k_vec
    from meer21cm.util import get_nd_slicer

    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _far_observer())
    sigma = 10.0
    slicer = get_nd_slicer()
    kx = np.asarray(fps.k_vec[0][slicer[0]], dtype=float)
    ky = np.asarray(fps.k_vec[1][slicer[1]], dtype=float)
    kz = np.asarray(fps.k_vec[2][slicer[2]], dtype=float)
    k_abs = np.sqrt(kx**2 + ky**2 + kz**2)
    khat = unit_khat_from_k_vec(fps.k_vec)
    nhat = np.zeros((16, 3))
    nhat[:, 2] = 1.0
    mean_b = mean_gaussian_beam_on_modes(
        k_abs, khat, nhat, [sigma], coherent=True, ell_max=12, nmu=48
    )
    lvl0 = gaussian_beam_attenuation(np.sqrt(kx**2 + ky**2 + 0.0 * kz), sigma)
    # In-zone k of the coarse lightcone (k σ ≲ 1); high-k modes on this
    # periodic test box need a much larger L expansion.
    ok = (k_abs > 0.02) & (k_abs < 0.10)
    rel = np.abs(mean_b[ok] / np.maximum(lvl0[ok], 1.0e-12) - 1.0)
    assert float(np.median(rel)) < 5e-3
    # level-0 helper still matches the closed form
    assert np.allclose(
        beam_out_mode_scale(fps, level=0, sigma_beam_in_mpc=sigma),
        lvl0**2,
    )


def test_mesh_window_beam_out_mode_scale_matches_direct():
    """Mesh matrix with a beam out_mode_scale equals the direct convolution."""
    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)
    from meer21cm.power_ops import gaussian_beam_attenuation
    from meer21cm.util import get_nd_slicer

    slicer = get_nd_slicer()
    kx = np.asarray(fps.k_vec[0][slicer[0]], dtype=float)
    ky = np.asarray(fps.k_vec[1][slicer[1]], dtype=float)
    kz = np.asarray(fps.k_vec[2][slicer[2]], dtype=float)
    beam = gaussian_beam_attenuation(np.sqrt(kx**2 + ky**2 + 0.0 * kz), 15.0) ** 2
    mat = build_mesh_window_matrix(
        fps, k_in, ells=ELLS, weights=weights, out_mode_scale=beam
    )
    model = mat.apply({0: theory_nodes})
    direct = _direct_mesh_model(fps, weights, k_in, theory_nodes, out_mode_scale=beam)
    for ell in ELLS:
        assert np.allclose(
            model[ell], direct[ell], rtol=1e-8, atol=1e-8 * np.max(np.abs(direct[ell]))
        ), f"ell={ell}: beam out_mode_scale matrix != direct"


def test_mean_beam_amplitude_on_cells_matches_direct():
    """Y_LM cell amplitude equals the brute-force Gaussian at one mode."""
    from meer21cm.multipole_ops import mean_beam_amplitude_on_cells

    rng = np.random.default_rng(5)
    nhat = rng.normal(size=(32, 3))
    nhat = nhat / np.linalg.norm(nhat, axis=1)[:, None]
    sigma_b = np.full(32, 11.0)
    k_hat = np.array([0.2, 0.5, np.sqrt(1.0 - 0.2**2 - 0.5**2)])
    k_abs = 0.055
    mu = nhat @ k_hat
    brute = np.exp(-0.5 * (k_abs * 11.0) ** 2 * (1.0 - mu**2))
    got = mean_beam_amplitude_on_cells(
        np.array([k_abs]),
        (np.array([k_hat[0]]), np.array([k_hat[1]]), np.array([k_hat[2]])),
        nhat,
        sigma_b,
        ell_max=12,
        nmu=64,
    )
    assert float(np.max(np.abs(got / brute - 1.0))) < 3e-4


def test_mesh_window_constant_bin_mass_equals_out_mode_scale():
    """Cell-independent B_b in out_bin_weights equals scalar B^2 out_mode_scale."""
    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)
    from meer21cm.power_ops import gaussian_beam_attenuation
    from meer21cm.util import get_nd_slicer

    slicer = get_nd_slicer()
    kx = np.asarray(fps.k_vec[0][slicer[0]], dtype=float)
    ky = np.asarray(fps.k_vec[1][slicer[1]], dtype=float)
    kz = np.asarray(fps.k_vec[2][slicer[2]], dtype=float)
    b_amp = gaussian_beam_attenuation(np.sqrt(kx**2 + ky**2 + 0.0 * kz), 15.0)
    # A spatially constant mass c scales xi by c^2, matching out_mode_scale=c^2.
    # Use one c for every bin so the two matrices are identical.
    c = 0.7
    n_out = len(np.asarray(fps.k1dbins)) - 1
    out_bin_weights = [c * weights for _ in range(n_out)]
    mat_b3 = build_mesh_window_matrix(
        fps, k_in, ells=ELLS, weights=weights, out_bin_weights=out_bin_weights
    )
    mat_b2 = build_mesh_window_matrix(
        fps,
        k_in,
        ells=ELLS,
        weights=weights,
        out_mode_scale=np.full_like(b_amp, c**2),
    )
    model_b3 = mat_b3.apply({0: theory_nodes})
    model_b2 = mat_b2.apply({0: theory_nodes})
    for ell in ELLS:
        assert np.allclose(
            model_b3[ell],
            model_b2[ell],
            rtol=1e-8,
            atol=1e-8 * np.max(np.abs(model_b2[ell])),
        ), f"ell={ell}: constant-mass B3 != scalar B^2"


def test_beam_mode_group_index_partitions_estimator_modes():
    """(|k| bin, |mu|) groups tile exactly the modes the estimator bins."""
    from meer21cm.multipole_ops import beam_mode_group_index

    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    n_out = len(np.asarray(fps.k1dbins)) - 1

    idx1, n1 = beam_mode_group_index(fps, n_mu=1)
    assert n1 == n_out
    idx3, n3 = beam_mode_group_index(fps, n_mu=3)
    assert n3 == 3 * n_out
    # same set of modes is covered, and each group maps back to its own bin
    assert np.array_equal(idx1 >= 0, idx3 >= 0)
    assert np.array_equal(idx3[idx3 >= 0] // 3, idx1[idx1 >= 0])
    counts = np.bincount(idx3[idx3 >= 0].ravel(), minlength=n3)
    for i in range(n_out):
        n_bin = int(np.sum(idx1 == i))
        if n_bin < 3:
            continue
        sub = counts[3 * i : 3 * i + 3]
        assert sub.sum() == n_bin
        # equal-count quantiles: no group may be empty for a populated bin
        assert sub.min() > 0


def test_cell_grid_los_reproduces_the_deposited_cube():
    """The kernel's zero mode must equal the cell sum built on cell_grid_los.

    beam_diagonal_correction subtracts a cell-space mean field from the
    exact one; if the LOS bookkeeping does not match what
    ngp_raw_cell_comb + los_xhat actually produce, the subtraction leaves
    a spurious ell>0 residual instead of removing one.
    """
    from meer21cm.multipole_ops import cell_grid_los
    from meer21cm.spherical import get_real_Ylm
    from meer21cm.window import ngp_raw_cell_comb

    ps = _shot_fps()
    n_cell = np.asarray(ps.pix_coor_in_box).reshape(-1, 3).shape[0]
    rng = np.random.default_rng(13)
    mass = rng.uniform(0.3, 1.0, n_cell)
    cube = ngp_raw_cell_comb(ps, particle_mass=mass)
    n_grid = float(np.prod(np.asarray(ps.box_ndim, int)))

    nhat_leg, inside = cell_grid_los(ps)
    for ell in (0, 2, 4):
        for m in range(-ell, ell + 1):
            ylm = get_real_Ylm(ell, m)
            from_cube = float(np.mean(cube * ylm(*ps.los_xhat)))
            y_cell = np.asarray(
                ylm(nhat_leg[:, 0], nhat_leg[:, 1], nhat_leg[:, 2]), float
            ) * np.ones(n_cell)
            from_cells = float(np.sum(mass * inside * y_cell)) / n_grid
            scale = max(abs(from_cube), 1e-30)
            assert (
                abs(from_cells - from_cube) < 1e-10 * scale
            ), f"ell={ell} m={m}: {from_cells:.6e} vs {from_cube:.6e}"


def test_mesh_window_diag_correction_is_a_pure_diagonal():
    """diag_correction adds T(k) P(k) only: no leakage between k_in shells."""
    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    rng = np.random.default_rng(11)
    corr = {
        (ell, m): rng.normal(size=np.asarray(fps.k_mode).shape) * 1e-3
        for ell in ELLS
        for m in range(-ell, ell + 1)
    }
    mat0 = build_mesh_window_matrix(fps, k_in, ells=ELLS, weights=weights)
    mat1 = build_mesh_window_matrix(
        fps, k_in, ells=ELLS, weights=weights, diag_correction=corr
    )
    delta = mat1.matrix - mat0.matrix
    # a kappa=0 term can only move power within a shell, so the added
    # matrix must vanish wherever the theory shell misses the output bin
    n_out = len(np.asarray(fps.k1dbins)) - 1
    k_mode = np.asarray(fps.k_mode, float).ravel()
    edges = np.concatenate(([0.0], 0.5 * (k_in[:-1] + k_in[1:]), [np.inf]))
    bin_idx = np.digitize(k_mode, np.asarray(fps.k1dbins, float)) - 1
    for j in range(len(k_in)):
        in_j = (k_mode >= edges[j]) & (k_mode < edges[j + 1])
        touched = set(np.unique(bin_idx[in_j]).tolist())
        for i in range(n_out):
            if i in touched:
                continue
            for i_ell in range(len(ELLS)):
                assert (
                    delta[i_ell * n_out + i, j] == 0.0
                ), f"diag_correction leaked into bin {i} from column {j}"
    assert np.max(np.abs(delta)) > 0.0, "diag_correction had no effect"


def test_exact_beam_legs_match_brute_force():
    """The Y_LM expansion of the exact zero-lag legs equals a direct cell sum.

    These legs are what a k-hat independent cell mass cannot reproduce for
    ell > 0: the estimator's Y_lm(nhat) leg couples to the beam's own L
    structure, so ell needs beam moments up to L = ell.
    """
    from meer21cm.multipole_ops import exact_beam_legs
    from meer21cm.spherical import get_real_Ylm

    rng = np.random.default_rng(7)
    n_cell = 400
    # a wide footprint: nhat spread ~60 deg about z, as on the 06 lightcone
    nhat = np.stack(
        [
            rng.uniform(-0.8, 0.8, n_cell),
            rng.uniform(-0.8, 0.8, n_cell),
            np.ones(n_cell),
        ],
        axis=1,
    )
    nhat /= np.linalg.norm(nhat, axis=1)[:, None]
    sigma_b = rng.choice(np.linspace(8.0, 24.0, 12), n_cell)
    cell_mass = rng.uniform(0.5, 1.0, n_cell)

    n_mode = 9
    kvec = rng.normal(size=(n_mode, 3))
    kvec /= np.linalg.norm(kvec, axis=1)[:, None]
    k_abs = rng.uniform(0.01, 0.07, n_mode)
    khat = (kvec[:, 0], kvec[:, 1], kvec[:, 2])

    ells = (0, 2, 4)
    legs = exact_beam_legs(
        k_abs, khat, nhat, sigma_b, cell_mass, ells=ells, l_max_beam=12, nmu=96
    )
    for n in range(n_mode):
        mu_b = nhat @ kvec[n]
        b = np.exp(-0.5 * k_abs[n] ** 2 * (1.0 - mu_b**2) * sigma_b**2)
        assert abs(float(legs[None][n]) / float(np.sum(cell_mass * b)) - 1.0) < 1e-3
        for ell in ells:
            for m in range(-ell, ell + 1):
                y_n = np.asarray(
                    get_real_Ylm(ell, m)(nhat[:, 0], nhat[:, 1], nhat[:, 2]), float
                ) * np.ones(n_cell)
                brute = float(np.sum(cell_mass * b * y_n))
                got = float(np.asarray(legs[(ell, m)])[n])
                assert abs(got - brute) <= 2e-3 * abs(brute) + 1e-9 * abs(
                    float(legs[None][n])
                ), f"ell={ell} m={m} mode {n}: {got:.6e} vs {brute:.6e}"


def test_beam_legs_l0_truncation_fails_above_monopole():
    """A k-hat independent cell mass only supplies L=0: exact for ell=0 only.

    This is the reason a scalar out_mode_scale (or any mean-field comb
    mass) cannot carry the beam into P2/P4.
    """
    from meer21cm.multipole_ops import exact_beam_legs

    rng = np.random.default_rng(8)
    n_cell = 400
    nhat = np.stack(
        [
            rng.uniform(-0.8, 0.8, n_cell),
            rng.uniform(-0.8, 0.8, n_cell),
            np.ones(n_cell),
        ],
        axis=1,
    )
    nhat /= np.linalg.norm(nhat, axis=1)[:, None]
    sigma_b = rng.choice(np.linspace(10.0, 26.0, 8), n_cell)
    cell_mass = np.ones(n_cell)
    kvec = rng.normal(size=(12, 3))
    kvec /= np.linalg.norm(kvec, axis=1)[:, None]
    k_abs = np.full(12, 0.06)
    khat = (kvec[:, 0], kvec[:, 1], kvec[:, 2])

    l_keep = (0, 2, 4, 6)
    ref = exact_beam_legs(
        k_abs, khat, nhat, sigma_b, cell_mass, ells=(0, 2), l_max_beam=12
    )
    trunc = {
        L: exact_beam_legs(
            k_abs, khat, nhat, sigma_b, cell_mass, ells=(0, 2), l_max_beam=L
        )
        for L in l_keep
    }
    err_mono = {
        L: float(np.max(np.abs(trunc[L][None] / ref[None] - 1.0))) for L in l_keep
    }
    err_quad = {
        L: max(
            float(np.max(np.abs(trunc[L][(2, m)] - ref[(2, m)])))
            / float(np.max(np.abs(ref[(2, m)])))
            for m in range(-2, 3)
        )
        for L in l_keep
    }
    # L=0 is the k-hat independent mean field.  Per mode it is already
    # ~16% wrong for the monopole leg (the shell average hides that) and
    # it misses the ell=2 legs almost entirely.
    assert err_mono[0] > 0.05, f"L=0 unexpectedly accurate for the monopole: {err_mono}"
    assert err_quad[0] > 0.5, f"L=0 unexpectedly accurate for ell=2: {err_quad}"
    # convergence needs L beyond ell, and must be monotone
    assert err_mono[2] < 0.01, err_mono
    assert err_quad[4] < 0.05, err_quad
    assert err_quad[6] < 0.005, err_quad
    for a, b in zip(l_keep[:-1], l_keep[1:]):
        assert err_quad[b] <= err_quad[a], (a, b, err_quad)


def test_mesh_window_mode_groups_reduce_to_ungrouped():
    """Splitting each |k| bin into |mu| groups is exact for identical kernels.

    The group fill must be additive with the *full* bin weight as the
    denominator, otherwise the shell average is silently rescaled.
    """
    from meer21cm.multipole_ops import beam_mode_group_index

    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)
    idx, n_group = beam_mode_group_index(fps, n_mu=3)

    mat_plain = build_mesh_window_matrix(fps, k_in, ells=ELLS, weights=weights)
    mat_grouped = build_mesh_window_matrix(
        fps,
        k_in,
        ells=ELLS,
        weights=weights,
        out_bin_weights=[weights] * n_group,
        out_group_index=idx,
    )
    model_plain = mat_plain.apply({0: theory_nodes})
    model_grouped = mat_grouped.apply({0: theory_nodes})
    for ell in ELLS:
        assert np.allclose(
            model_grouped[ell],
            model_plain[ell],
            rtol=1e-8,
            atol=1e-8 * np.max(np.abs(model_plain[ell])),
        ), f"ell={ell}: mu-grouped fill != ungrouped"


def test_mesh_window_input_groups_reduce_to_ungrouped():
    """Splitting the *theory* shell into |mu| groups is exact for one kernel.

    The input-mode beam path (``in_bin_weights``) sums groups into the
    same column, so with a q-independent cube it must reproduce the plain
    matrix bit for bit.  This pins the q-side bookkeeping (shell x group
    partition, additive fill) independently of any beam.
    """
    from meer21cm.multipole_ops import beam_input_mode_groups

    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, N_K_IN)
    theory_nodes = _theory_grid(k_in)
    idx, n_group = beam_input_mode_groups(fps, n_mu=3)
    assert n_group == 3
    assert set(np.unique(idx)) == {0, 1, 2}

    mat_plain = build_mesh_window_matrix(fps, k_in, ells=ELLS, weights=weights)
    mat_grouped = build_mesh_window_matrix(
        fps,
        k_in,
        ells=ELLS,
        weights=weights,
        in_bin_weights=lambda j, g: weights,
        in_group_index=idx,
    )
    model_plain = mat_plain.apply({0: theory_nodes})
    model_grouped = mat_grouped.apply({0: theory_nodes})
    for ell in ELLS:
        assert np.allclose(
            model_grouped[ell],
            model_plain[ell],
            rtol=1e-8,
            atol=1e-8 * np.max(np.abs(model_plain[ell])),
        ), f"ell={ell}: q-grouped fill != ungrouped"


def test_beam_input_cell_kernels_group_mean_is_the_exact_perp_moment():
    """u_b = tr(M) - n_b.M.n_b equals the group mean of q_perp,b² per cell.

    The cube can only hold one q, so the beam argument is averaged over
    the (shell, |mu|) group.  Doing that with the second-moment matrix is
    exact for the *argument* — no representative direction, so the
    azimuthal spread of q̂ about n̂_ref is kept.  This is the step that
    a flat-sky ⟨B⟩ over a |k| shell gets wrong.
    """
    from meer21cm.multipole_ops import beam_input_mode_groups

    fps = _shot_fps()
    rng = np.random.default_rng(4)
    nhat = rng.normal(size=(600, 3)) + np.array([0.0, 0.0, 6.0])
    nhat /= np.linalg.norm(nhat, axis=1)[:, None]
    idx, n_group = beam_input_mode_groups(fps, n_mu=3)

    k_in = np.geomspace(0.02, 0.12, 8)
    edges = np.concatenate(([0.0], 0.5 * (k_in[:-1] + k_in[1:]), [np.inf]))
    k_mode = np.asarray(fps.k_mode, float).ravel()
    q_vec = np.stack(
        [
            np.broadcast_to(np.asarray(c, float), fps.k_mode.shape).ravel()
            for c in np.meshgrid(*fps.k_vec, indexing="ij")
        ],
        axis=1,
    )
    g_flat = np.asarray(idx).ravel()
    cells = [0, 17, 123, 401]
    for j in (1, 4, 7):
        for g in range(n_group):
            sel = (k_mode >= edges[j]) & (k_mode < edges[j + 1]) & (g_flat == g)
            if not np.any(sel):
                continue
            qs = q_vec[sel]
            mmat = qs.T @ qs / qs.shape[0]
            u_moment = np.trace(mmat) - np.einsum("ci,ij,cj->c", nhat, mmat, nhat)
            for c in cells:
                brute = float(np.mean(np.sum(qs**2, axis=1) - (qs @ nhat[c]) ** 2))
                assert abs(float(u_moment[c]) - brute) <= 1e-10 * max(brute, 1e-30)


def test_beam_input_matrix_reduces_to_nobeam_without_a_beam():
    """beam_at_input with sigma_beam_ch=None is the plain MAS-out matrix."""
    from meer21cm.window import build_mesh_window_mas_out

    fps = _shot_fps()
    fps.sigma_beam_ch = None
    k_in = np.geomspace(0.02, 0.12, 6)
    theory_nodes = _theory_grid(k_in)
    counts = _mask(BOX_NDIM, BOX_LEN) + 0.1

    base = build_mesh_window_mas_out(fps, k_in, renorm_weights=counts, ells=ELLS)
    beam = build_mesh_window_mas_out(
        fps, k_in, renorm_weights=counts, ells=ELLS, beam_at_input=True, beam_n_mu=3
    )
    m_base = base.apply({0: theory_nodes})
    m_beam = beam.apply({0: theory_nodes})
    for ell in ELLS:
        assert np.allclose(
            m_beam[ell],
            m_base[ell],
            rtol=1e-8,
            atol=1e-8 * np.max(np.abs(m_base[ell])),
        ), f"ell={ell}: beam_at_input changed the no-beam matrix"


def test_mesh_window_leg_scale_multiplies_only_its_own_ell():
    """leg_scale rescales one leg product without touching the others.

    The beam correction is applied this way (a per-mode ratio on the whole
    kappa profile, not just its zero-lag value), so the plumbing has to be
    exactly multiplicative and ell-local.
    """
    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, 12)
    theory_nodes = _theory_grid(k_in)

    base = build_mesh_window_matrix(fps, k_in, ells=ELLS, weights=weights)
    rfft_shape = np.asarray(fps.k_mode).shape
    scale = {(2, m): np.full(rfft_shape, 3.0) for m in range(-2, 3)}
    scaled = build_mesh_window_matrix(
        fps, k_in, ells=ELLS, weights=weights, leg_scale=scale
    )
    m_base = base.apply({0: theory_nodes})
    m_scaled = scaled.apply({0: theory_nodes})
    assert np.allclose(m_scaled[2], 3.0 * m_base[2], rtol=1e-10, atol=0.0)
    for ell in (0, 4):
        assert np.allclose(m_scaled[ell], m_base[ell], rtol=1e-10, atol=0.0)

    with pytest.raises(ValueError):
        build_mesh_window_matrix(
            fps,
            k_in,
            ells=ELLS,
            weights=weights,
            leg_scale=scale,
            diag_correction={(0, 0): np.zeros(rfft_shape)},
        )


def test_beam_input_n_mu_4_partitions_all_theory_modes():
    """Production n_mu=4 covers every rFFT mode; groups are nonempty."""
    from meer21cm.multipole_ops import beam_input_mode_groups

    fps = _shot_fps()
    idx, n_group = beam_input_mode_groups(fps, n_mu=4)
    assert n_group == 4
    flat = np.asarray(idx).ravel()
    assert set(np.unique(flat)) == {0, 1, 2, 3}
    counts = np.bincount(flat, minlength=4)
    assert np.all(counts > 0)


def test_beam_input_n_mu_4_groups_split_q_perp():
    """Low-|μ| groups have larger ⟨q_⊥²⟩ than high-|μ| groups.

    That is the n_mu=4 leakage split: the same |q| is no longer assigned
    one shell-mean ⟨q_⊥²⟩ ~ (2/3) q² for every direction.
    """
    from meer21cm.multipole_ops import beam_input_mode_groups

    fps = _shot_fps()
    idx, n_group = beam_input_mode_groups(fps, n_mu=4)
    q_vec = np.stack(
        [
            np.broadcast_to(np.asarray(c, dtype=float), fps.k_mode.shape).ravel()
            for c in np.meshgrid(*fps.k_vec, indexing="ij")
        ],
        axis=1,
    )
    nref = np.array(
        [float(np.mean(np.asarray(c, dtype=float))) for c in fps.los_xhat],
        dtype=float,
    )
    nref = nref / float(np.linalg.norm(nref))
    g_flat = np.asarray(idx).ravel()
    qperp2 = []
    for g in range(n_group):
        qs = q_vec[g_flat == g]
        qperp2.append(float(np.mean(np.sum(qs**2, axis=1) - (qs @ nref) ** 2)))
    assert qperp2[0] > 1.2 * qperp2[-1], qperp2


def _beamed_ps_namespace():
    """Minimal duck type with a chromatic Gaussian for B5 cell-space tests."""
    from astropy.cosmology import Planck18
    from types import SimpleNamespace

    fps = _shot_fps()
    nu = np.linspace(900.0e6, 1050.0e6, 8)
    return SimpleNamespace(
        k_mode=fps.k_mode,
        k_vec=fps.k_vec,
        los_xhat=fps.los_xhat,
        box_len=fps.box_len,
        box_ndim=fps.box_ndim,
        box_origin=np.asarray(fps.los_observer, dtype=float),
        pix_coor_in_box=fps.pix_coor_in_box,
        pix_coor_in_cartesian=None,
        grid_scheme="cic",
        sigma_beam_ch=np.full(8, 0.4),
        sigma_beam_ch_in_mpc=np.linspace(8.0, 20.0, 8),
        nu=nu,
        freq_resol=float(np.diff(nu).mean()),
        pix_resol=1.0,
        astropy_cosmo_fiducial=Planck18,
    )


def test_beam_diag_additive_and_ratio_are_both_defined():
    """Both κ=0 corrections run; the quadrupole is not a no-op.

    Production is additive (beam_leg_scale=False).  The ratio form is
    still exact on the diagonal; it is not the default because it
    rescales the n_mu-split leakage.
    """
    from meer21cm.multipole_ops import beam_input_diagonal_correction

    ps = _beamed_ps_namespace()
    n_cell = int(np.asarray(ps.pix_coor_in_box).reshape(-1, 3).shape[0])
    k_in = np.geomspace(0.02, 0.12, 6)
    cell_mass = np.ones(n_cell, dtype=float)
    ratio = beam_input_diagonal_correction(
        ps, k_in, ells=(0, 2), n_mu=4, ratio=True, cell_mass=cell_mass
    )
    add = beam_input_diagonal_correction(
        ps, k_in, ells=(0, 2), n_mu=4, ratio=False, cell_mass=cell_mass
    )
    assert set(ratio) == set(add)
    assert (0, 0) in add and (2, 0) in add
    assert float(np.max(np.abs(add[(2, 0)]))) > 0.0
    assert float(np.max(np.abs(np.asarray(ratio[(2, 0)]) - 1.0))) > 1e-3


def test_beam_input_n_phi_1_matches_mu_only():
    """n_phi=1 is bit-identical to the |μ|-only grouping."""
    from meer21cm.multipole_ops import beam_input_mode_groups

    fps = _shot_fps()
    a, na = beam_input_mode_groups(fps, n_mu=4)
    b, nb = beam_input_mode_groups(fps, n_mu=4, n_phi=1)
    assert na == nb == 4
    assert np.array_equal(a, b)


def test_beam_input_n_phi_4_partitions_all_modes():
    """n_mu=4 × n_phi=4 covers every rFFT mode; all 16 groups are nonempty."""
    from meer21cm.multipole_ops import beam_input_mode_groups

    fps = _shot_fps()
    idx, n_group = beam_input_mode_groups(fps, n_mu=4, n_phi=4)
    assert n_group == 16
    flat = np.asarray(idx).ravel()
    assert set(np.unique(flat)) == set(range(16))
    assert np.all(np.bincount(flat, minlength=16) > 0)


def test_beam_input_phi_bins_have_different_M_axes():
    """Two φ bins at the same |μ| have different in-plane M principal axes."""
    from meer21cm.multipole_ops import beam_input_mode_groups

    fps = _shot_fps()
    idx, _n = beam_input_mode_groups(fps, n_mu=4, n_phi=4)
    nref = np.array(
        [float(np.mean(np.asarray(c, dtype=float))) for c in fps.los_xhat],
        dtype=float,
    )
    nref = nref / float(np.linalg.norm(nref))
    q_vec = np.stack(
        [
            np.broadcast_to(np.asarray(c, dtype=float), fps.k_mode.shape).ravel()
            for c in np.meshgrid(*fps.k_vec, indexing="ij")
        ],
        axis=1,
    )
    g_flat = np.asarray(idx).ravel()
    found = False
    for i_mu in range(4):
        axes = []
        for i_phi in range(4):
            g = i_mu * 4 + i_phi
            qs = q_vec[g_flat == g]
            if qs.shape[0] < 16:
                continue
            mmat = qs.T @ qs / qs.shape[0]
            proj = np.eye(3) - np.outer(nref, nref)
            mp = proj @ mmat @ proj
            _w, vecs = np.linalg.eigh(mp)
            axis = vecs[:, int(np.argmax(_w))]
            axis = axis - nref * float(axis @ nref)
            nrm = float(np.linalg.norm(axis))
            if nrm < 1e-12:
                continue
            axes.append(axis / nrm)
        if len(axes) < 2:
            continue
        align = abs(float(axes[0] @ axes[-1]))
        assert align < 0.95, (i_mu, align)
        found = True
        break
    assert found


def test_beam_ylm_labels_lmax2():
    """L≤2 even real Y_LM is 1 + 5 = 6 terms."""
    from meer21cm.multipole_ops import beam_ylm_labels

    labels = beam_ylm_labels(2)
    assert labels[0] == (0, 0)
    assert labels == [
        (0, 0),
        (2, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (2, 2),
    ]


def test_beam_ylm_s0_matches_truncated_exact_legs():
    """Σ_LM α_LM ⟨c_LM⟩ equals exact_beam_legs S^0 at the same L_max.

    f_L is frozen at the theory node and k_abs is set to that node so the
    two expressions are the same addition theorem.
    """
    from meer21cm.multipole_ops import (
        beam_cell_sigma_perp,
        beam_ylm_alpha,
        beam_ylm_labels,
        cell_grid_los,
        exact_beam_legs,
        gaussian_beam_legendre_moments,
    )
    from meer21cm.spherical import get_real_Ylm, unit_khat_from_k_vec

    ps = _beamed_ps_namespace()
    labels = beam_ylm_labels(2)
    nhat, sigma_b = beam_cell_sigma_perp(ps)
    _nhat_leg, inside = cell_grid_los(ps)
    e_b = np.ones(nhat.shape[0], dtype=float) * inside
    n_grid = float(np.prod(np.asarray(ps.box_ndim, dtype=int)))
    k_j = 0.05
    k_abs = np.full_like(np.asarray(ps.k_mode, dtype=float), k_j)
    khat = unit_khat_from_k_vec(ps.k_vec)
    s_ex = exact_beam_legs(
        k_abs,
        khat,
        nhat,
        sigma_b,
        e_b,
        ells=(0,),
        l_max_beam=2,
        norm=n_grid,
        nhat_leg=nhat,
    )
    alpha = beam_ylm_alpha(ps, labels)
    recon = np.zeros(np.asarray(ps.k_mode).size, dtype=float)
    for g, (L, M) in enumerate(labels):
        f_L = gaussian_beam_legendre_moments(k_j * sigma_b, (L,))[0]
        y = np.asarray(
            get_real_Ylm(L, M)(nhat[:, 0], nhat[:, 1], nhat[:, 2]), dtype=float
        )
        s0g = float(np.sum(e_b * f_L * y)) / n_grid
        recon = recon + alpha[g].ravel() * s0g
    assert np.allclose(recon, s_ex[None].ravel(), rtol=1e-8, atol=1e-10)


def test_in_group_scale_ones_matches_single_group():
    """A uniform in_group_scale is the same as one all-mode group."""
    weights = _mask(BOX_NDIM, BOX_LEN)
    fps = _make_fps(0, weights, _true_observer())
    k_in = np.geomspace(0.012, 0.16, 8)
    theory_nodes = _theory_grid(k_in)

    def kern(j, g):
        return weights

    idx = np.zeros(np.asarray(fps.k_mode).shape, dtype=np.int64)
    m_idx = build_mesh_window_matrix(
        fps, k_in, ells=ELLS, weights=weights, in_bin_weights=kern, in_group_index=idx
    )
    m_sc = build_mesh_window_matrix(
        fps,
        k_in,
        ells=ELLS,
        weights=weights,
        in_bin_weights=kern,
        in_group_scale=[np.ones(np.asarray(fps.k_mode).shape, dtype=float)],
    )
    a = m_idx.apply({0: theory_nodes})
    b = m_sc.apply({0: theory_nodes})
    for ell in ELLS:
        assert np.allclose(
            a[ell], b[ell], rtol=1e-10, atol=1e-10 * np.max(np.abs(a[ell]))
        ), f"ell={ell}: in_group_scale ones != single group"


def test_beam_ylm_diagonal_correction_quadrupole_nonzero():
    """L≤2 diagonal cubes miss cross terms and higher L; the additive is not 0."""
    from meer21cm.multipole_ops import beam_ylm_diagonal_correction

    ps = _beamed_ps_namespace()
    n_cell = int(np.asarray(ps.pix_coor_in_box).reshape(-1, 3).shape[0])
    cell_mass = np.ones(n_cell, dtype=float)
    corr = beam_ylm_diagonal_correction(
        ps, np.geomspace(0.02, 0.12, 6), ells=(0, 2), l_max_cube=2, cell_mass=cell_mass
    )
    assert (0, 0) in corr and (2, 0) in corr
    assert float(np.max(np.abs(corr[(2, 0)]))) > 0.0
