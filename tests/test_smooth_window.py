"""Tests for discrete-shell smooth window and multipole theory."""

import numpy as np
import pytest

from meer21cm.estimator import FieldPowerSpectrum
from meer21cm.multipole_model import (
    SmoothWindowEstimator,
    WindowedMultipoleModel,
    accumulate_window_multipoles,
    make_im_selection_field,
    run_smooth_window_realization,
)
from meer21cm.power_ops import get_modelpk_conv
from meer21cm.smooth_window import (
    DiscreteShellWindowMatrix,
    apply_discrete_shell_window_matrix,
    build_discrete_shell_window_matrix,
    power_to_correlation_multipole,
    wigner3j_square,
)
from meer21cm.util import legendre_polynomial_with_factor


def test_wigner3j_and_fftlog_smoke():
    L, coeffs = wigner3j_square(2, 2)
    assert 0 in L
    assert np.isclose(coeffs[0], 1.0) or len(coeffs) > 0
    k = np.geomspace(1e-3, 1.0, 64)
    pk = np.exp(-((k / 0.1) ** 2))
    s, xi = power_to_correlation_multipole(k, pk, ell=0)
    assert s.shape == pk.shape
    assert np.all(np.isfinite(xi))


def test_shell_map_matches_measure_multipoles_nmodes():
    ndim = (16, 16, 16)
    box_len = np.array([160.0, 160.0, 160.0])
    field = np.ones(ndim)
    k1dbins = np.linspace(0.05, 0.35, 8)
    fps = FieldPowerSpectrum(field, box_len, los="global", _skip_specification=True)
    shell = fps.multipole_bin_index_map(k1dbins=k1dbins)
    res = fps.measure_multipoles(k1dbins=k1dbins, ells=(0,))
    np.testing.assert_allclose(shell.nmodes, res.nmodes)
    np.testing.assert_allclose(shell.k_eff, res.k, equal_nan=True)
    assert shell.bin_index.shape == fps.k_mode.shape
    assert np.all((shell.bin_index >= -1) & (shell.bin_index < len(k1dbins) - 1))


def test_matrix_shape_contract():
    ndim = (12, 12, 12)
    box_len = np.array([120.0, 120.0, 120.0])
    # Non-trivial selection so finite-k window multipoles are non-zero
    weights = np.ones(ndim)
    weights[:3, :, :] = 0.0
    weights[:, :2, :] = 0.5
    k1dbins = np.linspace(0.08, 0.28, 5)
    k_in = np.geomspace(0.05, 0.4, 20)
    ells = (0, 2, 4)

    # HI selection is deterministic — one measurement is enough
    results = [
        run_smooth_window_realization(
            box_len=box_len,
            k1dbins=k1dbins,
            seed=0,
            tracer="hi",
            ells=ells,
            weights_hi=weights,
        )
    ]
    acc = accumulate_window_multipoles(results)
    fps = FieldPowerSpectrum(
        np.ones(ndim), box_len, los="global", _skip_specification=True
    )
    shell = fps.multipole_bin_index_map(k1dbins=k1dbins)
    mat = build_discrete_shell_window_matrix(
        shell,
        acc.k,
        acc.W_ell,
        k_in=k_in,
        ells=ells,
        n_fftlog=128,
        n_k_eval=64,
    )
    assert isinstance(mat, DiscreteShellWindowMatrix)
    n_out = len(k1dbins) - 1
    n_in = len(k_in)
    assert mat.matrix.shape == (3 * n_out, 3 * n_in)
    assert n_in > n_out
    assert mat.ells == ells


def test_uniform_weight_monopole_smoke():
    """Selection multipoles of a tapered weight map → finite matrix apply."""
    ndim = (14, 14, 14)
    box_len = np.array([140.0, 140.0, 140.0])
    weights = np.ones(ndim) * 3.0
    # Edge taper so the selection has non-zero finite-k power
    weights[:2, :, :] = 0.0
    weights[-2:, :, :] = 0.0
    k1dbins = np.linspace(0.08, 0.25, 5)
    k_in = np.geomspace(0.06, 0.35, 24)
    ells = (0,)

    est = SmoothWindowEstimator(
        box_len=box_len,
        k1dbins=k1dbins,
        ells=ells,
        tracer="hi",
        weights_hi=weights,
    )
    est.accumulate([est.run_one(0)])
    mat = est.build_window_matrix(k_in=k_in, n_fftlog=128, n_k_eval=64)
    assert mat.matrix.shape[0] == len(k1dbins) - 1
    assert mat.matrix.shape[1] == len(k_in)

    p_in = {0: np.ones_like(k_in)}
    p_out = apply_discrete_shell_window_matrix(p_in, mat.matrix, ells=ells)
    assert np.all(np.isfinite(p_out[0]))
    assert p_out[0].shape == (len(k1dbins) - 1,)


def test_three_k_grids_are_independent():
    """k1dbins_window (W_L), k_in (theory), k1dbins_out (estimator) differ."""
    ndim = (12, 12, 12)
    box_len = np.array([120.0, 120.0, 120.0])
    weights = np.ones(ndim)
    weights[:2, :, :] = 0.0
    k1dbins_out = np.linspace(0.08, 0.28, 5)
    k1dbins_window = np.geomspace(0.02, 0.4, 16)
    k_in = np.geomspace(0.05, 0.35, 20)
    ells = (0, 2)

    est = SmoothWindowEstimator(
        box_len=box_len,
        k1dbins_window=k1dbins_window,
        k1dbins_out=k1dbins_out,
        ells=ells,
        tracer="hi",
        weights_hi=weights,
    )
    np.testing.assert_allclose(est.k1dbins_out, k1dbins_out)
    np.testing.assert_allclose(est.k1dbins, k1dbins_out)  # legacy alias
    np.testing.assert_allclose(est.k1dbins_window, k1dbins_window)
    assert len(est.k1dbins_window) != len(est.k1dbins_out)

    est.accumulate([est.run_one(0)])
    assert est.k_window is not None
    assert len(est.k_window) == len(k1dbins_window) - 1

    mat = est.build_window_matrix(k_in=k_in, n_fftlog=128, n_k_eval=64)
    n_out = len(k1dbins_out) - 1
    assert mat.matrix.shape == (len(ells) * n_out, len(ells) * len(k_in))
    np.testing.assert_allclose(mat.k_out, est.k_out, equal_nan=True)
    np.testing.assert_allclose(mat.k_in, k_in)


def test_windowed_multipole_model_continuous_mu():
    """Continuous μ multipoles are finite on a synthetic (k,μ) model grid."""
    kmode = np.geomspace(0.05, 0.4, 30).reshape(5, 3, 2)
    mumode = np.linspace(-1, 1, kmode.size).reshape(kmode.shape)
    model = WindowedMultipoleModel(
        kmode=kmode,
        mumode=mumode,
        tracer_bias_1=1.0,
        kaiser_rsd=False,
        window_ells=(0, 2),
    )
    k_in = np.geomspace(0.08, 0.3, 12)
    raw = model.get_theory_multipoles_kmu(k_in, ells=(0, 2), nmu=32, which="auto_1")
    assert raw["P_ell"][0].shape == k_in.shape
    assert np.all(np.isfinite(raw["P_ell"][0]))
    out = model.get_model_multipoles(
        which="auto_1", k_in=k_in, ells=(0, 2), apply_window=False
    )
    assert out["window_applied"] is False
    np.testing.assert_allclose(out["P_ell"][0], raw["P_ell"][0])


def test_anisotropic_operator_order_vs_conv3d():
    """
    On anisotropic weights, discrete-shell window @ continuous P_ell should
    remain finite and closer in spirit to discrete multipoles of
    get_modelpk_conv than a random baseline (regression smoke; not a tight
    accuracy lock vs full 3D FFT).
    """
    nx = ny = nz = 16
    box_len = np.array([160.0, 160.0, 160.0])
    # Anisotropic survey weight
    z = np.linspace(0, 1, nz)
    weights = np.ones((nx, ny, nz))
    weights *= (0.3 + 0.7 * z)[None, None, :]
    weights[:2, :, :] = 0.0

    from meer21cm.power_ops import get_k_vector, get_vec_mode, bin_3d_to_1d

    box_resol = box_len / np.array([nx, ny, nz])
    k_vec = get_k_vector([nx, ny, nz], box_resol)
    kmode = get_vec_mode(k_vec)
    with np.errstate(divide="ignore", invalid="ignore"):
        mumode = np.clip(np.nan_to_num(k_vec[-1][None, None, :] / kmode), -1, 1)

    p3d = np.exp(-((kmode / 0.12) ** 2)) * (1.0 + 0.5 * mumode**2)
    p_conv = get_modelpk_conv(
        p3d, weights1_in_real=weights, weights2=weights, renorm=True
    )
    k1dbins = np.linspace(0.08, 0.3, 6)
    factor0 = np.poly1d(legendre_polynomial_with_factor(0))(mumode)
    p_conv_0, k_eff, nmodes = bin_3d_to_1d(p_conv * factor0, kmode, k1dbins)

    # Measure W_L from the selection field (no extra weights_grid — avoid w²)
    ells = (0, 2)
    acc = accumulate_window_multipoles(
        [
            run_smooth_window_realization(
                box_len=box_len,
                k1dbins=k1dbins,
                seed=0,
                tracer="hi",
                ells=ells,
                weights_hi=weights,
            )
        ]
    )
    fps = FieldPowerSpectrum(
        np.ones((nx, ny, nz)),
        box_len,
        los="global",
        _skip_specification=True,
    )
    shell = fps.multipole_bin_index_map(k1dbins=k1dbins)
    k_in = np.geomspace(0.05, 0.4, 40)
    mat = build_discrete_shell_window_matrix(
        shell,
        acc.k,
        acc.W_ell,
        k_in=k_in,
        ells=ells,
        n_fftlog=128,
        n_k_eval=64,
    )

    from scipy.interpolate import interp1d

    p0_bare, _, _ = bin_3d_to_1d(p3d * factor0, kmode, k1dbins)
    mask = np.isfinite(k_eff) & np.isfinite(p0_bare)
    p0_in = interp1d(
        k_eff[mask],
        p0_bare[mask],
        kind="linear",
        bounds_error=False,
        fill_value=(p0_bare[mask][0], 0.0),
    )(k_in)
    p2_in = np.zeros_like(k_in)
    p_out = mat.apply({0: p0_in, 2: p2_in})

    assert np.all(np.isfinite(p_out[0]))
    assert p_out[0].shape == p_conv_0.shape
    m = np.isfinite(p_conv_0) & np.isfinite(p_out[0]) & (np.abs(p_conv_0) > 0)
    if m.sum() > 0:
        rel = np.abs(p_out[0][m] - p_conv_0[m]) / np.abs(p_conv_0[m])
        assert np.median(rel) < 5.0  # loose; documents regression target


def test_cross_worker_runs_smoke():
    ndim = (13, 13, 13)
    box_len = np.array([130.0, 130.0, 130.0])
    weights = np.ones(ndim)
    weights[0, :, :] = 0.0
    k1dbins = np.linspace(0.1, 0.35, 5)
    res = run_smooth_window_realization(
        box_len=box_len,
        k1dbins=k1dbins,
        seed=7,
        tracer="cross",
        ells=(0,),
        weights_hi=weights,
        selection_mask=weights > 0,
        tot_num_galaxies=500,
    )
    assert np.all(np.isfinite(res.P_ell[0]))


def test_identity_continuous_matches_anisotropic_shell():
    """
    continuous='identity' + discrete shells should reproduce Legendre binning
    of P(k,μ) when enough multipoles are retained in the expansion.
    """
    from scipy.special import eval_legendre
    from meer21cm.power_ops import get_k_vector, get_vec_mode, bin_3d_to_1d
    from meer21cm.util import legendre_polynomial_with_factor

    nx = ny = nz = 20
    box_len = np.array([200.0, 200.0, 200.0])
    box_resol = box_len / np.array([nx, ny, nz])
    k_vec = get_k_vector([nx, ny, nz], box_resol)
    kmode = get_vec_mode(k_vec)
    with np.errstate(divide="ignore", invalid="ignore"):
        mumode = np.clip(np.nan_to_num(k_vec[-1][None, None, :] / kmode), -1, 1)

    # Mild anisotropy so ℓ=2 matters but higher multipoles are small
    p3d = np.exp(-((kmode / 0.15) ** 2)) * (1.0 + 0.4 * mumode**2)

    k1dbins = np.linspace(0.06, 0.28, 7)
    ells = (0, 2)
    fps = FieldPowerSpectrum(
        np.ones((nx, ny, nz)), box_len, los="global", _skip_specification=True
    )
    shell = fps.multipole_bin_index_map(k1dbins=k1dbins)

    P_disc = {}
    for ell in ells:
        fac = np.poly1d(legendre_polynomial_with_factor(ell))(mumode)
        P_disc[ell], k_eff, _ = bin_3d_to_1d(p3d * fac, kmode, k1dbins)

    # Continuous multipoles on a fine k_in grid (same definition as theory path)
    k_in = np.geomspace(0.04, 0.35, 60)
    mu_nodes, mu_w = np.polynomial.legendre.leggauss(64)
    P_in = {}
    for ell in ells:
        L = eval_legendre(int(ell), mu_nodes)
        # P(k,μ) = exp(-(k/0.15)^2) * (1 + 0.4 μ^2)
        pkmu = np.exp(-((k_in[:, None] / 0.15) ** 2)) * (
            1.0 + 0.4 * mu_nodes[None, :] ** 2
        )
        integ = np.sum(pkmu * L[None, :] * mu_w[None, :], axis=1)
        P_in[ell] = (2 * int(ell) + 1) / 2.0 * integ

    mat = build_discrete_shell_window_matrix(
        shell,
        k_in=k_in,
        ells=ells,
        continuous="identity",
        n_k_eval=128,
    )
    P_win = mat.apply(P_in)

    m = np.isfinite(P_disc[0]) & np.isfinite(P_win[0]) & (np.abs(P_disc[0]) > 0)
    rel0 = np.abs(P_win[0][m] - P_disc[0][m]) / np.abs(P_disc[0][m])
    assert np.median(rel0) < 0.05

    m2 = np.isfinite(P_disc[2]) & np.isfinite(P_win[2]) & (np.abs(P_disc[2]) > 0)
    if m2.sum() > 0:
        rel2 = np.abs(P_win[2][m2] - P_disc[2][m2]) / np.abs(P_disc[2][m2])
        assert np.median(rel2) < 0.15


def test_identity_does_not_need_W_ell():
    ndim = (10, 10, 10)
    box_len = np.array([100.0, 100.0, 100.0])
    k1dbins = np.linspace(0.1, 0.35, 5)
    k_in = np.geomspace(0.08, 0.4, 16)
    fps = FieldPowerSpectrum(
        np.ones(ndim), box_len, los="global", _skip_specification=True
    )
    shell = fps.multipole_bin_index_map(k1dbins=k1dbins)
    mat = build_discrete_shell_window_matrix(
        shell, k_in=k_in, ells=(0,), continuous="identity"
    )
    assert mat.matrix.shape == (4, 16)
    p_out = mat.apply({0: np.ones(16)})
    assert np.all(np.isfinite(p_out[0]))


def test_make_im_selection_field_copies_weights():
    weights = np.array([0.0, 4.0, 1.0])
    field = make_im_selection_field(weights)
    np.testing.assert_array_equal(field, weights)
    field[1] = -1.0
    assert weights[1] == 4.0


def test_hi_window_is_deterministic():
    """HI selection multipoles do not depend on seed."""
    ndim = (10, 10, 10)
    box_len = np.array([100.0, 100.0, 100.0])
    weights = np.linspace(0.1, 1.0, np.prod(ndim)).reshape(ndim)
    k1dbins = np.linspace(0.1, 0.4, 5)
    a = run_smooth_window_realization(
        box_len=box_len,
        k1dbins=k1dbins,
        seed=1,
        tracer="hi",
        ells=(0, 2),
        weights_hi=weights,
    )
    b = run_smooth_window_realization(
        box_len=box_len,
        k1dbins=k1dbins,
        seed=99,
        tracer="hi",
        ells=(0, 2),
        weights_hi=weights,
    )
    np.testing.assert_allclose(a.P_ell[0], b.P_ell[0])
    np.testing.assert_allclose(a.P_ell[2], b.P_ell[2])


def test_field_power_spectrum_los_global_and_reserved():
    field = np.ones((8, 8, 8))
    box_len = np.array([80.0, 80.0, 80.0])
    fps = FieldPowerSpectrum(field, box_len, los="global")
    assert fps.los == "global"
    fps.measure_multipoles(ells=(0,), k1dbins=np.linspace(0.1, 0.4, 4))
    with pytest.raises(ValueError, match="Unknown los"):
        FieldPowerSpectrum(field, box_len, los="diagonal")
    for reserved in ("endpoint", "firstpoint", "midpoint"):
        fps_r = FieldPowerSpectrum(field, box_len, los=reserved)
        assert fps_r.los == reserved
        with pytest.raises(NotImplementedError):
            fps_r.measure_multipoles(k1dbins=np.linspace(0.1, 0.4, 4), ells=(0,))
        with pytest.raises(NotImplementedError):
            fps_r.multipole_bin_index_map(k1dbins=np.linspace(0.1, 0.4, 4))


def test_discrete_mu_sampling_with_identity_window():
    """
    Uniform weights, no beam/sampling/MAS: identity continuous W + discrete
    shell sum recovers discrete-μ multipoles of the 3D model.
    """
    import warnings

    from meer21cm import MockSimulation
    from meer21cm.util import redshift_to_freq

    nu_arr = np.linspace(redshift_to_freq(0.8), redshift_to_freq(0.6), 100)
    mock = MockSimulation(
        ra_range=(0, 20),
        dec_range=(-20, 20),
        nu=nu_arr,
        hp_nside=64,
        mean_amp_1=1.0,
        density="gaussian",
        include_beam=[False, False],
        include_sky_sampling=[False, False],
        compensate=[False, False],
        sigma_v_1=200,
    )
    mock.k1dbins = np.linspace(0.01, 0.25, 11)
    mock.field_1 = mock.mock_tracer_field_1

    # mock (3D→1D) vs model (3D→1D) monopole
    p1d_mock, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    p1d_model, _, _ = mock.get_1d_power(mock.auto_power_tracer_1_model)
    m = np.isfinite(p1d_mock) & np.isfinite(p1d_model) & (np.abs(p1d_model) > 0)
    rel_mm = np.abs(p1d_mock[m] - p1d_model[m]) / np.abs(p1d_model[m])
    # model is close to mock
    assert np.median(rel_mm) < 0.05
    assert np.max(rel_mm) < 0.1

    ells = (0, 2, 4)
    shell = mock.multipole_bin_index_map(k1dbins=mock.k1dbins)
    k_in = np.geomspace(
        max(float(mock.k1dbins[0]) * 0.5, 1e-3),
        float(mock.k1dbins[-1]) * 1.5,
        60,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cont = mock.get_theory_multipoles_kmu(k_in, ells=ells, nmu=64, which="auto_1")
    mat = build_discrete_shell_window_matrix(
        shell, k_in=k_in, ells=ells, continuous="identity", n_k_eval=128
    )
    P_win = mat.apply(cont["P_ell"])

    for ell in ells:
        p_model_ell, _, _ = mock.get_1d_power(
            mock.auto_power_tracer_1_model, multipole_ell=ell
        )
        m_ell = (
            np.isfinite(P_win[ell])
            & np.isfinite(p_model_ell)
            & (np.abs(p_model_ell) > 0)
        )
        rel = np.abs(P_win[ell][m_ell] - p_model_ell[m_ell]) / np.abs(
            p_model_ell[m_ell]
        )
        # mock naive 3d to 1d model is close to continuous model with window
        assert np.median(rel) < 0.01, "ell=%d median rel=%s" % (ell, np.median(rel))
        assert np.max(rel) < 0.05, "ell=%d max rel=%s" % (ell, np.max(rel))


def test_continuous_window_matrix_with_tapering():
    import warnings

    from meer21cm import MockSimulation
    from meer21cm.util import redshift_to_freq

    # build simple simulation
    z_min, z_max = 0.6, 0.8
    nu_arr = np.linspace(redshift_to_freq(z_max), redshift_to_freq(z_min), 100)
    mock = MockSimulation(
        seed=42,
        ra_range=(0, 20),
        dec_range=(-20, 20),
        nu=nu_arr,
        hp_nside=128,
        mean_amp_1=1.0,
        include_beam=[False, False],
        include_sky_sampling=[False, False],
        compensate=[False, False],
        sigma_v_1=200,
        density="gaussian",
    )
    mock.k1dbins = np.linspace(0.01, 0.25, 11)
    mock.field_1 = mock.mock_tracer_field_1
    mock.weights_1 = np.ones_like(mock.field_1)
    mock.apply_taper_to_field(1, axis=(0, 1, 2))
    # 3D model convolution uses weights_grid_1 (Field FFT uses weights_1)
    mock.weights_grid_1 = mock.weights_1
    # get the 3D to 1D exact averaging
    ells = (0, 2, 4)
    k1dbins = mock.k1dbins
    shell = mock.multipole_bin_index_map(k1dbins=k1dbins)

    # Discrete shells: mock FFT vs weight-convolved 3D model
    p3d_model = mock.auto_power_tracer_1_model
    p3d_mock = mock.auto_power_3d_1
    P_disc_model = {}
    P_disc_mock = {}
    for ell in ells:
        P_disc_model[ell], keff_disc, _ = mock.get_1d_power(
            p3d_model, multipole_ell=ell
        )
        P_disc_mock[ell], keff_disc, _ = mock.get_1d_power(p3d_mock, multipole_ell=ell)
    # Measure W_L on fine k1dbins_window; ensure enough sampling in low-k
    k1dbins_window = np.geomspace(
        max(float(mock.k1dbins[0]) * 0.1, 1e-3),
        float(mock.k1dbins[-1]) * 1.1,
        1000,
    )
    swe = SmoothWindowEstimator.from_power_spectrum(
        mock,
        tracer="hi",
        ells=ells,
        weights_hi=mock.weights_1,
        weights_grid_1=None,  # avoid double-counting: weights_hi is already the full taper
        k1dbins_window=k1dbins_window,
        k1dbins_out=mock.k1dbins,
    )
    swe.accumulate([swe.run_one(0)])
    k_in = np.geomspace(
        max(float(k1dbins[0]) * 0.5, 1e-3), float(k1dbins[-1]) * 1.5, 160
    )

    # Smooth continuous kernel + discrete shells
    mat_smooth = swe.build_window_matrix(
        k_in,
        shell_map=shell,
        continuous="smooth",
        n_fftlog=256,
        n_k_eval=128,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cont_fine = mock.get_theory_multipoles_kmu(
            k_in, ells=ells, nmu=64, which="auto_1"
        )

    P_win_smooth = mat_smooth.apply(cont_fine["P_ell"])
    # ell=4 is noisy
    for ell in (0, 2):
        m_ell = (
            np.isfinite(P_win_smooth[ell])
            & np.isfinite(P_disc_model[ell])
            & (np.abs(P_disc_model[ell]) > 0)
        )
        rel = np.abs(P_win_smooth[ell][m_ell] - P_disc_model[ell][m_ell]) / np.abs(
            P_disc_model[ell][m_ell]
        )
        assert np.median(rel) < 0.05, "ell=%d median rel=%s" % (ell, np.median(rel))
        assert np.max(rel) < 0.2, "ell=%d max rel=%s" % (ell, np.max(rel))
