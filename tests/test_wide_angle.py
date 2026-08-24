"""Odd wide-angle matrix (wa_order=1) vs pypower layout."""

import numpy as np
import pytest

from meer21cm.estimator import FieldPowerSpectrum
from meer21cm.multipole import SmoothWindowEstimator, WindowedMultipoleModel
from meer21cm.window import (
    DiscreteShellWindowMatrix,
    build_discrete_shell_window_matrix,
)
from meer21cm.wide_angle import (
    derivative_matrix_nonuniform,
    odd_wide_angle_coefficients,
    power_spectrum_odd_wide_angle_matrix,
    propose_odd_wa_ells,
)


def _endpoint_los_m_beutler(d, nk=10, kmin=0.0, kmax=0.2):
    """Uniform-grid endpoint M from fbeutler/pk_tools (pypower test reference)."""
    m = np.zeros((nk * 5, nk * 3))
    dk = (kmax - kmin) / nk
    kp = np.array([kmin + i * dk + dk / 2.0 for i in range(nk)])
    k1 = (3.0 / (5.0 * d)) * (3.0 / kp)
    k2 = (1.0 / d) * (3.0 / 5.0) * (2.0 / kp)
    k3 = (1.0 / d) * (10.0 / 9.0) * (5.0 / kp)
    m[:nk, :nk] = np.identity(nk)
    m[2 * nk : 3 * nk, nk : 2 * nk] = np.identity(nk)
    m[4 * nk : 5 * nk, 2 * nk : 3 * nk] = np.identity(nk)
    m[nk : 2 * nk, nk : 2 * nk] = np.diag(k1)
    m[3 * nk : 4 * nk, nk : 2 * nk] = np.diag(k2)
    m[3 * nk : 4 * nk, 2 * nk : 3 * nk] = np.diag(k3)

    def _deriv(index1, index2, ik, pre_factor):
        delta_k = kp[1] - kp[0]
        norm = 2.0 * delta_k if 0 < ik < nk - 1 else delta_k
        if ik > 0:
            m[index1, index2 - 1] = -pre_factor / (d * norm)
        else:
            m[index1, index2] += -pre_factor / (d * norm)
        if ik < nk - 1:
            m[index1, index2 + 1] = pre_factor / (d * norm)
        else:
            m[index1, index2] += pre_factor / (d * norm)

    for ik in range(nk):
        _deriv(nk + ik, nk + ik, ik, 3.0 / 5.0)
        _deriv(3 * nk + ik, nk + ik, ik, -3.0 / 5.0)
        _deriv(3 * nk + ik, 2 * nk + ik, ik, 10.0 / 9.0)
    return m, kp


def test_odd_wide_angle_coefficients():
    ells, coeffs = odd_wide_angle_coefficients(1, wa_order=1, los="firstpoint")
    assert ells == [2]
    np.testing.assert_allclose(coeffs, [0.6])
    ells, coeffs = odd_wide_angle_coefficients(3, wa_order=1, los="firstpoint")
    assert ells == [2, 4]
    np.testing.assert_allclose(coeffs, [-0.6, 10.0 / 9.0])
    ells_e, coeffs_e = odd_wide_angle_coefficients(3, wa_order=1, los="endpoint")
    assert ells_e == ells
    np.testing.assert_allclose(coeffs_e, [-c for c in coeffs])
    with pytest.raises(ValueError, match="order 1"):
        odd_wide_angle_coefficients(1, wa_order=2)
    with pytest.raises(ValueError, match="odd poles"):
        odd_wide_angle_coefficients(2, wa_order=1)
    with pytest.raises(ValueError, match="line-of-sight"):
        odd_wide_angle_coefficients(1, los="midpoint")


def test_propose_odd_wa_ells():
    assert propose_odd_wa_ells((0, 2, 4)) == (1, 3, 5)
    assert propose_odd_wa_ells((0, 2)) == (1, 3)
    assert propose_odd_wa_ells((1, 3)) == ()


def test_derivative_matrix_nonuniform_uniform_grid():
    x = np.linspace(0.05, 0.25, 6)
    d = derivative_matrix_nonuniform(x)
    h = x[1] - x[0]
    np.testing.assert_allclose(d[1:-1, :-2].diagonal(), -0.5 / h, atol=1e-12)
    np.testing.assert_allclose(d[1:-1, 2:].diagonal(), 0.5 / h, atol=1e-12)


def test_wa_matrix_endpoint_vs_beutler_reference():
    d = 1.0
    ref, k = _endpoint_los_m_beutler(d, nk=10, kmin=0.0, kmax=0.2)
    m = power_spectrum_odd_wide_angle_matrix(
        k, ells_in=(0, 2, 4), ells_out=(0, 1, 2, 3, 4), d=d, los="endpoint"
    )
    n_k = k.size
    ells_out = (0, 1, 2, 3, 4)
    ells_in = (0, 2, 4)
    for i_out, ell_out in enumerate(ells_out):
        for i_in, ell_in in enumerate(ells_in):
            sl_out = slice(i_out * n_k, (i_out + 1) * n_k)
            sl_in = slice(i_in * n_k, (i_in + 1) * n_k)
            block = m[sl_out, sl_in]
            block_ref = ref[sl_out, sl_in]
            if ell_out % 2 == 0:
                np.testing.assert_allclose(block, block_ref, atol=1e-12)
                continue
            # Interior k: same central-difference stencil as Beutler; edges differ
            # (pypower / we use a quadratic one-sided derivative).
            np.testing.assert_allclose(
                np.diag(block)[1:-1], np.diag(block_ref)[1:-1], atol=1e-10, rtol=1e-8
            )
            np.testing.assert_allclose(
                block[1:-1, 1:-1], block_ref[1:-1, 1:-1], atol=1e-8, rtol=1e-6
            )


def test_wa_matrix_vs_pypower_if_available():
    pypower = pytest.importorskip("pypower")
    from pypower.wide_angle import (
        PowerSpectrumOddWideAngleMatrix,
        Projection,
    )

    ells = [0, 2, 4]
    kmin, kmax, nk = 0.0, 0.2, 10
    dk = (kmax - kmin) / nk
    k = np.array([i * dk + dk / 2.0 for i in range(nk)])
    projsin = [Projection(ell=ell, wa_order=0) for ell in ells]
    projsout = [Projection(ell=ell, wa_order=ell % 2) for ell in range(ells[-1] + 1)]
    wa = PowerSpectrumOddWideAngleMatrix(
        k, projsin, projsout=projsout, d=1.0, wa_orders=1, los="endpoint"
    )
    m = power_spectrum_odd_wide_angle_matrix(
        k, ells_in=ells, ells_out=(0, 1, 2, 3, 4), d=1.0, los="endpoint"
    )
    np.testing.assert_allclose(m, wa.value.T, atol=1e-12, rtol=1e-10)


def test_endpoint_negates_firstpoint_odd_blocks():
    k = np.linspace(0.05, 0.25, 8)
    ells_in = (0, 2, 4)
    ells_out = (0, 1, 2, 3, 4)
    m_fp = power_spectrum_odd_wide_angle_matrix(
        k, ells_in=ells_in, ells_out=ells_out, d=1.0, los="firstpoint"
    )
    m_ep = power_spectrum_odd_wide_angle_matrix(
        k, ells_in=ells_in, ells_out=ells_out, d=1.0, los="endpoint"
    )
    n_k = k.size
    for i, ell in enumerate(ells_out):
        block_fp = m_fp[i * n_k : (i + 1) * n_k]
        block_ep = m_ep[i * n_k : (i + 1) * n_k]
        if ell % 2 == 0:
            np.testing.assert_allclose(block_fp, block_ep)
        else:
            np.testing.assert_allclose(block_fp, -block_ep)


def _identity_shell_mat(ells_out, ells_in, k_in=None):
    if k_in is None:
        k_in = np.geomspace(0.08, 0.35, 12)
    fps = FieldPowerSpectrum(
        np.ones((8, 8, 8)),
        np.array([80.0, 80.0, 80.0]),
        los="endpoint",
        los_observer=(0.0, 0.0, 1.0e5),
        _skip_specification=True,
    )
    shell = fps.multipole_bin_index_map(k1dbins=np.linspace(0.1, 0.3, 5))
    return build_discrete_shell_window_matrix(
        shell,
        None,
        None,
        k_in=k_in,
        ells=ells_out,
        ells_in=ells_in,
        continuous="identity",
    )


def test_resum_input_odd_wide_angle_shape_and_apply():
    ells_even = (0, 2, 4)
    ells_in_full = tuple(sorted(set(ells_even) | set(propose_odd_wa_ells(ells_even))))
    k_in = np.geomspace(0.08, 0.35, 12)
    n_k = k_in.size
    mat = _identity_shell_mat(ells_even, ells_in_full, k_in=k_in)
    n_out = mat.k_out.size
    assert mat.matrix.shape == (len(ells_even) * n_out, len(ells_in_full) * n_k)
    assert mat.ells_in == ells_in_full
    assert mat.ells_out == ells_even

    rng = np.random.default_rng(0)
    p_even = {ell: rng.normal(size=n_k) for ell in ells_even}
    m_wa = power_spectrum_odd_wide_angle_matrix(
        k_in, ells_in=ells_even, ells_out=ells_in_full, d=2.5, los="firstpoint"
    )
    vec_even = np.concatenate([p_even[ell] for ell in ells_even])
    vec_full = m_wa @ vec_even
    p_full = {
        ell: vec_full[i * n_k : (i + 1) * n_k] for i, ell in enumerate(ells_in_full)
    }
    out_unresummed = mat.apply(p_full)

    mat.resum_input_odd_wide_angle(los="firstpoint", d=2.5, ells_even=ells_even)
    assert mat.ells_in == ells_even
    assert mat.matrix.shape == (len(ells_even) * n_out, len(ells_even) * n_k)
    out_resummed = mat.apply(p_even)
    for ell in ells_even:
        np.testing.assert_allclose(out_resummed[ell], out_unresummed[ell], atol=1e-12)


def test_resum_wa_d_infinity_matches_even_only_window():
    ells_even = (0, 2, 4)
    ells_in_full = tuple(sorted(set(ells_even) | set(propose_odd_wa_ells(ells_even))))
    k_in = np.geomspace(0.08, 0.35, 12)
    w_full = _identity_shell_mat(ells_even, ells_in_full, k_in=k_in)
    w_even = _identity_shell_mat(ells_even, ells_even, k_in=k_in)
    w_full.resum_input_odd_wide_angle(los="endpoint", d=1e12, ells_even=ells_even)
    np.testing.assert_allclose(w_full.matrix, w_even.matrix, atol=1e-8, rtol=1e-6)
    assert w_full.ells_in == ells_even


def test_smooth_window_estimator_wide_angle_identity():
    ndim = (8, 8, 8)
    box_len = np.array([80.0, 80.0, 80.0])
    weights = np.ones(ndim)
    k1dbins = np.linspace(0.1, 0.3, 5)
    k_in = np.geomspace(0.08, 0.35, 10)
    est = SmoothWindowEstimator(
        box_len=box_len,
        k1dbins=k1dbins,
        ells=(0, 2, 4),
        tracer="hi",
        weights_hi=weights,
        los="endpoint",
        los_observer=np.array([-1e4, 0.0, 0.0]),
        wide_angle=True,
        wa_d=1e4,
        wa_los="endpoint",
    )
    mat = est.build_window_matrix(k_in, continuous="identity", wide_angle=True)
    assert mat.ells_out == (0, 2, 4)
    assert mat.ells_in == (0, 2, 4)
    assert mat.matrix.shape == (3 * (len(k1dbins) - 1), 3 * len(k_in))


def test_windowed_model_apply_uses_ells_in_after_resum():
    k_in = np.geomspace(0.08, 0.35, 10)
    mat = _identity_shell_mat((0, 2), (0, 1, 2, 3), k_in=k_in)
    mat.resum_input_odd_wide_angle(los="firstpoint", d=50.0, ells_even=(0, 2))
    kmode = np.geomspace(0.05, 0.4, 30).reshape(5, 3, 2)
    mumode = np.linspace(-1, 1, kmode.size).reshape(kmode.shape)
    model = WindowedMultipoleModel(
        kmode=kmode,
        mumode=mumode,
        tracer_bias_1=1.0,
        kaiser_rsd=False,
        window_matrix=mat,
        window_ells=(0, 2),
    )
    out = model.get_model_multipoles(which="auto_1", nmu=16)
    assert out["window_applied"]
    assert tuple(out["ells"]) == (0, 2)
    assert set(out["P_ell"]) == {0, 2}
