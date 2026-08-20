"""Tests for real spherical harmonics and LOS unit vectors."""

import numpy as np
from scipy.special import eval_legendre

from meer21cm.spherical import (
    get_real_Ylm,
    unit_khat_from_k_vec,
    unit_los_from_observer,
    unit_vectors_from_components,
)


def test_y00_is_constant():
    Y00 = get_real_Ylm(0, 0, modules="scipy")
    rng = np.random.default_rng(0)
    n = rng.normal(size=(3, 16))
    n /= np.linalg.norm(n, axis=0, keepdims=True)
    vals = Y00(n[0], n[1], n[2])
    np.testing.assert_allclose(vals, 1.0 / np.sqrt(4.0 * np.pi))


def test_real_ylm_addition_theorem():
    rng = np.random.default_rng(1)
    k = rng.normal(size=3)
    n = rng.normal(size=3)
    k /= np.linalg.norm(k)
    n /= np.linalg.norm(n)
    mu = float(np.dot(k, n))
    for ell in (0, 1, 2, 3, 4):
        s = 0.0
        for m in range(-ell, ell + 1):
            Ylm = get_real_Ylm(ell, m, modules="scipy")
            s += float(Ylm(*k) * Ylm(*n))
        expected = (2 * ell + 1) / (4.0 * np.pi) * float(eval_legendre(ell, mu))
        np.testing.assert_allclose(s, expected, rtol=1e-10, atol=1e-12)


def test_get_real_ylm_rejects_bad_m_and_modules():
    import pytest

    with pytest.raises(ValueError, match="\\|m\\|"):
        get_real_Ylm(2, 3, modules="scipy")
    with pytest.raises(ValueError, match="modules"):
        get_real_Ylm(1, 0, modules="numpy")


def test_unit_los_and_khat():
    x_vec = (np.array([0.5, 1.5]), np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    nx, ny, nz = unit_los_from_observer(x_vec, (0.0, 0.0, 10.0))
    assert nx.shape == (2, 2, 2)
    norms = np.sqrt(nx**2 + ny**2 + nz**2)
    np.testing.assert_allclose(norms, 1.0)
    # far +z observer → n̂ ≈ z-hat
    np.testing.assert_allclose(nz, 1.0, atol=0.2)
    zx, zy, zz = unit_vectors_from_components(0.0, 0.0, 0.0)
    assert zx == 0.0 and zy == 0.0 and zz == 0.0
    k_vec = (
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
        np.array([0.0, 0.5]),
    )
    khx, khy, khz = unit_khat_from_k_vec(k_vec)
    assert khx.shape == (2, 2, 2)
    # k=0 mode
    assert khx[0, 0, 0] == 0.0
