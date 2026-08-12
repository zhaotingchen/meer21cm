"""
Real spherical harmonics and unit-vector helpers for Yamamoto multipoles.

``get_real_Ylm`` follows pypower / nbodykit (Hand et al. real-Ylm convention).
Sympy + numexpr are used when installed; otherwise scipy ``lpmv``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import eval_legendre, lpmv

__all__ = [
    "get_real_Ylm",
    "mean_legendre_over_los",
    "sample_los_unit_vectors",
    "unit_khat_from_k_vec",
    "unit_los_from_observer",
    "unit_vectors_from_components",
]


def get_real_Ylm(
    ell: int, m: int, modules: str | None = None
) -> Callable[..., NDArray]:
    """
    Return a function that evaluates the real spherical harmonic :math:`Y_{\\ell m}`.

    Adapted from pypower ``fft_power.get_real_Ylm`` / nbodykit.

    Parameters
    ----------
    ell : int
        Degree.
    m : int
        Order; ``|m| <= ell``.
    modules : {'sympy', 'scipy', None}, optional
        If ``None``, use sympy (+ numexpr) when available, else scipy.

    Returns
    -------
    Ylm : callable
        ``Ylm(xhat, yhat, zhat)`` on unit-normalised Cartesian components.
    """
    ell = int(ell)
    m = int(m)
    if abs(m) > ell:
        raise ValueError(f"Require |m| <= ell; got ell={ell}, m={m}")

    amp = np.sqrt((2 * ell + 1) / (4.0 * np.pi))
    if m != 0:
        fac = 1
        for n in range(ell - abs(m) + 1, ell + abs(m) + 1):
            fac *= n
        amp *= np.sqrt(2.0 / fac)

    sp = None
    if modules is None:
        try:
            import sympy as sp  # noqa: F401
        except ImportError:
            sp = None
        else:
            import sympy as sp
    elif "sympy" in str(modules):
        import sympy as sp
    elif "scipy" not in str(modules):
        raise ValueError('modules must be one of ["sympy", "scipy", None]')

    if sp is None:

        def Ylm(xhat, yhat, zhat):
            toret = amp * ((-1) ** m) * lpmv(abs(m), ell, zhat)
            phi = np.arctan2(yhat, xhat)
            if m < 0:
                toret = toret * np.sin(abs(m) * phi)
            elif m > 0:
                toret = toret * np.cos(m * phi)
            return toret

        Ylm.l = ell  # type: ignore[attr-defined]
        Ylm.m = m  # type: ignore[attr-defined]
        return Ylm

    x, y, z, r = sp.symbols("x y z r", real=True, positive=True)
    xhat_s, yhat_s, zhat_s = sp.symbols("xhat yhat zhat", real=True, positive=True)
    phi, theta = sp.symbols("phi theta")
    defs = [
        (sp.sin(phi), y / sp.sqrt(x**2 + y**2)),
        (sp.cos(phi), x / sp.sqrt(x**2 + y**2)),
        (sp.cos(theta), z / sp.sqrt(x**2 + y**2 + z**2)),
    ]
    expr = ((-1) ** m) * sp.assoc_legendre(ell, abs(m), sp.cos(theta))
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m) * phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m * phi))
    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x / r, xhat_s), (y / r, yhat_s), (z / r, zhat_s)])
    try:
        import numexpr  # noqa: F401
    except ImportError:
        numexpr = None
    Ylm = sp.lambdify(
        (xhat_s, yhat_s, zhat_s),
        expr,
        modules="numexpr" if numexpr is not None else ["scipy", "numpy"],
    )
    Ylm.expr = expr  # type: ignore[attr-defined]
    Ylm.l = ell  # type: ignore[attr-defined]
    Ylm.m = m  # type: ignore[attr-defined]
    return Ylm


def unit_vectors_from_components(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return unit vectors ``(x, y, z) / |r|``, with zeros where ``|r|=0``."""
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    z_np = np.asarray(z, dtype=float)
    norm = np.sqrt(x_np**2 + y_np**2 + z_np**2)
    out_x = np.zeros_like(x_np, dtype=float)
    out_y = np.zeros_like(y_np, dtype=float)
    out_z = np.zeros_like(z_np, dtype=float)
    good = norm > 0.0
    np.divide(x_np, norm, out=out_x, where=good)
    np.divide(y_np, norm, out=out_y, where=good)
    np.divide(z_np, norm, out=out_z, where=good)
    return out_x, out_y, out_z


def unit_los_from_observer(
    x_vec: Sequence[ArrayLike],
    los_observer: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    r"""
    Per-voxel line-of-sight unit vector on a Cartesian mesh.

    .. math::

        \hat n(x) = (x + x_{\mathrm{obs}}) / |x + x_{\mathrm{obs}}|

    ``x_vec`` is the 1D cell-centre coordinate tuple from
    :func:`~meer21cm.power_ops.get_x_vector` (``ij`` meshgrid).
    """
    observer = np.asarray(los_observer, dtype=float).reshape(3)
    xx, yy, zz = np.meshgrid(
        *[np.asarray(c, dtype=float) for c in x_vec], indexing="ij"
    )
    return unit_vectors_from_components(
        xx + observer[0], yy + observer[1], zz + observer[2]
    )


def unit_khat_from_k_vec(
    k_vec: Sequence[ArrayLike],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Unit wavevector :math:`\\hat k` on the rFFT lattice of ``k_vec``."""
    kx, ky, kz = np.meshgrid(
        *[np.asarray(c, dtype=float) for c in k_vec], indexing="ij"
    )
    return unit_vectors_from_components(kx, ky, kz)


def sample_los_unit_vectors(
    x_vec: Sequence[ArrayLike],
    los_observer: ArrayLike,
    los_weights: ArrayLike | None = None,
    n_los_samples: int = 1024,
    rng: np.random.Generator | int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Sample voxel line-of-sight unit vectors with optional weights.

    Returns ``(n_hat, sample_weights)`` with ``n_hat.shape == (n_sample, 3)``.
    Uses every voxel with positive weight when that count is ``<= n_los_samples``.
    """
    xh, yh, zh = unit_los_from_observer(x_vec, los_observer)
    hats = np.stack([xh.ravel(), yh.ravel(), zh.ravel()], axis=1)
    if los_weights is None:
        weights = np.ones(hats.shape[0], dtype=float)
    else:
        weights = np.asarray(los_weights, dtype=float).ravel()
        if weights.shape[0] != hats.shape[0]:
            raise ValueError(
                "los_weights size %s does not match voxel count %s"
                % (weights.shape[0], hats.shape[0])
            )
    norms = np.linalg.norm(hats, axis=1)
    good = (weights > 0.0) & np.isfinite(weights) & np.isfinite(norms) & (norms > 0.0)
    hats = hats[good]
    weights = weights[good]
    if hats.shape[0] == 0:
        raise ValueError("No voxels with positive los_weights")
    n_take = int(n_los_samples)
    if n_take < 1:
        raise ValueError("n_los_samples must be >= 1")
    if hats.shape[0] > n_take:
        generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        p = weights / weights.sum()
        idx = generator.choice(hats.shape[0], size=n_take, replace=False, p=p)
        hats = hats[idx]
        weights = weights[idx]
    return hats, weights


def mean_legendre_over_los(
    khat: tuple[ArrayLike, ArrayLike, ArrayLike] | Sequence[ArrayLike],
    n_hats: ArrayLike,
    ells: Sequence[int],
    sample_weights: ArrayLike | None = None,
) -> dict[int, NDArray[np.floating]]:
    r"""
    Per-mode voxel-averaged :math:`\langle\mathcal{L}_\ell(\hat k\cdot\hat n)\rangle`.

    ``khat`` is ``(kxhat, kyhat, kzhat)`` on the rFFT lattice. ``n_hats`` has
    shape ``(n_sample, 3)``.
    """
    khx = np.asarray(khat[0], dtype=float)
    khy = np.asarray(khat[1], dtype=float)
    khz = np.asarray(khat[2], dtype=float)
    hats = np.asarray(n_hats, dtype=float)
    if hats.ndim != 2 or hats.shape[1] != 3:
        raise ValueError("n_hats must have shape (n_sample, 3)")
    if sample_weights is None:
        w = np.ones(hats.shape[0], dtype=float)
    else:
        w = np.asarray(sample_weights, dtype=float).ravel()
        if w.shape[0] != hats.shape[0]:
            raise ValueError("sample_weights length must match n_hats")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("sample_weights must sum to a positive value")
    w = w / w_sum
    kh = np.stack((khx, khy, khz), axis=-1)
    mu = np.clip(kh @ hats.T, -1.0, 1.0)
    out: dict[int, NDArray[np.floating]] = {}
    for ell in ells:
        ell_i = int(ell)
        out[ell_i] = np.tensordot(eval_legendre(ell_i, mu), w, axes=([-1], [0]))
    return out
