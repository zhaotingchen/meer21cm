"""
Real spherical harmonics and unit-vector helpers for Yamamoto multipoles.

``get_real_Ylm`` follows pypower / nbodykit (Hand et al. real-Ylm convention).
Sympy + numexpr are used when installed; otherwise scipy ``lpmv``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import lpmv

__all__ = [
    "get_real_Ylm",
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
