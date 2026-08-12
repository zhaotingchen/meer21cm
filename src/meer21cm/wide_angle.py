r"""
Odd wide-angle theory matrix (wa_order=1).

Port of pypower ``odd_wide_angle_coefficients`` /
``PowerSpectrumOddWideAngleMatrix`` (arXiv:2106.06324). Maps even
:math:`wa=0` multipoles to odd :math:`wa=1` poles (and even identity
blocks) on a 1D :math:`k` grid. Meer21cm stores the dense matrix in
``(out, in)`` block layout.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "derivative_matrix_nonuniform",
    "odd_wide_angle_coefficients",
    "power_spectrum_odd_wide_angle_matrix",
    "propose_odd_wa_ells",
]


def odd_wide_angle_coefficients(
    ell: int, wa_order: int = 1, los: str = "firstpoint"
) -> tuple[list[int], list[float]]:
    r"""
    Coefficients of the odd wide-angle expansion (order 1 only).

    For firstpoint LOS; both factors flip sign for ``los='endpoint'``.
    """
    if int(wa_order) != 1:
        raise ValueError("Only wide-angle order 1 supported")
    ell = int(ell)
    if ell % 2 == 0:
        raise ValueError("Wide-angle order 1 produces only odd poles")
    los_s = str(los).lower()
    if los_s not in ("firstpoint", "endpoint"):
        raise ValueError("Only 'firstpoint' and 'endpoint' line-of-sight supported")

    def coefficient(ell_i: int) -> float:
        return ell_i * (ell_i + 1) / 2.0 / (2 * ell_i + 1)

    sign = (-1) ** (los_s == "endpoint")
    if ell == 1:
        return [ell + 1], [sign * coefficient(ell + 1)]
    return [ell - 1, ell + 1], [
        -sign * coefficient(ell - 1),
        sign * coefficient(ell + 1),
    ]


def propose_odd_wa_ells(ells_even: Sequence[int], wa_order: int = 1) -> tuple[int, ...]:
    """Odd :math:`\\ell` that wa_order=1 can produce from even input multipoles."""
    if int(wa_order) != 1:
        raise ValueError("Only wide-angle order 1 supported")
    ellsin = [int(e) for e in ells_even if int(e) % 2 == 0]
    if not ellsin:
        return ()
    out: list[int] = []
    for ellout in range(1, max(ellsin) + 2, 2):
        src, _ = odd_wide_angle_coefficients(ellout, wa_order=1, los="firstpoint")
        if any(ell in ellsin for ell in src):
            out.append(ellout)
    return tuple(out)


def derivative_matrix_nonuniform(x: ArrayLike) -> NDArray[np.floating]:
    r"""
    Second-order finite-difference matrix for :math:`d/dx` on an arbitrary 1D grid.

    Same stencil as pypower ``derivative_matrix_nonuniform``.
    """
    x_np = np.asarray(x, dtype=float)
    n = len(x_np)
    D = np.zeros((n, n), dtype=float)
    for i in range(1, n - 1):
        h0 = x_np[i] - x_np[i - 1]
        h1 = x_np[i + 1] - x_np[i]
        D[i, i - 1] = -h1 / (h0 * (h0 + h1))
        D[i, i] = (h1 - h0) / (h0 * h1)
        D[i, i + 1] = h0 / (h1 * (h0 + h1))
    x0, x1, x2 = x_np[0], x_np[1], x_np[2]
    D[0, 0] = (2 * x0 - x1 - x2) / ((x0 - x1) * (x0 - x2))
    D[0, 1] = (x0 - x2) / ((x1 - x0) * (x1 - x2))
    D[0, 2] = (x0 - x1) / ((x2 - x0) * (x2 - x1))
    xm2, xm1, xm0 = x_np[-3], x_np[-2], x_np[-1]
    D[-1, -3] = (xm0 - xm1) / ((xm2 - xm0) * (xm2 - xm1))
    D[-1, -2] = (xm0 - xm2) / ((xm1 - xm0) * (xm1 - xm2))
    D[-1, -1] = (2 * xm0 - xm2 - xm1) / ((xm0 - xm2) * (xm0 - xm1))
    return D


def power_spectrum_odd_wide_angle_matrix(
    k: ArrayLike,
    ells_in: Sequence[int],
    ells_out: Sequence[int] | None = None,
    d: float = 1.0,
    los: str = "firstpoint",
) -> NDArray[np.floating]:
    r"""
    Dense wa_order=1 matrix on a 1D :math:`k` grid, shape
    ``(len(ells_out) * n_k, len(ells_in) * n_k)`` (``out, in``).

    ``ells_in`` are even :math:`wa=0` theory multipoles. ``ells_out`` defaults
    to ``ells_in`` plus :func:`propose_odd_wa_ells`. Even→even blocks are
    identity; odd outputs are the pypower / Beutler :math:`1/d` and
    :math:`\partial_k` terms.
    """
    k_np = np.asarray(k, dtype=float)
    if k_np.ndim != 1 or k_np.size < 3:
        raise ValueError("k must be a 1D array of length >= 3")
    ells_in_t = tuple(int(e) for e in ells_in)
    if any(e % 2 for e in ells_in_t):
        raise ValueError("ells_in must be even wa=0 multipoles")
    if ells_out is None:
        ells_out_t = tuple(sorted(set(ells_in_t) | set(propose_odd_wa_ells(ells_in_t))))
    else:
        ells_out_t = tuple(int(e) for e in ells_out)
    los_s = str(los).lower()
    if los_s not in ("firstpoint", "endpoint"):
        raise ValueError("los must be 'firstpoint' or 'endpoint'")
    d = float(d)
    if d == 0.0:
        raise ValueError("wa_d / d must be nonzero")

    n_k = k_np.size
    n_in = len(ells_in_t)
    n_out = len(ells_out_t)
    matrix = np.zeros((n_out * n_k, n_in * n_k), dtype=float)
    eye = np.eye(n_k, dtype=float)
    deriv = derivative_matrix_nonuniform(k_np)

    for i_out, ell_out in enumerate(ells_out_t):
        for i_in, ell_in in enumerate(ells_in_t):
            if ell_out == ell_in and ell_out % 2 == 0:
                block = eye
            elif ell_out % 2 == 1:
                src, coeffs = odd_wide_angle_coefficients(
                    ell_out, wa_order=1, los=los_s
                )
                if ell_in not in src:
                    continue
                coeff = coeffs[src.index(ell_in)] / d
                if ell_in == ell_out + 1:
                    coeff_sb = -(ell_in + 1)
                else:
                    coeff_sb = ell_in
                # pypower stores tmp.T in (in, out); our (out, in) block is tmp.
                block = np.diag(coeff_sb * coeff / k_np) - coeff * deriv
            else:
                continue
            matrix[
                i_out * n_k : (i_out + 1) * n_k,
                i_in * n_k : (i_in + 1) * n_k,
            ] = block
    return matrix
