r"""
Pure helper functions for windowed multipole modelling.

Used by :mod:`meer21cm.multipole` and :mod:`meer21cm.window`.  Kept free of
the combined analysis class and of survey-window matrix algebra (that
lives in :mod:`meer21cm.window`).

Helpers here cover:

- fine theory / window-measure :math:`k` grids (``propose_k_in``, …);
- map-sampling and dish-beam transfers for the mesh window
  (``map_sampling_mode_scale``, ``beam_input_cell_kernels``, …);
- selection fields and pickleable workers for measuring survey-window
  multipoles :math:`W_L(k)` (``run_smooth_window_realization``, …).

Stateful classes (:class:`~meer21cm.multipole.MultipolePowerSpectrum`,
:class:`~meer21cm.multipole.WindowedMultipoleModel`,
:class:`~meer21cm.multipole.SmoothWindowEstimator`) live in
:mod:`meer21cm.multipole`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimator import FieldPowerSpectrum, MultipoleMeasurement
from .power_ops import (
    gaussian_beam_attenuation,
    power_weights_renorm,
    step_window_attenuation,
)
from .wide_angle import propose_odd_wa_ells

logger = logging.getLogger(__name__)

Tracer = Literal["hi", "gal", "cross"]


__all__ = [
    "AccumulatedWindow",
    "Tracer",
    "accumulate_window_multipoles",
    "beam_cell_sigma_perp",
    "beam_diagonal_correction",
    "beam_edge_cell_mass",
    "beam_input_cell_kernels",
    "beam_input_cell_masses",
    "beam_input_diagonal_correction",
    "beam_input_mode_groups",
    "beam_kernel_bin_masses",
    "beam_mode_group_index",
    "beam_out_mode_scale",
    "beam_ylm_alpha",
    "beam_ylm_cell_kernels",
    "beam_ylm_cell_masses",
    "beam_ylm_diagonal_correction",
    "beam_ylm_labels",
    "cell_grid_los",
    "cell_sampling_geometry",
    "cell_sampling_kernel",
    "cell_sampling_kernel_mu_rms",
    "exact_beam_legs",
    "gaussian_beam_legendre_moments",
    "make_galaxy_poisson_mean_density",
    "make_galaxy_poisson_random",
    "make_im_selection_field",
    "map_sampling_mode_scale",
    "mean_beam_amplitude_on_cells",
    "mean_gaussian_beam_on_modes",
    "propose_k1dbins_window",
    "propose_k_in",
    "propose_window_measure_ells",
    "run_smooth_window_realization",
]


def propose_window_measure_ells(
    ells: Sequence[int], *, wide_angle: bool = False
) -> tuple[int, ...]:
    """
    Multipole orders :math:`L` to measure for :math:`W_L(k)`.

    Follows pypower's ``CatalogSmoothWindow`` rule: even window multipoles
    up to :math:`2\\ell_{\\max}` for output multipoles ``ells``. With
    ``wide_angle=True``, also include odd :math:`L` needed for resum.
    """
    ells_t = tuple(int(e) for e in ells)
    even = tuple(e for e in ells_t if e % 2 == 0)
    if wide_angle and even:
        odds = propose_odd_wa_ells(even)
        max_ell = max(list(ells_t) + list(odds))
        extra_L = tuple(range(0, max_ell + 3))
        return tuple(sorted(set(ells_t) | set(odds) | set(extra_L)))
    max_ell = max(even) if even else max(ells_t)
    return tuple(range(0, 2 * max_ell + 1, 2))


def propose_k_in(
    k1dbins: ArrayLike,
    *,
    n: int = 80,
    low_factor: float = 0.5,
    high_factor: float = 1.5,
) -> NDArray[np.floating]:
    """
    Fine theory :math:`k_{\\mathrm{in}}` spanning estimator bin edges.

    Shared convention used by validation scripts and
    :func:`predict_windowed_multipoles`.
    """
    edges = np.asarray(k1dbins, dtype=float)
    if edges.size < 2:
        raise ValueError("k1dbins must have at least two edges")
    return np.geomspace(
        max(float(edges[0]) * float(low_factor), 1e-3),
        float(edges[-1]) * float(high_factor),
        int(n),
    )


def map_sampling_mode_scale(ps, *, z_resolved=True):
    r"""
    Map-sampling transfer :math:`|S(k)|^2` for the window-matrix ``mode_scale``.

    Propagation averages the fine box field over each HEALPix pixel ×
    frequency channel — a 3D bin constant in (angle, frequency) whose
    **comoving** size varies with redshift:

    .. math::

        D_\perp(z) = \theta_{\mathrm{pix}}\,\chi(z),\quad
        D_\parallel(z) = |\chi(\nu+\Delta\nu/2)-\chi(\nu-\Delta\nu/2)|.

    With ``z_resolved=False`` (level 0): the survey-mean kernel

    .. math::

        \mathrm{sinc}^2(k_x D_\perp/2)\,
        \mathrm{sinc}^2(k_y D_\perp/2)\,
        \mathrm{sinc}^2(k_z D_\parallel/2)

    using ``pix_resol_in_mpc`` / ``los_resol_in_mpc``.

    With ``z_resolved=True`` (level 1): transverse uses the mean-width
    sinc; radial uses the per-channel projected width

    .. math::

        D_{\parallel,\mathrm{eff}}(z)^2
        = D_\parallel(z)^2\langle\cos^2\theta\rangle
        + D_\perp(z)^2\langle\sin^2\theta\rangle,

    with ``θ`` the pixel angle from the box ``z`` axis, averaged with
    uniform channel weights.

    Returns an array on the rFFT grid of ``ps``.
    """
    from .util import freq_to_redshift, get_nd_slicer

    slicer = get_nd_slicer()
    kx = np.asarray(ps.k_vec[0][slicer[0]], dtype=float)
    ky = np.asarray(ps.k_vec[1][slicer[1]], dtype=float)
    kz = np.asarray(ps.k_vec[2][slicer[2]], dtype=float)
    if not z_resolved:
        dperp = float(ps.pix_resol_in_mpc)
        dpar = float(ps.los_resol_in_mpc)
        return (
            step_window_attenuation(kx, dperp, p=2)
            * step_window_attenuation(ky, dperp, p=2)
            * step_window_attenuation(kz, dpar, p=2)
        )

    cosmo = ps.astropy_cosmo_fiducial
    theta_pix = float(np.radians(ps.pix_resol))
    dnu = float(ps.freq_resol)
    z_lo = freq_to_redshift(ps.nu + 0.5 * dnu)
    z_hi = freq_to_redshift(ps.nu - 0.5 * dnu)
    chi_lo = cosmo.comoving_distance(z_lo).value
    chi_hi = cosmo.comoving_distance(z_hi).value
    dpar_ch = np.abs(chi_hi - chi_lo)
    chi_ch = 0.5 * (chi_lo + chi_hi)
    dperp_ch = theta_pix * chi_ch

    pos = np.asarray(ps.pix_coor_in_cartesian, dtype=float).reshape(-1, 3)
    nrm = np.linalg.norm(pos, axis=1)
    ok = nrm > 0
    cos2 = float(np.mean((pos[ok, 2] / nrm[ok]) ** 2))
    sin2 = 1.0 - cos2

    dperp_mean = float(ps.pix_resol_in_mpc)
    b_t = step_window_attenuation(kx, dperp_mean, p=2) * step_window_attenuation(
        ky, dperp_mean, p=2
    )
    out = np.zeros((kx.shape[0], ky.shape[1], kz.shape[2]))
    for c in range(len(chi_ch)):
        dpar_eff = float(np.sqrt(dpar_ch[c] ** 2 * cos2 + dperp_ch[c] ** 2 * sin2))
        out = out + b_t * step_window_attenuation(kz, dpar_eff, p=2)
    return out / len(chi_ch)


def _unique_pixel_nhat(ps) -> NDArray[np.floating]:
    """Unit LOS per unique sky pixel, in the box frame."""
    if getattr(ps, "_pix_coor_in_cartesian", None) is not None:
        pos = np.asarray(ps.pix_coor_in_cartesian, dtype=float).reshape(-1, 3)
    else:
        origin = np.asarray(getattr(ps, "box_origin", np.zeros(3)), dtype=float)
        pos = np.asarray(ps.pix_coor_in_box, dtype=float).reshape(
            -1, 3
        ) + origin.reshape(1, 3)
    n_ch = int(np.size(getattr(ps, "nu", 1)))
    if n_ch > 1 and pos.shape[0] % n_ch == 0:
        n_pix = pos.shape[0] // n_ch
        pos = pos.reshape(n_pix, n_ch, 3)[:, 0, :]
    chi = np.linalg.norm(pos, axis=1)
    nhat = np.zeros_like(pos)
    ok = chi > 0
    nhat[ok] = pos[ok] / chi[ok, None]
    return nhat


def gaussian_beam_legendre_moments(
    x,
    ells: Sequence[int],
    *,
    nmu: int = 64,
    power: int = 1,
) -> NDArray[np.floating]:
    r"""
    Legendre moments of a Gaussian beam in :math:`\mu=\hat k\cdot\hat n`.

    .. math::

        f_L(x)=\frac{2L+1}{2}\int_{-1}^{1}\mathrm{d}\mu\,
        \mathcal{L}_L(\mu)\,
        \exp\bigl(-\tfrac12\,\mathrm{power}\,x^2(1-\mu^2)\bigr).

    ``power=1`` is the amplitude :math:`B`; ``power=2`` is :math:`B^2`.
    """
    from scipy.special import eval_legendre

    x = np.asarray(x, dtype=float)
    ells_t = tuple(int(e) for e in ells)
    mus, wts = np.polynomial.legendre.leggauss(int(nmu))
    expo = np.exp(-0.5 * float(power) * x[..., None] ** 2 * (1.0 - mus**2))
    out = np.empty((len(ells_t),) + x.shape, dtype=float)
    for i, ell in enumerate(ells_t):
        pl = eval_legendre(int(ell), mus)
        out[i] = (
            0.5 * (2 * int(ell) + 1) * np.tensordot(expo, wts * pl, axes=([-1], [0]))
        )
    return out


def mean_gaussian_beam_on_modes(
    k_abs,
    khat,
    nhat,
    sigma_ch,
    *,
    coherent: bool = True,
    ell_max: int = 16,
    nmu: int = 64,
) -> NDArray[np.floating]:
    r"""
    Cell-averaged Gaussian beam via a real-:math:`Y_{LM}` addition theorem.

    :math:`\langle B\rangle` (``coherent=True``) or :math:`\langle B^2\rangle`
    (``coherent=False``) on the same grid as ``k_abs``.  ``khat`` is a
    3-tuple of arrays matching ``k_abs``; ``nhat`` is ``(n_pix, 3)``;
    ``sigma_ch`` is the per-channel comoving Gaussian :math:`\sigma`.
    """
    from .spherical import get_real_Ylm

    k_abs = np.asarray(k_abs, dtype=float)
    nhat = np.asarray(nhat, dtype=float).reshape(-1, 3)
    sigma_ch = np.atleast_1d(np.asarray(sigma_ch, dtype=float))
    power = 1 if coherent else 2
    ells = tuple(range(0, int(ell_max) + 1, 2))
    a_lm = {}
    for ell in ells:
        for m in range(-ell, ell + 1):
            ylm = get_real_Ylm(ell, m)
            a_lm[(ell, m)] = float(np.mean(ylm(nhat[:, 0], nhat[:, 1], nhat[:, 2])))

    k_flat = k_abs.ravel()
    k_pos = k_flat[k_flat > 0]
    k_hi = float(k_flat.max()) if k_flat.size else 0.0
    if k_pos.size == 0 or k_hi <= 0:
        return np.ones_like(k_abs)
    k_nodes = np.concatenate(
        ([0.0], np.geomspace(max(float(k_pos.min()), 1.0e-8), k_hi * 1.01, 255))
    )
    f_nodes = np.zeros((len(ells), k_nodes.size), dtype=float)
    for sig in sigma_ch:
        f_nodes = f_nodes + gaussian_beam_legendre_moments(
            k_nodes * float(sig), ells, nmu=int(nmu), power=power
        )
    f_nodes = f_nodes / float(sigma_ch.size)

    mean_b = np.zeros_like(k_abs, dtype=float)
    for i, ell in enumerate(ells):
        f_grid = np.interp(k_abs, k_nodes, f_nodes[i])
        proj = np.zeros_like(k_abs, dtype=float)
        for m in range(-ell, ell + 1):
            ylm = get_real_Ylm(ell, m)
            proj = proj + ylm(*khat) * a_lm[(ell, m)]
        mean_b = mean_b + f_grid * (4.0 * np.pi / (2 * ell + 1)) * proj
    return np.clip(mean_b, 0.0, 1.0)


def beam_out_mode_scale(
    ps,
    *,
    level: int = 1,
    coherent: bool = True,
    ell_max: int = 16,
    nmu: int = 64,
    sigma_beam_in_mpc: float | None = None,
    sigma_beam_ch_in_mpc: ArrayLike | None = None,
):
    r"""
    Beam transfer at the **output** Fourier mode, for ``out_mode_scale``.

    The sky-plane beam is applied to the discrete map *after* pixel ×
    channel binning, so it multiplies the output mode :math:`\mathbf k`
    inside the cell comb, not the theory mode :math:`\mathbf q`.  The
    returned array is the power-spectrum factor :math:`|\bar B(\mathbf k)|^2`
    (or :math:`\langle B^2\rangle` if ``coherent=False``) on the rFFT grid
    of ``ps``.

    Level 0: box-frame :math:`k_\perp=\sqrt{k_x^2+k_y^2}` and the
    channel-mean ``sigma_beam_in_mpc`` — the legacy
    :func:`~meer21cm.power_ops.gaussian_beam_attenuation` squared.

    Level 1: uniform average over map cells
    :math:`B_b=\exp[-\tfrac12 k^2(1-\mu_b^2)\sigma_{\perp,b}^2]`,
    evaluated with a real-:math:`Y_{LM}` addition theorem.  ``coherent``
    selects :math:`\langle B\rangle^2` versus :math:`\langle B^2\rangle`.
    """
    from .util import get_nd_slicer
    from .spherical import unit_khat_from_k_vec

    slicer = get_nd_slicer()
    kx = np.asarray(ps.k_vec[0][slicer[0]], dtype=float)
    ky = np.asarray(ps.k_vec[1][slicer[1]], dtype=float)
    kz = np.asarray(ps.k_vec[2][slicer[2]], dtype=float)

    if int(level) == 0:
        if sigma_beam_in_mpc is None:
            sigma_beam_in_mpc = getattr(ps, "sigma_beam_in_mpc", None)
        if sigma_beam_in_mpc is None:
            raise ValueError("beam_out_mode_scale level 0 needs sigma_beam_in_mpc")
        k_perp = np.sqrt(kx**2 + ky**2 + 0.0 * kz)
        b_amp = gaussian_beam_attenuation(k_perp, float(sigma_beam_in_mpc))
        return b_amp**2

    if int(level) != 1:
        raise ValueError(f"beam_out_mode_scale level must be 0 or 1; got {level}")

    if sigma_beam_ch_in_mpc is None:
        sigma_beam_ch_in_mpc = getattr(ps, "sigma_beam_ch_in_mpc", None)
    if sigma_beam_ch_in_mpc is None:
        if sigma_beam_in_mpc is None:
            sigma_beam_in_mpc = getattr(ps, "sigma_beam_in_mpc", None)
        if sigma_beam_in_mpc is None:
            raise ValueError("beam_out_mode_scale level 1 needs sigma_beam_ch_in_mpc")
        sigma_beam_ch_in_mpc = np.array([float(sigma_beam_in_mpc)])
    sig_ch = np.atleast_1d(np.asarray(sigma_beam_ch_in_mpc, dtype=float))
    k_abs = np.sqrt(kx**2 + ky**2 + kz**2)
    khat = unit_khat_from_k_vec(ps.k_vec)
    nhat = _unique_pixel_nhat(ps)
    mean_b = mean_gaussian_beam_on_modes(
        k_abs,
        khat,
        nhat,
        sig_ch,
        coherent=bool(coherent),
        ell_max=int(ell_max),
        nmu=int(nmu),
    )
    if coherent:
        return mean_b**2
    return mean_b


def beam_edge_cell_mass(ps):
    r"""
    Per-cell :math:`1/n_b` from the **binary-mask** beam-edge renormalization.

    :func:`~meer21cm.telescope.weighted_smoothing_healpix` with
    :math:`w=\mathbf{1}_\Omega` returns
    :math:`\mathcal{S}[s B]/n_b` with
    :math:`n_b=\mathcal{S}[\mathbf{1}_\Omega B]`.  The smoothed inverse-
    variance weights are discarded by the mock propagator, so this is a
    geometric amplitude on the deposit comb (``particle_mass`` of
    :func:`~meer21cm.window.ngp_raw_cell_comb`).  Interior cells have
    :math:`n_b\simeq 1`; the survey edge falls toward :math:`\sim 1/2`
    over one beam width.

    Returns a 1-D array matching ``ps.pix_coor_in_box`` (unit masses if
    ``sigma_beam_ch`` is ``None``).
    """
    import healpy as hp

    from .telescope import gaussian_beam_window

    n_cell = int(np.asarray(ps.pix_coor_in_box).reshape(-1, 3).shape[0])
    if getattr(ps, "sigma_beam_ch", None) is None:
        return np.ones(n_cell, dtype=float)

    nside = int(ps.hp_nside)
    pid = np.asarray(ps.pixel_id, dtype=np.int64).ravel()
    n_pix = int(pid.size)
    n_ch = int(np.size(ps.nu))
    npix_full = int(hp.nside2npix(nside))
    w_full = np.zeros(npix_full, dtype=np.float64)
    w_full[pid] = 1.0

    if hasattr(ps, "get_beam_window_ch"):
        bwin = np.asarray(ps.get_beam_window_ch(), dtype=np.float64)
    else:
        sigma = np.atleast_1d(np.asarray(ps.sigma_beam_ch, dtype=float))
        lmax = int(min(3 * nside - 1, getattr(ps, "b_ell_l_max", 8192)))
        sigma_rad = (sigma * ps.beam_unit).to("rad").value
        bwin = np.stack(
            [gaussian_beam_window(float(s), lmax) for s in sigma_rad], axis=0
        )
    if bwin.ndim == 1:
        bwin = np.broadcast_to(bwin, (n_ch, bwin.size))

    n_b = np.empty((n_pix, n_ch), dtype=float)
    cached = {}
    for ci in range(n_ch):
        key = tuple(np.round(bwin[ci], 12))
        if key not in cached:
            cached[key] = hp.smoothing(
                w_full,
                beam_window=np.asarray(bwin[ci], dtype=np.float64),
                iter=0,
                pol=False,
                use_weights=False,
            )
        n_b[:, ci] = cached[key][pid]
    n_b = np.maximum(n_b, 1.0e-12)
    mass = (1.0 / n_b).ravel()
    if mass.size == n_cell:
        return np.asarray(mass, dtype=float)
    if n_cell % mass.size == 0:
        return np.tile(mass, n_cell // mass.size)
    raise ValueError(
        f"beam_edge_cell_mass size {mass.size} incompatible with n_cell {n_cell}"
    )


def mean_beam_amplitude_on_cells(
    k_abs,
    khat,
    nhat,
    sigma_b,
    *,
    ell_max: int = 8,
    nmu: int = 32,
    n_mode_cap: int = 4000,
) -> NDArray[np.floating]:
    r"""
    Per-cell beam amplitude averaged over a set of output modes.

    :math:`B_b=\exp[-\tfrac12 k^2(1-\mu_b^2)\sigma_b^2]` averaged over
    the supplied :math:`(\mathbf k)` samples with the same real-
    :math:`Y_{LM}` expansion as :func:`mean_gaussian_beam_on_modes`.
    ``sigma_b`` is one Gaussian :math:`\sigma_\perp` per cell.
    """
    from .spherical import get_real_Ylm

    k_abs = np.asarray(k_abs, dtype=float).ravel()
    kx, ky, kz = (np.asarray(c, dtype=float).ravel() for c in khat)
    nhat = np.asarray(nhat, dtype=float).reshape(-1, 3)
    sigma_b = np.asarray(sigma_b, dtype=float).reshape(-1)
    if sigma_b.size != nhat.shape[0]:
        raise ValueError("sigma_b must match n_cell")
    if k_abs.size == 0:
        return np.ones(nhat.shape[0], dtype=float)
    if k_abs.size > int(n_mode_cap):
        rng = np.random.default_rng(0)
        take = rng.choice(k_abs.size, int(n_mode_cap), replace=False)
        k_abs, kx, ky, kz = k_abs[take], kx[take], ky[take], kz[take]
    sig_u, inv = np.unique(np.round(sigma_b, 12), return_inverse=True)
    ells = tuple(range(0, int(ell_max) + 1, 2))
    mass = np.zeros(nhat.shape[0], dtype=float)
    for ell in ells:
        pref = 4.0 * np.pi / (2 * ell + 1)
        f_sig = np.stack(
            [
                gaussian_beam_legendre_moments(k_abs * float(s), (ell,), nmu=int(nmu))[
                    0
                ]
                for s in sig_u
            ],
            axis=1,
        )
        for m in range(-ell, ell + 1):
            ylm = get_real_Ylm(ell, m)
            yk = np.broadcast_to(np.asarray(ylm(kx, ky, kz), dtype=float), k_abs.shape)
            mean_fY = np.mean(f_sig * yk[:, None], axis=0)
            yn = np.broadcast_to(
                np.asarray(ylm(nhat[:, 0], nhat[:, 1], nhat[:, 2]), dtype=float),
                nhat.shape[:1],
            )
            mass = mass + pref * yn * mean_fY[inv]
    return np.clip(mass, 0.0, 1.0)


def beam_cell_sigma_perp(ps):
    r"""
    Per-cell comoving beam width :math:`\sigma_{\perp,b}` and LOS.

    The channel is assigned by **true comoving distance**
    :math:`\chi_b=|\mathbf x_b-\mathbf x_{\mathrm{obs}}|`, not by the box
    :math:`z` coordinate: on a curved sky :math:`\sigma_\perp` is constant
    on observer-centred *spheres*, and a box :math:`z` slab of the 06
    footprint spans a factor :math:`\sim 2` in :math:`\chi`.

    Works for a chromatic beam as well — ``sigma_beam_ch_in_mpc`` already
    carries :math:`\chi(z_c)\,\sigma_\theta(\nu_c)` per channel, so
    :math:`\sigma_\theta\propto\lambda` needs no special case here.

    Returns ``(nhat, sigma_b)`` with shapes ``(n_cell, 3)``, ``(n_cell,)``.
    """
    from .util import freq_to_redshift

    geo = cell_sampling_geometry(ps)
    nhat = np.asarray(geo["nhat"], dtype=float)
    chi = np.asarray(geo["chi"], dtype=float)
    sig = np.atleast_1d(np.asarray(ps.sigma_beam_ch_in_mpc, dtype=float))
    z_lo = freq_to_redshift(ps.nu + 0.5 * ps.freq_resol)
    z_hi = freq_to_redshift(ps.nu - 0.5 * ps.freq_resol)
    chi_ch = 0.5 * (
        ps.astropy_cosmo_fiducial.comoving_distance(z_lo).value
        + ps.astropy_cosmo_fiducial.comoving_distance(z_hi).value
    )
    ic = np.argmin(np.abs(chi[:, None] - chi_ch[None, :]), axis=1)
    return nhat, sig[ic]


def cell_grid_los(ps):
    r"""
    Box-grid LOS :math:`\hat n` at the NGP voxel each map cell lands in.

    The estimator's :math:`Y_{\ell m}` leg is evaluated on the **grid**
    (``ps.los_xhat``), not at the map cell.  A quadrupole is a small
    residual of large cancellations, so for :math:`\ell>0` this sub-voxel
    distinction is not negligible and the beam-diagonal bookkeeping has to
    use the same :math:`\hat n` the cube kernel does.

    Returns ``(nhat_leg, inside)``: ``(n_cell, 3)`` and a bool mask of
    cells that actually deposit.
    """
    from .grid import particle_to_mesh_distance

    pix = np.asarray(ps.pix_coor_in_box, dtype=float).reshape(-1, 3)
    ndim = np.asarray(ps.box_ndim, dtype=int)
    s, idx = particle_to_mesh_distance(pix, np.asarray(ps.box_len, dtype=float), ndim)
    idx = np.asarray(idx, dtype=np.int64).T
    inside = np.all((idx >= 0) & (idx < ndim[None, :]), axis=1) & np.all(
        np.abs(np.asarray(s, dtype=float)) <= 0.5, axis=1
    )
    idx_c = np.clip(idx, 0, ndim[None, :] - 1)
    xhat = ps.los_xhat
    nhat = np.stack(
        [
            np.broadcast_to(np.asarray(c, dtype=float), tuple(ndim))[
                idx_c[:, 0], idx_c[:, 1], idx_c[:, 2]
            ]
            for c in xhat
        ],
        axis=1,
    )
    return nhat, inside


def beam_mode_group_index(ps, *, n_mu: int = 1):
    r"""
    Group estimator modes by output :math:`|k|` bin and by :math:`|\mu|`.

    :math:`\tilde B_b(\mathbf k)` depends on :math:`\hat k` through
    :math:`\mu_b=\hat k\cdot\hat n_b`, so a pure :math:`|k|`-shell average
    of the cell amplitudes throws away the beam quadrupole.  Splitting
    each shell into ``n_mu`` groups of :math:`|\mu|`
    (:math:`\mu=\hat k\cdot\hat n_{\mathrm{ref}}`, equal-count quantiles)
    restores it at the cost of ``n_mu`` times as many window kernels.

    ``n_mu = 1`` reproduces the plain per-bin grouping.

    Returns ``(index, n_group)`` with ``index`` an int array on the rFFT
    grid (``-1`` for modes outside the estimator bins).
    """
    k_mode = np.asarray(ps.k_mode, dtype=float)
    shape = k_mode.shape
    k_flat = k_mode.ravel()
    k1dbins = np.asarray(ps.k1dbins, dtype=float)
    n_out = int(k1dbins.size - 1)
    k1dweights = (
        np.ones_like(k_flat)
        if getattr(ps, "k1dweights", None) is None
        else np.asarray(ps.k1dweights, dtype=float).ravel()
    )
    bin_idx = np.digitize(k_flat, k1dbins) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_out) & (k1dweights > 0) & (k_flat > 0)

    n_mu_i = max(1, int(n_mu))
    index = np.full(k_flat.shape, -1, dtype=np.int64)
    if n_mu_i == 1:
        index[valid] = bin_idx[valid]
        return index.reshape(shape), n_out

    from .spherical import unit_khat_from_k_vec

    khat = unit_khat_from_k_vec(ps.k_vec)
    nref = np.array(
        [float(np.mean(np.asarray(c, dtype=float))) for c in ps.los_xhat], dtype=float
    )
    nrm = float(np.linalg.norm(nref))
    nref = nref / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0])
    mu_abs = np.abs(
        sum(np.asarray(khat[i], dtype=float).ravel() * nref[i] for i in range(3))
    )
    for i in range(n_out):
        sel = valid & (bin_idx == i)
        if not np.any(sel):
            continue
        m = mu_abs[sel]
        # equal-count |mu| quantiles so every kernel has comparable weight
        edges = np.quantile(m, np.linspace(0.0, 1.0, n_mu_i + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        sub = np.clip(np.digitize(m, edges[1:-1]), 0, n_mu_i - 1)
        index[sel] = i * n_mu_i + sub
    return index.reshape(shape), n_out * n_mu_i


def exact_beam_legs(
    k_abs,
    khat,
    nhat,
    sigma_b,
    cell_mass,
    *,
    ells: Sequence[int] = (0, 2, 4),
    l_max_beam: int | None = None,
    nmu: int = 64,
    norm: float = 1.0,
    nhat_leg=None,
):
    r"""
    Exact zero-lag Yamamoto legs of a beamed selection.

    .. math::

        S^{\ell m}(\mathbf k)
        =\frac{1}{N}\sum_b c_b\,\tilde B_b(\mathbf k)\,Y_{\ell m}(\hat n_b)
        =\sum_{LM}\frac{4\pi}{2L+1}Y_{LM}(\hat k)\,M^{\ell m}_{LM}(k),

    with :math:`M^{\ell m}_{LM}(k)=\frac1N\sum_b c_b f_L(k\sigma_b)
    Y_{LM}(\hat n_b)Y_{\ell m}(\hat n_b)` and :math:`f_L` the Legendre
    moments of the Gaussian (:func:`gaussian_beam_legendre_moments`).
    Costs cell moments only — no FFTs, and no loop over modes.

    ``nhat`` is the LOS the **beam** smears about (the map cell); pass
    ``nhat_leg`` when the estimator's :math:`Y_{\ell m}` leg lives on a
    different LOS (the box grid — see :func:`cell_grid_los`).

    ``norm`` is :math:`N`.  Returns ``{None: S^0} | {(ell, m): S^{lm}}``,
    each shaped like ``k_abs``.
    """
    from .spherical import get_real_Ylm

    k_abs = np.asarray(k_abs, dtype=float)
    shape = k_abs.shape
    k_flat = k_abs.ravel()
    kx, ky, kz = (
        np.broadcast_to(np.asarray(c, dtype=float), shape).ravel() for c in khat
    )
    nhat = np.asarray(nhat, dtype=float).reshape(-1, 3)
    sigma_b = np.asarray(sigma_b, dtype=float).reshape(-1)
    cell_mass = np.asarray(cell_mass, dtype=float).reshape(-1)
    n_cell = nhat.shape[0]
    if sigma_b.size != n_cell or cell_mass.size != n_cell:
        raise ValueError("sigma_b and cell_mass must match n_cell")

    ells_t = tuple(int(e) for e in ells)
    if l_max_beam is None:
        # the ell leg couples to L <= ell, but per mode the series needs a
        # couple of extra orders to converge (see the unit test)
        l_max_beam = max(ells_t) + 4
    ells_L = tuple(range(0, int(l_max_beam) + 1, 2))

    n_leg = nhat if nhat_leg is None else np.asarray(nhat_leg, dtype=float)
    if n_leg.shape != (n_cell, 3):
        raise ValueError("nhat_leg must match nhat")
    y_cell = {
        (e, m): np.broadcast_to(
            np.asarray(
                get_real_Ylm(e, m)(n_leg[:, 0], n_leg[:, 1], n_leg[:, 2]), dtype=float
            ),
            (n_cell,),
        )
        for e in ells_t
        for m in range(-e, e + 1)
    }
    keys = [None] + list(y_cell)

    sig_u, i_sig = np.unique(np.round(sigma_b, 12), return_inverse=True)
    k_u, i_k = np.unique(np.round(k_flat, 10), return_inverse=True)
    out = {key: np.zeros(k_flat.size, dtype=float) for key in keys}
    for L in ells_L:
        pref = 4.0 * np.pi / (2 * L + 1)
        f_tab = gaussian_beam_legendre_moments(
            k_u[:, None] * sig_u[None, :], (L,), nmu=int(nmu)
        )[0]
        for M in range(-L, L + 1):
            ylm = get_real_Ylm(L, M)
            y_cell_LM = np.broadcast_to(
                np.asarray(ylm(nhat[:, 0], nhat[:, 1], nhat[:, 2]), dtype=float),
                (n_cell,),
            )
            y_k = np.broadcast_to(
                np.asarray(ylm(kx, ky, kz), dtype=float), k_flat.shape
            )
            for key in keys:
                fac = cell_mass * y_cell_LM
                if key is not None:
                    fac = fac * y_cell[key]
                mom = np.bincount(i_sig, weights=fac, minlength=sig_u.size) / float(
                    norm
                )
                out[key] += pref * y_k * (f_tab @ mom)[i_k]
    return {key: val.reshape(shape) for key, val in out.items()}


def beam_diagonal_correction(
    ps,
    *,
    ells: Sequence[int] = (0, 2, 4),
    masses=None,
    group_index=None,
    l_max_beam: int | None = None,
    nmu: int = 64,
    cell_mass=None,
):
    r"""
    Exact zero-lag (diagonal) beam response minus the mean-field one.

    The Yamamoto :math:`\ell` leg is
    :math:`S^{\ell m}(\mathbf k)=\langle u_{\mathbf k}Y_{\ell m}(\hat n)\rangle`
    with :math:`u_{\mathbf k}=w\,\tilde B(\mathbf k)`, and the kernel's
    :math:`\boldsymbol\kappa=0` value is :math:`S^{\ell m}S^{0}`.  Via the
    addition theorem

    .. math::

        S^{\ell m}(\mathbf k)=\sum_{LM}\frac{4\pi}{2L+1}Y_{LM}(\hat k)\,
        M^{\ell m}_{LM}(k),\quad
        M^{\ell m}_{LM}(k)=\bigl\langle w f_L(k\sigma)
        Y_{LM}(\hat n)Y_{\ell m}(\hat n)\bigr\rangle,

    so it costs cell moments, no FFTs.

    A cell mass that does not depend on :math:`\hat k` (any B1/B2/B3
    mean field) only ever supplies the :math:`L=0` term.  That is exact
    for :math:`\ell=0` and badly wrong above it — the estimator's
    :math:`Y_{\ell m}(\hat n)` leg couples to the beam's own :math:`L`
    structure, and :math:`\ell` needs :math:`L\le\ell`.  Measured on the
    06 lightcone, the mean field
    delivers only \(0.27\)–\(0.40\) of the exact \(\ell=2\) diagonal at
    high \(k\).

    Returns ``{(ell, m): dS}`` with ``dS`` on the rFFT grid, to be
    **added** to the kernel's :math:`\boldsymbol\kappa=0` term, i.e. to
    ``FFT[w Y_lm] FFT[w]^*`` before the :math:`Y_{\ell m}(\hat k)`
    contraction.  It is a difference of two cell-space expressions, so
    the cell-vs-grid convention of :math:`\hat n` cancels to first order
    and there is no division by a possibly vanishing mean field.
    """
    from .spherical import get_real_Ylm, unit_khat_from_k_vec

    ells_t = tuple(int(e) for e in ells)
    nhat, sigma_b = beam_cell_sigma_perp(ps)
    n_cell = nhat.shape[0]
    e_b = (
        beam_edge_cell_mass(ps)
        if cell_mass is None
        else np.asarray(cell_mass, dtype=float)
    )
    if e_b.shape != (n_cell,):
        raise ValueError(f"cell_mass shape {e_b.shape} != n_cell {n_cell}")
    if masses is None:
        masses, group_index = beam_kernel_bin_masses(ps, group_index=group_index)
    masses = np.asarray(masses, dtype=float)
    g_flat = np.asarray(group_index, dtype=np.int64).ravel()

    n_grid = float(np.prod(np.asarray(ps.box_ndim, dtype=int)))
    k_mode = np.asarray(ps.k_mode, dtype=float)
    shape = k_mode.shape
    # the estimator's Y_lm leg is evaluated on the box grid, the beam
    # smears about the map cell: both legs must use the grid LOS so the
    # mean-field subtraction matches the cube kernel exactly
    nhat_leg, inside = cell_grid_los(ps)
    e_b = e_b * inside
    s_ex = exact_beam_legs(
        k_mode,
        unit_khat_from_k_vec(ps.k_vec),
        nhat,
        sigma_b,
        e_b,
        ells=ells_t,
        l_max_beam=l_max_beam,
        nmu=int(nmu),
        norm=n_grid,
        nhat_leg=nhat_leg,
    )

    # mean field: the k-hat independent cell mass the cube kernel deposits
    keys = [k for k in s_ex if k is not None]
    y_cell = {
        key: np.broadcast_to(
            np.asarray(
                get_real_Ylm(*key)(nhat_leg[:, 0], nhat_leg[:, 1], nhat_leg[:, 2]),
                dtype=float,
            ),
            (n_cell,),
        )
        for key in keys
    }
    s_mf = {key: np.zeros(k_mode.size, dtype=float) for key in [None] + keys}
    for g in range(int(masses.shape[0])):
        sel = g_flat == g
        if not np.any(sel):
            continue
        w_g = e_b * masses[g]
        s_mf[None][sel] = float(np.sum(w_g)) / n_grid
        for key in keys:
            s_mf[key][sel] = float(np.sum(w_g * y_cell[key])) / n_grid

    valid = (g_flat >= 0).astype(float).reshape(shape)
    s0_mf = s_mf[None].reshape(shape)
    return {
        key: (s_ex[key] * s_ex[None] - s_mf[key].reshape(shape) * s0_mf) * valid
        for key in keys
    }


def beam_kernel_bin_masses(
    ps, *, n_mu: int = 1, ell_max: int = 8, nmu: int = 32, group_index=None
):
    r"""
    Per-group, per-cell beam amplitudes for the B3 kernel.

    Row :math:`g` is :math:`\langle\tilde B_b(\mathbf k)\rangle` averaged
    over the estimator modes of group :math:`g`
    (:func:`beam_mode_group_index`).  With ``n_mu > 1`` the groups are
    :math:`(|k|\text{ bin}, |\mu|)` cells, so the mass cube carries the
    beam's :math:`\hat k` anisotropy instead of only its shell mean.

    Pass ``masses[g] * beam_edge_cell_mass(ps)`` as ``particle_mass`` of
    :func:`~meer21cm.window.ngp_raw_cell_comb` for that group, or set
    ``beam_in_kernel=True`` on
    :func:`~meer21cm.window.build_mesh_window_mas_out`.

    Returns ``(masses, group_index)`` with ``masses`` of shape
    ``(n_group, n_cell)``.  Unit masses if ``sigma_beam_ch`` is ``None``.
    """
    from .spherical import unit_khat_from_k_vec

    n_cell = int(np.asarray(ps.pix_coor_in_box).reshape(-1, 3).shape[0])
    if group_index is None:
        group_index, n_group = beam_mode_group_index(ps, n_mu=n_mu)
    else:
        group_index = np.asarray(group_index, dtype=np.int64)
        n_group = int(group_index.max()) + 1
    if getattr(ps, "sigma_beam_ch", None) is None:
        return np.ones((n_group, n_cell), dtype=float), group_index

    nhat, sigma_b = beam_cell_sigma_perp(ps)
    if nhat.shape[0] != n_cell:
        raise ValueError(f"cell geometry n_cell {nhat.shape[0]} != pix_coor {n_cell}")

    k_mode = np.asarray(ps.k_mode, dtype=float).ravel()
    khat = unit_khat_from_k_vec(ps.k_vec)
    khat_r = tuple(np.asarray(c, dtype=float).ravel() for c in khat)
    g_flat = group_index.ravel()
    masses = np.ones((n_group, n_cell), dtype=float)
    for g in range(n_group):
        sel = g_flat == g
        if not np.any(sel):
            continue
        masses[g] = mean_beam_amplitude_on_cells(
            k_mode[sel],
            tuple(c[sel] for c in khat_r),
            nhat,
            sigma_b,
            ell_max=int(ell_max),
            nmu=int(nmu),
        )
    return masses, group_index


def _nref_from_los(ps):
    """Mean line-of-sight unit vector from the estimator ``los_xhat`` grid."""
    nref = np.array(
        [float(np.mean(np.asarray(c, dtype=float))) for c in ps.los_xhat], dtype=float
    )
    nrm = float(np.linalg.norm(nref))
    return nref / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0])


def _phi_around_nref(khat, nref):
    """Azimuth of :math:`\\hat q` around :math:`\\hat n_{\\mathrm{ref}}`."""
    tmp = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(nref[0])) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    e1 = tmp - nref * float(tmp @ nref)
    e1 = e1 / float(np.linalg.norm(e1))
    e2 = np.cross(nref, e1)
    q1 = sum(np.asarray(khat[i], dtype=float).ravel() * e1[i] for i in range(3))
    q2 = sum(np.asarray(khat[i], dtype=float).ravel() * e2[i] for i in range(3))
    return np.arctan2(q2, q1)


def beam_input_mode_groups(ps, *, n_mu: int = 4, n_phi: int = 1):
    r"""
    Group the **theory** modes :math:`\mathbf q` by :math:`|\mu|` and
    optionally azimuth :math:`\phi` around :math:`\hat n_{\mathrm{ref}}`.

    The beam multiplies the field *before* the selection, so in the mesh
    window it belongs at the inner (theory) mode :math:`\mathbf q`, not at
    the output mode.  A real-space selection cube can only hold one
    :math:`\mathbf q`, so the theory shell is split into ``n_mu``
    equal-count :math:`|\mu|` groups
    (:math:`\mu=\hat q\cdot\hat n_{\mathrm{ref}}`) and each group gets its
    own beamed cube.  ``n_mu = 1`` keeps only the shell mean.
    ``n_phi > 1`` further splits each :math:`|\mu|` group into equal-count
    azimuth bins (index ``i_mu * n_phi + i_phi``).  ``n_phi = 1`` is
    identical to the :math:`|\mu|`-only map.

    The groups partition **all** theory modes (nothing is dropped), so
    summing them reproduces the ungrouped matrix.

    Returns ``(index, n_group)``; ``index`` is an int array on the rFFT
    grid.
    """
    from .spherical import unit_khat_from_k_vec

    shape = np.asarray(ps.k_mode, dtype=float).shape
    n_mu_i = max(1, int(n_mu))
    n_phi_i = max(1, int(n_phi))
    if n_mu_i == 1 and n_phi_i == 1:
        return np.zeros(shape, dtype=np.int64), 1

    khat = unit_khat_from_k_vec(ps.k_vec)
    nref = _nref_from_los(ps)
    if n_mu_i == 1:
        i_mu = np.zeros(int(np.prod(shape)), dtype=np.int64)
    else:
        mu_abs = np.abs(
            sum(np.asarray(khat[i], dtype=float).ravel() * nref[i] for i in range(3))
        )
        edges = np.quantile(mu_abs, np.linspace(0.0, 1.0, n_mu_i + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        i_mu = np.clip(np.digitize(mu_abs, edges[1:-1]), 0, n_mu_i - 1)
    if n_phi_i == 1:
        return i_mu.reshape(shape).astype(np.int64), n_mu_i

    phi = _phi_around_nref(khat, nref)
    pedges = np.quantile(phi, np.linspace(0.0, 1.0, n_phi_i + 1))
    pedges[0], pedges[-1] = -np.inf, np.inf
    i_phi = np.clip(np.digitize(phi, pedges[1:-1]), 0, n_phi_i - 1)
    index = i_mu * n_phi_i + i_phi
    return index.reshape(shape).astype(np.int64), n_mu_i * n_phi_i


def beam_input_cell_masses(
    ps,
    k_in,
    *,
    n_mu: int = 4,
    n_phi: int = 1,
    mode_scale=None,
    cell_mass=None,
):
    r"""
    Per-cell beam masses of the input-mode kernel (``beam_at_input``).

    Shared by :func:`beam_input_cell_kernels` (which deposits them) and
    :func:`beam_input_diagonal_correction` (which needs the same masses
    to subtract the model's own :math:`\kappa=0` term).  See
    :func:`beam_input_cell_kernels` for the derivation.

    Returns ``(index, shell_edges, mass_fn)``; ``mass_fn(j, g)`` gives the
    length-``n_cell`` mass for column ``j`` and group ``g`` (``None`` if
    that shell/group intersection is empty).
    """
    k_in_np = np.asarray(k_in, dtype=float)
    index, _n_group = beam_input_mode_groups(ps, n_mu=n_mu, n_phi=n_phi)
    g_flat = index.ravel()

    if cell_mass is None:
        cell_mass = beam_edge_cell_mass(ps)
    mass0 = np.asarray(cell_mass, dtype=float)
    edges = np.concatenate(([0.0], 0.5 * (k_in_np[:-1] + k_in_np[1:]), [np.inf]))

    if getattr(ps, "sigma_beam_ch", None) is None:

        def mass_fn(j, g):
            return mass0

        return index, edges, mass_fn

    nhat, sigma_b = beam_cell_sigma_perp(ps)
    k_mode = np.asarray(ps.k_mode, dtype=float).ravel()
    q_vec = np.stack(
        [
            np.broadcast_to(np.asarray(c, dtype=float), ps.k_mode.shape).ravel()
            for c in np.meshgrid(*ps.k_vec, indexing="ij")
        ],
        axis=1,
    )
    ms = (
        np.ones_like(k_mode)
        if mode_scale is None
        else np.asarray(mode_scale, dtype=float).ravel()
    )
    half = 0.5 * sigma_b**2

    def mass_fn(j, g):
        sel = (k_mode >= edges[j]) & (k_mode < edges[j + 1]) & (g_flat == g)
        if not np.any(sel):
            return None
        wq = ms[sel]
        tot = float(np.sum(wq))
        if not np.isfinite(tot) or tot <= 0:
            wq = np.ones_like(wq)
            tot = float(wq.size)
        qs = q_vec[sel]
        mmat = (qs.T * wq) @ qs / tot
        u_b = np.trace(mmat) - np.einsum("ci,ij,cj->c", nhat, mmat, nhat)
        return mass0 * np.exp(-np.clip(u_b, 0.0, None) * half)

    return index, edges, mass_fn


def beam_input_cell_kernels(
    ps,
    k_in,
    *,
    n_mu: int = 4,
    n_phi: int = 1,
    mode_scale=None,
    cell_mass=None,
):
    r"""
    Input-mode beam kernels for :func:`~meer21cm.window.build_mesh_window_matrix`.

    The observed map cell :math:`b` sees the field smoothed by its own
    beam, so the estimator leg is

    .. math::

        G_{\ell m}(\mathbf k,\mathbf q)
        = \sum_b w_b\,Y_{\ell m}(\hat n_b)\,
          \tilde B_b(\mathbf q)\,e^{i(\mathbf q-\mathbf k)\cdot\mathbf x_b},
        \qquad
        \tilde B_b(\mathbf q)=e^{-q_{\perp,b}^2\sigma_b^2/2},

    with :math:`q_{\perp,b}^2 = q^2-(\mathbf q\cdot\hat n_b)^2` — the beam
    argument uses the **cell's own** LOS :math:`\hat n_b` and comoving
    width :math:`\sigma_b`, so curved sky and chromaticity are both exact
    per cell.  Both legs carry :math:`\tilde B_b(\mathbf q)`, so the
    beamed selection cube is just the NGP deposit of
    :math:`m_b\tilde B_b(\mathbf q)` and the ordinary
    :math:`\kappa=\mathbf k-\mathbf q` convolution structure survives.

    Because the cube must be a single real field, :math:`\mathbf q` is
    replaced by its group mean *inside the exponent*.  That mean is exact
    (not a representative direction): with
    :math:`M=\langle\mathbf q\mathbf q^{\mathsf T}\rangle` over the modes
    of shell :math:`j` and group :math:`g`,

    .. math::

        \langle q_{\perp,b}^2\rangle = \mathrm{tr}\,M
        - \hat n_b^{\mathsf T} M\,\hat n_b ,

    so the azimuthal spread of :math:`\hat q` about
    :math:`\hat n_{\mathrm{ref}}` is captured at second order for free.
    Only the residual curvature
    :math:`\langle e^{-u}\rangle\ne e^{-\langle u\rangle}` within a group
    is dropped; it converges away with ``n_mu``.

    What a :math:`|\mu|` grouping can **not** do is carry the beam's
    azimuthal structure.  Each cell has its own :math:`\hat n_b`, so
    :math:`\tilde B_b(\mathbf q)` is not symmetric about
    :math:`\hat n_{\mathrm{ref}}` and two modes with the same
    :math:`|\mu|` but different azimuth see different beams.  One cube per
    group averages that away, and the loss does **not** shrink with
    ``n_mu``.  The per-mode diagonal of
    :func:`beam_input_diagonal_correction` restores that azimuthal
    structure; the cube then only has to carry the leakage.

    Parameters
    ----------
    k_in :
        Theory :math:`|k|` nodes — the matrix columns.  Voronoi shells
        (the same edges :func:`~meer21cm.window.build_mesh_window_matrix`
        uses) set the :math:`|q|` membership.
    n_mu :
        Number of :math:`|\mu|` groups (:func:`beam_input_mode_groups`).
    mode_scale :
        Optional same-:math:`q` transfer used to weight the group mean of
        :math:`M` (should be the ``mode_scale`` passed to the matrix).
    cell_mass :
        Per-cell mass multiplying :math:`\tilde B_b` (default
        :func:`beam_edge_cell_mass`).

    Returns
    -------
    (index, kernel_fn) :
        ``index`` for ``in_group_index``; ``kernel_fn(j, g)`` returns the
        real-space cube for column ``j`` and group ``g`` (``None`` when
        that shell/group intersection is empty).
    """
    from .window import ngp_raw_cell_comb

    index, _edges, mass_fn = beam_input_cell_masses(
        ps, k_in, n_mu=n_mu, n_phi=n_phi, mode_scale=mode_scale, cell_mass=cell_mass
    )

    def kernel_fn(j, g):
        mass = mass_fn(j, g)
        if mass is None:
            return None
        return ngp_raw_cell_comb(ps, particle_mass=mass)

    return index, kernel_fn


def beam_input_diagonal_correction(
    ps,
    k_in,
    *,
    ells: Sequence[int] = (0, 2, 4),
    n_mu: int = 4,
    n_phi: int = 1,
    mode_scale=None,
    cell_mass=None,
    l_max_beam: int | None = None,
    nmu: int = 64,
    ratio: bool = False,
    clip: float = 8.0,
):
    r"""
    Exact per-mode beam diagonal minus the input-mode group cube's own diagonal.

    The mesh kernel's :math:`\boldsymbol\kappa=0` (i.e.
    :math:`\mathbf q=\mathbf k`) term is
    :math:`S^{\ell m}(\mathbf k)S^{0}(\mathbf k)` with
    :math:`S^{\ell m}=\langle w\tilde B(\mathbf k)Y_{\ell m}(\hat n)\rangle`.
    A real-space cube can only hold one :math:`\hat k`, so
    :func:`beam_input_cell_kernels` deposits the group mean and loses the
    beam's azimuthal structure.  Here both sides are evaluated in cell
    space — exact via :func:`exact_beam_legs` (a :math:`Y_{LM}` addition
    theorem, no FFTs), group mean via the *same* masses the cube uses —
    and the difference is returned to be **added** to the kernel's
    :math:`\boldsymbol\kappa=0` term.

    Being a difference of two cell-space expressions there is no division
    by a possibly vanishing mean field, and the group cube keeps carrying
    the leakage.

    Parameters
    ----------
    ratio :
        Return the exact/model **ratio** instead of the difference, to be
        multiplied onto the whole :math:`\boldsymbol\kappa` profile rather
        than added at :math:`\boldsymbol\kappa=0`.  The extra assumption
        is that the beam's directional response varies slowly across the
        width of the window kernel — true when the survey is large
        compared with :math:`1/k`, and it is what lets the correction
        reach the leakage as well as the diagonal.  Still exact at
        :math:`\boldsymbol\kappa=0`.
    clip :
        Ratios are clipped to :math:`\pm` this, and set to 1 where the
        model leg is below :math:`10^{-3}` of its peak.

    Returns
    -------
    dict
        ``{(ell, m): dS}`` on the rFFT grid (a difference, or a ratio if
        ``ratio``).
    """
    from .spherical import get_real_Ylm, unit_khat_from_k_vec

    ells_t = tuple(int(e) for e in ells)
    index, edges, mass_fn = beam_input_cell_masses(
        ps, k_in, n_mu=n_mu, n_phi=n_phi, mode_scale=mode_scale, cell_mass=cell_mass
    )
    g_flat = np.asarray(index, dtype=np.int64).ravel()
    n_group = int(g_flat.max()) + 1

    nhat, sigma_b = beam_cell_sigma_perp(ps)
    nhat_leg, inside = cell_grid_los(ps)
    e_b = (
        beam_edge_cell_mass(ps)
        if cell_mass is None
        else np.asarray(cell_mass, dtype=float)
    ) * inside

    n_grid = float(np.prod(np.asarray(ps.box_ndim, dtype=int)))
    k_mode = np.asarray(ps.k_mode, dtype=float)
    shape = k_mode.shape
    s_ex = exact_beam_legs(
        k_mode,
        unit_khat_from_k_vec(ps.k_vec),
        nhat,
        sigma_b,
        e_b,
        ells=ells_t,
        l_max_beam=l_max_beam,
        nmu=int(nmu),
        norm=n_grid,
        nhat_leg=nhat_leg,
    )

    keys = [key for key in s_ex if key is not None]
    y_cell = {
        key: np.broadcast_to(
            np.asarray(
                get_real_Ylm(*key)(nhat_leg[:, 0], nhat_leg[:, 1], nhat_leg[:, 2]),
                dtype=float,
            ),
            e_b.shape,
        )
        for key in keys
    }

    k_flat = k_mode.ravel()
    shell_of = np.clip(np.digitize(k_flat, edges[1:-1]), 0, len(edges) - 2)
    s_mf = {key: np.zeros(k_flat.size, dtype=float) for key in [None] + keys}
    for j in range(len(edges) - 1):
        for g in range(n_group):
            sel = (shell_of == j) & (g_flat == g)
            if not np.any(sel):
                continue
            mass = mass_fn(j, g)
            if mass is None:
                continue
            # mass_fn already carries cell_mass; only the deposit mask is left
            w_g = np.asarray(mass, dtype=float) * inside
            s_mf[None][sel] = float(np.sum(w_g)) / n_grid
            for key in keys:
                s_mf[key][sel] = float(np.sum(w_g * y_cell[key])) / n_grid

    s0_mf = s_mf[None].reshape(shape)
    if not ratio:
        return {
            key: s_ex[key] * s_ex[None] - s_mf[key].reshape(shape) * s0_mf
            for key in keys
        }

    out = {}
    for key in keys:
        num = s_ex[key] * s_ex[None]
        den = s_mf[key].reshape(shape) * s0_mf
        floor = 1e-3 * float(np.max(np.abs(den)))
        r = np.ones_like(num)
        ok = np.abs(den) > floor
        r[ok] = num[ok] / den[ok]
        out[key] = np.clip(r, -clip, clip)
    return out


def beam_ylm_labels(l_max: int = 2):
    """Even beam multipoles :math:`(L,M)` up to ``l_max`` (real :math:`Y_{LM}`)."""
    l_max_i = int(l_max)
    return [(L, M) for L in range(0, l_max_i + 1, 2) for M in range(-L, L + 1)]


def beam_ylm_alpha(ps, labels):
    r"""
    Theory weights :math:`\alpha_{LM}(\hat q)=(4\pi/(2L+1))Y_{LM}(\hat q)`.

    The Gaussian beam addition theorem is
    :math:`\tilde B_b(\mathbf q)=\sum_{LM}\alpha_{LM}(\hat q)\,
    f_L(|q|\sigma_b)\,Y_{LM}(\hat n_b)`.  A diagonal :math:`Y_{LM}`
    cube uses :math:`T(q)\propto\alpha_{LM}(\hat q)^2`.
    """
    from .spherical import get_real_Ylm, unit_khat_from_k_vec

    khat = unit_khat_from_k_vec(ps.k_vec)
    shape = np.broadcast_shapes(*(np.asarray(c).shape for c in khat))
    out = []
    for L, M in labels:
        y = np.broadcast_to(
            np.asarray(get_real_Ylm(int(L), int(M))(*khat), dtype=float), shape
        )
        out.append((4.0 * np.pi / (2 * int(L) + 1)) * np.asarray(y, dtype=float))
    return np.stack(out, axis=0)


def beam_ylm_cell_masses(
    ps,
    k_in,
    *,
    l_max: int = 2,
    cell_mass=None,
    nmu: int = 64,
):
    r"""
    Per-cell masses of a diagonal :math:`Y_{LM}` beam cube.

    :math:`c_{LM,j}(b)=m_b\,f_L(k_{\mathrm{in}}[j]\,\sigma_b)\,
    Y_{LM}(\hat n_b)`.  ``f_L`` is frozen at the theory node (the same
    Voronoi-shell convention as :func:`beam_input_cell_kernels`).
    """
    from .spherical import get_real_Ylm

    labels = beam_ylm_labels(l_max)
    if cell_mass is None:
        cell_mass = beam_edge_cell_mass(ps)
    mass0 = np.asarray(cell_mass, dtype=float)
    k_in_np = np.asarray(k_in, dtype=float)
    if getattr(ps, "sigma_beam_ch", None) is None:

        def mass_fn(j, g):
            L, M = labels[g]
            if L != 0:
                return np.zeros_like(mass0)
            y00 = float(get_real_Ylm(0, 0)(1.0, 0.0, 0.0))
            return mass0 * y00

        return labels, mass_fn

    nhat, sigma_b = beam_cell_sigma_perp(ps)
    y_cell = {
        (L, M): np.asarray(
            get_real_Ylm(int(L), int(M))(nhat[:, 0], nhat[:, 1], nhat[:, 2]),
            dtype=float,
        )
        for L, M in labels
    }

    def mass_fn(j, g):
        L, M = labels[g]
        f_L = gaussian_beam_legendre_moments(
            float(k_in_np[j]) * sigma_b, (L,), nmu=int(nmu)
        )[0]
        return mass0 * f_L * y_cell[(L, M)]

    return labels, mass_fn


def beam_ylm_cell_kernels(
    ps,
    k_in,
    *,
    l_max: int = 2,
    cell_mass=None,
    nmu: int = 64,
):
    r"""
    Diagonal :math:`Y_{LM}` cubes for :func:`~meer21cm.window.build_mesh_window_matrix`.

    Pair with ``in_group_scale[g] = \alpha_{LM}(\hat q)^2``
    (:func:`beam_ylm_alpha`).  Each theory column is a weighted sum over
    all shell modes, not a partition — this is why the cubes cannot go
    through ``in_group_index``.
    """
    from .window import ngp_raw_cell_comb

    labels, mass_fn = beam_ylm_cell_masses(
        ps, k_in, l_max=l_max, cell_mass=cell_mass, nmu=nmu
    )

    def kernel_fn(j, g):
        return ngp_raw_cell_comb(ps, particle_mass=mass_fn(j, g))

    return labels, kernel_fn


def beam_ylm_diagonal_correction(
    ps,
    k_in,
    *,
    ells: Sequence[int] = (0, 2, 4),
    l_max_cube: int = 2,
    l_max_beam: int | None = None,
    cell_mass=None,
    nmu: int = 64,
):
    r"""
    Exact per-mode beam diagonal minus the diagonal-:math:`Y_{LM}` cube's own.

    The cube fill at :math:`\boldsymbol\kappa=0` is
    :math:`\sum_{LM}
    \langle c_{LM} Y_{\ell m}\rangle\langle c_{LM}\rangle
    \alpha_{LM}(\hat k)^2`
    (no :math:`LM\neq L'M'` cross terms).  The additive term restores
    :func:`exact_beam_legs` (default :math:`L\le\max\ell+4`).
    """
    from .spherical import get_real_Ylm, unit_khat_from_k_vec

    ells_t = tuple(int(e) for e in ells)
    labels, mass_fn = beam_ylm_cell_masses(
        ps, k_in, l_max=l_max_cube, cell_mass=cell_mass, nmu=nmu
    )
    alpha = beam_ylm_alpha(ps, labels)

    nhat, sigma_b = beam_cell_sigma_perp(ps)
    nhat_leg, inside = cell_grid_los(ps)
    e_b = (
        beam_edge_cell_mass(ps)
        if cell_mass is None
        else np.asarray(cell_mass, dtype=float)
    ) * inside

    n_grid = float(np.prod(np.asarray(ps.box_ndim, dtype=int)))
    k_mode = np.asarray(ps.k_mode, dtype=float)
    shape = k_mode.shape
    s_ex = exact_beam_legs(
        k_mode,
        unit_khat_from_k_vec(ps.k_vec),
        nhat,
        sigma_b,
        e_b,
        ells=ells_t,
        l_max_beam=l_max_beam,
        nmu=int(nmu),
        norm=n_grid,
        nhat_leg=nhat_leg,
    )
    keys = [key for key in s_ex if key is not None]
    y_cell = {
        key: np.broadcast_to(
            np.asarray(
                get_real_Ylm(*key)(nhat_leg[:, 0], nhat_leg[:, 1], nhat_leg[:, 2]),
                dtype=float,
            ),
            e_b.shape,
        )
        for key in keys
    }

    k_in_np = np.asarray(k_in, dtype=float)
    edges = np.concatenate(([0.0], 0.5 * (k_in_np[:-1] + k_in_np[1:]), [np.inf]))
    k_flat = k_mode.ravel()
    shell_of = np.clip(np.digitize(k_flat, edges[1:-1]), 0, len(edges) - 2)
    prod = {key: np.zeros(k_flat.size, dtype=float) for key in keys}
    for j in range(len(k_in_np)):
        sel = shell_of == j
        if not np.any(sel):
            continue
        for g in range(len(labels)):
            mass = np.asarray(mass_fn(j, g), dtype=float) * inside
            s0 = float(np.sum(mass)) / n_grid
            a2 = (np.asarray(alpha[g], dtype=float).ravel() ** 2)[sel]
            for key in keys:
                slm = float(np.sum(mass * y_cell[key])) / n_grid
                prod[key][sel] += slm * s0 * a2
    return {key: s_ex[key] * s_ex[None] - prod[key].reshape(shape) for key in keys}


def cell_sampling_geometry(ps):
    r"""
    Per-cell line-of-sight and comoving top-hat widths.

    ``pix_coor_in_cartesian`` is assigned to the nearest frequency
    channel via comoving distance.  Returns ``nhat`` ``(n_cell, 3)``,
    ``dperp`` and ``dpar`` ``(n_cell,)``.
    """
    from .util import freq_to_redshift

    if getattr(ps, "_pix_coor_in_cartesian", None) is not None:
        pos = np.asarray(ps.pix_coor_in_cartesian, dtype=float).reshape(-1, 3)
    else:
        origin = np.asarray(getattr(ps, "box_origin", np.zeros(3)), dtype=float)
        pos = np.asarray(ps.pix_coor_in_box, dtype=float).reshape(
            -1, 3
        ) + origin.reshape(1, 3)
    chi = np.linalg.norm(pos, axis=1)
    nhat = np.zeros_like(pos)
    ok = chi > 0
    nhat[ok] = pos[ok] / chi[ok, None]

    cosmo = ps.astropy_cosmo_fiducial
    theta_pix = float(np.radians(ps.pix_resol))
    dnu = float(ps.freq_resol)
    chi_lo = cosmo.comoving_distance(freq_to_redshift(ps.nu + 0.5 * dnu)).value
    chi_hi = cosmo.comoving_distance(freq_to_redshift(ps.nu - 0.5 * dnu)).value
    dpar_ch = np.abs(chi_hi - chi_lo)
    chi_ch = 0.5 * (chi_lo + chi_hi)
    dperp_ch = theta_pix * chi_ch
    ic = np.argmin(np.abs(chi[:, None] - chi_ch[None, :]), axis=1)
    return {
        "nhat": nhat,
        "dperp": np.asarray(dperp_ch[ic], dtype=float),
        "dpar": np.asarray(dpar_ch[ic], dtype=float),
        "chi": chi,
        "dperp_ch": np.asarray(dperp_ch, dtype=float),
        "dpar_ch": np.asarray(dpar_ch, dtype=float),
    }


def cell_sampling_kernel(q_hat, q_abs, nhat, dperp, dpar):
    r"""
    Per-cell map-sampling kernel :math:`\hat S_b(\mathbf q)`.

    Each cell is a top-hat of radial width ``dpar`` and angular side
    ``dperp`` in its own frame :math:`\hat n_b`.  The two transverse
    axes split :math:`q_\perp` evenly (the diagnostic split; swapping
    it changes :math:`\langle|S|^2\rangle` by :math:`<10^{-3}`).
    """
    q_hat = np.asarray(q_hat, dtype=float).reshape(3)
    qn = float(np.linalg.norm(q_hat))
    if qn <= 0:
        raise ValueError("q_hat must be a non-zero 3-vector")
    q_hat = q_hat / qn
    q_abs = float(q_abs)
    nhat = np.asarray(nhat, dtype=float).reshape(-1, 3)
    dperp = np.asarray(dperp, dtype=float).reshape(-1)
    dpar = np.asarray(dpar, dtype=float).reshape(-1)
    mu = nhat @ q_hat
    q_par = q_abs * mu
    q_perp = q_abs * np.sqrt(np.maximum(1.0 - mu**2, 0.0))
    rad = np.sinc(q_par * dpar / (2.0 * np.pi))
    tr = np.sinc((q_perp / np.sqrt(2.0)) * dperp / (2.0 * np.pi)) ** 2
    return rad * tr


def cell_sampling_kernel_mu_rms(q_abs, dperp, dpar, *, nmu: int = 16):
    r"""
    Shell-frozen per-cell amplitude :math:`\sqrt{\langle\hat S_b^2\rangle_\mu}`.

    :math:`\hat S_b` in the cell frame depends on :math:`|q|` and
    :math:`\mu=\hat q\cdot\hat n_b` only.  Gauss–Legendre average over
    :math:`\mu\in[-1,1]`.
    """
    q_abs = float(q_abs)
    dperp = np.asarray(dperp, dtype=float).reshape(-1)
    dpar = np.asarray(dpar, dtype=float).reshape(-1)
    mus, wts = np.polynomial.legendre.leggauss(int(nmu))
    s2 = np.zeros(dperp.shape[0], dtype=float)
    for mu, w in zip(mus, wts):
        q_par = q_abs * float(mu)
        q_perp = q_abs * float(np.sqrt(max(1.0 - mu * mu, 0.0)))
        s = np.sinc(q_par * dpar / (2.0 * np.pi)) * (
            np.sinc((q_perp / np.sqrt(2.0)) * dperp / (2.0 * np.pi)) ** 2
        )
        s2 = s2 + float(w) * s**2
    return np.sqrt(np.maximum(0.5 * s2, 0.0))


def propose_k1dbins_window(
    k1dbins_out: ArrayLike,
    *,
    k_min: float | None = None,
    n: int = 1000,
    low_factor: float = 0.1,
    high_factor: float = 1.1,
) -> NDArray[np.floating]:
    """
    Log-spaced bin edges for measuring :math:`W_L(k)`.

    The lower edge is clipped to ``k_min`` when supplied (e.g. the box
    fundamental mode) so empty low-``k`` shells are not Hankel-extrapolated.
    """
    edges_out = np.asarray(k1dbins_out, dtype=float)
    k_lo = max(float(edges_out[0]) * low_factor, 1e-3)
    if k_min is not None:
        k_lo = max(k_lo, float(k_min))
    k_hi = float(edges_out[-1]) * high_factor
    return np.geomspace(k_lo, k_hi, int(n))


def _excess_zero_mode_window_power(swe) -> float:
    """
    Pair-count excess :math:`\\max(W(0)-W(k_{\\mathrm{fund}}), 0)`.

    Measured :math:`W_L` bins start at the box fundamental; FFTLog edge
    extrapolation already carries some of the low-``k`` peak. Passing the
    full :meth:`~SmoothWindowEstimator.zero_mode_window_power` as ``W_zero``
    therefore double-counts for sharp footprints — only the excess spike is
    missing from the Hankel transform.
    """
    w0 = float(swe.zero_mode_window_power())
    w_first = 0.0
    if swe.W_ell is not None and 0 in swe.W_ell and swe.k is not None:
        k_w = np.asarray(swe.k, dtype=float)
        w0k = np.asarray(swe.W_ell[0], dtype=float)
        finite = np.isfinite(k_w) & np.isfinite(w0k) & (k_w > 0)
        if np.any(finite):
            w_first = float(w0k[finite][0])
    return max(w0 - w_first, 0.0)


@dataclass
class AccumulatedWindow:
    """Ensemble-averaged window multipoles from :func:`accumulate_window_multipoles`."""

    k: NDArray[np.floating]
    nmodes: NDArray[np.floating]
    ells: tuple[int, ...]
    W_ell: dict[int, NDArray[np.floating]]
    W_ell_std: dict[int, NDArray[np.floating]]
    n_realizations: int


# ---------------------------------------------------------------------------
# Selection / random fields for window estimation
# ---------------------------------------------------------------------------


def make_im_selection_field(weights: ArrayLike) -> NDArray[np.floating]:
    r"""
    IM selection / weight cube used to measure survey-window multipoles.

    The smooth-window matrix needs the multipoles of the selection that
    multiplies the data cube (mask, inverse-variance map, …), not of
    cosmological signal or of white noise. This helper returns that weight
    field as a float array for
    :class:`~meer21cm.estimator.FieldPowerSpectrum`.
    """
    return np.asarray(weights, dtype=float).copy()


def make_galaxy_poisson_mean_density(
    selection_mask: ArrayLike,
    dndz_box: ArrayLike | None = None,
    mean_density: float | None = None,
    tot_num_galaxies: float | None = None,
) -> NDArray[np.floating]:
    """
    Build per-voxel Poisson means for galaxy randoms.

    ``selection_mask`` is typically ``(counts_in_box > 0)``. If ``dndz_box`` is
    given it multiplies the mask. Normalisation priority:

    1. ``tot_num_galaxies`` — scale so ``sum(mean) == tot_num_galaxies``;
    2. else ``mean_density`` — constant amplitude on the (unweighted) mask;
    3. else leave the (mask × dndz) field unscaled.
    """
    mask = np.asarray(selection_mask, dtype=float)
    if dndz_box is None:
        mean = mask.copy()
    else:
        mean = mask * np.asarray(dndz_box, dtype=float)
    total = mean.sum()
    if total <= 0:
        raise ValueError("Galaxy random mean density is zero everywhere")
    if tot_num_galaxies is not None:
        mean = mean * (float(tot_num_galaxies) / total)
    elif mean_density is not None:
        mean = mask * float(mean_density)
    return mean


def make_galaxy_poisson_random(
    mean_density: ArrayLike, seed: int
) -> NDArray[np.floating]:
    """Poisson-sample a galaxy number-count field from per-voxel means."""
    rng = np.random.default_rng(seed)
    return rng.poisson(np.asarray(mean_density, dtype=float)).astype(float)


def accumulate_window_multipoles(
    results: Sequence[MultipoleMeasurement] | Iterable[MultipoleMeasurement],
) -> AccumulatedWindow:
    """
    Average a list of multipole measurements into window multipoles.

    Parameters
    ----------
    results : sequence of MultipoleMeasurement
        Outputs of :meth:`~meer21cm.estimator.FieldPowerSpectrum.measure_multipoles`
        or :func:`run_smooth_window_realization`.
    """
    results_list = list(results)
    if not results_list:
        raise ValueError("No results to accumulate")
    first = results_list[0]
    ells = tuple(first.ells)
    k = np.asarray(first.k, dtype=float)
    nmodes = np.asarray(first.nmodes, dtype=float)
    stack: dict[int, list[NDArray[np.floating]]] = {ell: [] for ell in ells}
    for res in results_list:
        for ell in ells:
            stack[ell].append(np.asarray(res.P_ell[ell], dtype=float))
    W_ell = {ell: np.mean(stack[ell], axis=0) for ell in ells}
    W_ell_std = {
        ell: (
            np.std(stack[ell], axis=0, ddof=1)
            if len(stack[ell]) > 1
            else np.zeros_like(k)
        )
        for ell in ells
    }
    return AccumulatedWindow(
        k=k,
        nmodes=nmodes,
        ells=ells,
        W_ell=W_ell,
        W_ell_std=W_ell_std,
        n_realizations=len(results_list),
    )


def run_smooth_window_realization(
    box_len: ArrayLike,
    k1dbins: ArrayLike | None = None,
    seed: int = 0,
    tracer: Tracer | str = "hi",
    ells: Sequence[int] = (0, 2, 4),
    los: str = "global",
    los_observer: ArrayLike | None = None,
    weights_hi: ArrayLike | None = None,
    selection_mask: ArrayLike | None = None,
    dndz_box: ArrayLike | None = None,
    tot_num_galaxies: float | None = None,
    mean_density: float | None = None,
    weights_grid_1: ArrayLike | None = None,
    weights_grid_2: ArrayLike | None = None,
    mean_center_1: bool = False,
    mean_center_2: bool = False,
    unitless_1: bool = False,
    unitless_2: bool = False,
    k1dbins_window: ArrayLike | None = None,
) -> MultipoleMeasurement:
    """
    Pickleable worker for smooth-window multipoles.

    For ``tracer='hi'``, measures multipoles of the IM **selection / weight
    field** ``weights_hi`` (deterministic; ``seed`` is unused). For
    ``tracer='gal'``, measures the galaxy **selection** (mask × optional
    dndz), optionally Poisson-sampled at density ``tot_num_galaxies`` with
    amplitude restored to the selection. For ``tracer='cross'``,
    cross-correlates the HI selection with that galaxy selection field.
    All three paths rescale by :func:`~meer21cm.power_ops.power_weights_renorm`
    so :math:`W_L` matches the data estimator's :math:`\\sum w^2` convention.

    ``k1dbins_window`` (preferred) or legacy ``k1dbins`` are the **fine**
    bin edges used only to measure :math:`W_L(k)`. They are independent of
    estimator ``k_out`` / ``ps.k1dbins``.

    Intended for external batching (no pool is opened here).
    """
    if k1dbins_window is not None:
        k1dbins = k1dbins_window
    if k1dbins is None:
        raise TypeError("k1dbins_window (or legacy k1dbins) is required")
    tracer_s = str(tracer).lower()
    ells_t = tuple(int(e) for e in ells)

    field_1: NDArray[np.floating] | None
    field_2: NDArray[np.floating] | None
    w1: ArrayLike | None
    w2: ArrayLike | None

    if tracer_s in ("hi", "cross"):
        if weights_hi is None:
            raise ValueError("weights_hi is required for tracer=%r" % tracer_s)
        # Survey window = PS of the selection that multiplies the data cube.
        field_1 = make_im_selection_field(weights_hi)
        w1 = weights_grid_1
    else:
        field_1 = None
        w1 = None

    # Galaxy selection weight cube (same role as weights_hi for IM). Do *not*
    # bake tot_num_galaxies into the FFT amplitude — that would scale W_L by
    # λ² = (N_gal / ∑sel)² relative to the HI-style PS(selection).
    gal_sel: NDArray[np.floating] | None = None
    if tracer_s in ("gal", "cross"):
        if selection_mask is None:
            if weights_hi is not None:
                selection_mask = np.asarray(weights_hi) > 0
            else:
                raise ValueError("selection_mask is required for tracer=%r" % tracer_s)
        gal_sel = make_galaxy_poisson_mean_density(
            selection_mask,
            dndz_box=dndz_box,
            mean_density=mean_density,
            tot_num_galaxies=None,
        )
        gal_seed = seed if tracer_s == "gal" else seed + 10_000_003
        if tot_num_galaxies is not None:
            # Poisson-sample the selection at density ∝ tot_num, then restore
            # selection amplitude so ⟨field⟩ ≈ gal_sel (HI-equivalent).
            lam = float(tot_num_galaxies) / float(np.asarray(gal_sel).sum())
            if lam <= 0.0:
                raise ValueError("tot_num_galaxies must be positive")
            field_2 = make_galaxy_poisson_random(gal_sel * lam, gal_seed) / lam
        else:
            field_2 = np.asarray(gal_sel, dtype=float).copy()
        w2 = weights_grid_2
    else:
        field_2 = None
        w2 = None

    if tracer_s == "hi":
        assert field_1 is not None
        fps = FieldPowerSpectrum(
            field_1,
            box_len,
            weights_1=w1,
            mean_center_1=mean_center_1,
            unitless_1=unitless_1,
            los=los,
            los_observer=los_observer,
            _skip_specification=True,
        )
        meas = fps.measure_multipoles(which="auto_1", k1dbins=k1dbins, ells=ells_t)
        # Selection is stored as the FFT *field* with optional grid weights;
        # FPS renorm then sees only weights_1 (often None → 1). Rescale so W_L
        # includes R[w_eff]=N/∑w_eff² matching the data estimator.
        w_eff = _window_effective_weights(field_1, w1)
        return _rescale_window_multipoles(meas, w_eff, w_eff, float(fps.renorm_ps_1))
    if tracer_s == "gal":
        assert field_2 is not None and gal_sel is not None
        fps = FieldPowerSpectrum(
            field_2,
            box_len,
            weights_1=w2,
            mean_center_1=mean_center_2,
            unitless_1=unitless_2,
            los=los,
            los_observer=los_observer,
            _skip_specification=True,
        )
        meas = fps.measure_multipoles(which="auto_1", k1dbins=k1dbins, ells=ells_t)
        # Same R rescale as HI: target weights are the selection cube (× grid).
        w_eff = _window_effective_weights(gal_sel, w2)
        return _rescale_window_multipoles(meas, w_eff, w_eff, float(fps.renorm_ps_1))
    if tracer_s == "cross":
        assert field_1 is not None and field_2 is not None and gal_sel is not None
        fps = FieldPowerSpectrum(
            field_1,
            box_len,
            weights_1=w1,
            field_2=field_2,
            weights_2=w2,
            mean_center_1=mean_center_1,
            mean_center_2=mean_center_2,
            unitless_1=unitless_1,
            unitless_2=unitless_2,
            los=los,
            los_observer=los_observer,
            _skip_specification=True,
        )
        meas = fps.measure_multipoles(which="cross", k1dbins=k1dbins, ells=ells_t)
        w_eff_1 = _window_effective_weights(field_1, w1)
        w_eff_2 = _window_effective_weights(gal_sel, w2)
        return _rescale_window_multipoles(
            meas, w_eff_1, w_eff_2, float(fps.renorm_ps_cross)
        )
    raise ValueError("Unknown tracer %r; expected 'hi', 'gal', or 'cross'" % tracer_s)


def _window_effective_weights(
    selection: ArrayLike, weights_grid: ArrayLike | None
) -> NDArray[np.floating]:
    """Multiplicative weights whose Fourier transform enters the window PS."""
    sel = np.asarray(selection, dtype=float)
    if weights_grid is None:
        return sel
    return sel * np.asarray(weights_grid, dtype=float)


def _rescale_window_multipoles(
    meas: MultipoleMeasurement,
    weights_1: ArrayLike,
    weights_2: ArrayLike,
    renorm_already: float,
) -> MultipoleMeasurement:
    """
    Multiply measured window multipoles so they include
    ``power_weights_renorm(weights_1, weights_2)``.

    The data estimator keeps that renorm on the observed PS; baking the same
    factor into ``W_L`` makes the Hankel window :math:`O(1)` without dropping
    ``∑w²`` normalisation on the estimator side.
    """
    r_want = float(power_weights_renorm(weights_1, weights_2))
    scale = r_want / float(renorm_already) if renorm_already != 0.0 else r_want
    if not np.isfinite(scale) or scale == 1.0:
        return meas
    meas.P_ell = {
        int(ell): scale * np.asarray(p, dtype=float) for ell, p in meas.P_ell.items()
    }
    return meas


def _resolve_window_k_edges(
    k1dbins: ArrayLike | None,
    k1dbins_window: ArrayLike | None,
    k1dbins_out: ArrayLike | None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Resolve fine window-measure edges and coarse estimator ``k_out`` edges.

    Legacy ``k1dbins`` alone sets both (old behaviour). Prefer explicit
    ``k1dbins_window`` (fine :math:`W_L` bins) and ``k1dbins_out`` (estimator
    / matrix rows; usually ``ps.k1dbins``).
    """
    if k1dbins_out is None:
        k1dbins_out = k1dbins
    if k1dbins_window is None:
        k1dbins_window = k1dbins if k1dbins is not None else k1dbins_out
    if k1dbins_out is None:
        raise TypeError(
            "k1dbins_out (or legacy k1dbins) is required for estimator k_out bins"
        )
    return (
        np.asarray(k1dbins_window, dtype=float),
        np.asarray(k1dbins_out, dtype=float),
    )
