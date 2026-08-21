r"""
Multipole theory and selection-based survey-window estimation.

Yamamoto-matched alternative to 3D :func:`~meer21cm.power_ops.get_modelpk_conv`
(global plane-parallel modelling stays on that 3D path, then
:meth:`~meer21cm.power.PowerSpectrum.get_1d_power`):

1. Measure window multipoles :math:`W_L(k)` on a **fine**
   ``k1dbins_window`` via :class:`SmoothWindowEstimator`.
2. Build a discrete-shell matrix
   :class:`~meer21cm.window.DiscreteShellWindowMatrix` that maps
   continuous theory     :math:`P_{\ell'}(k_{\mathrm{in}})` onto coarse
   estimator bins :math:`P_\ell(k_{\mathrm{out}})` with the same
   discrete-:math:`\\mu` projector as
   :meth:`~meer21cm.power.PowerSpectrum.get_1d_power` (the leading-order
   Yamamoto binning; identity :math:`W` matches 3D→1D). ``los='global'``
   is not a window path.
3. Evaluate convolved multipoles with :class:`WindowedMultipoleModel`
   or the one-shot :func:`predict_windowed_multipoles`.

HI windows use the selection that multiplies the data cube (not white noise).
Galaxy windows use the same selection-weight cube (mask × optional
:math:`\mathrm{d}N/\mathrm{d}z`); ``tot_num_galaxies`` only sets a Poisson
sampling density used to *estimate* that selection (then amplitude is
restored). Field multipoles use
:class:`~meer21cm.estimator.FieldPowerSpectrum` (Yamamoto
``firstpoint`` / ``endpoint`` only; ``los='global'`` raises). Default
3D modelling on
:class:`~meer21cm.power.PowerSpectrum` is unchanged.

Window multipoles are scaled by the same weight-squared renorm
:func:`~meer21cm.power_ops.power_weights_renorm` used by the data
estimator, so :math:`W_L` is :math:`O(1)` (:math:`Q_0(s\to 0)\sim 1`)
while the estimator still divides by :math:`\sum w^2`. Pass ``W_zero``
(the :math:`k=0` selection power) into
:meth:`SmoothWindowEstimator.build_window_matrix` to restore the pair-count
term that measured :math:`W_L` bins omit. Matrix algebra lives in
:mod:`meer21cm.window`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .estimator import (
    FieldPowerSpectrum,
    MultipoleMeasurement,
    MultipoleShellMap,
)
from .model import ModelPowerSpectrum
from .power_ops import power_weights_renorm
from .window import (
    DiscreteShellWindowMatrix,
    WindowEllMap,
    apply_discrete_shell_window_matrix,
    build_discrete_shell_window_matrix,
    build_mesh_window_matrix,
    require_yamamoto_los,
    window_zero_mode_power,
)
from .wide_angle import propose_odd_wa_ells

import warnings

logger = logging.getLogger(__name__)

Tracer = Literal["hi", "gal", "cross"]


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


def _excess_zero_mode_window_power(swe: "SmoothWindowEstimator") -> float:
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


def predict_windowed_multipoles(
    ps,
    *,
    continuous: Literal["identity", "smooth"] = "smooth",
    k_in: ArrayLike | None = None,
    ells: Sequence[int] = (0, 2, 4),
    which: str = "auto_1",
    nmu: int = 64,
    los: str | None = None,
    los_observer: ArrayLike | None = None,
    k1dbins: ArrayLike | None = None,
    k1dbins_window: ArrayLike | None = None,
    n_window_bins: int = 1000,
    k1dweights: ArrayLike | None = None,
    weights_hi: ArrayLike | None = None,
    tracer: Tracer = "hi",
    n_fftlog: int = 512,
    n_k_eval: int = 128,
    W_zero: float | Literal["excess"] | None = None,
    box_volume: float | None = None,
    theory_scale: ArrayLike | None = None,
    mode_scale: ArrayLike | None = None,
    n_k_in: int = 80,
    wide_angle: bool | None = None,
    wa_d: float | None = None,
    wa_los: str | None = None,
) -> dict[str, Any]:
    """
    One-shot continuous :math:`P_\\ell(k_{\\mathrm{in}})` × discrete-shell
    window → estimator multipoles. The outer discrete-shell sum applies the
    discrete-:math:`\\mu` projector
    :math:`(2\\ell+1)\\mathcal{L}_\\ell(\\mu)` with
    :math:`\\mu=\\hat k\\cdot\\hat n_{\\mathrm{ref}}` (the local-LOS reference
    direction, box centre); identity :math:`W` matches 3D→1D.

    Unifies the identity / smooth prediction glue used by the Yamamoto
    validation scripts: build (or skip) a
    :class:`SmoothWindowEstimator`, form the shell map, build
    :class:`~meer21cm.window.DiscreteShellWindowMatrix`, evaluate
    continuous theory via
    :meth:`~meer21cm.model.ModelPowerSpectrum.get_theory_multipoles_kmu`,
    and apply the matrix.

    Parameters
    ----------
    ps :
        :class:`~meer21cm.power.PowerSpectrum`-like object with
        ``k1dbins``, ``multipole_bin_index_map``, and
        ``get_theory_multipoles_kmu``.
    continuous : {'identity', 'smooth'}, default 'smooth'
        Discrete-shell continuous layer. ``'identity'`` needs no measured
        :math:`W_L` (no survey convolution). ``'smooth'``
        accumulates one HI selection realisation.
    k_in : array_like, optional
        Fine theory nodes. Defaults to :func:`propose_k_in` on ``k1dbins``.
    W_zero : float or {'excess'}, optional
        Forwarded to
        :meth:`SmoothWindowEstimator.build_window_matrix`. ``'excess'``
        uses :func:`_excess_zero_mode_window_power` (pair-count spike minus
        the first measured :math:`W_0` bin). Ignored for ``continuous=
        'identity'``.
    theory_scale : array_like, optional
        Multiplicative kernel on continuous :math:`P_\\ell(k_{\\mathrm{in}})`
        before the window (same length as ``k_in``). Prefer putting
        same-k grid transfers in ``mode_scale`` instead.
    mode_scale : array_like, optional
        Per-Cartesian-mode theory transfer (same shape as ``ps`` Fourier
        ``k_mode`` / shell map). Forwarded to
        :func:`~meer21cm.window.build_discrete_shell_window_matrix`
        so MAS / gridding compensation multiplies inside the discrete-shell
        sum.
    box_volume, n_fftlog, n_k_eval, wide_angle, wa_d, wa_los :
        Forwarded to the matrix builder.

    Returns
    -------
    dict
        ``k`` (``k_out``), ``P_ell``, ``k_in``, ``P_ell_unconvolved``,
        ``ells``, ``window_matrix``, ``W_zero`` (resolved float or
        ``None``), and ``continuous`` (the raw
        :meth:`~meer21cm.model.ModelPowerSpectrum.get_theory_multipoles_kmu`
        dict).
    """
    continuous_s = str(continuous).lower()
    if continuous_s not in ("identity", "smooth"):
        raise ValueError("continuous must be 'identity' or 'smooth'")

    ells_t = tuple(int(e) for e in ells)
    k1dbins_out = np.asarray(ps.k1dbins if k1dbins is None else k1dbins, dtype=float)
    if k_in is None:
        k_in_arr = propose_k_in(k1dbins_out, n=n_k_in)
    else:
        k_in_arr = np.asarray(k_in, dtype=float)

    los_use = getattr(ps, "los", "endpoint") if los is None else str(los)
    require_yamamoto_los(los_use)
    obs_saved = getattr(ps, "los_observer", None)
    los_saved = getattr(ps, "los", None)
    if los_observer is not None:
        ps.los_observer = np.asarray(los_observer, dtype=float)
    ps.los = los_use

    try:
        mat: DiscreteShellWindowMatrix
        w_zero_resolved: float | None = None
        if continuous_s == "identity":
            ells_out = ells_t
            ells_in = ells_t
            do_wa = bool(wide_angle)
            even = tuple(e for e in ells_out if e % 2 == 0)
            if do_wa:
                if not even:
                    raise ValueError(
                        "wide_angle requires at least one even output multipole"
                    )
                odds = propose_odd_wa_ells(even)
                ells_in = tuple(sorted(set(even) | set(odds)))
            shell = ps.multipole_bin_index_map(
                k1dbins=k1dbins_out,
                k1dweights=k1dweights,
                los=los_use,
            )
            mat = build_discrete_shell_window_matrix(
                shell,
                k_in=k_in_arr,
                ells=ells_out,
                ells_in=ells_in,
                continuous="identity",
                n_k_eval=n_k_eval,
                mode_scale=mode_scale,
            )
            if do_wa:
                d_wa = wa_d
                if d_wa is None:
                    raise ValueError("wa_d is required when wide_angle=True")
                los_wa = (
                    wa_los
                    if wa_los is not None
                    else (
                        los_use
                        if los_use in ("firstpoint", "endpoint")
                        else "firstpoint"
                    )
                )
                mat.resum_input_odd_wide_angle(
                    los=los_wa, d=float(d_wa), ells_even=even
                )
        else:
            k_mode = getattr(ps, "k_mode", None)
            k_fund = None
            if k_mode is not None:
                kpos = np.asarray(k_mode, dtype=float)
                kpos = kpos[np.isfinite(kpos) & (kpos > 0)]
                if kpos.size:
                    k_fund = float(np.min(kpos))
            if k1dbins_window is None:
                k1dbins_window = propose_k1dbins_window(
                    k1dbins_out, k_min=k_fund, n=n_window_bins
                )
            w_hi = weights_hi
            if w_hi is None:
                w_hi = getattr(ps, "weights_1", None)
            swe = SmoothWindowEstimator.from_power_spectrum(
                ps,
                tracer=tracer,
                ells=ells_t,
                weights_hi=w_hi,
                weights_grid_1=None,
                k1dbins_window=k1dbins_window,
                k1dbins_out=k1dbins_out,
                los=los_use,
                los_observer=getattr(ps, "los_observer", None),
                wide_angle=bool(wide_angle) if wide_angle is not None else False,
                wa_d=wa_d,
                wa_los=wa_los,
            )
            swe.accumulate([swe.run_one(0)])
            shell = ps.multipole_bin_index_map(
                k1dbins=k1dbins_out,
                k1dweights=k1dweights,
                los=los_use,
            )
            if W_zero is None:
                w_zero_resolved = None
            elif isinstance(W_zero, str) and str(W_zero).lower() == "excess":
                w_zero_resolved = _excess_zero_mode_window_power(swe)
            else:
                w_zero_resolved = float(W_zero)
            build_kw: dict[str, Any] = dict(
                shell_map=shell,
                continuous="smooth",
                n_fftlog=n_fftlog,
                n_k_eval=n_k_eval,
            )
            if w_zero_resolved is not None:
                build_kw["W_zero"] = w_zero_resolved
            if box_volume is not None:
                build_kw["box_volume"] = float(box_volume)
            elif w_zero_resolved is not None:
                build_kw["box_volume"] = float(
                    np.prod(np.asarray(ps.box_len, dtype=float))
                )
            if wide_angle is not None:
                build_kw["wide_angle"] = bool(wide_angle)
            if wa_d is not None:
                build_kw["wa_d"] = wa_d
            if wa_los is not None:
                build_kw["wa_los"] = wa_los
            if mode_scale is not None:
                build_kw["mode_scale"] = mode_scale
            mat = swe.build_window_matrix(k_in_arr, **build_kw)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cont = ps.get_theory_multipoles_kmu(
                k_in_arr, ells=ells_t, nmu=nmu, which=which
            )
        P_in = {ell: np.asarray(cont["P_ell"][ell], dtype=float) for ell in ells_t}
        if theory_scale is not None:
            scale = np.asarray(theory_scale, dtype=float)
            if scale.shape != k_in_arr.shape:
                raise ValueError(
                    "theory_scale must have the same shape as k_in "
                    f"(got {scale.shape}, expected {k_in_arr.shape})"
                )
            P_in = {ell: P_in[ell] * scale for ell in ells_t}
        P_win = mat.apply(P_in)
    finally:
        if los_saved is not None:
            ps.los = los_saved
        if los_observer is not None:
            ps.los_observer = obs_saved

    return {
        "k": np.asarray(mat.k_out, dtype=float),
        "P_ell": P_win,
        "k_in": k_in_arr,
        "P_ell_unconvolved": cont["P_ell"],
        "ells": ells_t,
        "window_matrix": mat,
        "W_zero": w_zero_resolved,
        "continuous": cont,
    }


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
    so :math:`W_L` matches the data estimator's :math:`\sum w^2` convention.

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


class SmoothWindowEstimator:
    """
    Measure survey-window multipoles from the HI selection and/or galaxy randoms.

    Three distinct :math:`k` grids (do not conflate them):

    - ``k1dbins_window`` — fine bin **edges** used only to measure
      :math:`W_L(k)` (stored centres in :attr:`k_window` / :attr:`k` after
      :meth:`accumulate`). Should resolve the low-``k`` window peak.
    - ``k_in`` — fine theory nodes, passed to :meth:`build_window_matrix`
      (matrix columns).
    - ``k1dbins_out`` — coarse estimator bin **edges** (legacy
      ``ps.k1dbins``); shell map / matrix rows (:attr:`k_out`).

    Legacy ``k1dbins`` alone still sets both window and out edges (old
    behaviour). Prefer passing ``k1dbins_window`` and ``k1dbins_out``
    separately.

    For HI, :meth:`run_one` measures multipoles of ``weights_hi`` (the field
    that multiplies the data cube) and rescales them by
    :func:`~meer21cm.power_ops.power_weights_renorm` so :math:`W_L` is
    :math:`O(1)` while the data estimator still applies the same renorm.
    Galaxy / cross use the selection cube the same way; ``tot_num_galaxies``
    only sets optional Poisson sampling density (amplitude restored).
    ``seed`` matters only for those Poisson draws.

    Does **not** open an MPI or multiprocessing pool. Callers map
    :meth:`get_arg_list_for_seeds` (or :func:`run_smooth_window_realization`)
    externally, then :meth:`accumulate`. Use :meth:`build_window_matrix` to
    turn accumulated :math:`W_L(k)` (or an identity :math:`|k|`-rebin
    kernel) into a
    :class:`~meer21cm.window.DiscreteShellWindowMatrix`.

    Parameters
    ----------
    box_len : array_like
        Box lengths in Mpc.
    k1dbins : array_like, optional
        Legacy: if ``k1dbins_window`` / ``k1dbins_out`` are omitted, used for
        **both** window measurement and estimator ``k_out``.
    k1dbins_window : array_like, optional
        Fine bin edges for measuring :math:`W_L(k)`.
    k1dbins_out : array_like, optional
        Coarse estimator bin edges (matrix ``k_out``; usually ``ps.k1dbins``).
    ells : sequence, default (0, 2, 4)
        Multipoles to measure / store.
    los : {'endpoint', 'firstpoint'}, default 'endpoint'
        Yamamoto LOS for
        :class:`~meer21cm.estimator.FieldPowerSpectrum`.
        ``los='global'`` is rejected (use the 3D ``get_1d_power`` path).
    los_observer : array_like, optional
        Observer position for local Yamamoto LOS (Mpc). Defaults to a far
        observer along :math:`+z` so :math:`\\hat n_{\\mathrm{ref}}=\\hat z`
        (plane-parallel Yamamoto; identity :math:`W` matches 3D→1D).
        ``from_power_spectrum`` uses ``ps.los_observer`` or ``ps.box_origin``
        when present.
    tracer : {'hi', 'gal', 'cross'}
        HI selection auto, galaxy randoms auto, or HI×gal cross window.
    weights_hi : ndarray, optional
        IM selection / weight cube whose multipoles define the HI window.
    selection_mask : ndarray, optional
        Galaxy footprint (e.g. ``counts_in_box > 0``).
    dndz_box : ndarray, optional
        Per-voxel dN/dz weight for galaxy randoms.
    tot_num_galaxies : float, optional
        If set, Poisson-sample the galaxy selection at this expected count,
        then divide by ``N/∑sel`` so the FFT field has selection amplitude
        (not number-count amplitude). Omit for a deterministic selection FFT.
    mean_density : float, optional
        Constant mean density on the mask (alternative amplitude for the
        selection cube when ``tot_num_galaxies`` is omitted).
    weights_grid_1, weights_grid_2 : ndarray, optional
        Optional extra FFT grid weights (same role as estimator
        ``weights_1/2``). Leave ``None`` when ``weights_hi`` is already the
        full multiplicative selection (avoid double-counting).
    wide_angle : bool, default False
        If True, measure extra odd :math:`W_L` and resum wa_order=1 into
        the discrete-shell matrix (even theory input only).
    wa_d : float, optional
        Comoving distance to the effective redshift (Mpc) for wide-angle.
    wa_los : {'firstpoint', 'endpoint'}, optional
        Wide-angle LOS convention. Defaults to :attr:`los` when local.
    """

    def __init__(
        self,
        box_len,
        k1dbins=None,
        ells=(0, 2, 4),
        los="endpoint",
        los_observer=None,
        tracer="hi",
        weights_hi=None,
        selection_mask=None,
        dndz_box=None,
        tot_num_galaxies=None,
        mean_density=None,
        weights_grid_1=None,
        weights_grid_2=None,
        mean_center_1=False,
        mean_center_2=False,
        unitless_1=False,
        unitless_2=False,
        k1dbins_window=None,
        k1dbins_out=None,
        wide_angle=False,
        wa_d=None,
        wa_los=None,
    ):
        self.box_len = np.asarray(box_len, dtype=float)
        self.k1dbins_window, self.k1dbins_out = _resolve_window_k_edges(
            k1dbins, k1dbins_window, k1dbins_out
        )
        # Legacy alias: estimator / shell-map edges (not the W_L measure grid).
        self.k1dbins = self.k1dbins_out
        self.ells = tuple(int(e) for e in ells)
        self.los = require_yamamoto_los(los)
        if los_observer is None:
            # Far +z: n_ref = z, so identity W matches get_1d_power of the 3D cube.
            self.los_observer = np.array([0.0, 0.0, 1.0e12], dtype=float)
        else:
            self.los_observer = np.asarray(los_observer, dtype=float)
        self.wide_angle = bool(wide_angle)
        self.wa_d = None if wa_d is None else float(wa_d)
        self.wa_los = None if wa_los is None else str(wa_los).lower()
        self._ells_measure = propose_window_measure_ells(
            self.ells, wide_angle=self.wide_angle
        )
        self.tracer = str(tracer).lower()
        self.weights_hi = weights_hi
        self.selection_mask = selection_mask
        self.dndz_box = dndz_box
        self.tot_num_galaxies = tot_num_galaxies
        self.mean_density = mean_density
        self.weights_grid_1 = weights_grid_1
        self.weights_grid_2 = weights_grid_2
        self.mean_center_1 = mean_center_1
        self.mean_center_2 = mean_center_2
        self.unitless_1 = unitless_1
        self.unitless_2 = unitless_2

        self.k = None
        self.nmodes = None
        self.W_ell = None
        self.W_ell_std = None
        self.n_realizations = 0
        self.window_matrix: DiscreteShellWindowMatrix | None = None
        self.k_in = None
        self.shell_map: MultipoleShellMap | None = None

    @property
    def k_window(self) -> NDArray[np.floating] | None:
        """Measured :math:`W_L` wavenumber centres (after :meth:`accumulate`)."""
        return self.k

    @property
    def k_out(self) -> NDArray[np.floating] | None:
        """Estimator bin centres from the shell map / built window matrix."""
        if self.window_matrix is not None:
            return self.window_matrix.k_out
        if self.shell_map is not None:
            return np.asarray(self.shell_map.k_eff, dtype=float)
        return None

    @classmethod
    def from_power_spectrum(cls, ps, tracer="hi", ells=(0, 2, 4), **kwargs):
        """
        Build from a :class:`~meer21cm.power.PowerSpectrum`-like object.

        Uses ``weights_field_1`` (else ``counts_in_box``) as the HI selection
        field for window multipoles, and ``(selection > 0)`` as the default
        galaxy mask.

        ``k1dbins_out`` defaults to ``ps.k1dbins``. Pass a finer
        ``k1dbins_window`` for measuring :math:`W_L`; if omitted, it falls
        back to the same edges as ``k1dbins_out`` (legacy behaviour).
        """
        # Resolve the HI selection lazily so that objects without survey
        # pixel data (e.g. survey-free simulation boxes) never trigger the
        # ``counts_in_box`` computation when ``weights_hi`` is given.
        weights_hi = kwargs.pop("weights_hi", None)
        if weights_hi is None:
            weights_hi = getattr(ps, "weights_field_1", None)
        if weights_hi is None:
            weights_hi = getattr(ps, "counts_in_box", None)
        if weights_hi is None:
            weights_hi = getattr(ps, "weights_1", None)
        selection_mask = kwargs.pop(
            "selection_mask",
            None if weights_hi is None else (np.asarray(weights_hi) > 0),
        )
        k1dbins = kwargs.pop("k1dbins", None)
        k1dbins_out = kwargs.pop(
            "k1dbins_out", k1dbins if k1dbins is not None else ps.k1dbins
        )
        k1dbins_window = kwargs.pop("k1dbins_window", None)
        k_mode = getattr(ps, "k_mode", None)
        k_fund = None
        if k_mode is not None:
            kpos = np.asarray(k_mode, dtype=float)
            kpos = kpos[np.isfinite(kpos) & (kpos > 0)]
            if kpos.size:
                k_fund = float(np.min(kpos))
        if k1dbins_window is not None and k_fund is not None:
            kw = np.asarray(k1dbins_window, dtype=float).copy()
            kw[0] = max(float(kw[0]), k_fund)
            k1dbins_window = kw
        kwargs.setdefault("los", getattr(ps, "los", "endpoint"))
        if "los_observer" not in kwargs:
            los_observer = getattr(ps, "los_observer", None)
            if los_observer is None:
                los_observer = getattr(ps, "box_origin", None)
            kwargs["los_observer"] = los_observer
        if kwargs.get("wide_angle") and "wa_d" not in kwargs:
            origin = np.asarray(getattr(ps, "box_origin", 0.0), dtype=float).reshape(-1)
            if origin.size == 1:
                origin = np.full(3, float(origin.flat[0]))
            elif origin.size != 3:
                origin = np.pad(origin, (0, max(0, 3 - origin.size)))[:3]
            box_len_ps = np.asarray(ps.box_len, dtype=float).reshape(-1)
            kwargs["wa_d"] = float(np.linalg.norm(origin + 0.5 * box_len_ps))
        return cls(
            box_len=ps.box_len,
            k1dbins=k1dbins,
            k1dbins_window=k1dbins_window,
            k1dbins_out=k1dbins_out,
            ells=ells,
            tracer=tracer,
            weights_hi=weights_hi,
            selection_mask=selection_mask,
            weights_grid_1=kwargs.pop(
                "weights_grid_1", getattr(ps, "weights_grid_1", None)
            ),
            weights_grid_2=kwargs.pop(
                "weights_grid_2", getattr(ps, "weights_grid_2", None)
            ),
            **kwargs,
        )

    def _worker_kwargs(self):
        return dict(
            box_len=self.box_len,
            k1dbins_window=self.k1dbins_window,
            tracer=self.tracer,
            ells=self._ells_measure,
            los=self.los,
            los_observer=self.los_observer,
            weights_hi=self.weights_hi,
            selection_mask=self.selection_mask,
            dndz_box=self.dndz_box,
            tot_num_galaxies=self.tot_num_galaxies,
            mean_density=self.mean_density,
            weights_grid_1=self.weights_grid_1,
            weights_grid_2=self.weights_grid_2,
            mean_center_1=self.mean_center_1,
            mean_center_2=self.mean_center_2,
            unitless_1=self.unitless_1,
            unitless_2=self.unitless_2,
        )

    def get_arg_list_for_seeds(self, seed_list):
        """
        Build pickleable argument tuples for external batch mapping.

        Each tuple is ``(kwargs_dict, seed)`` suitable for
        ``run_smooth_window_realization(**kwargs, seed=seed)``.
        """
        base = self._worker_kwargs()
        return [(dict(base), int(seed)) for seed in seed_list]

    def run_one(self, seed):
        """Run a single realization (convenience wrapper)."""
        return run_smooth_window_realization(seed=seed, **self._worker_kwargs())

    def accumulate(self, results):
        """Average realization results into ``W_ell`` / :attr:`k_window`."""
        acc = accumulate_window_multipoles(results)
        self.k = acc.k
        self.nmodes = acc.nmodes
        self.W_ell = acc.W_ell
        self.W_ell_std = acc.W_ell_std
        self.n_realizations = acc.n_realizations
        return acc

    def zero_mode_window_power(self) -> float:
        """
        :math:`k=0` window power of the selection, same convention as :math:`W_L`.

        Added as a constant :math:`Q_0^{\\mathrm{DC}}=W(0)/V` when building
        the smooth matrix (the measured :math:`W_L` bins start at the
        fundamental and never include this mode).
        """
        volume = float(np.prod(np.asarray(self.box_len, dtype=float)))
        tracer = self.tracer
        w1 = None
        w2 = None
        if tracer in ("hi", "cross"):
            if self.weights_hi is None:
                return 0.0
            w1 = _window_effective_weights(self.weights_hi, self.weights_grid_1)
        if tracer in ("gal", "cross"):
            mask = self.selection_mask
            if mask is None and self.weights_hi is not None:
                mask = np.asarray(self.weights_hi) > 0
            if mask is None:
                return 0.0
            gal_sel = make_galaxy_poisson_mean_density(
                mask,
                dndz_box=self.dndz_box,
                mean_density=self.mean_density,
                tot_num_galaxies=None,
            )
            w2 = _window_effective_weights(gal_sel, self.weights_grid_2)
        if tracer == "hi":
            return window_zero_mode_power(w1, volume)
        if tracer == "gal":
            return window_zero_mode_power(w2, volume)
        return float(np.mean(w1) * np.mean(w2) * volume * power_weights_renorm(w1, w2))

    def make_shell_map(
        self,
        k1dweights=None,
    ) -> MultipoleShellMap:
        """
        Build a :class:`~meer21cm.estimator.MultipoleShellMap` for ``k_out``.

        Uses :attr:`k1dbins_out` (legacy estimator edges), not the fine
        :attr:`k1dbins_window` used to measure :math:`W_L`. The map is
        :math:`|k|`-shell membership for the discrete-shell window.
        :attr:`~meer21cm.estimator.MultipoleShellMap.mu` is
        :math:`\\hat k\\cdot\\hat n_{\\mathrm{ref}}` (the local-LOS reference
        direction, box centre) and is used as the discrete-:math:`\\mu`
        projector.
        """
        shape = None
        for candidate in (
            self.weights_hi,
            self.selection_mask,
            self.weights_grid_1,
            self.weights_grid_2,
        ):
            if candidate is not None:
                shape = np.asarray(candidate).shape
                break
        if shape is None:
            raise ValueError(
                "Cannot infer box_ndim for shell map; provide weights_hi, "
                "selection_mask, or weights_grid_1/2"
            )
        field = np.ones(shape, dtype=float)
        fps = FieldPowerSpectrum(
            field,
            self.box_len,
            los=self.los,
            los_observer=self.los_observer,
            _skip_specification=True,
        )
        self.shell_map = fps.multipole_bin_index_map(
            k1dbins=self.k1dbins_out,
            k1dweights=k1dweights,
            los=self.los,
        )
        return self.shell_map

    def build_window_matrix(
        self,
        k_in,
        shell_map: MultipoleShellMap | None = None,
        n_fftlog=512,
        continuous="smooth",
        wide_angle=None,
        wa_d=None,
        wa_los=None,
        **kwargs,
    ) -> DiscreteShellWindowMatrix:
        """
        Build a discrete-shell window matrix ``W_{ℓℓ'}(k_out, k_in)``.

        Parameters
        ----------
        k_in : array_like
            Fine theory :math:`k` nodes (matrix columns; independent of
            :attr:`k1dbins_window` and :attr:`k1dbins_out`).
        shell_map : MultipoleShellMap, optional
            Estimator shells for ``k_out``. Defaults to
            :meth:`make_shell_map` from :attr:`k1dbins_out`.
        continuous : {'smooth', 'identity'}, default 'smooth'
            ``'identity'`` needs no accumulated ``W_ell`` (no survey
            convolution). ``'smooth'`` requires :meth:`accumulate`
            first (uses measured :attr:`k_window` / :attr:`W_ell`).
            The outer discrete-shell sum always applies the
            discrete-:math:`\\mu` projector
            (:math:`\\mu=\\hat k\\cdot\\hat n_{\\mathrm{ref}}`, the leading-order
            Yamamoto binning / 3D→1D sampling).
        wide_angle : bool, optional
            If True, include wa_order=1 odd theory columns then resum so
            :meth:`~meer21cm.window.DiscreteShellWindowMatrix.apply`
            takes even Kaiser :math:`P_\\ell` only.
        wa_d : float, optional
            Comoving distance to the effective redshift (Mpc). Defaults to
            :attr:`wa_d` stored on this estimator.
        wa_los : {'firstpoint', 'endpoint'}, optional
            Wide-angle LOS. Defaults to :attr:`wa_los` or the estimator
            :attr:`los`.
        """
        continuous_s = str(continuous).lower()
        if continuous_s == "smooth" and (self.W_ell is None or self.k is None):
            raise RuntimeError(
                "Accumulate window multipoles before building a smooth matrix"
            )
        if shell_map is None:
            shell_map = self.shell_map
        if shell_map is None:
            shell_map = self.make_shell_map()
        self.k_in = np.asarray(k_in, dtype=float)
        do_wa = self.wide_angle if wide_angle is None else bool(wide_angle)
        ells_out = self.ells
        ells_in = ells_out
        even = tuple(e for e in ells_out if e % 2 == 0)
        if do_wa:
            if not even:
                raise ValueError(
                    "wide_angle requires at least one even output multipole"
                )
            odds = propose_odd_wa_ells(even)
            ells_in = tuple(sorted(set(even) | set(odds)))
            d_wa = self.wa_d if wa_d is None else float(wa_d)
            if d_wa is None:
                raise ValueError("wa_d is required when wide_angle=True")
            los_wa = wa_los if wa_los is not None else self.wa_los
            if los_wa is None:
                los_wa = (
                    self.los if self.los in ("firstpoint", "endpoint") else "firstpoint"
                )
        self.window_matrix = build_discrete_shell_window_matrix(
            shell_map,
            None if continuous_s == "identity" else self.k_window,
            None if continuous_s == "identity" else self.W_ell,
            k_in=self.k_in,
            ells=ells_out,
            ells_in=ells_in,
            continuous=continuous_s,
            n_fftlog=n_fftlog,
            **kwargs,
        )
        if do_wa:
            self.window_matrix.resum_input_odd_wide_angle(
                los=los_wa, d=d_wa, ells_even=even
            )
        return self.window_matrix


class WindowedMultipoleModel(ModelPowerSpectrum):
    """
    Continuous multipole theory with an optional discrete-shell window matrix.

    Starts from :meth:`~meer21cm.model.ModelPowerSpectrum.power_kmu` (cosmo +
    RSD only; **no** beam, map sampling, or MAS compensation — those belong in
    the window), forms unconvolved multipoles at fine :math:`k_{\\mathrm{in}}`
    by a continuous :math:`\\mu` integral, then optionally applies a
    :class:`~meer21cm.window.DiscreteShellWindowMatrix` (discrete
    :math:`\\mu` sampling via box-centre :math:`\\hat n_{\\mathrm{ref}}`;
    the leading-order Yamamoto binning, matching 3D→1D for identity
    :math:`W`). ``los='global'`` is not a window path.
    Does **not** use :func:`~meer21cm.power_ops.get_modelpk_conv`.

    Parameters
    ----------
    window_matrix : DiscreteShellWindowMatrix or ndarray, optional
        Pre-built discrete-shell window matrix.
    window_ells : sequence of int, default (0, 2, 4)
        Default multipoles for theory and matrix blocks.
    """

    def __init__(
        self,
        *args: Any,
        window_matrix: DiscreteShellWindowMatrix | ArrayLike | None = None,
        window_ells: Sequence[int] = (0, 2, 4),
        **params: Any,
    ) -> None:
        super().__init__(*args, **params)
        self.window_ells = tuple(int(e) for e in window_ells)
        self._window_matrix_obj: DiscreteShellWindowMatrix | None = None
        self._window_matrix_raw: NDArray[np.floating] | None = None
        if window_matrix is not None:
            self.set_window_matrix(window_matrix)

    @property
    def window_matrix(self) -> NDArray[np.floating] | None:
        """Dense matrix ``(n_ell * n_out, n_ell * n_in)``, or ``None``."""
        if self._window_matrix_obj is not None:
            return self._window_matrix_obj.matrix
        return self._window_matrix_raw

    @property
    def k_out(self) -> NDArray[np.floating] | None:
        """Estimator bin centres from the attached discrete-shell matrix."""
        if self._window_matrix_obj is not None:
            return self._window_matrix_obj.k_out
        return None

    @property
    def k_in(self) -> NDArray[np.floating] | None:
        """Fine theory ``k_in`` nodes from the attached discrete-shell matrix."""
        if self._window_matrix_obj is not None:
            return self._window_matrix_obj.k_in
        return None

    @property
    def k_in_window(self) -> NDArray[np.floating] | None:
        """Alias for :attr:`k_in` (historical name)."""
        return self.k_in

    def set_window_matrix(
        self, window_matrix: DiscreteShellWindowMatrix | ArrayLike
    ) -> None:
        """Attach a discrete-shell window matrix (object or raw ndarray)."""
        if isinstance(window_matrix, DiscreteShellWindowMatrix):
            self._window_matrix_obj = window_matrix
            self._window_matrix_raw = None
            self.window_ells = tuple(window_matrix.ells_out)
        else:
            self._window_matrix_obj = None
            self._window_matrix_raw = np.asarray(window_matrix, dtype=float)

    def build_window_matrix_from_shell(
        self,
        shell_map: MultipoleShellMap,
        k_window: ArrayLike,
        W_ell: WindowEllMap,
        k_in: ArrayLike,
        ells: Sequence[int] | None = None,
        wide_angle: bool = False,
        wa_d: float | None = None,
        wa_los: str | None = None,
        **kwargs: Any,
    ) -> DiscreteShellWindowMatrix:
        """Build and attach a :class:`DiscreteShellWindowMatrix`."""
        if ells is None:
            ells = self.window_ells
        ells_out = tuple(int(e) for e in ells)
        ells_in = kwargs.pop("ells_in", None)
        even = tuple(e for e in ells_out if e % 2 == 0)
        if wide_angle:
            if not even:
                raise ValueError(
                    "wide_angle requires at least one even output multipole"
                )
            odds = propose_odd_wa_ells(even)
            ells_in = tuple(sorted(set(even) | set(odds)))
            if wa_d is None:
                raise ValueError("wa_d is required when wide_angle=True")
            if wa_los is None:
                wa_los = "firstpoint"
        result = build_discrete_shell_window_matrix(
            shell_map,
            k_window,
            W_ell,
            k_in=k_in,
            ells=ells_out,
            ells_in=ells_in,
            **kwargs,
        )
        if wide_angle:
            result.resum_input_odd_wide_angle(los=wa_los, d=float(wa_d), ells_even=even)
        self.set_window_matrix(result)
        return result

    def get_theory_multipoles_kmu(
        self,
        k_in: ArrayLike,
        ells: Sequence[int] | None = None,
        nmu: int = 64,
        which: str = "auto_1",
    ) -> dict[str, Any]:
        """
        Continuous multipoles; defaults ``ells`` to :attr:`window_ells`.

        See :meth:`ModelPowerSpectrum.get_theory_multipoles_kmu`.
        """
        if ells is None:
            ells = self.window_ells
        return super().get_theory_multipoles_kmu(k_in, ells=ells, nmu=nmu, which=which)

    def get_model_multipoles(
        self,
        which: str = "auto_1",
        k_in: ArrayLike | None = None,
        ells: Sequence[int] | None = None,
        nmu: int = 64,
        apply_window: bool = True,
    ) -> dict[str, Any]:
        """
        Theory multipoles, optionally convolved with the window matrix.

        Parameters
        ----------
        which : {'auto_1', 'auto_2', 'cross'}
            Tracer combination.
        k_in : array_like, optional
            Fine theory ``k``. Defaults to ``k_in`` stored on the attached
            :class:`DiscreteShellWindowMatrix`.
        ells : sequence of int, optional
            Multipoles (must match the matrix if ``apply_window``).
        nmu : int, default 64
            Gauss–Legendre nodes for the continuous :math:`\\mu` integral.
        apply_window : bool, default True
            If True and a window matrix is set, apply it.
        """
        if ells is None:
            ells = self.window_ells
        if k_in is None:
            k_in = self.k_in_window
        if k_in is None:
            raise ValueError(
                "k_in is required when no DiscreteShellWindowMatrix is attached"
            )

        ells_theory = ells
        if apply_window and self._window_matrix_obj is not None:
            ells_theory = self._window_matrix_obj.ells_in
        raw = self.get_theory_multipoles_kmu(
            k_in, ells=ells_theory, nmu=nmu, which=which
        )
        if apply_window and self.window_matrix is not None:
            if self._window_matrix_obj is not None:
                convolved = self._window_matrix_obj.apply(raw["P_ell"])
                k_out = self._window_matrix_obj.k_out
                nmodes = self._window_matrix_obj.nmodes
                ells_out = self._window_matrix_obj.ells_out
            else:
                convolved = apply_discrete_shell_window_matrix(
                    raw["P_ell"], self.window_matrix, ells=ells
                )
                k_out = None
                nmodes = None
                ells_out = tuple(int(e) for e in ells)
            return {
                "k": k_out if k_out is not None else raw["k"],
                "k_in": raw["k"],
                "nmodes": nmodes,
                "P_ell": convolved,
                "P_ell_unconvolved": raw["P_ell"],
                "ells": ells_out,
                "window_applied": True,
            }
        return {
            "k": raw["k"],
            "k_in": raw["k"],
            "nmodes": None,
            "P_ell": raw["P_ell"],
            "ells": raw["ells"],
            "window_applied": False,
        }
