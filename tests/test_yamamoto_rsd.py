"""Regression tests mirroring ``misc/yamamoto/rsd/sim_rsd_yamamoto.py`` checks."""

from __future__ import annotations

import numpy as np

from meer21cm import MockSimulation
from meer21cm.multipole_model import predict_windowed_multipoles, propose_k_in
from meer21cm.util import redshift_to_freq

ELLS = (0, 2, 4)
SEEDS = (0, 1, 2)
R_OVER_L = 1.0e5
BOX_LEN = (400.0, 400.0, 400.0)
BOX_NDIM = (64, 64, 64)

# Faster Hankel / window grids than the misc validation script.
N_FFTLOG = 128
N_K_EVAL = 64
N_WINDOW_BINS = 400
N_LOS_SAMPLES = 256
NMU = 32
K_IN_N = 60

Z_LO, Z_HI = 0.4, 0.8
N_CH = 64
RA_RANGE = (142.0, 178.0)
DEC_RANGE = (-12.0, 12.0)
# Match ``misc/yamamoto/rsd/sim_healpix_rsd_yamamoto.py --bigger-box``.
_BIGGER_BOX_ANG_SCALE = 1.3
_BIGGER_BOX_Z_SCALE = 2.0
_TRANS_RESCALE = float(np.sqrt(3.0 * 2.0))
PS_DOWNRES_T, PS_DOWNRES_R = 2.0 * _TRANS_RESCALE, 1.0
HP_NSIDE = 128

# Median rel-error limits (median over three seeds, then per multipole).
_FAR_IDENTITY_LIMITS = {0: 0.06, 2: 0.18, 4: 0.45}
_LC_SMOOTH_LIMITS = {0: 0.25, 2: 0.55, 4: 0.85}


def _survey_geometry(*, bigger_box: bool = False):
    """Return ``(z_lo, z_hi, ra_range, dec_range)``; bigger-box matches the misc script."""
    z_lo, z_hi = float(Z_LO), float(Z_HI)
    ra = (float(RA_RANGE[0]), float(RA_RANGE[1]))
    dec = (float(DEC_RANGE[0]), float(DEC_RANGE[1]))
    if not bigger_box:
        return z_lo, z_hi, ra, dec
    z_mid = 0.5 * (z_lo + z_hi)
    z_half = 0.5 * (z_hi - z_lo) * _BIGGER_BOX_Z_SCALE
    z_lo, z_hi = z_mid - z_half, z_mid + z_half
    ra_c = 0.5 * (ra[0] + ra[1])
    ra_h = 0.5 * (ra[1] - ra[0]) * _BIGGER_BOX_ANG_SCALE
    dec_c = 0.5 * (dec[0] + dec[1])
    dec_h = 0.5 * (dec[1] - dec[0]) * _BIGGER_BOX_ANG_SCALE
    return z_lo, z_hi, (ra_c - ra_h, ra_c + ra_h), (dec_c - dec_h, dec_c + dec_h)


def _median_rel_diff(a, b) -> float:
    a_np = np.asarray(a, dtype=float)
    b_np = np.asarray(b, dtype=float)
    mask = np.isfinite(a_np) & np.isfinite(b_np) & (np.abs(b_np) > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.median(np.abs(a_np[mask] / b_np[mask] - 1.0)))


def _default_k_in(mock) -> np.ndarray:
    return propose_k_in(mock.k1dbins, n=K_IN_N)


def _k1dbins_from_box(mock, n_bins: int = 8) -> np.ndarray:
    kmode = np.asarray(mock.k_mode, dtype=float)
    kpos = kmode[kmode > 0]
    dk = float(kpos.min())
    kmax = 0.7 * float(kpos.max())
    n_bins = max(2, min(n_bins, int(np.floor(kmax / dk))))
    return np.linspace(dk, kmax, n_bins + 1)


def _make_periodic_rsd_box(*, seed: int, los_observer) -> MockSimulation:
    mock = MockSimulation(
        density="gaussian",
        kaiser_rsd=True,
        rsd_from_field=True,
        parallel_plane=False,
        sigma_v_1=0.0,
        tracer_bias_1=1.0,
        mean_amp_1=1.0,
        model_k_from_field=True,
        seed=int(seed),
        los="endpoint",
        los_observer=los_observer,
    )
    mock.box_len = np.asarray(BOX_LEN, dtype=float)
    mock.box_ndim = np.asarray(BOX_NDIM, dtype=int)
    mock.propagate_field_k_to_model()
    mock.box_origin = np.asarray(los_observer, dtype=float)
    mock.k1dbins = np.linspace(0.04, 0.28, 8)
    mock.field_1 = mock.mock_tracer_field_1
    mock.weights_1 = np.ones_like(mock.field_1, dtype=float)
    return mock


def _make_lightcone_rsd(seed: int, *, bigger_box: bool = False) -> MockSimulation:
    z_lo, z_hi, ra_range, dec_range = _survey_geometry(bigger_box=bigger_box)
    # Keep channel spacing when Δz stretches (``--bigger-box`` uses ×2 in z).
    n_ch = int(round(N_CH * (z_hi - z_lo) / (Z_HI - Z_LO)))
    nu = np.linspace(redshift_to_freq(z_hi), redshift_to_freq(z_lo), n_ch)
    mock = MockSimulation(
        nu=nu,
        hp_nside=HP_NSIDE,
        ra_range=ra_range,
        dec_range=dec_range,
        density="gaussian",
        kaiser_rsd=True,
        rsd_from_field=True,
        parallel_plane=False,
        sigma_v_1=0.0,
        tracer_bias_1=1.0,
        mean_amp_1=1.0,
        seed=int(seed),
        model_k_from_field=True,
        include_beam=[False, False],
        include_sky_sampling=[False, False],
        compensate=[False, False],
        los="global",
    )
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.downres_factor_transverse = PS_DOWNRES_T
    mock.downres_factor_radial = PS_DOWNRES_R
    mock.get_enclosing_box()
    mock.field_1 = mock.mock_tracer_field_1
    mock.weights_1 = np.asarray(mock.counts_in_box, dtype=float)
    mock.k1dbins = _k1dbins_from_box(mock)
    return mock


def _far_observer_obs():
    L = np.asarray(BOX_LEN, dtype=float)
    R = float(R_OVER_L) * float(np.max(L))
    return (-0.5 * L[0], -0.5 * L[1], R)


def _measure_endpoint(mock: MockSimulation, los_observer):
    mock.los = "endpoint"
    mock.los_observer = np.asarray(los_observer, dtype=float)
    return mock.measure_multipoles(which="auto_1", k1dbins=mock.k1dbins, ells=ELLS)


def _identity_window_prediction(mock: MockSimulation, *, los_observer):
    out = predict_windowed_multipoles(
        mock,
        continuous="identity",
        k_in=_default_k_in(mock),
        ells=ELLS,
        nmu=NMU,
        los="endpoint",
        los_observer=los_observer,
        n_los_samples=N_LOS_SAMPLES,
        n_k_eval=N_K_EVAL,
    )
    return out["P_ell"]


def _smooth_window_prediction(mock: MockSimulation, *, los_observer):
    out = predict_windowed_multipoles(
        mock,
        continuous="smooth",
        k_in=_default_k_in(mock),
        ells=ELLS,
        nmu=NMU,
        los="endpoint",
        los_observer=los_observer,
        n_los_samples=N_LOS_SAMPLES,
        n_window_bins=N_WINDOW_BINS,
        n_fftlog=N_FFTLOG,
        n_k_eval=N_K_EVAL,
    )
    return out["P_ell"]


def _median_over_seeds(seed_rels: list[dict[int, float]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for ell in ELLS:
        vals = [r[ell] for r in seed_rels if np.isfinite(r[ell])]
        out[ell] = float(np.median(vals)) if vals else float("nan")
    return out


def _run_seeds_serial(build_and_compare):
    rels = []
    for seed in SEEDS:
        rels.append(build_and_compare(seed))
    return _median_over_seeds(rels)


def test_far_observer_rsd_identity_window():
    """Far-observer periodic box: Yamamoto mock vs identity discrete-shell W."""

    def one_seed(seed: int) -> dict[int, float]:
        obs = _far_observer_obs()
        mock = _make_periodic_rsd_box(seed=seed, los_observer=obs)
        meas = _measure_endpoint(mock, obs)
        P_win = _identity_window_prediction(mock, los_observer=obs)
        return {e: _median_rel_diff(meas.P_ell[e], P_win[e]) for e in ELLS}

    med = _run_seeds_serial(one_seed)
    for ell, limit in _FAR_IDENTITY_LIMITS.items():
        assert med[ell] < limit, "ell=%d median rel=%s" % (ell, med[ell])


def _lightcone_smooth_window_medians(*, bigger_box: bool = False) -> dict[int, float]:
    def one_seed(seed: int) -> dict[int, float]:
        mock = _make_lightcone_rsd(seed, bigger_box=bigger_box)
        mock.apply_taper_to_field(1, axis=(0, 1, 2))
        mock.weights_grid_1 = mock.weights_1
        obs = np.asarray(mock.box_origin, dtype=float)
        meas = _measure_endpoint(mock, obs)
        P_win = _smooth_window_prediction(mock, los_observer=obs)
        return {e: _median_rel_diff(meas.P_ell[e], P_win[e]) for e in ELLS}

    return _run_seeds_serial(one_seed)


def test_lightcone_rsd_smooth_window():
    """HealPix lightcone with taper: Yamamoto mock vs smooth discrete-shell W."""
    med = _lightcone_smooth_window_medians(bigger_box=False)
    for ell, limit in _LC_SMOOTH_LIMITS.items():
        assert med[ell] < limit, "ell=%d median rel=%s" % (ell, med[ell])


def test_lightcone_rsd_smooth_window_bigger_box():
    """Same as ``test_lightcone_rsd_smooth_window`` with misc ``--bigger-box`` geometry."""
    med = _lightcone_smooth_window_medians(bigger_box=True)
    for ell, limit in _LC_SMOOTH_LIMITS.items():
        assert med[ell] < limit, "ell=%d median rel=%s" % (ell, med[ell])
