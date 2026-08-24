"""
Yamamoto estimator + windowed multipole model on one object.

:class:`MultipolePowerSpectrum` is the path-(3)+(4) analogue of
:class:`~meer21cm.power.PowerSpectrum`: same survey I/O and lightcone
gridding, local-LOS multipoles from
:meth:`~meer21cm.estimator.FieldPowerSpectrum.measure_multipoles`, and a
windowed theory vector from :class:`~meer21cm.multipole_model.WindowedMultipoleModel`.

The expensive window operator (selection, beam, map sampling, MAS) is
**not** built on access.  Call :meth:`run_window_matrix` (serial) or
:meth:`get_arg_list_for_window_columns` + :func:`run_window_column` +
:meth:`accumulate_window_columns` (external multiprocessing / MPI).
Changing beam or survey settings flags the cached matrix stale; theory
:math:`P(k,\\mu)` updates automatically and is applied as ``W @ P_ell(k_in)``.

Mesh windows map isotropic :math:`P_0(k_{\\mathrm{in}})` only
(``ells_in = (0,)``).  Fitting Kaiser :math:`P_2,P_4` uses
``window='smooth'``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import windows

from .estimator import FieldPowerSpectrum, MultipoleMeasurement
from .grid import LightconeGriddingMixin, fourier_window_for_assignment
from .multipole_model import (
    SmoothWindowEstimator,
    WindowedMultipoleModel,
    map_sampling_mode_scale,
    propose_k_in,
    run_smooth_window_realization,
)
from .power import _FieldModelGlueMixin
from .util import tagging
from .window import (
    DiscreteShellWindowMatrix,
    accumulate_mesh_window_matrices,
    build_mesh_window_mas_out,
    build_mesh_window_matrix,
    list_mesh_window_columns,
    propose_mesh_k_in,
    run_mesh_window_columns,
)

logger = logging.getLogger(__name__)

WindowKind = Literal["mesh", "smooth"]

_WINDOW_NOT_BUILT = (
    "Window matrix has not been built. Call run_window_matrix() or "
    "get_arg_list_for_window_columns() then accumulate_window_columns()."
)
_WINDOW_STALE = (
    "Cached window matrix is stale (beam or survey settings changed). "
    "Call run_window_matrix() to rebuild."
)

_WINDOW_WORKER: MultipolePowerSpectrum | None = None
_INNER_TAPER_B5_WARNED = False


def init_window_column_worker(mps: MultipolePowerSpectrum) -> None:
    """Pool initializer: store the parent object for :func:`run_window_column`."""
    global _WINDOW_WORKER
    _WINDOW_WORKER = mps


def run_window_column(kwargs: dict, columns) -> DiscreteShellWindowMatrix:
    """
    Pickleable worker for one mesh (or smooth) window column chunk.

    ``kwargs`` is one dict from
    :meth:`MultipolePowerSpectrum.get_arg_list_for_window_columns`.
    For the mesh path with beam kernels, pass
    ``initializer=init_window_column_worker, initargs=(mps,)`` so the
    worker can reuse the parent's setup; ``kwargs`` may then omit
    unpickleable callables and ``run_window_column`` uses the stored
    object.
    """
    if kwargs.get("use_worker_object"):
        if _WINDOW_WORKER is None:
            raise RuntimeError(
                "run_window_column needs init_window_column_worker(mps) "
                "when use_worker_object is True"
            )
        return _WINDOW_WORKER._fill_window_columns(columns)
    kind = str(kwargs.get("kind", "mesh")).lower()
    if kind == "mesh":
        return run_mesh_window_columns(kwargs["build_kwargs"], columns)
    raise ValueError(f"unsupported window kind for column worker: {kind!r}")


class MultipolePowerSpectrum(
    _FieldModelGlueMixin,
    LightconeGriddingMixin,
    FieldPowerSpectrum,
    WindowedMultipoleModel,
):
    """
    Combined Yamamoto field estimator and windowed multipole model.

    Same constructor pattern as :class:`~meer21cm.power.PowerSpectrum`
    (Specification survey kwargs, then box / field / model).  Defaults
    differ: ``los='endpoint'``, and beam / sky sampling / MAS compensation
    are **off** on the legacy 3D ``get_model_power_i`` path — those
    operators live in the window matrix.

    Parameters
    ----------
    window : {'mesh', 'smooth'}, default 'mesh'
        Window backend.  ``'mesh'`` is the lightcone FFT window
        (:func:`~meer21cm.window.build_mesh_window_mas_out` when there is
        no post-deposit taper; inner-mode
        :func:`~meer21cm.window.build_mesh_window_matrix` when
        ``window_taper_axes`` is set).  Theory input is the isotropic
        monopole.  ``'smooth'`` is the Hankel / Wigner path-(4) matrix
        (Kaiser :math:`P_\\ell` in, including optional wide-angle resum).
    beam_n_mu, beam_at_input, beam_leg_scale, ...
        Forwarded to the mesh B5 beam model.  Production defaults match
        tests 06/07 (``n_mu=4``, additive :math:`\\kappa=0`).
    """

    _window_cache_names = ("_window_matrix_obj", "_window_matrix_raw")

    def __init__(
        self,
        field_1=None,
        box_len=None,
        weights_field_1=None,
        weights_grid_1=None,
        mean_center_1=False,
        unitless_1=False,
        field_2=None,
        weights_field_2=None,
        weights_grid_2=None,
        mean_center_2=False,
        unitless_2=False,
        k1dbins=None,
        kmode=None,
        mumode=None,
        tracer_bias_1=1.0,
        sigma_v_1=0.0,
        tracer_bias_2=None,
        sigma_v_2=0.0,
        include_beam=None,
        fog_profile="lorentz",
        cross_coeff=1.0,
        model_k_from_field=True,
        mean_amp_1=1.0,
        mean_amp_2=1.0,
        sampling_resol=None,
        include_sky_sampling=None,
        downres_factor_transverse=1.2,
        downres_factor_radial=2.0,
        init_box_from_map_data=False,
        box_buffkick=5,
        compensate=None,
        taper_func=windows.blackmanharris,
        kaiser_rsd=True,
        grid_scheme="nnb",
        interlace_shift=0.0,
        num_particle_per_pixel=1,
        seed=None,
        kperpbins=None,
        kparabins=None,
        flat_sky=False,
        flat_sky_padding=[0, 0, 0],
        k1dweights=None,
        los="endpoint",
        los_observer=None,
        window="mesh",
        window_ells=(0, 2, 4),
        beam_at_input=True,
        beam_n_mu=4,
        beam_n_phi=1,
        beam_diag_correction=True,
        beam_leg_scale=False,
        beam_in_kernel=False,
        beam_ylm=False,
        beam_ylm_lmax=2,
        theory_nmu=64,
        n_k_in=80,
        window_taper_axes=(),
        wide_angle=False,
        **params,
    ):
        if seed is None:
            seed = np.random.randint(0, 2**32)
        self.seed = seed
        self.num_particle_per_pixel = num_particle_per_pixel
        if field_1 is None:
            if "box_ndim" in params.keys():
                field_1 = np.ones(params["box_ndim"])
            else:
                field_1 = np.ones([1, 1, 1])
        if box_len is None:
            box_len = np.array([1, 1, 1])
        if include_beam is None:
            include_beam = [False, False]
        if include_sky_sampling is None:
            include_sky_sampling = [False, False]
        if compensate is None:
            compensate = [False, False]
        WindowedMultipoleModel.__init__(
            self,
            kmode=kmode,
            mumode=mumode,
            tracer_bias_1=tracer_bias_1,
            sigma_v_1=sigma_v_1,
            tracer_bias_2=tracer_bias_2,
            sigma_v_2=sigma_v_2,
            include_beam=include_beam,
            fog_profile=fog_profile,
            cross_coeff=cross_coeff,
            weights_field_1=weights_field_1,
            weights_field_2=weights_field_2,
            weights_grid_1=weights_grid_1,
            weights_grid_2=weights_grid_2,
            mean_amp_1=mean_amp_1,
            mean_amp_2=mean_amp_2,
            sampling_resol=sampling_resol,
            include_sky_sampling=include_sky_sampling,
            kaiser_rsd=kaiser_rsd,
            compensate=compensate,
            window_ells=window_ells,
            **params,
        )
        self.model_k_from_field = model_k_from_field
        FieldPowerSpectrum.__init__(
            self,
            field_1,
            box_len,
            weights_1=weights_grid_1,
            mean_center_1=mean_center_1,
            unitless_1=unitless_1,
            field_2=field_2,
            weights_2=weights_grid_2,
            mean_center_2=mean_center_2,
            unitless_2=unitless_2,
            los=los,
            los_observer=los_observer,
            _skip_specification=True,
        )
        if model_k_from_field:
            self.propagate_field_k_to_model()
        self.k1dbins = k1dbins
        self.kperpbins = kperpbins
        self.kparabins = kparabins
        self.downres_factor_transverse = downres_factor_transverse
        self.downres_factor_radial = downres_factor_radial
        init_attr = [
            "_rot_mat_sky_to_box",
            "_pix_coor_in_cartesian",
            "_counts_in_box",
            "_flat_sky",
            "_box_origin",
            "_box_voxel_redshift",
        ]
        for attr in init_attr:
            setattr(self, attr, None)
        self.upgrade_sampling_from_gridding = False
        self.box_buffkick = box_buffkick
        self.taper_func = taper_func
        if init_box_from_map_data:
            self.get_enclosing_box()
        self.grid_scheme = grid_scheme
        self.interlace_shift = interlace_shift
        self.flat_sky = flat_sky
        self.flat_sky_padding = flat_sky_padding
        self.k1dweights = k1dweights

        self._window_kind = str(window).lower()
        if self._window_kind not in ("mesh", "smooth"):
            raise ValueError("window must be 'mesh' or 'smooth'")
        self._beam_at_input = bool(beam_at_input)
        self._beam_n_mu = int(beam_n_mu)
        self._beam_n_phi = int(beam_n_phi)
        self._beam_diag_correction = bool(beam_diag_correction)
        self._beam_leg_scale = bool(beam_leg_scale)
        self._beam_in_kernel = bool(beam_in_kernel)
        self._beam_ylm = bool(beam_ylm)
        self._beam_ylm_lmax = int(beam_ylm_lmax)
        self.theory_nmu = int(theory_nmu)
        self.n_k_in = int(n_k_in)
        self._window_taper_axes = tuple(int(a) for a in window_taper_axes)
        self.wide_angle = bool(wide_angle)
        self._window_k_in: NDArray[np.floating] | None = None
        self._window_stale = False
        self._data_multipoles = None
        self._model_multipoles = None

    def clean_cache(self, attr):
        """Flag the window matrix stale instead of wiping it."""
        names = list(attr)
        kept = []
        hit_window = False
        for name in names:
            if name in self._window_cache_names or name == "_window_matrix":
                hit_window = True
            else:
                kept.append(name)
        if hit_window and getattr(self, "_window_matrix_obj", None) is not None:
            self._window_stale = True
        super().clean_cache(kept)

    def _warn_window_status(self) -> None:
        if self._window_matrix_obj is None and self._window_matrix_raw is None:
            logger.warning(_WINDOW_NOT_BUILT)
            return
        if self._window_stale:
            logger.warning(_WINDOW_STALE)

    @property
    @tagging("beam", "box", "nu", "window")
    def window_matrix(self) -> NDArray[np.floating] | None:
        """Dense window matrix, or ``None`` if :meth:`run_window_matrix` has not been called."""
        self._warn_window_status()
        return WindowedMultipoleModel.window_matrix.fget(self)

    @property
    def window_kind(self) -> str:
        """``'mesh'`` or ``'smooth'``."""
        return self._window_kind

    @window_kind.setter
    def window_kind(self, value: str) -> None:
        kind = str(value).lower()
        if kind not in ("mesh", "smooth"):
            raise ValueError("window must be 'mesh' or 'smooth'")
        self._window_kind = kind
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    def beam_n_mu(self) -> int:
        return self._beam_n_mu

    @beam_n_mu.setter
    def beam_n_mu(self, value: int) -> None:
        self._beam_n_mu = int(value)
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    def beam_at_input(self) -> bool:
        return self._beam_at_input

    @beam_at_input.setter
    def beam_at_input(self, value: bool) -> None:
        self._beam_at_input = bool(value)
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    def window_taper_axes(self) -> tuple[int, ...]:
        return self._window_taper_axes

    @window_taper_axes.setter
    def window_taper_axes(self, value) -> None:
        self._window_taper_axes = tuple(int(a) for a in value)
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    @tagging("box", "field_1")
    def data_multipoles(self) -> MultipoleMeasurement:
        """Yamamoto (or global) data multipoles; built on first access."""
        if self._data_multipoles is None:
            self._data_multipoles = self.measure_multipoles(
                which="auto_1",
                k1dbins=self.k1dbins,
                ells=self.window_ells,
                k1dweights=self.k1dweights,
            )
        return self._data_multipoles

    @property
    @tagging("cosmo_model", "nu", "kmode", "mumode", "tracer_1", "rsd")
    def model_multipoles(self) -> dict[str, Any] | None:
        """Windowed theory multipoles ``W @ P_ell(k_in)`` from current ``power_kmu``."""
        self._warn_window_status()
        if self._window_matrix_obj is None and self._window_matrix_raw is None:
            return None
        if self._model_multipoles is None:
            self._model_multipoles = self.get_model_multipoles(
                which="auto_1",
                nmu=self.theory_nmu,
                apply_window=True,
            )
        return self._model_multipoles

    def _resolve_k_in(self) -> NDArray[np.floating]:
        if self._window_k_in is not None:
            return np.asarray(self._window_k_in, dtype=float)
        if self.k1dbins is None:
            raise ValueError("k1dbins is required before building the window matrix")
        if self.window_kind == "mesh":
            return propose_mesh_k_in(self, n=self.n_k_in)
        return propose_k_in(self.k1dbins, n=self.n_k_in)

    def _renorm_weights(self) -> NDArray[np.floating]:
        w = getattr(self, "weights_1", None)
        if w is None:
            w = getattr(self, "weights_grid_1", None)
        if w is None:
            return np.ones(self.box_ndim, dtype=float)
        return np.asarray(w, dtype=float)

    def _selection_weights(self) -> NDArray[np.floating]:
        w = getattr(self, "weights_field_1", None)
        if w is None:
            w = self._renorm_weights()
        return np.asarray(w, dtype=float)

    def _taper_cube(self, shape) -> NDArray[np.floating] | None:
        axes = self._window_taper_axes
        if not axes:
            return None
        taper = np.ones(shape, dtype=float)
        ndim = np.asarray(shape, dtype=int)
        for ax in axes:
            t = np.asarray(self.taper_func(int(ndim[ax])), dtype=float)
            slicer = [None, None, None]
            slicer[int(ax)] = slice(None)
            taper = taper * t[tuple(slicer)]
        return taper

    def _window_mode_scale(self) -> NDArray[np.floating] | None:
        if not getattr(self, "has_resol", False):
            return None
        try:
            return map_sampling_mode_scale(self, z_resolved=True)
        except Exception:
            logger.debug("map_sampling_mode_scale unavailable; mode_scale=None")
            return None

    def _use_inner_mode_taper(self) -> bool:
        """Post-deposit taper is inner-mode (04), not MAS-out × (T×NGP)."""
        return bool(self._window_taper_axes)

    def _inner_mode_scale(self) -> NDArray[np.floating]:
        w_mas2 = fourier_window_for_assignment(self.box_ndim, self.grid_scheme) ** 2
        samp = self._window_mode_scale()
        if samp is None:
            return w_mas2
        return w_mas2 * samp

    def _inner_mode_taper_weights(self) -> NDArray[np.floating]:
        """Estimator :math:`w(x)`: CIC counts after ``apply_taper_to_field``.

        ``window_taper_axes`` selects this inner-mode branch; it does not
        remultiply :math:`T`.  Call :meth:`apply_taper_to_field` (or set
        ``weights_1`` to already-tapered CIC) before building the window.
        """
        return np.asarray(self._renorm_weights(), dtype=float)

    def _warn_inner_taper_no_b5(self) -> None:
        global _INNER_TAPER_B5_WARNED
        if getattr(self, "sigma_beam_ch", None) is None:
            return
        if _INNER_TAPER_B5_WARNED:
            return
        _INNER_TAPER_B5_WARNED = True
        logger.warning(
            "Post-deposit taper uses inner-mode (tapered CIC weights); "
            "B5 beam_at_input is not applied on this path."
        )

    def _use_mas_out(self) -> bool:
        if self._use_inner_mode_taper():
            return False
        pix = getattr(self, "_pix_coor_in_cartesian", None)
        if pix is None:
            return False
        return str(self.grid_scheme).lower() != "nnb"

    def _use_beam(self) -> bool:
        return (
            bool(self.beam_at_input)
            and getattr(self, "sigma_beam_ch", None) is not None
        )

    def _fill_window_columns(self, columns=None) -> DiscreteShellWindowMatrix:
        k_in = self._resolve_k_in()
        ells = self.window_ells
        if self.window_kind == "smooth":
            swe = SmoothWindowEstimator.from_power_spectrum(
                self,
                ells=ells,
                wide_angle=self.wide_angle,
            )
            if columns is not None:
                raise ValueError(
                    "smooth window has no mesh columns; use run_window_matrix() "
                    "or SmoothWindowEstimator.get_arg_list_for_seeds"
                )
            swe.accumulate([swe.run_one(0)])
            return swe.build_window_matrix(k_in, continuous="smooth")

        mode_scale = self._window_mode_scale()
        if self._use_inner_mode_taper():
            # Data: CIC then T on weights (apply_taper_to_field).  MAS-out
            # with T×NGP is the wrong operator when T varies on CIC scales.
            self._warn_inner_taper_no_b5()
            weights = self._inner_mode_taper_weights()
            return build_mesh_window_matrix(
                self,
                k_in,
                weights=weights,
                ells=ells,
                mode_scale=self._inner_mode_scale(),
                renorm_weights=weights,
                columns=columns,
            )

        if self._use_mas_out():
            return build_mesh_window_mas_out(
                self,
                k_in,
                renorm_weights=self._renorm_weights(),
                ells=ells,
                mode_scale=mode_scale,
                beam_at_input=self._use_beam(),
                beam_in_kernel=self._beam_in_kernel and not self._use_beam(),
                beam_n_mu=self._beam_n_mu,
                beam_n_phi=self._beam_n_phi,
                beam_diag_correction=self._beam_diag_correction,
                beam_leg_scale=self._beam_leg_scale,
                beam_ylm=self._beam_ylm,
                beam_ylm_lmax=self._beam_ylm_lmax,
                columns=columns,
            )

        weights = self._selection_weights()
        return build_mesh_window_matrix(
            self,
            k_in,
            weights=weights,
            ells=ells,
            mode_scale=mode_scale,
            renorm_weights=self._renorm_weights(),
            columns=columns,
        )

    def run_window_matrix(self, columns=None) -> DiscreteShellWindowMatrix:
        """
        Build the window matrix in this process (serial).

        For column-parallel mesh builds, use
        :meth:`get_arg_list_for_window_columns` and
        :meth:`accumulate_window_columns` instead.
        """
        mat = self._fill_window_columns(columns=columns)
        self.set_window_matrix(mat)
        self._window_k_in = np.asarray(mat.k_in, dtype=float)
        self._window_stale = False
        self._model_multipoles = None
        return mat

    def _list_mesh_columns(self):
        k_in = self._resolve_k_in()
        n_in = int(np.asarray(k_in).size)
        if (
            self.window_kind != "mesh"
            or not self._use_beam()
            or not self._use_mas_out()
        ):
            return list(range(n_in))
        from .multipole_model import beam_edge_cell_mass, beam_input_cell_kernels

        mode_scale = self._window_mode_scale()
        edge = beam_edge_cell_mass(self)
        gi, kernel = beam_input_cell_kernels(
            self,
            k_in,
            n_mu=self._beam_n_mu,
            n_phi=self._beam_n_phi,
            mode_scale=mode_scale,
            cell_mass=edge,
        )
        k_mode = np.asarray(self.k_mode, dtype=float).ravel()
        k_in_np = np.asarray(k_in, dtype=float)
        edges = np.concatenate(([0.0], 0.5 * (k_in_np[:-1] + k_in_np[1:]), [np.inf]))
        in_shell = [
            (k_mode >= edges[j]) & (k_mode < edges[j + 1]) for j in range(len(k_in_np))
        ]
        return list_mesh_window_columns(
            n_in,
            in_group_index=gi,
            in_bin_weights=kernel,
            in_shell=in_shell,
        )

    def get_arg_list_for_window_columns(
        self, n_chunks: int | None = None
    ) -> list[tuple[dict, Any]]:
        """
        Pickleable ``(kwargs, columns)`` tuples for external mapping.

        Mesh without beam kernels: ``kwargs`` is a dict for
        :func:`~meer21cm.window.run_mesh_window_columns`.  Mesh with B5
        beam kernels sets ``use_worker_object=True`` so the pool must be
        started with :func:`init_window_column_worker`.
        """
        if self.window_kind == "smooth":
            swe = SmoothWindowEstimator.from_power_spectrum(
                self, ells=self.window_ells, wide_angle=self.wide_angle
            )
            return swe.get_arg_list_for_seeds([0])

        cols = self._list_mesh_columns()
        n = max(1, int(n_chunks) if n_chunks is not None else len(cols) or 1)
        n = min(n, max(1, len(cols)))
        chunks: list[list] = []
        if cols:
            size, extra = divmod(len(cols), n)
            idx = 0
            for i in range(n):
                take = size + (1 if i < extra else 0)
                if take:
                    chunks.append(list(cols[idx : idx + take]))
                    idx += take
        if self._use_inner_mode_taper():
            self._warn_inner_taper_no_b5()
            weights = self._inner_mode_taper_weights()
            build_kwargs = dict(
                ps=self,
                k_in=self._resolve_k_in(),
                weights=weights,
                ells=self.window_ells,
                mode_scale=self._inner_mode_scale(),
                renorm_weights=weights,
            )
            return [
                ({"kind": "mesh", "build_kwargs": build_kwargs}, ch) for ch in chunks
            ]
        if self._use_beam() and self._use_mas_out():
            kwargs = {"use_worker_object": True, "kind": "mesh"}
            return [(dict(kwargs), ch) for ch in chunks]
        k_in = self._resolve_k_in()
        weights = self._selection_weights()
        build_kwargs = dict(
            ps=self,
            k_in=k_in,
            weights=weights,
            ells=self.window_ells,
            mode_scale=self._window_mode_scale(),
            renorm_weights=self._renorm_weights(),
        )
        return [({"kind": "mesh", "build_kwargs": build_kwargs}, ch) for ch in chunks]

    def accumulate_window_columns(
        self, results: Sequence[DiscreteShellWindowMatrix]
    ) -> DiscreteShellWindowMatrix:
        """Sum column chunks and attach the window matrix."""
        mat = accumulate_mesh_window_matrices(list(results))
        self.set_window_matrix(mat)
        self._window_k_in = np.asarray(mat.k_in, dtype=float)
        self._window_stale = False
        self._model_multipoles = None
        return mat

    @classmethod
    def from_power_spectrum(cls, ps, **kwargs) -> MultipolePowerSpectrum:
        """
        Copy box, field, weights, bins, LOS, and lightcone state from a
        PowerSpectrum-like object (including :class:`~meer21cm.mock.MockSimulation`).
        """
        weights_grid_1 = kwargs.pop(
            "weights_grid_1",
            getattr(ps, "weights_grid_1", getattr(ps, "weights_1", None)),
        )
        weights_grid_2 = kwargs.pop(
            "weights_grid_2",
            getattr(ps, "weights_grid_2", getattr(ps, "weights_2", None)),
        )
        spec: dict[str, Any] = {}
        skymap = getattr(ps, "skymap", None)
        if skymap is not None and "skymap" not in kwargs:
            spec["skymap"] = skymap
        for key in (
            "nu",
            "sigma_beam_ch",
            "beam_model",
            "beam_unit",
            "ra_range",
            "dec_range",
            "precision",
            "batch_number",
            "mean_amp_1",
            "mean_amp_2",
            "sigma_v_1",
            "sigma_v_2",
        ):
            if key in kwargs:
                continue
            try:
                val = getattr(ps, key)
            except Exception:
                continue
            if val is not None:
                spec[key] = val
        obj = cls(
            field_1=ps.field_1,
            box_len=ps.box_len,
            weights_field_1=getattr(ps, "weights_field_1", None),
            weights_grid_1=weights_grid_1,
            mean_center_1=getattr(ps, "mean_center_1", False),
            unitless_1=getattr(ps, "unitless_1", False),
            field_2=getattr(ps, "field_2", None),
            weights_field_2=getattr(ps, "weights_field_2", None),
            weights_grid_2=weights_grid_2,
            mean_center_2=getattr(ps, "mean_center_2", False),
            unitless_2=getattr(ps, "unitless_2", False),
            k1dbins=getattr(ps, "k1dbins", None),
            tracer_bias_1=getattr(ps, "tracer_bias_1", 1.0),
            tracer_bias_2=getattr(ps, "tracer_bias_2", None),
            kaiser_rsd=kwargs.pop("kaiser_rsd", getattr(ps, "kaiser_rsd", True)),
            grid_scheme=kwargs.pop("grid_scheme", getattr(ps, "grid_scheme", "nnb")),
            k1dweights=getattr(ps, "k1dweights", None),
            los=kwargs.pop("los", getattr(ps, "los", "endpoint")),
            los_observer=kwargs.pop("los_observer", getattr(ps, "los_observer", None)),
            **spec,
            **kwargs,
        )
        obj.box_ndim = ps.box_ndim
        for name in (
            "downres_factor_transverse",
            "downres_factor_radial",
            "kperpbins",
            "kparabins",
        ):
            val = getattr(ps, name, None)
            if val is not None:
                setattr(obj, name, val)
        for name in (
            "_pix_coor_in_cartesian",
            "_counts_in_box",
            "_box_origin",
            "_box_voxel_redshift",
            "_rot_mat_sky_to_box",
            "_flat_sky",
        ):
            val = getattr(ps, name, None)
            if val is not None:
                setattr(obj, name, val)
        if getattr(ps, "has_resol", False):
            obj.has_resol = True
        if getattr(ps, "k1dweights", None) is not None:
            obj.k1dweights = ps.k1dweights
        w1 = getattr(ps, "weights_1", None)
        if w1 is not None:
            obj.weights_1 = w1
        return obj
