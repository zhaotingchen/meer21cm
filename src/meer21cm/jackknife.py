"""
This module contains the jackknife covariance estimation class and related functions.

In data analysis, you should have already used :class:`meer21cm.power.PowerSpectrum`
to calculate the power spectrum of the data.
In that case, you can pass the :class:`meer21cm.power.PowerSpectrum` instance as an
input to :class:`meer21cm.jackknife.JackknifeCovariance`.
The jackknife class then takes into account the settings of the power spectrum
instance (gridding, beam, weights, tapering, k-binning, etc.), removes one sky
patch at a time, recomputes the power spectrum of each jackknife realisation,
and estimates the covariance matrix of the power spectrum from the scatter
of the realisations.

The survey volume is split into ``ra_patch_num x dec_patch_num x los_patch_num``
patches. The map patches are computed with
:meth:`meer21cm.dataanalysis.Specification.get_jackknife_patches`,
and the galaxies are assigned to the same patches with
:meth:`meer21cm.dataanalysis.Specification.get_gal_patch_labels`.
Along the line-of-sight, the patches are defined as bins in frequency for the
HI map, while the galaxy positions are stored as redshifts;
``get_gal_patch_labels`` converts each galaxy redshift to its 21cm frequency,
:math:`\\nu_g = f_{21} / (1 + z_g)`, taken from the :attr:freq_gal property,
and digitizes it into the same frequency bins, so that map voxels and galaxies 
belonging to the same comoving slab are always removed together. 
Note that bins that are linear in frequency are not linear in
redshift, so digitizing the galaxy redshifts into linear z-bins would not
match the map patches; the line-of-sight binning is always done in frequency.
You can specify the line-of-sight range either as ``nu_range`` (in Hz) or as
``z_range``; the two are mutually exclusive and internally everything is
converted to frequency.

Two important implementation details:

1. The enclosing rectangular box is computed in each realisation before
   the jackknife mask is applied, using the full ``W_HI`` window and the same
   ``seed`` as the input instance. This guarantees that every realisation
   (and the data measurement itself) lives on the identical Cartesian grid and
   k-modes. The jackknife masking is applied by zeroing the map data 
   and the pixel weights ``w_HI`` inside the removed patch; 
   the gridded weights returned by the regridding then automatically define 
   the jackknifed window on the Cartesian grid. ``W_HI`` itself is left untouched, 
   because the pixel coordinates used by the gridding routines are tied to the 
   ``W_HI`` footprint used to build the box.

2. Every jackknife realisation is computed on a fresh
   :class:`meer21cm.power.PowerSpectrum` instance constructed from the attributes
   of the input instance (following the same pattern as
   :class:`meer21cm.transfer.TransferFunction`). The input instance is never
   modified, so there is no conflict between the actual data measurement and
   the covariance calculation.

The calculation of the realisations supports parallelisation, either with
``multiprocessing`` or with MPI via :class:`mpi4py.futures.MPIPoolExecutor`.

A minimum example:

.. code-block:: python

    >>> from meer21cm.jackknife import JackknifeCovariance
    >>> jc = JackknifeCovariance(ps, ra_patch_num=4, dec_patch_num=3, los_patch_num=4)
    >>> results = jc.run(type="cross")  # or type="auto" for the HI auto-power
    >>> cov, mean_p1d = jc.get_covariance(results)
"""
import numpy as np
from multiprocessing import Pool

from .power import PowerSpectrum
from .util import f_21
from .transfer import required_attrs

# attributes of the input PowerSpectrum instance that are propagated
# to the per-realisation instances, on top of ``meer21cm.transfer.required_attrs``.
# note that ra_range and dec_range are not propagated:
# Specification.__init__ invokes trim_map_to_range, which would trim the map of
# the per-realisation instances to the ranges, whereas the input map data must be
# reproduced exactly as it is (the input instance may have been used with an
# untrimmed map, and trimming would change the enclosing box and the k-modes).
extra_required_attrs = [
    "seed",
    "downres_factor_radial",
    "downres_factor_transverse",
]


class JackknifeCovariance:
    """
    A class to estimate the covariance matrix of the 1D power spectrum
    via jackknife resampling of the survey volume.

    For each jackknife realisation, one patch is removed from the data:
    the HI map data and pixel weights ``w_HI`` are zeroed inside the patch,
    and the galaxies inside the patch are removed from the catalogue.
    The (jackknifed) map, weights and galaxy catalogue are then regridded
    onto the Cartesian box and the power spectrum is recomputed with the
    same settings as the input instance. A minimum example:

    .. code-block:: python

        >>> jc = JackknifeCovariance(ps, 4, 3, 4)
        >>> results = jc.run(type="cross")
        >>> cov, mean_p1d = jc.get_covariance(results)

    If ``type`` is ``"auto"``, each realisation returns the 1D auto-power of the
    HI map (``auto_power_3d_1`` binned to 1D). If ``type`` is ``"cross"``, each
    realisation returns the 1D HI x galaxy cross-power (``cross_power_3d`` binned
    to 1D), and optionally also the two auto-powers.

    Note that the input instance ``ps`` is never modified: each realisation
    is computed on a fresh instance built from the attributes of ``ps``
    (same pattern as :class:`meer21cm.transfer.TransferFunction`), so the data
    stored in ``ps`` is not affected by the covariance calculation.

    Parameters
    ----------
    ps: :class:`meer21cm.power.PowerSpectrum`
        The power spectrum instance used for the data measurement.
        It must have the map data (and the galaxy catalogue for ``type="cross"``)
        already read in, and the gridding settings
        (``downres_factor_transverse``, ``downres_factor_radial``,
        ``box_buffkick``, ``grid_scheme``, ...) already set.
        For ``type="cross"``, the instance must provide
        :meth:`meer21cm.dataanalysis.Specification.get_gal_patch_labels`.
    ra_patch_num: int
        The number of patch grids in the right ascension direction.
    dec_patch_num: int
        The number of patch grids in the declination direction.
    los_patch_num: int
        The number of patch grids in the line-of-sight direction.
        The line-of-sight is binned linearly in frequency.
    ra_range: tuple, default None
        The range of the right ascension in degrees. Default uses ``ps.ra_range``.
    dec_range: tuple, default None
        The range of the declination in degrees. Default uses ``ps.dec_range``.
    nu_range: tuple, default None
        The range of the frequency in Hz used for the line-of-sight split.
        Mutually exclusive with ``z_range``. If neither is provided, default uses
        ``[ps.nu.min() - ps.freq_resol/2, ps.nu.max() + ps.freq_resol/2]``,
        matching :meth:`meer21cm.dataanalysis.Specification.get_jackknife_patches`.
    z_range: tuple, default None
        The redshift range used for the line-of-sight split, as an alternative
        to ``nu_range``. It is converted internally to a frequency range via
        :math:`\\nu = f_{21}/(1+z)`, and the split is performed linearly in
        frequency for both the map and the galaxies.
    weights_scheme: str, default "gridded"
        The scheme used to assign the power spectrum weights of each
        realisation. Make sure this matches what you did for the data
        measurement.

        If ``"gridded"``, the field and grid weights of the HI map are the
        gridded jackknifed pixel weights ``w_HI`` (second output of
        ``grid_data_to_field``), and the galaxy weights are controlled by
        ``gal_grid_weights``.

        If ``"counts"``, the weights follow the cookbook convention:
        the HI grid weights are ``counts_in_box`` (inverse noise variance
        weighting) and the HI field weights are None; the galaxy field
        weights are the binary occupancy ``counts_in_box > 0`` and the
        galaxy grid weights are uniform. Note that in each realisation
        ``counts_in_box`` is recomputed with
        :meth:`meer21cm.power.PowerSpectrum.get_counts_in_box` after
        zeroing the pixel weights inside the removed patch, so that the
        patch carries zero weight.
    apply_taper: bool, default True
        Whether to apply the taper function ``ps.taper_func`` to the gridded
        weights of each field in each realisation
        (see :meth:`meer21cm.power.PowerSpectrum.apply_taper_to_field`).
        Note that, unlike in :class:`meer21cm.transfer.TransferFunction`,
        the gridded weights are recomputed from scratch in each realisation,
        so the taper must be re-applied here; make sure this setting matches
        what you did for the data measurement.
    taper_axis: tuple, default (2,)
        The box axes along which the taper is applied.
        Default is the z-axis, which is approximately the line-of-sight.
    gal_grid_weights: str, default "binary"
        Only used if ``weights_scheme="gridded"``.
        The scheme for the grid/field weights of the galaxy field.
        If ``"binary"``, the weights are the binary window
        ``(counts_in_box > 0) * (gridded jackknifed w_HI > 0)``,
        i.e. the default galaxy window of
        :meth:`meer21cm.power.PowerSpectrum.grid_gal_to_field` with the removed
        patch excluded. If ``"gridded"``, the gridded galaxy weights returned by
        ``grid_gal_to_field`` are used instead.
    min_patch_weight_fraction: float, default 0.0
        Patches whose fraction of the total map weight ``(w_HI * W_HI)``
        is not greater than this value are skipped when ``run`` is called with
        ``patch_indices=None``. The default skips only completely empty patches
        (e.g. patches fully outside the survey footprint), whose removal would
        produce a realisation identical to the data and artificially shrink
        the covariance.
    pool: str, default "multiprocessing"
        The pool to use for parallelisation.
        Can be "multiprocessing", "mpi", or "serial" (no parallelisation).
    num_process: int, default None
        The number of processes to use for parallelisation.
        If not provided, the number of processes is set to the number of
        cores available.
    """

    def __init__(
        self,
        ps: PowerSpectrum,
        ra_patch_num: int,
        dec_patch_num: int,
        los_patch_num: int,
        ra_range: tuple | None = None,
        dec_range: tuple | None = None,
        nu_range: tuple | None = None,
        z_range: tuple | None = None,
        weights_scheme: str = "counts",
        apply_taper: bool = True,
        taper_axis: tuple = (2,),
        gal_grid_weights: str = "binary",
        min_patch_weight_fraction: float = 0.0,
        pool: str = "multiprocessing",
        num_process: int | None = None,
    ):
        self.ps = ps
        self.ra_patch_num = ra_patch_num
        self.dec_patch_num = dec_patch_num
        self.los_patch_num = los_patch_num
        if ra_range is None:
            ra_range = ps.ra_range
        if dec_range is None:
            dec_range = ps.dec_range
        self.ra_range = tuple(ra_range)
        self.dec_range = tuple(dec_range)
        if (nu_range is not None) and (z_range is not None):
            raise ValueError(
                "nu_range and z_range are mutually exclusive, provide only one"
            )
        if z_range is not None:
            assert z_range[0] < z_range[1], "z_range[0] must be less than z_range[1]"
            nu_range = (f_21 / (1 + z_range[1]), f_21 / (1 + z_range[0]))
        if nu_range is None:
            nu_range = (
                ps.nu.min() - ps.freq_resol / 2,
                ps.nu.max() + ps.freq_resol / 2,
            )
        self.nu_range = tuple(nu_range)
        if weights_scheme not in ("gridded", "counts"):
            raise ValueError(f"Invalid weights_scheme: {weights_scheme}")
        self.weights_scheme = weights_scheme
        self.apply_taper = apply_taper
        self.taper_axis = tuple(taper_axis)
        if gal_grid_weights not in ("binary", "gridded"):
            raise ValueError(f"Invalid gal_grid_weights: {gal_grid_weights}")
        self.gal_grid_weights = gal_grid_weights
        self.min_patch_weight_fraction = min_patch_weight_fraction
        self.pool = pool
        self.num_process = num_process
        self.patch_indices_used = None

    @property
    def num_patches(self):
        """
        The total number of jackknife patches,
        ``ra_patch_num * dec_patch_num * los_patch_num``.
        """
        return self.ra_patch_num * self.dec_patch_num * self.los_patch_num

    @property
    def z_range(self):
        """
        The redshift range corresponding to ``self.nu_range``.
        """
        return (f_21 / self.nu_range[1] - 1, f_21 / self.nu_range[0] - 1)

    def get_patch_masks(self):
        """
        The jackknife patch masks of the map, computed with
        :meth:`meer21cm.dataanalysis.Specification.get_jackknife_patches`
        and flattened along the patch dimensions.
        ``mask[i] = True`` marks the pixels that are removed in
        realisation ``i``.

        Returns
        -------
        mask_arr: np.ndarray of bool
            The patch masks, of shape ``(num_patches, *ps.W_HI.shape)``.
        """
        mask_arr = self.ps.get_jackknife_patches(
            self.ra_patch_num,
            self.dec_patch_num,
            self.los_patch_num,
            ra_range=self.ra_range,
            dec_range=self.dec_range,
            nu_range=self.nu_range,
        )
        return mask_arr.reshape((self.num_patches,) + self.ps.W_HI.shape)

    def get_gal_patch_labels(self):
        """
        The jackknife patch label of each galaxy in ``ps``, computed with
        :meth:`meer21cm.dataanalysis.Specification.get_gal_patch_labels`.
        The galaxy redshifts are converted internally to 21cm line frequencies
        and digitized into the same frequency bins as the map patches, so the
        galaxy labels match the map patch indices exactly.
        Galaxy ``g`` is removed in realisation ``i`` if ``label[g] == i``.

        Returns
        -------
        label: np.ndarray of int
            Patch index for each galaxy, -1 for galaxies outside the ranges.
        """
        return self.ps.get_gal_patch_labels(
            self.ra_patch_num,
            self.dec_patch_num,
            self.los_patch_num,
            ra_range=self.ra_range,
            dec_range=self.dec_range,
            nu_range=self.nu_range,
        )

    def patch_weight_fraction(self):
        """
        The fraction of the total map weight ``(w_HI * W_HI)`` contained
        in each patch. Useful for identifying (nearly) empty patches and
        for checking how uniform the jackknife split is.

        Returns
        -------
        frac: np.ndarray
            The weight fraction of each patch, of shape ``(num_patches,)``.
        """
        # nan_to_num: real maps can store nan outside the sampled window,
        # which would otherwise propagate and flag every patch as empty
        w = np.nan_to_num(self.ps.w_HI) * self.ps.W_HI
        masks = self.get_patch_masks()
        return np.array([(w * m).sum() for m in masks]) / w.sum()

    def get_ps_instance_attr_dict(self):
        """
        Generate the attribute dictionary for the per-realisation
        power spectrum instances.
        It reads the attributes listed in
        :data:`meer21cm.transfer.required_attrs` and
        :data:`meer21cm.jackknife.extra_required_attrs`
        from the input power spectrum instance.
        Note that ``seed`` is propagated so that the enclosing box
        (which uses random sampling of the pixels) is identical in every
        realisation and identical to the one of the data measurement.

        Returns
        -------
        attr_dict: dict
            The attribute dictionary for the per-realisation instances.
        """
        attr_dict = {}
        for attr in required_attrs + extra_required_attrs:
            attr_dict[attr] = getattr(self.ps, attr)
        return attr_dict

    def get_default_patch_indices(self):
        """
        The indices of the patches used by default in ``run``:
        all patches whose weight fraction is greater than
        ``min_patch_weight_fraction``.

        Returns
        -------
        patch_indices: np.ndarray of int
        """
        frac = self.patch_weight_fraction()
        return np.where(frac > self.min_patch_weight_fraction)[0]

    def get_arg_list_for_parallel_auto(
        self, patch_indices, return_power_3d=False, return_auto_power=False
    ):
        """
        Generate a list of arguments for parallelisation of the auto-power runs.
        This list is then used for ``pool.starmap``.

        Parameters
        ----------
        patch_indices: list
            The indices of the patches to remove, one per realisation.
        return_power_3d: bool, default False
            Whether to also return the 3D power spectrum of each realisation.
        return_auto_power: bool, default False
            Dummy argument to keep consistency with the cross runs.

        Returns
        -------
        arg_list: list
            The list of arguments for the parallelisation.
        """
        masks = self.get_patch_masks()
        arg_list = []
        for j in patch_indices:
            attr_dict = self.get_ps_instance_attr_dict()
            arg_list.append(
                (
                    attr_dict,
                    masks[j],
                    self.ps.k1dweights,
                    self.weights_scheme,
                    self.apply_taper,
                    self.taper_axis,
                    return_power_3d,
                )
            )
        return arg_list

    def get_arg_list_for_parallel_cross(
        self, patch_indices, return_power_3d=False, return_auto_power=False
    ):
        """
        Generate a list of arguments for parallelisation of the cross-power runs.
        This list is then used for ``pool.starmap``.

        Parameters
        ----------
        patch_indices: list
            The indices of the patches to remove, one per realisation.
        return_power_3d: bool, default False
            Whether to also return the 3D cross-power spectrum of each realisation.
        return_auto_power: bool, default False
            Whether to also return the 1D auto-power spectra of the two fields
            of each realisation.

        Returns
        -------
        arg_list: list
            The list of arguments for the parallelisation.
        """
        if self.ps.ra_gal.size == 0:
            raise ValueError(
                "the galaxy catalogue of the input instance is empty, "
                "cannot run type='cross'"
            )
        masks = self.get_patch_masks()
        labels = self.get_gal_patch_labels()
        ra_gal = self.ps.ra_gal
        dec_gal = self.ps.dec_gal
        z_gal = self.ps.z_gal
        arg_list = []
        for j in patch_indices:
            attr_dict = self.get_ps_instance_attr_dict()
            keep_gal = labels != j
            gal_radecz = (ra_gal[keep_gal], dec_gal[keep_gal], z_gal[keep_gal])
            arg_list.append(
                (
                    attr_dict,
                    masks[j],
                    gal_radecz,
                    self.ps.k1dweights,
                    self.weights_scheme,
                    self.apply_taper,
                    self.taper_axis,
                    self.gal_grid_weights,
                    return_power_3d,
                    return_auto_power,
                )
            )
        return arg_list

    def run(
        self,
        patch_indices=None,
        type="cross",
        return_power_3d=False,
        return_auto_power=False,
    ):
        """
        Run the jackknife realisations.

        Note that, ``run`` automatically uses a parallel pool to loop over
        the patches (unless ``pool="serial"``).
        If you believe the parallel behaviour is not as expected,
        you can manually extract the argument list and map the function yourself.
        For example:

        .. code-block:: python

            >>> jc = JackknifeCovariance(ps, 4, 3, 4)
            >>> results = jc.run(type="auto")

        is the same as:

        .. code-block:: python

            >>> jc = JackknifeCovariance(ps, 4, 3, 4)
            >>> arg_list = jc.get_arg_list_for_parallel_auto(
            ...     jc.get_default_patch_indices()
            ... )
            >>> results = []
            >>> with Pool(jc.num_process) as pool:
            >>>     for result_i in pool.starmap(run_jackknife_auto, arg_list):
            >>>         results.append(result_i)

        Parameters
        ----------
        patch_indices: list, default None
            The indices of the patches to remove, one per realisation.
            If None, all patches with weight fraction greater than
            ``min_patch_weight_fraction`` are used
            (see :meth:`get_default_patch_indices`).
            The indices actually used are stored in ``self.patch_indices_used``.
        type: str, default "cross"
            The type of power spectrum to compute for each realisation.
            Can be "cross" (HI x galaxy) or "auto" (HI x HI).
        return_power_3d: bool, default False
            Whether to also return the 3D power spectrum of each realisation.
        return_auto_power: bool, default False
            Only used if ``type="cross"``. Whether to also return the 1D
            auto-power spectra of the HI map and of the galaxy field of
            each realisation.

        Returns
        -------
        results_arr: list
            The list of results, with each element being a sublist for the
            result of one realisation.
            The first element of each sublist is the 1D power spectrum.
            If ``return_power_3d`` is True, the next element is the 3D power
            spectrum. If ``return_auto_power`` is True (cross only), the next
            two elements are the 1D auto-power of field 1 (HI) and of
            field 2 (galaxy).
        """
        if type == "cross":
            run_func = run_jackknife_cross
        elif type == "auto":
            run_func = run_jackknife_auto
        else:
            raise ValueError(f"Invalid type: {type}")
        if patch_indices is None:
            patch_indices = self.get_default_patch_indices()
        if len(patch_indices) == 0:
            raise ValueError(
                "no jackknife patch to use. Check that the patch splits overlap "
                "the survey footprint (see self.patch_weight_fraction()) and "
                "that ra_range, dec_range and nu_range/z_range are correct."
            )
        self.patch_indices_used = np.asarray(patch_indices)
        arg_func = getattr(self, f"get_arg_list_for_parallel_{type}")
        arg_list = arg_func(patch_indices, return_power_3d, return_auto_power)
        if self.pool == "serial":
            return [run_func(*args) for args in arg_list]
        if self.pool == "multiprocessing":
            pool_func = Pool
        elif self.pool == "mpi":
            from mpi4py.futures import MPIPoolExecutor

            pool_func = MPIPoolExecutor
        else:
            raise ValueError(f"Invalid pool: {self.pool}")
        results_arr = []
        with pool_func(self.num_process) as pool:
            for result_i in pool.starmap(run_func, arg_list):
                results_arr.append(result_i)
        return results_arr

    @staticmethod
    def jackknife_covariance(power_arr):
        r"""
        The delete-one jackknife covariance matrix of the input realisations,

        .. math::
            C = \frac{N - 1}{N}
            \sum_{j=1}^{N} (P_j - \bar{P}) (P_j - \bar{P})^T

        where :math:`N` is the number of realisations and
        :math:`\bar{P}` is the mean of the realisations.

        Parameters
        ----------
        power_arr: np.ndarray
            The 1D power spectra of the realisations,
            of shape ``(num_realisations, num_kbins)``.

        Returns
        -------
        cov: np.ndarray
            The jackknife covariance matrix,
            of shape ``(num_kbins, num_kbins)``.
        mean: np.ndarray
            The mean of the realisations, of shape ``(num_kbins,)``.
        """
        power_arr = np.asarray(power_arr)
        num_jack = power_arr.shape[0]
        if num_jack < 2:
            raise ValueError(
                f"at least 2 realisations are needed, got {num_jack}"
            )
        mean = power_arr.mean(axis=0)
        delta = power_arr - mean[None, :]
        cov = (num_jack - 1) / num_jack * np.einsum("ja,jb->ab", delta, delta)
        return cov, mean

    def get_covariance(self, results_arr, element=0):
        """
        Compute the jackknife covariance matrix from the output of ``run``.

        Parameters
        ----------
        results_arr: list
            The output of :meth:`run`.
        element: int, default 0
            The element of each result sublist to use.
            Default is the first element, i.e. the 1D power spectrum
            of the requested type.

        Returns
        -------
        cov: np.ndarray
            The jackknife covariance matrix.
        mean: np.ndarray
            The mean of the realisations.
        """
        power_arr = np.array([result[element] for result in results_arr])
        return self.jackknife_covariance(power_arr)


# this must be pickleable inputs for multiprocessing
def _check_k1dweights(jk, k_sel_3d_to_1d):
    """
    Check that the input ``k1dweights`` (``k_sel``) matches the k-mode grid
    of the realisation. A mismatched array (e.g. built from ``ps.k_mode``
    before ``ps.get_enclosing_box()`` was called, when the box is still
    the placeholder unit box and the k arrays have shape ``(1, 1, 1)``) would
    otherwise broadcast silently and typically zero out every k-bin,
    turning the whole 1D power spectrum into nan.
    """
    if k_sel_3d_to_1d is None:
        return
    k_sel_3d_to_1d = np.asarray(k_sel_3d_to_1d)
    if k_sel_3d_to_1d.shape != jk.k_mode.shape:
        raise ValueError(
            f"k1dweights (k_sel) has shape {k_sel_3d_to_1d.shape}, but the "
            f"k-mode grid of the jackknife realisation has shape "
            f"{jk.k_mode.shape}. k1dweights must be computed on the same "
            "enclosing box as the data measurement: call "
            "ps.get_enclosing_box() (with the final gridding settings) "
            "before building k_sel from ps.k_mode / k_perp / k_para, "
            "and then set ps.k1dweights = k_sel."
        )


def _grid_jackknifed_map(jk, patch_mask, return_kept_counts=False):
    """
    Apply the jackknife mask to the map data and pixel weights of a
    per-realisation instance, and grid the jackknifed map to the Cartesian box.

    The enclosing box is computed before masking, from the full ``W_HI``
    window, so that every realisation lives on the identical grid and k-modes.
    ``W_HI`` is left untouched (the pixel coordinates used for the gridding are
    tied to the ``W_HI`` footprint used to compute the box); the masking enters
    through the map data, the pixel weights ``w_HI``, and consequently the
    gridded weights, which define the jackknifed window on the box.

    Parameters
    ----------
    jk: :class:`meer21cm.power.PowerSpectrum`
        The per-realisation instance.
    patch_mask: np.ndarray of bool
        The mask of the removed patch, True for removed pixels.
    return_kept_counts: bool, default False
        Whether to also compute the number of kept map voxels falling in
        each box cell (the jackknife-consistent analogue of
        ``counts_in_box``, needed for ``weights_scheme="counts"``).
        This requires a second gridding pass.

    Returns
    -------
    map_rg: np.ndarray
        The gridded jackknifed map.
    weights_rg: np.ndarray
        The gridded jackknifed weights.
    counts_keep_rg: np.ndarray or None
        The number of kept map voxels per box cell,
        or None if ``return_kept_counts`` is False.
    """
    # identical box for all realisations: computed from the full window
    # with the seed propagated from the input instance
    jk.get_enclosing_box()
    keep = (~np.asarray(patch_mask, dtype=bool)) * (jk.W_HI > 0)
    keep = keep.astype(jk.data.dtype)
    # nan_to_num: real maps can store nan outside the sampled window;
    # nan in the pixel weights would propagate through the gridding and
    # turn the whole power spectrum into nan
    jk.data = np.nan_to_num(jk.data) * keep
    jk.weights_map_pixel = np.nan_to_num(jk.weights_map_pixel) * keep
    map_rg, weights_rg, _ = jk.grid_data_to_field()
    counts_keep_rg = None
    if return_kept_counts:
        # the jackknife-consistent analogue of ``counts_in_box``:
        # get_counts_in_box grids the pixel weights ``w_HI`` (single particle
        # pass, no interlacing), which at this point are already jackknifed,
        # so this reduces exactly to ``ps.counts_in_box`` when no patch
        # is removed
        counts_keep_rg = jk.get_counts_in_box()
    return map_rg, weights_rg, counts_keep_rg


def run_jackknife_auto(
    ps_attr_dict,
    patch_mask,
    k_sel_3d_to_1d=None,
    weights_scheme="gridded",
    apply_taper=True,
    taper_axis=(2,),
    return_power_3d=False,
):
    """
    Compute the 1D auto-power spectrum of the HI map for one jackknife
    realisation, with the patch given by ``patch_mask`` removed.

    Parameters
    ----------
    ps_attr_dict: dict
        The attribute dictionary to initialize the per-realisation instance.
    patch_mask: np.ndarray of bool
        The mask of the removed patch, True for removed pixels.
    k_sel_3d_to_1d: np.ndarray, default None
        The weights for averaging the 3D power spectrum k-modes to
        the 1D power spectrum.
    weights_scheme: str, default "gridded"
        The scheme used to assign the power spectrum weights,
        see :class:`JackknifeCovariance`.
    apply_taper: bool, default True
        Whether to apply the taper function to the gridded weights.
    taper_axis: tuple, default (2,)
        The box axes along which the taper is applied.
    return_power_3d: bool, default False
        Whether to also return the 3D power spectrum.

    Returns
    -------
    result: list
        The list of results. The first element is the 1D auto-power spectrum.
        If ``return_power_3d`` is True, the second element is the
        3D auto-power spectrum.
    """
    jk = PowerSpectrum(**ps_attr_dict)
    map_rg, weights_rg, counts_keep_rg = _grid_jackknifed_map(
        jk, patch_mask, return_kept_counts=(weights_scheme == "counts")
    )
    _check_k1dweights(jk, k_sel_3d_to_1d)
    jk.field_1 = map_rg
    if weights_scheme == "counts":
        # cookbook convention: inverse noise variance weighting with the
        # (jackknifed) number of map voxels per box cell
        jk.weights_grid_1 = counts_keep_rg
        jk.weights_field_1 = None
    elif weights_scheme == "gridded":
        jk.weights_field_1 = weights_rg
        jk.weights_grid_1 = weights_rg
    else:
        raise ValueError(f"Invalid weights_scheme: {weights_scheme}")
    # gridding may have overridden these, restore the input settings
    jk.mean_center_1 = ps_attr_dict["mean_center_1"]
    jk.unitless_1 = ps_attr_dict["unitless_1"]
    jk.include_beam = ps_attr_dict["include_beam"]
    jk.include_sky_sampling = ps_attr_dict["include_sky_sampling"]
    if apply_taper:
        jk.apply_taper_to_field(1, axis=list(taper_axis))
    p1d_auto, _, _ = jk.get_1d_power(
        "auto_power_3d_1", k1dweights=k_sel_3d_to_1d
    )
    result = [p1d_auto]
    if return_power_3d:
        result.append(jk.auto_power_3d_1)
    return result


def run_jackknife_cross(
    ps_attr_dict,
    patch_mask,
    gal_radecz,
    k_sel_3d_to_1d=None,
    weights_scheme="gridded",
    apply_taper=True,
    taper_axis=(2,),
    gal_grid_weights="binary",
    return_power_3d=False,
    return_auto_power=False,
):
    """
    Compute the 1D HI x galaxy cross-power spectrum for one jackknife
    realisation, with the patch given by ``patch_mask`` removed from the map
    and the galaxies inside the patch removed from the catalogue.

    Parameters
    ----------
    ps_attr_dict: dict
        The attribute dictionary to initialize the per-realisation instance.
    patch_mask: np.ndarray of bool
        The mask of the removed patch, True for removed pixels.
    gal_radecz: tuple of np.ndarray
        The (ra, dec, z) of the galaxies kept in this realisation.
    k_sel_3d_to_1d: np.ndarray, default None
        The weights for averaging the 3D power spectrum k-modes to
        the 1D power spectrum.
    weights_scheme: str, default "gridded"
        The scheme used to assign the power spectrum weights,
        see :class:`JackknifeCovariance`.
    apply_taper: bool, default True
        Whether to apply the taper function to the gridded weights
        of both fields.
    taper_axis: tuple, default (2,)
        The box axes along which the taper is applied.
    gal_grid_weights: str, default "binary"
        The scheme for the grid/field weights of the galaxy field,
        see :class:`JackknifeCovariance`.
    return_power_3d: bool, default False
        Whether to also return the 3D cross-power spectrum.
    return_auto_power: bool, default False
        Whether to also return the 1D auto-power spectra of the two fields.

    Returns
    -------
    result: list
        The list of results. The first element is the 1D cross-power spectrum.
        If ``return_power_3d`` is True, the next element is the 3D cross-power
        spectrum. If ``return_auto_power`` is True, the next two elements are
        the 1D auto-power of field 1 (HI) and of field 2 (galaxy).
    """
    jk = PowerSpectrum(**ps_attr_dict)
    map_rg, weights_rg, counts_keep_rg = _grid_jackknifed_map(
        jk, patch_mask, return_kept_counts=(weights_scheme == "counts")
    )
    _check_k1dweights(jk, k_sel_3d_to_1d)
    # jackknifed galaxy catalogue
    ra_gal, dec_gal, z_gal = gal_radecz
    jk._ra_gal = np.asarray(ra_gal)
    jk._dec_gal = np.asarray(dec_gal)
    jk._z_gal = np.asarray(z_gal)
    gal_map_rg, gal_weights_rg, _ = jk.grid_gal_to_field()
    # field 1: HI map
    jk.field_1 = map_rg
    if weights_scheme == "counts":
        # cookbook convention: inverse noise variance weighting with the
        # (jackknifed) number of map voxels per box cell
        jk.weights_grid_1 = counts_keep_rg
        jk.weights_field_1 = None
    elif weights_scheme == "gridded":
        jk.weights_field_1 = weights_rg
        jk.weights_grid_1 = weights_rg
    else:
        raise ValueError(f"Invalid weights_scheme: {weights_scheme}")
    jk.mean_center_1 = ps_attr_dict["mean_center_1"]
    jk.unitless_1 = ps_attr_dict["unitless_1"]
    # field 2: galaxy number counts
    # (grid_gal_to_field has already set mean_center_2=True, unitless_2=True)
    jk.field_2 = gal_map_rg
    if weights_scheme == "counts":
        # cookbook convention: binary occupancy of the (jackknifed) lightcone
        # as field weights, uniform grid weights
        jk.weights_field_2 = (counts_keep_rg > 0).astype(gal_map_rg.dtype)
        jk.weights_grid_2 = np.ones_like(gal_map_rg)
    elif gal_grid_weights == "binary":
        # default galaxy window (box region sampled by the lightcone),
        # restricted to the region kept by the jackknife
        weights_g = jk.weights_field_2 * (weights_rg > 0)
        jk.weights_field_2 = weights_g
        jk.weights_grid_2 = weights_g
    elif gal_grid_weights == "gridded":
        jk.weights_field_2 = gal_weights_rg
        jk.weights_grid_2 = gal_weights_rg
    else:
        raise ValueError(f"Invalid gal_grid_weights: {gal_grid_weights}")
    jk.include_beam = ps_attr_dict["include_beam"]
    jk.include_sky_sampling = ps_attr_dict["include_sky_sampling"]
    if apply_taper:
        jk.apply_taper_to_field(1, axis=list(taper_axis))
        jk.apply_taper_to_field(2, axis=list(taper_axis))
    p1d_cross, _, _ = jk.get_1d_power("cross_power_3d", k1dweights=k_sel_3d_to_1d)
    result = [p1d_cross]
    if return_power_3d:
        result.append(jk.cross_power_3d)
    if return_auto_power:
        p1d_auto_1, _, _ = jk.get_1d_power(
            "auto_power_3d_1", k1dweights=k_sel_3d_to_1d
        )
        p1d_auto_2, _, _ = jk.get_1d_power(
            "auto_power_3d_2", k1dweights=k_sel_3d_to_1d
        )
        result.append(p1d_auto_1)
        result.append(p1d_auto_2)
    return result