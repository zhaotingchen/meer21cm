from time import thread_time_ns
import numpy as np
from meer21cm.util import pca_clean, dft_matrix, inv_dft_matrix
from meer21cm.mock import MockSimulation
from multiprocessing import Pool


def fft_matrix(mat, norm="backward"):
    r"""
    Perform the Fourier transform of a matrix.

    .. math::
        \tilde{M} = \mathcal{F} M \mathcal{F}^{-1}

    where :math:`\mathcal{F}` is the Fourier transform matrix.
    See also :func:`meer21cm.util.dft_matrix`.

    Parameters
    ----------
        mat: np.ndarray
            The matrix to be transformed.
        norm: str, default "backward"
            The normalization of the Fourier transform.

    Returns
    -------
        mat_fft: np.ndarray
            The Fourier transformed matrix.
    """

    return (
        dft_matrix(mat.shape[0], norm=norm)
        @ mat
        @ inv_dft_matrix(mat.shape[0], norm=norm)
    )


def get_pca_matrix(map, N_fg, weights, mean_center_map):
    r"""
    Get the PCA matrix that operates on the map data.

    The eigendecomposition of the map data gives the eigenvectors :math:`v_{i}`,
    which can be used to form the source mixing matrix A so that

    .. math::
        A = \{v_1, v_2, ..., v_{N_fg}\}

    The PCA matrix is then

    .. math::
        R = I - A A^T

    The residual data vector after cleaning is then

    .. math::
        r_{ija} = \sum_{b} R_{ab} m_{ijb}

    where :math:`R_{ab}` is the PCA matrix, and :math:`m_{ijb}` is the uncleaned map data.

    Parameters
    ----------
        map: np.ndarray
            The map data.
        N_fg: int
            The number of foreground components.
        weights: np.ndarray
            The weights of the map data.
        mean_center_map: bool
            Whether to mean center the map data.

    Returns
    -------

    """
    _, A_mat = pca_clean(
        map,
        N_fg,
        weights=weights,
        mean_center=mean_center_map,
        return_A=True,
    )
    R_mat = np.eye(map.shape[-1]) - A_mat @ A_mat.T
    return R_mat


def analytic_transfer_function(clean_mat_1, clean_mat_2=None):
    r"""
    Calculate the analytic transfer function of a clean matrix.
    See Chen 2025 [1] for derivations.

    For a foreground cleaning matrix :math:`R_{ab}`,
    the residual data vector for power spectrum estimation is :math:`r_{ija} = \sum_b R_{ab} m_{ijb}`,
    where i,j are the pixel indices, and a,b are the frequency indices.

    Under the flat sky approximation, the signal loss, as well as the mode-mixing,
    is along the line-of-sight (k_para) direction.
    The unnormalised window function matrix is

    .. math::
        H_{ab} = |\tilde{R}^1_{ab} (\tilde{R}^2_{ab})^*|_{\rm Re}

    The corresponding signal loss is :math:`\sum_b H_{ab}`, and the analytical
    transfer function is the inverse of the signal loss.

    After normalisation, the window function matrix is

    .. math::
        W = {\rm diag}\Big(\sum_b H_{ab}\Big)^{-1} H

    Parameters
    ----------
        clean_mat_1: np.ndarray
            The clean matrix that applies to the data vector.
        clean_mat_2: np.ndarray, optional
            The clean matrix that applies to the second data vector for cross-correlation.
            If not provided, it is assumed to be the same as clean_mat_1 for auto-correlation.

    Returns
    -------
        transfer_func: np.ndarray
            The analytical transfer function.
        Wab: np.ndarray
            The normalised window function matrix.

    References
    ----------
    .. [1] Chen, Z.,, "A quadratic estimator view of the transfer function correction in intensity mapping surveys", https://ui.adsabs.harvard.edu/abs/2025MNRAS.542L...1C/abstract.
    """
    assert (clean_mat_1.ndim == 2) and (clean_mat_1.shape[0] == clean_mat_1.shape[1])
    if clean_mat_2 is None:
        clean_mat_2 = clean_mat_1
    assert np.allclose(clean_mat_1.shape, clean_mat_2.shape)
    num_k = clean_mat_1.shape[0] // 2 + 1
    R_mat_fourier_1 = fft_matrix(clean_mat_1)
    R_mat_fourier_2 = fft_matrix(clean_mat_2)
    Hab = (np.conj(R_mat_fourier_1) * R_mat_fourier_2).real
    Hab = Hab[:num_k, :num_k]
    signal_loss = Hab.sum(1)
    renorm_mat = np.diag(1 / signal_loss)
    Wab = renorm_mat @ Hab
    return 1 / signal_loss, Wab


required_attrs = [
    "wproj",
    "num_pix_x",
    "num_pix_y",
    "nu",
    "data",
    "map_has_sampling",
    "weights_map_pixel",
    "cosmo",
    "sigma_beam_ch",
    "ps_type",
    "cold",
    "backend",
    "omega_hi",
    "tracer_bias_1",
    "tracer_bias_2",
    "sigma_v_1",
    "sigma_v_2",
    "include_beam",
    "fog_profile",
    "weights_field_1",
    "weights_field_2",
    "weights_grid_1",
    "weights_grid_2",
    "renorm_weights_1",
    "renorm_weights_2",
    "renorm_weights_cross",
    "mean_amp_1",
    "mean_amp_2",
    "kaiser_rsd",
    "sigma_z_1",
    "sigma_z_2",
    "sampling_resol",
    "box_buffkick",
    "num_particle_per_pixel",
    "mean_center_1",
    "mean_center_2",
    "unitless_1",
    "unitless_2",
    "compensate",
    "taper_func",
    "grid_scheme",
    "interlace_shift",
    "flat_sky",
    "flat_sky_padding",
    "kperpbins",
    "kparabins",
    "k1dbins",
    "include_sky_sampling",
    "k1dweights",
]


class TransferFunction:
    """
    A class to perform transfer function analysis.
    """

    def __init__(
        self,
        ps,
        N_fg,
        R_mat=None,
        uncleaned_data=None,
        highres_sim=3,
        upres_transverse=4,
        upres_radial=4,
        mean_center_map=True,
        pca_map_weights=None,
        parallel_plane=True,
        rsd_from_field=False,
        discrete_source_dndz=np.ones_like,
        recalculate_pca_with_injection=True,
        pool="multiprocessing",
        num_process=None,
    ):
        self.ps = ps
        self.N_fg = N_fg
        self.highres_sim = highres_sim
        self.upres_transverse = upres_transverse
        self.upres_radial = upres_radial
        self.mean_center_map = mean_center_map
        if pca_map_weights is None:
            pca_map_weights = self.ps.weights_map_pixel
        self.pca_map_weights = pca_map_weights
        self.parallel_plane = parallel_plane
        self.rsd_from_field = rsd_from_field
        self.discrete_source_dndz = discrete_source_dndz
        if uncleaned_data is None:
            uncleaned_data = ps.data
        self.uncleaned_data = uncleaned_data
        self.recalculate_pca_with_injection = recalculate_pca_with_injection
        self.pool = pool
        self.num_process = num_process
        self.R_mat = R_mat

    def get_mock_instance_attr_dict(self, seed):
        attr_dict = {}
        for attr in required_attrs:
            attr_dict[attr] = getattr(self.ps, attr)
        attr_dict["seed"] = seed
        if hasattr(self.ps, "ra_gal"):
            attr_dict["num_discrete_source"] = self.ps.ra_gal.size
        attr_dict["highres_sim"] = self.highres_sim
        attr_dict["parallel_plane"] = self.parallel_plane
        attr_dict["rsd_from_field"] = self.rsd_from_field
        attr_dict["discrete_source_dndz"] = self.discrete_source_dndz
        attr_dict["downres_factor_radial"] = 1 / self.upres_radial
        attr_dict["downres_factor_transverse"] = 1 / self.upres_transverse
        attr_dict["kmax"] = self.ps.kmax * self.highres_sim
        attr_dict["num_kpoints"] = self.ps.num_kpoints * self.highres_sim
        return attr_dict

    def get_arg_list_for_parallel_null(
        self,
        seed_list,
        return_power_3d=False,
        return_power_1d=False,
    ):
        arg_list = []
        for seed in seed_list:
            mock_attr_dict = self.get_mock_instance_attr_dict(seed)
            arg_list.append(
                (
                    mock_attr_dict,
                    self.ps.field_1,
                    self.ps.downres_factor_radial,
                    self.ps.downres_factor_transverse,
                    self.ps.weights_field_1,
                    self.ps.weights_grid_1,
                    self.ps.weights_field_2,
                    self.ps.weights_grid_2,
                    self.ps.k1dweights,
                    return_power_3d,
                )
            )
        return arg_list

    def get_arg_list_for_parallel_cross(
        self, seed_list, return_power_3d=False, return_power_1d=False
    ):
        arg_list = []
        for seed in seed_list:
            mock_attr_dict = self.get_mock_instance_attr_dict(seed)
            arg_list.append(
                (
                    mock_attr_dict,
                    self.ps.downres_factor_radial,
                    self.ps.downres_factor_transverse,
                    self.ps.weights_field_1,
                    self.ps.weights_grid_1,
                    self.ps.weights_field_2,
                    self.ps.weights_grid_2,
                    self.ps.k1dweights,
                    self.recalculate_pca_with_injection,
                    self.N_fg,
                    self.pca_map_weights,
                    self.mean_center_map,
                    self.R_mat,
                    self.uncleaned_data,
                    return_power_3d,
                    return_power_1d,
                )
            )
        return arg_list

    def get_arg_list_for_parallel_auto(
        self, seed_list, return_power_3d=False, return_power_1d=False
    ):
        arg_list = []
        for seed in seed_list:
            mock_attr_dict = self.get_mock_instance_attr_dict(seed)
            arg_list.append(
                (
                    mock_attr_dict,
                    self.ps.downres_factor_radial,
                    self.ps.downres_factor_transverse,
                    self.ps.weights_field_1,
                    self.ps.weights_grid_1,
                    self.ps.k1dweights,
                    self.recalculate_pca_with_injection,
                    self.N_fg,
                    self.pca_map_weights,
                    self.mean_center_map,
                    self.R_mat,
                    self.uncleaned_data,
                    return_power_3d,
                    return_power_1d,
                )
            )
        return arg_list

    def run(
        self, seed_list, type="cross", return_power_3d=False, return_power_1d=False
    ):
        if type == "cross":
            run_func = run_tf_calculation_cross
        elif type == "auto":
            run_func = run_tf_calculation_auto
        elif type == "null":
            run_func = run_null_test
        else:
            raise ValueError(f"Invalid type: {type}")
        arg_func = getattr(self, f"get_arg_list_for_parallel_{type}")
        arg_list = arg_func(seed_list, return_power_3d, return_power_1d)
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


# this must be pickleable inputs for multiprocessing
def run_tf_calculation_cross(
    mock_attr_dict,
    downres_factor_radial,
    downres_factor_transverse,
    weights_field_1,
    weights_grid_1,
    weights_field_2,
    weights_grid_2,
    k_sel_3d_to_1d,
    recalculate_pca_with_injection,
    N_fg,
    pca_map_weights,
    mean_center_map,
    R_mat=None,
    uncleaned_data=0.0,
    return_power_3d=False,
    return_power_1d=False,
):
    mock = MockSimulation(**mock_attr_dict)
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.downres_factor_radial = downres_factor_radial
    mock.downres_factor_transverse = downres_factor_transverse
    mock.get_enclosing_box()
    mock_map_rg, _, _ = mock.grid_data_to_field()
    mockgal_map_rg, _, _ = mock.grid_gal_to_field()
    mock.field_1 = mock_map_rg
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    mock.field_2 = mockgal_map_rg
    mock.weights_field_2 = weights_field_2
    mock.weights_grid_2 = weights_grid_2
    # get 1d power
    pmock_1d_cross, _, _ = mock.get_1d_power(
        mock.cross_power_3d, k1dweights=k_sel_3d_to_1d
    )
    if return_power_3d:
        power_3d_uncleaned = mock.cross_power_3d
    # perform PCA cleaning
    map_to_clean = mock.data
    if recalculate_pca_with_injection:
        map_to_clean = uncleaned_data + mock.data
    R_mat_mock = R_mat
    if R_mat is None:
        R_mat_mock = get_pca_matrix(
            map_to_clean, N_fg, pca_map_weights, mean_center_map
        )
    mock_map_cleaned = np.einsum("ij,abj->abi", R_mat_mock, mock.data)
    mock.data = mock_map_cleaned
    # regrid the cleaned data
    mock_map_rg, _, _ = mock.grid_data_to_field()
    mock.field_1 = mock_map_rg
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    # get 1d power again
    pmock_1d_cross_cleaned, _, _ = mock.get_1d_power(
        mock.cross_power_3d, k1dweights=k_sel_3d_to_1d
    )
    tf_1d_i = pmock_1d_cross_cleaned / pmock_1d_cross
    result = [tf_1d_i]
    if return_power_3d:
        result.append(power_3d_uncleaned)
        result.append(mock.cross_power_3d)
    if return_power_1d:
        result.append(pmock_1d_cross)
        result.append(pmock_1d_cross_cleaned)
    return result


def run_tf_calculation_auto(
    mock_attr_dict,
    downres_factor_radial,
    downres_factor_transverse,
    weights_field_1,
    weights_grid_1,
    k_sel_3d_to_1d,
    recalculate_pca_with_injection,
    N_fg,
    pca_map_weights,
    mean_center_map,
    R_mat=None,
    uncleaned_data=0.0,
    return_power_3d=False,
    return_power_1d=False,
):
    mock = MockSimulation(**mock_attr_dict)
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.downres_factor_radial = downres_factor_radial
    mock.downres_factor_transverse = downres_factor_transverse
    mock.get_enclosing_box()
    mock_map_rg_before, _, _ = mock.grid_data_to_field()
    mock.field_1 = mock_map_rg_before
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    # get 1d power
    pmock_1d_before, _, _ = mock.get_1d_power(
        mock.auto_power_3d_1, k1dweights=k_sel_3d_to_1d
    )
    if return_power_3d:
        power_3d_uncleaned = mock.auto_power_3d_1
    # perform PCA cleaning
    map_to_clean = mock.data
    if recalculate_pca_with_injection:
        map_to_clean = uncleaned_data + mock.data
    R_mat_mock = R_mat
    if R_mat is None:
        R_mat_mock = get_pca_matrix(
            map_to_clean, N_fg, pca_map_weights, mean_center_map
        )
    mock_map_cleaned = np.einsum("ij,abj->abi", R_mat_mock, mock.data)
    mock.data = mock_map_cleaned
    # regrid the cleaned data
    mock_map_rg_cleaned, _, _ = mock.grid_data_to_field()
    mock.field_1 = mock_map_rg_before
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    mock.field_2 = mock_map_rg_cleaned
    mock.weights_field_2 = weights_field_1
    mock.weights_grid_2 = weights_grid_1
    mock.unitless_2 = False
    mock.mean_center_2 = False

    # get 1d power again
    pmock_1d_cross_cleaned, _, _ = mock.get_1d_power(
        mock.cross_power_3d, k1dweights=k_sel_3d_to_1d
    )
    tf_1d_i = pmock_1d_cross_cleaned / pmock_1d_before
    result = [tf_1d_i]
    if return_power_3d:
        result.append(power_3d_uncleaned)
        result.append(mock.cross_power_3d)
    if return_power_1d:
        result.append(pmock_1d_before)
        result.append(pmock_1d_cross_cleaned)
    return result


def run_null_test(
    mock_attr_dict,
    hi_map_rg,
    downres_factor_radial,
    downres_factor_transverse,
    weights_field_1,
    weights_grid_1,
    weights_field_2,
    weights_grid_2,
    k_sel_3d_to_1d,
    return_power_3d=False,
):
    mock = MockSimulation(**mock_attr_dict)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.downres_factor_radial = downres_factor_radial
    mock.downres_factor_transverse = downres_factor_transverse
    mock.get_enclosing_box()
    mockgal_map_rg, _, _ = mock.grid_gal_to_field()
    mock.field_1 = hi_map_rg
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    mock.field_2 = mockgal_map_rg
    mock.weights_field_2 = weights_field_2
    mock.weights_grid_2 = weights_grid_2
    pmock_1d_cross, _, _ = mock.get_1d_power(
        mock.cross_power_3d, k1dweights=k_sel_3d_to_1d
    )
    result = [pmock_1d_cross]
    if return_power_3d:
        result.append(mock.cross_power_3d)
    return result
