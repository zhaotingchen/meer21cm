import numpy as np
from meer21cm.transfer import analytic_transfer_function, pca_clean, get_pca_matrix
from meer21cm import MockSimulation
from meer21cm.transfer import TransferFunction
import pytest


def test_analytic_transfer_function():
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        k1dbins=np.linspace(0.05, 1.5, 11),
    )
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.use_flat_sky_box()
    mock.propagate_field_k_to_model()
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.use_flat_sky_box()
    mock.propagate_field_k_to_model()
    mock.field_1 = mock.mock_tracer_field_1
    porig_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    fg = np.ones_like(mock.field_1) * 20
    fg *= ((mock.nu / 408 / 1e6) ** (-2.7))[None, None, :]
    res_map, A_mat = pca_clean(fg + mock.field_1, 3, return_A=True)
    R_mat = np.eye(mock.nu.size) - A_mat @ A_mat.T
    tf, wab = analytic_transfer_function(R_mat)
    mock.field_1 = res_map
    pdata_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1 * tf[None, None, :])
    pnotf_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    no_tf_deviation = np.abs(pnotf_1d - porig_1d) / porig_1d
    tf_deviation = np.abs(pdata_1d - porig_1d) / porig_1d
    assert tf_deviation.mean() < 5e-2
    assert no_tf_deviation.mean() > tf_deviation.mean()
    assert np.diagonal(wab)[0] < 0.1
    assert np.abs(np.diagonal(wab)[6:].mean() - 1) < 0.01


def test_numerical_transfer_function():
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        k1dbins=np.linspace(0.05, 1.5, 11),
        flat_sky=True,
        tracer_bias_2=1.0,
        num_discrete_source=10000,
    )
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.use_flat_sky_box()
    mock.propagate_field_k_to_model()
    mock.field_1 = mock.mock_tracer_field_1
    porig_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    fg = np.ones_like(mock.field_1) * 20
    fg *= ((mock.nu / 408 / 1e6) ** (-2.7))[None, None, :]
    res_map, A_mat = pca_clean(fg + mock.field_1, 3, return_A=True, mean_center=True)
    R_mat = np.eye(mock.nu.size) - A_mat @ A_mat.T
    R_mat_test = get_pca_matrix(
        fg + mock.field_1, 3, weights=np.ones_like(mock.field_1), mean_center_map=True
    )
    assert np.allclose(R_mat, R_mat_test)
    mock.field_1 = res_map
    pnotf_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    tf_true = pnotf_1d / porig_1d
    mock.data = fg + mock.field_1
    tf = TransferFunction(
        mock,
        N_fg=3,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        num_process=3,
    )
    results_arr = tf.run(
        range(10), type="auto", return_power_3d=True, return_power_1d=True
    )
    results_arr = np.array([results_arr[i][0] for i in range(10)])
    tf_1d = results_arr.mean((0))
    assert np.abs(tf_true - tf_1d).mean() < 5e-2
    mock.propagate_mock_tracer_to_gal_cat()
    results_arr = tf.run(
        range(10), type="cross", return_power_3d=True, return_power_1d=True
    )
    results_arr = np.array([results_arr[i][0] for i in range(10)])
    tf_1d = results_arr.mean((0))
    assert np.abs(tf_true - tf_1d).mean() < 5e-2
    tf.pool = "mpi"
    results_arr = tf.run(range(10), type="null", return_power_3d=True)
    results_arr = np.array([results_arr[i][0] for i in range(10)])
    null_ps_1d = results_arr.mean((0))
    # note null is cross, so missing one temperature unit
    avg = (mock.average_hi_temp * null_ps_1d / porig_1d).mean()
    assert np.abs(avg) < 1e-2
    with pytest.raises(ValueError):
        tf.run(range(10), type="test", return_power_3d=False, return_power_1d=False)
    tf.pool = "test"
    with pytest.raises(ValueError):
        tf.run(range(10), type="auto", return_power_3d=False, return_power_1d=False)
