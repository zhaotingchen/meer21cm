from meer21cm.mock import MockSimulation
import numpy as np
from meer21cm.fg import ForegroundSimulation
from meer21cm.util import pcaclean, dft_matrix, center_to_edges


raminMK, ramaxMK = 334, 357
decminMK, decmaxMK = -35, -26.5
ra_range_MK = (raminMK, ramaxMK)
dec_range_MK = (decminMK, decmaxMK)
temp = MockSimulation(
    ra_range=ra_range_MK,
    dec_range=dec_range_MK,
    tracer_bias_1=1.5,
    mean_amp_1="average_hi_temp",
    density="gaussian",
    seed=42,
)

fg = ForegroundSimulation(
    sp_uni_indx=-2.55,
    wproj=temp.wproj,
    num_pix_x=temp.num_pix_x,
    num_pix_y=temp.num_pix_y,
)
fg_map = fg.fg_cube(temp.nu)

kperpedges = np.linspace(0.0, 0.5, 11)
kparaedges = np.linspace(0, 3, 16)

F_mat = dft_matrix(temp.nu.size, norm="forward")
F_mat_inv = np.linalg.inv(F_mat)


def run_one_realization(seed):
    mock = MockSimulation(
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
        tracer_bias_1=1.5,
        mean_amp_1="average_hi_temp",
        density="gaussian",
        seed=seed,
    )
    mock.box_len = np.array(
        [
            mock.pix_resol_in_mpc * mock.num_pix_x,
            mock.pix_resol_in_mpc * mock.num_pix_y,
            mock.los_resol_in_mpc * mock.nu.size,
        ]
    )
    mock.box_ndim = np.array([mock.num_pix_x, mock.num_pix_y, mock.nu.size])
    mock.propagate_field_k_to_model()
    mock.weights_1 = np.ones_like(mock.W_HI)
    mock.field_1 = mock.mock_tracer_field_1
    mock.kperpbins = kperpedges
    mock.kparabins = kparaedges
    power_model = mock.auto_power_tracer_1_model
    np.save(f"temp/power_model_{seed}.npy", power_model)
    field_before = mock.mock_tracer_field_1.copy() + fg_map
    field_after, A_mat = pcaclean(field_before, 40, return_A=True)
    R_mat = np.eye(field_before.shape[-1]) - A_mat @ A_mat.T
    np.save(f"temp/R_mat_{seed}.npy", R_mat)
    mock.field_1 = field_after
    power_out = mock.auto_power_3d_1
    np.save(f"temp/power_cleaned_{seed}.npy", power_out)
    R_mat_fourier = F_mat @ R_mat @ F_mat_inv
    np.save(f"temp/R_mat_fourier_{seed}.npy", R_mat_fourier)
    power_model_conv = np.einsum("ij,abj->abi", np.abs(R_mat_fourier) ** 2, power_model)
    np.save(f"temp/power_model_conv_{seed}.npy", power_model_conv)
    return 1
