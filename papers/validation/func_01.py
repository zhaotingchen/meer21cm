import numpy as np
from meer21cm.mock import MockSimulation
from meer21cm.grid import (
    project_particle_to_regular_grid,
    shot_noise_correction_from_gridding,
    fourier_window_for_assignment,
)


def get_3d_power(seed, window, rsd):
    mock = MockSimulation(
        band="L",
        survey="meerklass_2021",
        kaiser_rsd=rsd,
        seed=seed,
        tracer_bias_2=1.0,
        num_discrete_source=1e6,
    )
    mock._box_len = np.array([1000, 1000, 1000])
    mock._box_ndim = np.array([400, 400, 400])
    mock.propagate_field_k_to_model()
    # hack
    mock._box_voxel_redshift = np.ones(mock.box_ndim) * mock.z
    mock_pos = mock.mock_tracer_position_in_box.copy()
    # change box ndim
    mock._box_ndim = np.array([50, 50, 50])
    mock.propagate_field_k_to_model()
    gal_counts, _, _ = project_particle_to_regular_grid(
        mock_pos,
        mock.box_len,
        mock.box_ndim,
        average=False,
        grid_scheme=window,
    )
    pg3d = mock.auto_power_3d_1
    sn = shot_noise_correction_from_gridding(
        mock.box_ndim,
        window,
    )
    pmod3d = mock.auto_power_tracer_1_model
    mock.field_1 = gal_counts / gal_counts.mean() - 1
    pg3d = mock.auto_power_3d_1
    sn = (
        shot_noise_correction_from_gridding(
            mock.box_ndim,
            window,
        )
        * np.prod(mock.box_len)
        / gal_counts.sum()
    )
    return pg3d, sn, pmod3d
