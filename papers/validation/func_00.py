import numpy as np
from meer21cm.mock import MockSimulation
from meer21cm.grid import project_particle_to_regular_grid


def get_3d_power(
    seed,
):
    mock = MockSimulation(
        band="L",
        survey="meerklass_2021",
        kaiser_rsd=True,
        seed=seed,
        tracer_bias_2=1.0,
        num_discrete_source=1e6,
    )
    mock._box_len = np.array([1000, 1000, 1000])
    mock._box_ndim = np.array([50, 50, 50])
    mock.propagate_field_k_to_model()
    mock.field_1 = mock.mock_tracer_field_1
    mock.weights_1 = np.ones_like(mock.field_1)
    pdata3d = mock.auto_power_3d_1
    pmod3d = mock.auto_power_tracer_1_model
    # hack
    mock._box_voxel_redshift = np.ones(mock.box_ndim) * mock.z
    _, _, gal_counts = project_particle_to_regular_grid(
        mock.mock_tracer_position_in_box,
        mock.box_len,
        mock.box_ndim,
    )
    mock.field_1 = gal_counts / gal_counts.mean() - 1
    pg3d = mock.auto_power_3d_1 - (np.prod(mock.box_len) / gal_counts.sum())
    return pdata3d, pmod3d, pg3d
