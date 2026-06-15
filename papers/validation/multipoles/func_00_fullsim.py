# 00: just mock simulation with multipole moments and cylindrical binning
import numpy as np
from meer21cm.mock import MockSimulation
from specs_fullsim import *
from meer21cm.grid import (
    project_particle_to_regular_grid,
    shot_noise_correction_from_gridding,
)
from scipy.interpolate import interp1d
from meer21cm.power import get_shot_noise_galaxy
from multiprocessing import Pool
from meer21cm.power import bin_3d_to_cy, bin_3d_to_1d


def get_k_modes():
    mock = get_mock(0)
    mock.get_enclosing_box()
    kperp_1 = mock.k_perp.copy()
    kpara_1 = mock.k_para.copy()
    kmode_1 = mock.k_mode.copy()
    kvec_1 = mock.k_vec.copy()
    return kperp_1, kpara_1, kmode_1, kvec_1


def get_3d_power_multipole(
    seed,
):
    mock = get_mock(seed)
    mock.sigma_beam_ch = sigma_beam_ch
    mock.include_beam = [True, False]
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    mock.get_enclosing_box()
    mock.field_1 = mock.mock_tracer_field_1
    mock.weights_1 = np.ones_like(mock.field_1)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    pdata3d = mock.auto_power_3d_1
    phimod3d = mock.auto_power_tracer_1_model
    _, _, gal_counts = project_particle_to_regular_grid(
        mock.mock_tracer_position_in_box,
        mock.box_len,
        mock.box_ndim,
    )
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)
    mock.field_2 = gal_counts
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = np.ones_like(gal_counts)
    mock.apply_taper_to_field(2, axis=[0, 1, 2])
    mock.mean_center_2 = True
    mock.unitless_2 = True
    mock.compensate = [False, False]
    shot_noise = get_shot_noise_galaxy(
        gal_counts, mock.box_len, mock.weights_grid_2, mock.weights_field_2
    )
    pg3d = mock.auto_power_3d_2 - shot_noise
    pgmod3d = mock.auto_power_tracer_2_model
    pcross3d = mock.cross_power_3d
    pcrossmod3d = mock.cross_power_tracer_model

    # Setup k-selection for anisotropic binning
    kvec = mock.k_vec
    k_xy_sel = (
        (np.abs(kvec[0]) < 0.04)[:, None, None]
        * (np.abs(kvec[1]) < 0.04)[None, :, None]
        * (np.abs(kvec[2]) < 10)[None, None, :]
    )
    k_xy_sel[0] = 0.0
    k_xy_sel[:, 0] = 0.0
    k_xy_sel[:, :, 0] = 0.0

    # Use the repo's built-in multipole estimator; k-weights set per-mock
    kvec = mock.k_vec
    k_xy_sel = (
        (np.abs(kvec[0]) < 0.04)[:, None, None]
        * (np.abs(kvec[1]) < 0.04)[None, :, None]
        * (np.abs(kvec[2]) < 10)[None, None, :]
    )
    k_xy_sel[0] = 0.0
    k_xy_sel[:, 0] = 0.0
    k_xy_sel[:, :, 0] = 0.0
    mock.k1dweights = k_xy_sel.astype(float)

    mock.downres_factor_transverse = sim_upres_transverse
    mock.downres_factor_radial = sim_upres_radial
    mock.get_enclosing_box()
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()
    mock.downres_factor_transverse = ps_downres_transverse
    mock.downres_factor_radial = ps_downres_radial
    mock.get_enclosing_box()

    # Re-grid maps and galaxy boxes to regular fields (match func_fullsim.py)
    himap_rg, _, _ = mock.grid_data_to_field()
    galmap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)

    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    mock.include_sky_sampling = [True, False]
    mock.compensate = [True, True]
    mock.include_beam = [True, False]

    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box > 0) * mock.counts_in_box).astype("float")
    mock.apply_taper_to_field(2, axis=[0, 1, 2])

    shot_noise = get_shot_noise_galaxy(
        galmap_rg,
        mock.box_len,
        mock.weights_grid_2,
        mock.weights_field_2,
    ) * shot_noise_correction_from_gridding(mock.box_ndim, mock.grid_scheme)

    pdata3d = mock.auto_power_3d_1
    phimod3d = mock.auto_power_tracer_1_model
    pg3d = mock.auto_power_3d_2 - shot_noise
    pgmod3d = mock.auto_power_tracer_2_model
    pcross3d = mock.cross_power_3d
    pcrossmod3d = mock.cross_power_tracer_model

    # Now compute multipoles for all tracers (HI, model, galaxy, cross)
    def get_multipole(power3d, ell):
        power_ell, keff_i, nmodes_i = mock.get_1d_power(
            power3d,
            k1dbins=k1dbins,
            multipole_ell=ell,
        )
        return power_ell, keff_i, nmodes_i

    pdata_ell0, keff, nmodes = get_multipole(pdata3d, 0)
    pdata_ell2, _, _ = get_multipole(pdata3d, 2)
    pdata_ell4, _, _ = get_multipole(pdata3d, 4)

    phimod_ell0, _, _ = get_multipole(phimod3d, 0)
    phimod_ell2, _, _ = get_multipole(phimod3d, 2)
    phimod_ell4, _, _ = get_multipole(phimod3d, 4)

    # Compute multipole moments for galaxies (ell=0,2,4)
    pg_ell0, _, _ = get_multipole(pg3d, 0)
    pg_ell2, _, _ = get_multipole(pg3d, 2)
    pg_ell4, _, _ = get_multipole(pg3d, 4)

    pgmod_ell0, _, _ = get_multipole(pgmod3d, 0)
    pgmod_ell2, _, _ = get_multipole(pgmod3d, 2)
    pgmod_ell4, _, _ = get_multipole(pgmod3d, 4)

    # Compute cross multipole moments (ell=0,2,4)
    pcross_ell0, _, _ = get_multipole(pcross3d, 0)
    pcross_ell2, _, _ = get_multipole(pcross3d, 2)
    pcross_ell4, _, _ = get_multipole(pcross3d, 4)

    pcrossmod_ell0, _, _ = get_multipole(pcrossmod3d, 0)
    pcrossmod_ell2, _, _ = get_multipole(pcrossmod3d, 2)
    pcrossmod_ell4, _, _ = get_multipole(pcrossmod3d, 4)

    # Compute 2D cylindrical power spectra
    pdata_cy = bin_power_cy(
        pdata3d[None], mock.k_perp, mock.k_para, kperpbins, kparabins
    )[0]
    phimod_cy = bin_power_cy(
        phimod3d[None], mock.k_perp, mock.k_para, kperpbins, kparabins
    )[0]
    pg_cy = bin_power_cy(pg3d[None], mock.k_perp, mock.k_para, kperpbins, kparabins)[0]
    pgmod_cy = bin_power_cy(
        pgmod3d[None], mock.k_perp, mock.k_para, kperpbins, kparabins
    )[0]
    pcross_cy = bin_power_cy(
        pcross3d[None], mock.k_perp, mock.k_para, kperpbins, kparabins
    )[0]
    pcrossmod_cy = bin_power_cy(
        pcrossmod3d[None], mock.k_perp, mock.k_para, kperpbins, kparabins
    )[0]

    print("Seed {} completed".format(seed))

    return (
        pdata_ell0,
        pdata_ell2,
        pdata_ell4,
        phimod_ell0,
        phimod_ell2,
        phimod_ell4,
        pg_ell0,
        pg_ell2,
        pg_ell4,
        pgmod_ell0,
        pgmod_ell2,
        pgmod_ell4,
        pcross_ell0,
        pcross_ell2,
        pcross_ell4,
        pcrossmod_ell0,
        pcrossmod_ell2,
        pcrossmod_ell4,
        pdata_cy,
        phimod_cy,
        pg_cy,
        pgmod_cy,
        pcross_cy,
        pcrossmod_cy,
        keff,
        nmodes,
    )


def main():
    pdata_ell0_arr = []
    pdata_ell2_arr = []
    pdata_ell4_arr = []
    phimod_ell0_arr = []
    phimod_ell2_arr = []
    phimod_ell4_arr = []

    pg_ell0_arr = []
    pg_ell2_arr = []
    pg_ell4_arr = []
    pgmod_ell0_arr = []
    pgmod_ell2_arr = []
    pgmod_ell4_arr = []

    pcross_ell0_arr = []
    pcross_ell2_arr = []
    pcross_ell4_arr = []
    pcrossmod_ell0_arr = []
    pcrossmod_ell2_arr = []
    pcrossmod_ell4_arr = []

    pdata_cy_arr = []
    phimod_cy_arr = []
    pg_cy_arr = []
    pgmod_cy_arr = []
    pcross_cy_arr = []
    pcrossmod_cy_arr = []

    keff_list = []
    nmodes_list = []

    with Pool(16) as p:
        for (
            pdata_ell0,
            pdata_ell2,
            pdata_ell4,
            phimod_ell0,
            phimod_ell2,
            phimod_ell4,
            pg_ell0,
            pg_ell2,
            pg_ell4,
            pgmod_ell0,
            pgmod_ell2,
            pgmod_ell4,
            pcross_ell0,
            pcross_ell2,
            pcross_ell4,
            pcrossmod_ell0,
            pcrossmod_ell2,
            pcrossmod_ell4,
            pdata_cy,
            phimod_cy,
            pg_cy,
            pgmod_cy,
            pcross_cy,
            pcrossmod_cy,
            keff,
            nmodes,
        ) in p.map(get_3d_power_multipole, range(512)):

            pdata_ell0_arr.append(pdata_ell0)
            pdata_ell2_arr.append(pdata_ell2)
            pdata_ell4_arr.append(pdata_ell4)
            phimod_ell0_arr.append(phimod_ell0)
            phimod_ell2_arr.append(phimod_ell2)
            phimod_ell4_arr.append(phimod_ell4)

            pg_ell0_arr.append(pg_ell0)
            pg_ell2_arr.append(pg_ell2)
            pg_ell4_arr.append(pg_ell4)
            pgmod_ell0_arr.append(pgmod_ell0)
            pgmod_ell2_arr.append(pgmod_ell2)
            pgmod_ell4_arr.append(pgmod_ell4)

            pcross_ell0_arr.append(pcross_ell0)
            pcross_ell2_arr.append(pcross_ell2)
            pcross_ell4_arr.append(pcross_ell4)
            pcrossmod_ell0_arr.append(pcrossmod_ell0)
            pcrossmod_ell2_arr.append(pcrossmod_ell2)
            pcrossmod_ell4_arr.append(pcrossmod_ell4)

            pdata_cy_arr.append(pdata_cy)
            phimod_cy_arr.append(phimod_cy)
            pg_cy_arr.append(pg_cy)
            pgmod_cy_arr.append(pgmod_cy)
            pcross_cy_arr.append(pcross_cy)
            pcrossmod_cy_arr.append(pcrossmod_cy)

            keff_list.append(keff)
            nmodes_list.append(nmodes)

    # Convert to arrays
    pdata_ell0_arr = np.array(pdata_ell0_arr)
    pdata_ell2_arr = np.array(pdata_ell2_arr)
    pdata_ell4_arr = np.array(pdata_ell4_arr)
    phimod_ell0_arr = np.array(phimod_ell0_arr)[0][None]
    phimod_ell2_arr = np.array(phimod_ell2_arr)[0][None]
    phimod_ell4_arr = np.array(phimod_ell4_arr)[0][None]

    pg_ell0_arr = np.array(pg_ell0_arr)
    pg_ell2_arr = np.array(pg_ell2_arr)
    pg_ell4_arr = np.array(pg_ell4_arr)
    pgmod_ell0_arr = np.array(pgmod_ell0_arr)[0][None]
    pgmod_ell2_arr = np.array(pgmod_ell2_arr)[0][None]
    pgmod_ell4_arr = np.array(pgmod_ell4_arr)[0][None]

    pcross_ell0_arr = np.array(pcross_ell0_arr)
    pcross_ell2_arr = np.array(pcross_ell2_arr)
    pcross_ell4_arr = np.array(pcross_ell4_arr)
    pcrossmod_ell0_arr = np.array(pcrossmod_ell0_arr)[0][None]
    pcrossmod_ell2_arr = np.array(pcrossmod_ell2_arr)[0][None]
    pcrossmod_ell4_arr = np.array(pcrossmod_ell4_arr)[0][None]

    pdata_cy_arr = np.array(pdata_cy_arr)
    phimod_cy_arr = np.array(phimod_cy_arr)[0][None]
    pg_cy_arr = np.array(pg_cy_arr)
    pgmod_cy_arr = np.array(pgmod_cy_arr)[0][None]
    pcross_cy_arr = np.array(pcross_cy_arr)
    pcrossmod_cy_arr = np.array(pcrossmod_cy_arr)[0][None]

    keff = keff_list[0]
    nmodes = nmodes_list[0]

    np.savez(
        "/users/dtassie/foregroundsims/meer21cm/papers/validation/00_multipole.npz",
        pdata_ell0_arr=pdata_ell0_arr,
        pdata_ell2_arr=pdata_ell2_arr,
        pdata_ell4_arr=pdata_ell4_arr,
        phimod_ell0_arr=phimod_ell0_arr,
        phimod_ell2_arr=phimod_ell2_arr,
        phimod_ell4_arr=phimod_ell4_arr,
        pg_ell0_arr=pg_ell0_arr,
        pg_ell2_arr=pg_ell2_arr,
        pg_ell4_arr=pg_ell4_arr,
        pgmod_ell0_arr=pgmod_ell0_arr,
        pgmod_ell2_arr=pgmod_ell2_arr,
        pgmod_ell4_arr=pgmod_ell4_arr,
        pcross_ell0_arr=pcross_ell0_arr,
        pcross_ell2_arr=pcross_ell2_arr,
        pcross_ell4_arr=pcross_ell4_arr,
        pcrossmod_ell0_arr=pcrossmod_ell0_arr,
        pcrossmod_ell2_arr=pcrossmod_ell2_arr,
        pcrossmod_ell4_arr=pcrossmod_ell4_arr,
        pdata_cy_arr=pdata_cy_arr,
        phimod_cy_arr=phimod_cy_arr,
        pg_cy_arr=pg_cy_arr,
        pgmod_cy_arr=pgmod_cy_arr,
        pcross_cy_arr=pcross_cy_arr,
        pcrossmod_cy_arr=pcrossmod_cy_arr,
        keff=keff,
        nmodes=nmodes,
    )


if __name__ == "__main__":
    # run the simulations
    main()
