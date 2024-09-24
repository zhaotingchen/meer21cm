import numpy as np
import pytest
from meer21cm.mock import (
    gen_random_gal_pos,
    run_poisson_mock,
    HISimulation,
    run_lognormal_mock,
    gen_clustering_gal_pos,
    MockSimulation,
    hi_mass_to_flux_profile,
)
from meer21cm.util import hod_obuljen18
from astropy.cosmology import Planck18, WMAP1
from meer21cm.util import himf_pars_jones18, center_to_edges, f_21
from unittest.mock import patch
import matplotlib.pyplot as plt
import sys
from meer21cm.power import PowerSpectrum


def test_matter_mock(test_W):
    # default is Planck18, so use WMAP1 to test
    # if cosmo is properly updated throughout
    k1dedges = np.geomspace(0.05, 1.5, 20)

    mock = MockSimulation(
        cosmo=WMAP1,
        k1dbins=k1dedges,
        model_k_from_field=True,
        upgrade_sampling_from_gridding=True,
        kaiser_rsd=False,
    )
    mock.map_has_sampling = test_W * np.ones_like(mock.nu)[None, None, :]
    mock.get_enclosing_box()
    # underlying code has been tested in grid
    # simply test invoking
    mock.box_origin
    mock.box_len
    mock.box_resol
    mock.box_ndim
    mock.rot_mat_sky_to_box
    mock.pix_coor_in_cartesian
    mock.pix_coor_in_box
    # test input and output power consistency
    mock.k1dbins = k1dedges

    mock.field_1 = mock.mock_matter_field
    pfield_i, keff, nmodes = mock.get_1d_power(
        "auto_power_3d_1",
    )

    pmatter3d = mock.matter_power_spectrum_fnc(mock.kmode)
    pm1d, _, _ = mock.get_1d_power(
        pmatter3d,
    )
    avg_deviation = np.sqrt(
        ((np.abs((pfield_i - pm1d) / pm1d)) ** 2 * nmodes).sum() / nmodes.sum()
    )
    assert avg_deviation < 2e-1
    # test RSD
    mock.kaiser_rsd = True
    # mock.get_mock_matter_field()
    mock.field_1 = mock.mock_matter_field
    pfield_i_rsd, keff, nmodes = mock.get_1d_power(
        "auto_power_3d_1",
    )
    pm1d_rsd, _, _ = mock.get_1d_power(mock.auto_power_matter_model)
    avg_deviation = np.sqrt(
        ((np.abs((pfield_i_rsd - pm1d_rsd) / pm1d_rsd)) ** 2 * nmodes).sum()
        / nmodes.sum()
    )
    assert avg_deviation < 2e-1


@pytest.mark.parametrize("tracer_i", [(1), (2)])
def test_tracer_mock(test_W, tracer_i):
    k1dedges = np.geomspace(0.05, 1.5, 20)

    mock = MockSimulation(
        tracer_bias_1=1.5,
        tracer_bias_2=1.9,
        cosmo=WMAP1,
        k1dbins=k1dedges,
        kaiser_rsd=True,
        # mock is generated on the grid so no sampling effects
        include_sampling=[False, False],
        downres_factor_transverse=0.8,
        downres_factor_radial=0.8,
        model_k_from_field=True,
        upgrade_sampling_from_gridding=True,
        # mean_amp_1='average_hi_temp',
    )
    setattr(mock, "mean_amp_" + str(tracer_i), "average_hi_temp")
    mock.map_has_sampling = test_W * np.ones_like(mock.nu)[None, None, :]
    # mock.get_mock_tracer_field()
    mock.field_1 = mock.mock_tracer_field_1
    mock.field_2 = mock.mock_tracer_field_2

    pfield_1_rsd, keff, nmodes = mock.get_1d_power(
        "auto_power_3d_1",
    )
    pfield_2_rsd, keff, nmodes = mock.get_1d_power(
        "auto_power_3d_2",
    )
    pfield_c_rsd, keff, nmodes = mock.get_1d_power(
        "cross_power_3d",
    )

    pmod_1, _, _ = mock.get_1d_power((mock.auto_power_tracer_1_model))
    pmod_2, _, _ = mock.get_1d_power((mock.auto_power_tracer_2_model))
    pmod_c, _, _ = mock.get_1d_power((mock.cross_power_tracer_model))
    pfield = [pfield_1_rsd, pfield_2_rsd, pfield_c_rsd]
    pmod = [pmod_1, pmod_2, pmod_c]
    for i in range(3):
        avg_deviation = np.sqrt(
            ((np.abs((pfield[i] - pmod[i]) / pmod[i])) ** 2 * nmodes).sum()
            / nmodes.sum()
        )
        # the accuracy is not good due to large variance of a single realization
        # multiple realizations are tested in test_pipeline.py
        assert avg_deviation < 1


def test_tracer_position():
    raminGAMA, ramaxGAMA = 339, 351
    decminGAMA, decmaxGAMA = -35, -30
    ra_range = (raminGAMA, ramaxGAMA)
    dec_range = (decminGAMA, decmaxGAMA)
    with pytest.raises(ValueError):
        mock = MockSimulation(
            ra_range=ra_range,
            dec_range=dec_range,
            kaiser_rsd=True,
            seed=42,
            discrete_base_field="3",
        )
    # now do a proper mock based on tracer 2
    mock = MockSimulation(
        ra_range=ra_range,
        dec_range=dec_range,
        kaiser_rsd=True,
        discrete_base_field=2,
        target_relative_to_num_g=2.5,
    )
    mock.data = np.ones(mock.W_HI.shape)
    mock.w_HI = np.ones(mock.W_HI.shape)
    mock.counts = np.ones(mock.W_HI.shape)
    mock.trim_map_to_range()
    mock.downres_factor_radial = 1 / 2.0
    mock.downres_factor_transverse = 1 / 2.0
    mock.get_enclosing_box()
    mock.tracer_bias_2 = 1.9
    mock.num_discrete_source = 2700
    mock.propagate_mock_tracer_to_gal_cat()
    # ensure that galaxies are there
    assert len(mock.ra_mock_tracer) > mock.num_discrete_source
    assert len(mock.dec_mock_tracer) > mock.num_discrete_source
    assert len(mock.z_mock_tracer) > mock.num_discrete_source
    assert (mock.mock_inside_range).sum() == mock.num_discrete_source
    assert len(mock.ra_gal) == mock.num_discrete_source
    # test of power spectrum is performed in pipeline tests
    # test warining raised
    mock.target_relative_to_num_g = 0.1
    with pytest.warns(UserWarning):
        mock.propagate_mock_tracer_to_gal_cat()


def test_hi_mass_to_flux():
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range_MK = (raminMK, ramaxMK)
    dec_range_MK = (decminMK, decmaxMK)
    num_g = 10000
    mock = MockSimulation(
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
        tracer_bias_1=1.5,
        tracer_bias_2=1.9,
        num_discrete_source=num_g,
        target_relative_to_num_g=1.1,
    )
    mock.propagate_mock_tracer_to_gal_cat()
    # some random galaxies
    himass_g = np.random.uniform(9, 11, mock.z_mock_tracer.size)
    hifluxd_ch = hi_mass_to_flux_profile(
        himass_g,
        mock.z_mock_tracer,
        mock.nu,
        cosmo=mock.cosmo,
        seed=mock.seed,
    )
    # approximate from 1705.04210
    approx_mass = (
        hifluxd_ch.sum(0)
        * mock.freq_resol
        * mock.luminosity_distance(mock.z_mock_tracer).value ** 2
        * 49.7
    )
    ratio = approx_mass / 10**himass_g
    # scaling should be exact
    assert np.allclose(ratio.std(), 0)
    # mean is about 1
    assert np.abs(1 - ratio.mean()) < 5e-2
    # now with velocity
    hifluxd_ch = hi_mass_to_flux_profile(
        himass_g,
        mock.z_mock_tracer,
        mock.nu,
        cosmo=mock.cosmo,
        seed=mock.seed,
        tf_slope=3.66,
        tf_zero=1.6,
        no_vel=False,
    )
    # approximate from 1705.04210
    approx_mass = (
        hifluxd_ch.sum(0)
        * mock.freq_resol
        * mock.luminosity_distance(mock.z_mock_tracer).value ** 2
        * 49.7
    )
    ratio = approx_mass / 10**himass_g
    # scaling should be exact
    assert np.allclose(ratio.std(), 0)
    # mean is about 1
    assert np.abs(1 - ratio.mean()) < 5e-2
    # with enough galaxy should be symmetric
    average_profile = hifluxd_ch.mean(-1)
    # mid point should be peak
    assert np.argmax(average_profile) == average_profile.size // 2
    # symmetric
    assert np.max(np.abs(average_profile[::-1] - average_profile)) < 1e-5


def test_auto_mmin(test_wproj, test_nu, test_W, test_GAMA_range):
    num_g = 10000
    hisim = HISimulation(
        nu=test_nu,
        wproj=test_wproj,
        num_g=num_g,
        num_pix_x=test_W.shape[0],
        num_pix_y=test_W.shape[1],
        density="lognormal",
        verbose=False,
        do_stack=False,
        x_dim=test_W.shape[0],
        y_dim=test_W.shape[1],
        ignore_double_counting=False,
        return_indx_and_weight=False,
        seed=42,
        himf_pars=himf_pars_jones18(Planck18.h / 0.7),
    )
    hisim.get_gal_pos()
    mmin_1 = hisim.mmin
    mmin_halo = hisim.mmin_halo
    hisim.auto_mmin = hod_obuljen18
    hisim.get_gal_pos()
    mmin_2 = hisim.mmin
    assert mmin_1 != mmin_2


def test_hisim_class(test_wproj, test_nu, test_W, test_GAMA_range):
    num_g = 10000
    hisim = HISimulation(
        nu=test_nu,
        wproj=test_wproj,
        num_g=num_g,
        num_pix_x=test_W.shape[0],
        num_pix_y=test_W.shape[1],
        density="poisson",
        verbose=False,
        do_stack=False,
        x_dim=test_W.shape[0],
        y_dim=test_W.shape[1],
        ignore_double_counting=False,
        return_indx_and_weight=False,
        seed=42,
        himf_pars=himf_pars_jones18(Planck18.h / 0.7),
    )
    assert np.allclose(hisim.W_HI, np.ones_like(hisim.W_HI))
    # test no cache
    hisim.get_hi_map()
    hisim.get_gal_pos(cache=True)
    assert hisim.do_stack == False
    assert hisim.seed == 42
    ra_g, dec_g, inside_range = (hisim.ra_g_mock, hisim.dec_g_mock, hisim.inside_range)
    # test when no range is specified, the sizes match with n_g
    assert len(ra_g) == num_g
    assert len(dec_g) == num_g
    assert len(inside_range) == num_g
    assert (inside_range).mean() == 1
    z_g_mock = hisim.z_g_mock
    assert ((z_g_mock - hisim.z_ch.min()) >= 0).mean() == 1
    assert ((z_g_mock - hisim.z_ch.max()) <= 0).mean() == 1
    hisim.get_hifluxdensity_ch()
    hisim.get_hi_map()


def test_import_error(test_wproj, test_nu, test_W, test_GAMA_range):
    num_g = 10000
    hisim = HISimulation(
        nu=test_nu,
        wproj=test_wproj,
        num_g=num_g,
        num_pix_x=test_W.shape[0],
        num_pix_y=test_W.shape[1],
        density="lognormal",
        W_HI=test_W,
        verbose=False,
        do_stack=False,
        x_dim=test_W.shape[0],
        y_dim=test_W.shape[1],
        ignore_double_counting=False,
        return_indx_and_weight=False,
        seed=42,
        himf_pars=himf_pars_jones18(Planck18.h / 0.7),
    )
    hisim.get_gal_pos()
    with pytest.raises(ValueError):
        hisim = HISimulation(
            nu=test_nu,
            wproj=test_wproj,
            num_g=num_g,
            num_pix_x=test_W.shape[0],
            num_pix_y=test_W.shape[1],
            density="newthing",
            W_HI=test_W,
            verbose=False,
            do_stack=False,
            x_dim=test_W.shape[0],
            y_dim=test_W.shape[1],
            ignore_double_counting=False,
            return_indx_and_weight=False,
            seed=42,
            himf_pars=himf_pars_jones18(Planck18.h / 0.7),
        )


@pytest.mark.parametrize(
    "i, test_gal_func", [(0, gen_random_gal_pos), (1, gen_clustering_gal_pos)]
)
def test_gen_random_gal_pos(
    i, test_gal_func, test_wproj, test_nu, test_W, test_GAMA_range
):
    num_g = 1000
    if i == 0:
        ra_g, dec_g, inside_range = test_gal_func(test_wproj, test_W[:, :, 0], num_g)
        # test when no range is specified, the sizes match with n_g
        assert len(ra_g) == num_g
        assert len(dec_g) == num_g
        assert len(inside_range) == num_g
        assert (inside_range).mean() == 1.0
    elif i == 1:
        ra_g, dec_g, z_g_mock, inside_range, mmin_halo = test_gal_func(
            test_nu, Planck18, test_wproj, num_g, test_W
        )
        assert len(ra_g) >= num_g
        assert len(dec_g) >= num_g
        assert len(inside_range) >= num_g
        assert (inside_range).mean() == 1.0
    # test when range specified, the number matches and the selected galaxies are in range
    if i == 0:
        ra_g, dec_g, inside_range = test_gal_func(
            test_wproj,
            test_W[:, :, 0],
            num_g,
            ra_range=test_GAMA_range[0],
            dec_range=test_GAMA_range[1],
        )
        assert inside_range.sum() == num_g
    elif i == 1:
        ra_g, dec_g, z_g_mock, inside_range, mmin_halo = test_gal_func(
            test_nu,
            Planck18,
            test_wproj,
            num_g,
            test_W,
            seed=42,
            ra_range=test_GAMA_range[0],
            dec_range=test_GAMA_range[1],
            kaiser_rsd=True,
        )
        assert inside_range.sum() >= num_g
    assert np.mean(ra_g[inside_range] > test_GAMA_range[0][0]) == 1
    assert np.mean(ra_g[inside_range] < test_GAMA_range[0][1]) == 1
    assert np.mean(dec_g[inside_range] > test_GAMA_range[1][0]) == 1
    assert np.mean(dec_g[inside_range] < test_GAMA_range[1][1]) == 1
    # currently there is just no good way for default_rng to work with pytest.
    # saving this for later
    seed_default_rng = np.random.randint(1, 10000)
    ra_g, dec_g, inside_range = gen_random_gal_pos(
        test_wproj,
        test_W[:, :, 0],
        num_g,
        ra_range=test_GAMA_range[0],
        dec_range=test_GAMA_range[1],
        seed=seed_default_rng,
    )
    assert inside_range.sum() == num_g
    assert np.mean(ra_g[inside_range] > test_GAMA_range[0][0]) == 1
    assert np.mean(ra_g[inside_range] < test_GAMA_range[0][1]) == 1
    assert np.mean(dec_g[inside_range] > test_GAMA_range[1][0]) == 1
    assert np.mean(dec_g[inside_range] < test_GAMA_range[1][1]) == 1


@pytest.mark.parametrize(
    "i, test_mock_func", [(0, run_poisson_mock), (1, run_lognormal_mock)]
)
def test_gal_pos_in_mock(
    i, test_mock_func, test_wproj, test_W, test_nu, test_GAMA_range
):
    num_g = 10000
    (
        himap_g,
        ra_g,
        dec_g,
        z_g,
        indx_1_g,
        indx_2_g,
        gal_which_ch,
        hifluxd_in,
        inside_range,
    ) = test_mock_func(
        test_nu,
        num_g,
        himf_pars_jones18(Planck18.h / 0.7),
        test_wproj,
        W_HI=test_W,
        seed=42,
        no_vel=True,
        mmin=10.5,
        verbose=False,
        do_stack=False,
        x_dim=test_W.shape[0],
        y_dim=test_W.shape[1],
        ignore_double_counting=False,
        return_indx_and_weight=False,
        ra_range=test_GAMA_range[0],
        dec_range=test_GAMA_range[1],
    )
    nu_edges = center_to_edges(test_nu)
    assert np.allclose(himap_g[:, :, 0].shape, test_W[:, :, 0].shape)
    assert himap_g.shape[-1] == len(test_nu)
    assert inside_range.sum() == num_g
    assert np.mean(ra_g[inside_range] > test_GAMA_range[0][0]) == 1
    assert np.mean(ra_g[inside_range] < test_GAMA_range[0][1]) == 1
    assert np.mean(dec_g[inside_range] > test_GAMA_range[1][0]) == 1
    assert np.mean(dec_g[inside_range] < test_GAMA_range[1][1]) == 1
    # test z_g_mock distribution is not needed as the underlying function is tested in util
    assert (f_21 / (1 + z_g) >= nu_edges[gal_which_ch]).mean() == 1
    assert (f_21 / (1 + z_g) <= nu_edges[gal_which_ch + 1]).mean() == 1
    # ra,dec mapping to index should be tested in util
    # when no velocity, the flux should be one channel
    assert len(hifluxd_in) == 1

    # generate only one galaxy
    num_g = 1
    # fix z in case it falls in the edge
    if i == 0:
        (
            himap_g,
            ra_g,
            dec_g,
            z_g,
            indx_1_g,
            indx_2_g,
            gal_which_ch,
            hifluxd_in,
            inside_range,
        ) = test_mock_func(
            test_nu,
            num_g,
            himf_pars_jones18(Planck18.h / 0.7),
            test_wproj,
            W_HI=test_W,
            seed=42,
            no_vel=False,
            tf_slope=3.66,
            tf_zero=1.6,
            mmin=10.5,
            verbose=False,
            do_stack=False,
            x_dim=test_W.shape[0],
            y_dim=test_W.shape[1],
            ignore_double_counting=False,
            return_indx_and_weight=False,
            fix_z=f_21 / test_nu.mean() - 1,
            fix_ra_dec=(350, -30),
        )
        assert z_g == f_21 / test_nu.mean() - 1
        assert ra_g == 350
        assert dec_g == -30
        assert (himap_g > 0).sum() == len(hifluxd_in[hifluxd_in > 0])
        assert np.allclose((himap_g[himap_g > 0]).sum(), hifluxd_in.sum())
        assert himap_g[indx_1_g, indx_2_g, gal_which_ch] > 0
        # hiflux should have larger width than velocity so some empty channel
        assert (hifluxd_in == 0).sum() > 0


@pytest.mark.parametrize(
    "i, test_mock_func", [(0, run_poisson_mock), (1, run_lognormal_mock)]
)
def test_raise_error(i, test_mock_func, test_wproj, test_W, test_nu, test_GAMA_range):
    num_g = 1
    with pytest.raises(ValueError):
        test_mock_func(
            test_nu,
            num_g,
            himf_pars_jones18(Planck18.h / 0.7),
            test_wproj,
            W_HI=test_W[:, 0, 0],
            seed=42,
            no_vel=True,
            mmin=10.5,
            verbose=False,
            do_stack=False,
            x_dim=test_W.shape[0],
            y_dim=test_W.shape[1],
            ignore_double_counting=False,
            return_indx_and_weight=False,
            ra_range=test_GAMA_range[0],
            dec_range=test_GAMA_range[1],
        )
    with pytest.raises(ValueError):
        test_mock_func(
            test_nu,
            num_g,
            himf_pars_jones18(Planck18.h / 0.7),
            test_wproj,
            W_HI=np.zeros_like(test_W),
            seed=42,
            no_vel=True,
            mmin=10.5,
            verbose=False,
            do_stack=False,
            x_dim=test_W.shape[0],
            y_dim=test_W.shape[1],
            ignore_double_counting=False,
            return_indx_and_weight=False,
            ra_range=test_GAMA_range[0],
            dec_range=test_GAMA_range[1],
        )
    if i == 1:
        with pytest.raises(ValueError):
            test_mock_func(
                test_nu,
                num_g,
                himf_pars_jones18(Planck18.h / 0.7),
                test_wproj,
                kaiser_rsd=True,
            )


def test_rsd_error(test_wproj, test_W, test_nu, test_GAMA_range):
    num_g = 1
    with pytest.raises(ValueError):
        _ = gen_clustering_gal_pos(
            test_nu,
            Planck18,
            test_wproj,
            num_g,
            test_W,
            ra_range=test_GAMA_range[0],
            dec_range=test_GAMA_range[1],
            kaiser_rsd=True,
        )


@pytest.mark.parametrize("test_mock_func", [(run_poisson_mock), (run_lognormal_mock)])
def test_plt(test_mock_func, test_wproj, test_W, test_nu, test_GAMA_range):
    plt.switch_backend("Agg")
    num_g = 100
    test_mock_func(
        test_nu,
        num_g,
        himf_pars_jones18(Planck18.h / 0.7),
        test_wproj,
        W_HI=test_W,
        seed=42,
        no_vel=False,
        tf_slope=3.66,
        tf_zero=1.6,
        mmin=10.5,
        verbose=True,
        do_stack=False,
        x_dim=test_W.shape[0],
        y_dim=test_W.shape[1],
        ignore_double_counting=False,
        return_indx_and_weight=False,
        fix_z=np.ones(num_g) * f_21 / test_nu.mean() - 1,
        fix_ra_dec=(np.ones(num_g) * 350, np.ones(num_g) * (-30.0)),
    )
    plt.close("all")


def test_mock_healpix(test_wproj, test_W, test_nu, test_GAMA_range):
    nu = test_nu[:3]
    num_g = 2
    (
        himap_g,
        ra_g,
        dec_g,
        z_g,
        indx_1_g,
        indx_2_g,
        gal_which_ch,
        hifluxd_in,
        inside_range,
    ) = run_poisson_mock(
        nu,
        num_g,
        himf_pars_jones18(Planck18.h / 0.7),
        test_wproj,
        W_HI=test_W,
        seed=42,
        no_vel=True,
        mmin=10.5,
        verbose=False,
        do_stack=False,
        x_dim=test_W.shape[0],
        y_dim=test_W.shape[1],
        ignore_double_counting=False,
        return_indx_and_weight=False,
        ra_range=test_GAMA_range[0],
        dec_range=test_GAMA_range[1],
        fast_ang_pos=False,
    )
    nu_edges = center_to_edges(nu)

    assert np.allclose(himap_g[:, :, 0].shape, test_W[:, :, 0].shape)
    assert himap_g.shape[-1] == len(nu)
    assert inside_range.sum() == num_g
    assert np.mean(ra_g[inside_range] > test_GAMA_range[0][0]) == 1
    assert np.mean(ra_g[inside_range] < test_GAMA_range[0][1]) == 1
    assert np.mean(dec_g[inside_range] > test_GAMA_range[1][0]) == 1
    assert np.mean(dec_g[inside_range] < test_GAMA_range[1][1]) == 1
    # test z_g_mock distribution is not needed as the underlying function is tested in util
    assert (f_21 / (1 + z_g) >= nu_edges[gal_which_ch]).mean() == 1
    assert (f_21 / (1 + z_g) <= nu_edges[gal_which_ch + 1]).mean() == 1
    # ra,dec mapping to index should be tested in util
    # when no velocity, the flux should be one channel
    assert len(hifluxd_in) == 1

    # generate only one galaxy
    num_g = 1
    nu = test_nu[:10]
    # fix z in case it falls in the edge
    (
        himap_g,
        ra_g,
        dec_g,
        z_g,
        indx_1_g,
        indx_2_g,
        gal_which_ch,
        hifluxd_in,
        inside_range,
    ) = run_poisson_mock(
        nu,
        num_g,
        himf_pars_jones18(Planck18.h / 0.7),
        test_wproj,
        W_HI=test_W,
        seed=42,
        no_vel=False,
        tf_slope=3.66,
        tf_zero=1.6,
        mmin=10.5,
        verbose=False,
        do_stack=False,
        x_dim=test_W.shape[0],
        y_dim=test_W.shape[1],
        ignore_double_counting=False,
        return_indx_and_weight=False,
        fix_z=f_21 / nu.mean() - 1,
        fast_ang_pos=False,
    )
    assert (himap_g > 0).sum() == len(hifluxd_in[hifluxd_in > 0])
    assert np.allclose((himap_g[himap_g > 0]).sum(), hifluxd_in.sum())
    assert himap_g[indx_1_g, indx_2_g, gal_which_ch] > 0
    # hiflux should have larger width than velocity so some empty channel
    assert (hifluxd_in == 0).sum() > 0


@pytest.mark.parametrize(
    "i, test_mock_func", [(0, run_poisson_mock), (1, run_lognormal_mock)]
)
def test_invoke_stack(i, test_mock_func, test_wproj, test_W, test_nu, test_GAMA_range):
    plt.switch_backend("Agg")
    # generate only one galaxy
    num_g = 100
    # fix z in case it falls in the edge
    (
        himap_g,
        ra_g,
        dec_g,
        z_g,
        indx_1_g,
        indx_2_g,
        gal_which_ch,
        hifluxd_in,
        inside_range,
        stack_3D_map,
        stack_3D_weight,
        x_edges,
        ang_edges,
    ) = test_mock_func(
        test_nu,
        num_g,
        himf_pars_jones18(Planck18.h / 0.7),
        test_wproj,
        W_HI=test_W,
        seed=42,
        no_vel=True,
        mmin=10.5,
        verbose=True,
        x_dim=test_W.shape[0],
        y_dim=test_W.shape[1],
        do_stack=True,
        ignore_double_counting=True,
        return_indx_and_weight=False,
        ra_range=test_GAMA_range[0],
        dec_range=test_GAMA_range[1],
    )
    plt.close("all")
