import numpy as np
import pytest
from meer21cm.mock import (
    gen_random_gal_pos,
    run_poisson_mock,
    HISimulation,
    run_lognormal_mock,
    gen_clustering_gal_pos,
)
from astropy.cosmology import Planck18
from hiimtool.basic_util import himf_pars_jones18, centre_to_edges, f_21
from unittest.mock import patch
import matplotlib.pyplot as plt
import sys

python_ver = sys.version_info


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
    hisim.get_hifluxdensity_ch(cache=True)
    hisim.get_hi_map(cache=True)


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
    if python_ver < (3, 9):
        with pytest.raises(ImportError):
            hisim.get_gal_pos()
    else:
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
        if python_ver < (3, 9):
            return 1
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
    if i == 1 and python_ver < (3, 9):
        return 1
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
    nu_edges = centre_to_edges(test_nu)
    assert np.allclose(himap_g[:, :, 0].shape, test_W[:, :, 0].shape)
    assert himap_g.shape[-1] == len(test_nu)
    assert inside_range.sum() == num_g
    assert np.mean(ra_g[inside_range] > test_GAMA_range[0][0]) == 1
    assert np.mean(ra_g[inside_range] < test_GAMA_range[0][1]) == 1
    assert np.mean(dec_g[inside_range] > test_GAMA_range[1][0]) == 1
    assert np.mean(dec_g[inside_range] < test_GAMA_range[1][1]) == 1
    # test z_g_mock distribution is not needed as the underlying function is tested in hiimtool
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
        assert (himap_g[himap_g > 0]).sum() == hifluxd_in.sum()
        assert himap_g[indx_1_g, indx_2_g, gal_which_ch] > 0
        # hiflux should have larger width than velocity so some empty channel
        assert (hifluxd_in == 0).sum() > 0


@pytest.mark.parametrize(
    "i, test_mock_func", [(0, run_poisson_mock), (1, run_lognormal_mock)]
)
def test_raise_error(i, test_mock_func, test_wproj, test_W, test_nu, test_GAMA_range):
    num_g = 1
    if i == 1 and python_ver < (3, 9):
        return 1
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


@pytest.mark.parametrize("test_mock_func", [(run_poisson_mock), (run_lognormal_mock)])
def test_plt(test_mock_func, test_wproj, test_W, test_nu, test_GAMA_range):
    if test_mock_func is run_lognormal_mock and python_ver < (3, 9):
        return 1
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
    nu_edges = centre_to_edges(nu)

    assert np.allclose(himap_g[:, :, 0].shape, test_W[:, :, 0].shape)
    assert himap_g.shape[-1] == len(nu)
    assert inside_range.sum() == num_g
    assert np.mean(ra_g[inside_range] > test_GAMA_range[0][0]) == 1
    assert np.mean(ra_g[inside_range] < test_GAMA_range[0][1]) == 1
    assert np.mean(dec_g[inside_range] > test_GAMA_range[1][0]) == 1
    assert np.mean(dec_g[inside_range] < test_GAMA_range[1][1]) == 1
    # test z_g_mock distribution is not needed as the underlying function is tested in hiimtool
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
    if python_ver < (3, 9):
        return 1
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
