import numpy as np
import pytest
from astropy.cosmology import Planck18, WMAP1
from meer21cm.util import *
import sys
from scipy.special import erf


def test_tagging():
    @tagging("test")
    def foo():
        pass

    assert foo.tags == ("test",)


def test_find_property_with_tags():
    class Foo:
        def __init__(
            self,
            xinit=1,
        ):
            self.x = xinit
            self.dependency_dict = find_property_with_tags(self)

        @property
        @tagging("test")
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

        @property
        def y(self):
            return self._y

        @y.setter
        def y(self, value):
            self._y = value

    foo = Foo(1)
    foo.x = 2
    assert foo.dependency_dict == {"x": ("test",)}


def test_center_to_edges():
    outarr = center_to_edges(np.linspace(0.5, 9.5, 10))
    assert np.allclose(outarr, np.linspace(0, 10, 11))


def test_find_ch_id():
    ch_id = find_ch_id(np.array([0.1, 0.6, 1.7, 3.5]), np.array([0, 1, 2]))
    assert np.allclose(ch_id, np.array([0, 1, 2, 3]))


def test_omega_hi_to_average_temp():
    # just for tests
    omega_hi = 1e-4
    z = 0
    HzoverH0 = (Planck18.H(z) / Planck18.H0).to("").value
    tbar_old = 0.18 * omega_hi * Planck18.h * (1 + z) ** 2 / HzoverH0
    tbar = omega_hi_to_average_temp(omega_hi, z=z, cosmo=Planck18)
    assert (np.abs(tbar - tbar_old) / tbar) < 1e-1
    # test another cosmology
    HzoverH0 = (WMAP1.H(z) / WMAP1.H0).to("").value
    tbar_old = 0.18 * omega_hi * WMAP1.h * (1 + z) ** 2 / HzoverH0
    tbar = omega_hi_to_average_temp(omega_hi, z=z, cosmo=WMAP1)
    assert (np.abs(tbar - tbar_old) / tbar) < 1e-1


def test_freq_redshift():
    assert freq_to_redshift(f_21) == 0.0
    assert freq_to_redshift(f_21 / 2) == 1.0
    assert redshift_to_freq(0.0) == f_21
    assert redshift_to_freq(1.0) == f_21 / 2


def test_get_ang_between_coord():
    ra1 = np.zeros(11)
    ra2 = np.array([0])
    dec1 = np.linspace(-30, -40, 11)
    dec2 = np.array([80])
    ang = get_ang_between_coord(ra1, dec1, ra2, dec2)
    assert np.allclose(ang.ravel(), dec2 - dec1)


def test_generate_colored_noise():
    rand_arr = [
        generate_colored_noise(100, 100, lambda k: np.ones_like(k)) for i in range(1000)
    ]
    rand_arr = np.array(rand_arr)
    assert np.allclose(rand_arr.mean(), 0.0)
    assert np.abs(rand_arr.std() - 1.0) < 0.2


def test_get_default_args():
    def test_func(x, arg1=1):
        return 1

    defaults = get_default_args(test_func)
    assert len(defaults) == 1
    for k, v in defaults.items():
        assert k == "arg1"
        assert v == 1


def test_get_wcs_coor(test_wproj, test_wcs):
    with pytest.raises(Exception) as e_info:
        get_wcs_coor(test_wcs, np.arange(10), np.arange(10))
    get_wcs_coor(test_wproj, np.arange(10), np.arange(10))


def test_pcaclean():
    test_arr = np.random.normal(size=(10))
    with pytest.raises(Exception) as e_info:
        pcaclean(test_arr, 1, return_analysis=True)
    test_arr = np.random.normal(size=(200, 200, 10))
    C, eignumb, eigenval, V = pcaclean(test_arr, 1, return_analysis=True)
    assert (np.abs(C - np.eye((test_arr.shape[-1]))) < 0.1).mean() == 1
    test_arr = np.random.normal(size=(10, 200, 200))
    # test renorm
    C, eignumb, eigenval, V = pcaclean(
        test_arr,
        1,
        weights=2 * np.ones_like(test_arr),
        return_analysis=True,
        los_axis=0,
        mean_centre=True,
    )
    assert C.shape == (10, 10)
    assert V.shape == (10, 10)
    assert (np.abs(C - np.eye((test_arr.shape[0]))) < 0.1).mean() == 1
    assert np.allclose(eignumb, np.linspace(1, 10, 10))
    assert np.std(eigenval) < 0.1
    res_arr = pcaclean(
        test_arr,
        1,
        return_analysis=False,
        los_axis=0,
        weights=2 * np.ones_like(test_arr),
        mean_centre=True,
    )
    assert res_arr.shape == test_arr.shape
    assert np.abs((res_arr).mean()) < 1e-3
    res_arr, A_mat = pcaclean(
        test_arr,
        1,
        return_analysis=False,
        los_axis=0,
        mean_centre=True,
        return_A=True,
    )
    res_arr = pcaclean(
        test_arr,
        1,
        return_analysis=False,
        los_axis=0,
        weights=2 * np.ones_like(test_arr),
        mean_centre=True,
        mean_centre_weights=np.ones_like(test_arr),
    )
    assert res_arr.shape == test_arr.shape
    assert np.abs((res_arr).mean()) < 1e-3


def test_radec_to_indx(test_wproj):
    indx_i, indx_j = radec_to_indx(0, -30, test_wproj, to_int=True)
    indx_1, indx_2 = radec_to_indx(0, -30, test_wproj, to_int=False)
    assert np.round(indx_1) == indx_i
    assert np.round(indx_2) == indx_j


def test_rebin_spectrum():
    test_spectrum = np.zeros(503)
    test_spectrum[503 // 2] = 1.0
    test_rebin = rebin_spectrum(test_spectrum, rebin_width=3)
    assert test_rebin.sum() == 1 / 3
    assert test_rebin.size == 503 // 3
    test_rebin = rebin_spectrum(test_spectrum, rebin_width=3, mode="sum")
    assert test_rebin.sum() == 1
    test_rebin = rebin_spectrum(test_spectrum, rebin_width=13, mode="sum")
    assert test_rebin.sum() == 1


def test_hod_obuljen18():
    mass_1 = np.log10(hod_obuljen18(11.27))
    mass_2 = np.log10(10**9.52 * np.exp(-1) / Planck18.h)
    assert np.allclose(mass_1, mass_2)


def test_check_unit_equiv():
    assert check_unit_equiv(units.m, units.cm)
    assert not check_unit_equiv(units.m, units.K)


def test_jy_to_kelvin():
    omega = np.random.uniform(0.01, 1)
    freq = np.random.uniform(0.01, 1) * 1e9
    test = jy_to_kelvin(1, omega, freq)
    test2 = (
        (
            1
            * units.Jy
            / omega
            * (constants.c / freq / units.Hz) ** 2
            / 2
            / constants.k_B
        )
        .to("K")
        .value
    )
    assert np.allclose(test, test2)


def test_busy_function_simple():
    xarr = np.linspace(-10, 10, 101)
    assert np.allclose(busy_function_simple(xarr, 2, 1, 0, 0), erf(-(xarr**2)) + 1)


def test_find_indx_for_subarr():
    arr1 = np.arange(100)
    arr2 = np.arange(1000)
    assert np.allclose(find_indx_for_subarr(arr1, arr2), np.arange(100))
    arr2[0] = 1
    with pytest.raises(AssertionError):
        find_indx_for_subarr(arr1, arr2)


def test_himf():
    h_70 = Planck18.h / 0.7
    mmin = 6
    himf_pars = himf_pars_jones18(h_70)
    nhi, omegahi, psn = cal_himf(himf_pars, mmin, Planck18)
    assert np.allclose(nhi, 0.13980687586146462)
    assert np.allclose(omegahi, 0.00036495842278914405)
    assert np.allclose(psn, 150.93927719814297)
    minput = np.linspace(mmin, 11, 500)
    nhi_cumu = cumu_nhi_from_himf(minput, mmin, himf_pars)
    assert np.allclose(nhi_cumu[0], 0.0)
    assert np.allclose(nhi_cumu[-1], nhi)


def uniform_pdf(x):
    return np.ones_like(x)


def uniform_cdf(x):
    return x


def test_sample_from_dist():
    test_sample = sample_from_dist(uniform_pdf, 0, 1, size=1000000, cdf=False, seed=42)
    count, _ = np.histogram(test_sample, bins=10)
    assert (
        np.abs((count - len(test_sample) / 10) / (len(test_sample) / 10)) > 0.01
    ).sum() == 0
    test_sample = sample_from_dist(uniform_cdf, 0, 1, size=1000000, cdf=True, seed=42)
    count, _ = np.histogram(test_sample, bins=10)
    assert (
        np.abs((count - len(test_sample) / 10) / (len(test_sample) / 10)) > 0.01
    ).sum() == 0


def test_tully_fisher():
    assert np.allclose(tully_fisher(np.ones(100), 0, 2), 1e2)
    assert np.allclose(tully_fisher(np.ones(100), 1, 2, inv=True), 1e-2)
