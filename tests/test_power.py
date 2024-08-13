import numpy as np
from meer21cm.power import *
from powerbox import PowerBox
import pytest


def test_get_k_vector():
    box_len = np.array([100, 200, 60])
    box_dim = np.array([10, 20, 6])
    box_resol = box_len / box_dim
    kvec = get_k_vector(box_dim, box_resol)
    for i in range(3):
        kvec_i = kvec[i]
        assert np.abs(kvec_i).max() == np.pi / box_resol[i]
        assert np.abs(kvec_i[kvec_i != 0]).min() == 2 * np.pi / box_len[i]


def test_get_vec_mode():
    vec1 = np.linspace(0, 5, 6)
    vec2 = np.linspace(8, 10, 3)
    vec3 = np.linspace(15, 19, 5)
    mode1 = get_vec_mode((vec1, vec2, vec3))
    mode2 = np.sqrt(
        vec1[:, None, None] ** 2 + vec2[None, :, None] ** 2 + vec3[None, None, :] ** 2
    )
    assert np.allclose(mode1, mode2)


def test_get_fourier_density():
    rand_arr = np.random.normal(size=(50, 50, 50))
    rand_fourier = get_fourier_density(rand_arr, norm="ortho")
    assert np.abs(np.std(rand_fourier) - 1.0) < 5e-2
    assert np.abs(np.mean(rand_fourier)) < 5e-2
    assert np.abs((np.abs(rand_fourier) ** 2).mean() - 1) < 5e-2
    rand_arr += 1
    rand_fourier = get_fourier_density(
        rand_arr,
        mean_center=True,
        unitless=True,
        norm="ortho",
    )
    assert np.abs(np.std(rand_fourier) - 1.0) < 5e-2
    assert np.abs(np.mean(rand_fourier)) < 5e-2
    assert np.abs((np.abs(rand_fourier) ** 2).mean() - 1) < 5e-2


def test_get_power_spectrum():
    complex_rand = np.random.normal(size=100000) + 1j * np.random.normal(size=100000)
    complex_rand /= np.sqrt(2)
    power_3d = get_power_spectrum(complex_rand, [1, 1, 1])
    assert np.abs(power_3d.mean() - 1) < 2e-2
    assert np.abs(power_3d.std() - 1) < 2e-2
    spindx = np.random.uniform(-3, 0)
    box_len = np.array([100, 50, 100])
    box_dim = np.array([100, 200, 60])
    box_resol = box_len / box_dim
    kvec = get_k_vector(box_dim, box_resol)
    kmode = get_vec_mode(kvec)
    pb = PowerBox(
        box_dim,
        lambda k: k ** (spindx),
        dim=3,
        boxlength=box_len,
    )
    delta_x = pb.delta_x()
    delta_fourier = get_fourier_density(delta_x)
    mean_power = (
        (get_power_spectrum(delta_fourier, box_len) / kmode ** (spindx))[kmode > 0]
    ).mean()
    # current lack of precision seems to be a powerbox bug?
    assert np.abs(mean_power - 1) < 0.1
    # test poisson galaxies
    delta_x = np.zeros(100**3)
    num_g = 1000
    rand_choice = np.random.choice(np.arange(100**3), num_g, replace=False)
    delta_x[rand_choice] += 1.0
    delta_x = delta_x.reshape((100, 100, 100))
    delta_fourier = get_fourier_density(delta_x, mean_center=True, unitless=True)
    power_3d = get_power_spectrum(delta_fourier, np.array([100, 100, 100]))
    power_sn = 1e6 / num_g
    assert np.abs(power_3d.mean() / power_sn - 1) < 5e-2
    assert np.abs(power_3d.std() / power_sn - 1) < 5e-2


def test_MapPowerSpectrum():
    box_len = np.array([100, 50, 100])
    box_dim = np.array([100, 200, 60])
    box_resol = box_len / box_dim
    kvec = get_k_vector(box_dim, box_resol)
    kmode = get_vec_mode(kvec)
    delta_x = np.zeros(box_dim).ravel()
    num_g = 10000
    rand_choice = np.random.choice(np.arange(np.prod(box_dim)), num_g, replace=False)
    delta_x[rand_choice] += 1.0
    delta_x = delta_x.reshape(box_dim)
    sn = get_shot_noise(
        delta_x,
        box_len,
    )
    ps = MapPowerSpectrum(
        delta_x,
        box_len,
        remove_sn_1=True,
        unitless_1=True,
        mean_center_1=True,
    )
    for i in range(3):
        assert np.allclose(ps.k_vec[i], kvec[i])
    assert np.allclose(ps.k_mode, kmode)
    ps.k_perp
    ps.k_para
    ps.fourier_field_2
    ps.auto_power_3d_2
    ps.cross_power_3d
    power = ps.auto_power_3d_1
    assert np.abs(power.mean()) < 1

    ps = MapPowerSpectrum(
        delta_x,
        box_len,
        remove_sn_1=True,
        unitless_1=True,
        mean_center_1=True,
        field_2=delta_x,
        remove_sn_2=True,
        mean_center_2=True,
        unitless_2=True,
    )
    power = ps.auto_power_3d_2
    assert np.abs(power.mean()) < 1
    power = ps.cross_power_3d
    assert np.abs((power.mean() - sn) / sn) < 2e-2


def test_get_shot_noise():
    # test poisson galaxies
    delta_x = np.zeros(100**3)
    num_g = 1000
    rand_choice = np.random.choice(np.arange(100**3), num_g, replace=False)
    delta_x[rand_choice] += 1.0
    delta_x = delta_x.reshape((100, 100, 100))
    box_len = np.array([100, 100, 100])
    power = get_shot_noise(
        delta_x,
        box_len,
    )
    power_sn = 1e6 / num_g
    assert power == power_sn


def test_raise_error():
    delta_x = np.ones([100, 100, 100])
    box_len = [1, 1, 1]
    delta_2 = np.ones([2, 2, 2])
    with pytest.raises(AssertionError):
        ps = MapPowerSpectrum(delta_x, box_len, field_2=delta_2)


def test_bin_functions():
    power_3d = np.ones([10, 10, 5])
    k_perp = np.linspace(0, 99, 100).reshape((10, 10))
    kperp_edges = np.linspace(0, 100, 11)
    power_cy = bin_3d_to_cy(
        power_3d,
        k_perp,
        kperp_edges,
    )
    assert np.allclose(power_cy.shape, [5, 10])
    assert np.allclose(power_cy, np.ones_like(power_cy))
    power_3d = np.linspace(0, 9, 10)
    k_mode = np.linspace(90, 99, 10)
    k1d_edges = np.linspace(90, 100, 11)
    ps1d, ps1derr, nmodes, k1deff = bin_3d_to_1d(
        power_3d,
        k_mode,
        k1d_edges,
        error=True,
    )
    assert np.allclose(ps1d, np.linspace(0, 9, 10))
    assert np.allclose(k1deff, k_mode)
    assert np.allclose(nmodes, np.ones_like(nmodes))
    assert np.allclose(ps1derr, np.zeros_like(nmodes))
    ps1d, k1deff = bin_3d_to_1d(
        power_3d,
        k_mode,
        k1d_edges,
        error=False,
    )


def test_get_gaussian_noise_floor():
    box_len = np.array([80, 50, 100])
    box_dim = np.array([100, 200, 41])
    box_resol = box_len / box_dim
    rand_noise = np.random.normal(size=box_dim)
    ps = MapPowerSpectrum(
        rand_noise,
        box_len,
        remove_sn_1=False,
        unitless_1=False,
        mean_center_1=False,
    )
    power = ps.auto_power_3d_1
    floor1 = ps.auto_power_3d_1.mean()
    floor2 = get_gaussian_noise_floor(
        1,
        box_dim,
        box_volume=np.prod(ps.box_len),
    )
    assert np.abs((floor1 - floor2) / floor1) < 2e-2
    # test weights
    counts = np.random.randint(1, 100, size=rand_noise.shape)
    rand_noise = rand_noise / np.sqrt(counts)
    ps = MapPowerSpectrum(
        rand_noise,
        box_len,
        remove_sn_1=False,
        unitless_1=False,
        mean_center_1=False,
    )
    power = ps.auto_power_3d_1
    floor1 = ps.auto_power_3d_1.mean()
    floor2 = get_gaussian_noise_floor(
        1,
        box_dim,
        box_volume=np.prod(ps.box_len),
        counts=counts,
    )
    assert np.abs((floor1 - floor2) / floor1) < 2e-2


def test_get_independent_fourier_modes():
    rand_int = np.random.randint(2, 50)
    box_dim = np.array([2 * rand_int, 2 * rand_int, 2 * rand_int])
    indep_modes = get_independent_fourier_modes(box_dim)
    assert indep_modes.sum() == np.prod(box_dim) - (np.prod(box_dim - 1) // 2)
    box_dim += 1
    indep_modes = get_independent_fourier_modes(box_dim)
    assert indep_modes.sum() == np.prod(box_dim) // 2 + 1


def test_ModelPowerSpectrum():
    # test fog
    model = ModelPowerSpectrum()
    assert np.allclose(model.fog_term(1e7), np.ones(len(model.kmode)))
    model.mumode = np.ones_like(model.kmode)
    assert np.allclose(model.fog_term(np.inf), np.zeros(len(model.kmode)))

    # test matter power with no rsd
    model = ModelPowerSpectrum()
    model.get_model_power()
    matter_ps_real = model.matter_power_spectrum_fnc(model.kmode)
    assert np.allclose(model.auto_power_matter, matter_ps_real)

    # add rsd, test kaiser term
    model.mumode = np.ones_like(model.kmode)
    model.get_model_power()
    matter_ps_rsd = model.auto_power_matter
    assert np.allclose(matter_ps_rsd / matter_ps_real, (1 + model.f_growth) ** 2)

    # test tracer with no rsd but with bias
    model = ModelPowerSpectrum(tracer_bias_1=2.0)
    model.get_model_power()
    tracer_ps_rsd = model.auto_power_tracer_1
    assert np.allclose(tracer_ps_rsd, matter_ps_real * 4)

    # add rsd
    model.mumode = np.ones_like(model.kmode)
    model.matter_only_rsd = True
    model.get_model_power()
    tracer_ps_rsd = model.auto_power_tracer_1
    assert np.allclose(tracer_ps_rsd / matter_ps_real, (1 + model.f_growth) ** 2 * 4)

    # true rsd
    model.matter_only_rsd = False
    model.get_model_power()
    tracer_ps_rsd = model.auto_power_tracer_1
    assert np.allclose(
        tracer_ps_rsd / matter_ps_real, (1 + model.f_growth / 2.0) ** 2 * 4
    )

    # test 2 tracers with no rsd but with bias
    model = ModelPowerSpectrum(
        tracer_bias_1=2.0,
        tracer_bias_2=3.0,
        cross_coeff=0.5,
    )
    model.get_model_power()
    tracer_ps_rsd = model.auto_power_tracer_1
    assert np.allclose(tracer_ps_rsd, matter_ps_real * 4)
    tracer_ps_rsd = model.auto_power_tracer_2
    assert np.allclose(tracer_ps_rsd, matter_ps_real * 9)

    # add rsd
    model.mumode = np.ones_like(model.kmode)
    model.matter_only_rsd = True
    model.get_model_power()
    tracer_ps_rsd = model.auto_power_tracer_2
    cross_ps_rsd = model.cross_power_tracer
    assert np.allclose(tracer_ps_rsd / matter_ps_real, (1 + model.f_growth) ** 2 * 9)
    assert np.allclose(
        cross_ps_rsd / matter_ps_real, (1 + model.f_growth) ** 2 * 6 - 6 + 6 * 0.5
    )

    # true rsd
    model.matter_only_rsd = False
    model.get_model_power()
    tracer_ps_rsd = model.auto_power_tracer_2
    cross_ps_rsd = model.cross_power_tracer
    assert np.allclose(
        tracer_ps_rsd / matter_ps_real, (1 + model.f_growth / 3.0) ** 2 * 9
    )
    assert np.allclose(
        cross_ps_rsd / matter_ps_real,
        (1 + model.f_growth / 2.0) * (1 + model.f_growth / 3.0) * 6 - 6 + 6 * 0.5,
    )
