import numpy as np
from meer21cm.power import *
from powerbox import PowerBox


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
    assert np.abs(np.std(rand_fourier) - 1.0) < 1e-2
    assert np.abs(np.mean(rand_fourier)) < 1e-2
    assert np.abs((np.abs(rand_fourier) ** 2).mean() - 1) < 1e-2
    rand_arr += 1
    rand_fourier = get_fourier_density(
        rand_arr,
        mean_center=True,
        unitless=True,
        norm="ortho",
    )
    assert np.abs(np.std(rand_fourier) - 1.0) < 1e-2
    assert np.abs(np.mean(rand_fourier)) < 1e-2
    assert np.abs((np.abs(rand_fourier) ** 2).mean() - 1) < 1e-2


def test_get_power_spectrum():
    complex_rand = np.random.normal(size=100000) + 1j * np.random.normal(size=100000)
    complex_rand /= np.sqrt(2)
    power_3d = get_power_spectrum(complex_rand, [1, 1, 1])
    assert np.abs(power_3d.mean() - 1) < 1e-2
    assert np.abs(power_3d.std() - 1) < 1e-2
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
    assert np.abs(power_3d.mean() / power_sn - 1) < 1e-2
    assert np.abs(power_3d.std() / power_sn - 1) < 1e-2
