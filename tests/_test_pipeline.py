import numpy as np
from meer21cm.pipeline import MockObservation
from meer21cm.mock import lamb_21
from astropy import constants, units


def test_unit_convert(test_nu, test_wproj):
    mockobs = MockObservation(
        test_nu,
        test_wproj,
        seed=42,
    )
    jy_2_k = mockobs.jy_to_kelvin
    pix_area = mockobs.pix_area
    z_ch = mockobs.z_ch
    k_2_jy = (
        (
            (2 * constants.k_B * units.K / (lamb_21 * (1 + (z_ch))) ** 2)
            * (pix_area * np.pi**2 / 180**2)
        )
        .to("Jy")
        .value
    )
    assert np.allclose(jy_2_k, 1 / k_2_jy)
    assert mockobs.freq_to_vel(mockobs.freq_resol) == mockobs.vel_resol
    assert mockobs.vel_to_freq(mockobs.vel_resol) == mockobs.freq_resol


def test_noise_sigma(test_nu, test_wproj):
    mockobs = MockObservation(
        test_nu,
        test_wproj,
        seed=42,
        time_resol=2,
        num_pix_x=1,
        num_pix_y=1,
        pix_counts=np.ones_like(test_nu)[None, None, :],
    )
    sigma_n = mockobs.thermal_noise_sigma()
    mockobs.get_noise_map()
