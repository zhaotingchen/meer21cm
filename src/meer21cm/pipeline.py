from astropy import constants, units
import numpy as np
from .cosmology import CosmologyCalculator
from .mock import HISimulation
from .fg import ForegroundSimulation
from .telescope import cmb_temperature, galaxy_temperature, receiver_temperature_meerkat


class MockObservation(HISimulation, ForegroundSimulation, CosmologyCalculator):
    def __init__(
        self,
        nu,
        wproj,
        time_resol=None,
        **settings,
    ):
        super().__init__(
            nu=nu,
            wproj=wproj,
            **settings,
        )
        super(HISimulation, self).__init__(nu=nu, wproj=wproj, **settings)
        super(ForegroundSimulation, self).__init__(nu=nu, wproj=wproj, **settings)
        self.time_resol = time_resol
        self.z = self.z_ch.mean()
        self.__dict__.update(settings)

    def vel_to_freq(self, vel):
        return vel / self.vel_resol * self.freq_resol

    def freq_to_vel(self, freq):
        return freq / self.freq_resol * self.vel_resol

    @property
    def jy_to_kelvin(self):
        pix_area = units.deg**2 * self.pix_area
        equiv = units.brightness_temperature(self.nu * units.Hz)
        return (1 * units.Jy / pix_area).to(units.K, equivalencies=equiv).value

    def thermal_noise_sigma(self):
        sys_temp = (
            cmb_temperature(self.nu)
            + galaxy_temperature(self.nu)
            + receiver_temperature_meerkat(self.nu)
        )
        sigma_n = sys_temp / np.sqrt(2 * self.freq_resol * self.time_resol)
        return sigma_n
