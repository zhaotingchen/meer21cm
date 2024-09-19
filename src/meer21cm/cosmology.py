import numpy as np
import camb
import astropy
from meer21cm import Specification
from scipy.interpolate import interp1d
from meer21cm.util import omega_hi_to_average_temp, tagging


class CosmologyCalculator(Specification):
    """
    The class for storing the cosmological model used for calculation.

    The underlying cosmological model is defined via :class:`astropy.cosmology.LambdaCDM` with all the background
    properties calculated via ``astropy``.

    The matter density fluctuation is calculated using ``camb``.
    """

    def __init__(
        self,
        nonlinear="none",
        kmax=2.0,
        kmin=1e-4,
        omegahi=5e-4,
        **params,
    ):
        super().__init__(**params)
        self.nonlinear = nonlinear
        self.kmax = kmax
        self.kmin = kmin
        self._matter_power_spectrum_fnc = None
        self.omegahi = omegahi

    @property
    def nonlinear(self):
        """
        What nonlinear input model to use for camb.
        Set to ``'none'`` for linear matter power
        and ``'both'`` for nonlinear power.
        """
        return self._nonlinear

    @nonlinear.setter
    def nonlinear(self, value):
        self._nonlinear = value
        # cosmology changed, clear cache
        self.clean_cache(self.cosmo_dep_attr)

    @property
    def average_hi_temp(self):
        """
        The average HI brightness temperature in Kelvin.
        """
        tbar = omega_hi_to_average_temp(self.omegahi, z=self.z, cosmo=self)
        return tbar

    @Specification.cosmo.setter
    def cosmo(self, value):
        cosmo = value
        if isinstance(value, str):
            cosmo = getattr(astropy.cosmology, value)
        self._cosmo = cosmo
        self.ns = cosmo.meta["n"]
        self.sigma8 = cosmo.meta["sigma8"]
        self.tau = cosmo.meta["tau"]
        self.Oc0 = cosmo.meta["Oc0"]
        # there is probably a more elegant way of doing this, but I dont know how
        # maybe just inheriting astropy cosmology class?
        for key in cosmo.__dir__():
            if key[0] != "_":
                self.__dict__.update({key: getattr(cosmo, key)})
        self.camb_pars = self.get_camb_pars()
        # cosmology changed, clear cache
        self.clean_cache(self.cosmo_dep_attr)

    def get_camb_pars(self):
        """
        The associated :class:`camb.model.CAMBparams` set by the input cosmology.
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.H0.value,
            ombh2=self.Ob0 * self.h**2,
            omch2=self.Oc0 * self.h**2,
            omk=self.Ok0,
            mnu=self.m_nu.value.sum(),
            nnu=self.Neff,
            TCMB=self.Tcmb0.value,
            tau=self.tau,
        )
        # rescale As based on input sigma8
        As_fid = 2e-9
        pars.InitPower.set_params(As=As_fid, ns=self.ns)
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        results = camb.get_results(pars)
        s8_fid = results.get_sigma8_0()
        self.As = As_fid * self.sigma8**2 / s8_fid**2
        pars.InitPower.set_params(
            As=self.As,
            ns=self.ns,
        )
        pars.set_matter_power(redshifts=[self.z], kmax=2.0)
        results = camb.get_results(pars)
        self.f_growth = results.get_fsigma8()[0] / results.get_sigma8()[0]
        self.sigma_8_z = results.get_sigma8()[0]
        return pars

    @property
    @tagging("cosmo", "nu")
    def matter_power_spectrum_fnc(self):
        """
        Interpolation function for the real-space isotropic matter power spectrum.
        """
        if self._matter_power_spectrum_fnc is None:
            self.get_matter_power_spectrum()
        return self._matter_power_spectrum_fnc

    def get_matter_power_spectrum(self):
        """
        Invoke to calculate the interpolcation function for the matter
        power spectrum in real space.
        """
        pars = self.camb_pars
        pars.set_matter_power(
            redshifts=np.unique(np.array([self.z, 0.0])).tolist(),
            kmax=self.kmax / self.h,
        )
        pars.NonLinear = getattr(camb.model, "NonLinear_" + self.nonlinear)
        results = camb.get_results(pars)
        kh, z, pk = results.get_matter_power_spectrum(
            minkh=self.kmin / self.h, maxkh=self.kmax / self.h, npoints=200
        )
        karr = kh * self.h
        pkarr = pk[-1] / self.h**3
        matter_power_func = interp1d(
            karr, pkarr, bounds_error=False, fill_value="extrapolate"
        )
        self._matter_power_spectrum_fnc = matter_power_func

    # weights_1 and weights_2 are later used in power spectrum
    @property
    def weights_1(self):
        """
        The weights in real space grids for tracer 1.
        """
        return self._weights_1

    @property
    def weights_2(self):
        """
        The weights in real space grids for tracer 2.
        """
        return self._weights_2

    @weights_1.setter
    def weights_1(self, value):
        # if weight is updated, clear fourier field
        self._weights_1 = value
        if "field_1_dep_attr" in dir(self):
            self.clean_cache(self.field_1_dep_attr)
        if "tracer_1_dep_attr" in dir(self):
            self.clean_cache(self.tracer_1_dep_attr)

    @weights_2.setter
    def weights_2(self, value):
        # if weight is updated, clear fourier field
        self._weights_2 = value
        if "field_2_dep_attr" in dir(self):
            self.clean_cache(self.field_2_dep_attr)
        if "tracer_2_dep_attr" in dir(self):
            self.clean_cache(self.tracer_2_dep_attr)
