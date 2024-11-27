import numpy as np
import camb
import astropy
from meer21cm import Specification
from scipy.interpolate import interp1d
from meer21cm.util import omega_hi_to_average_temp, tagging
from astropy.cosmology import Planck18, w0waCDM
import baccoemu


class CosmologyParameters:
    """
    The class for storing cosmological parameters, and settings for computing matter power
    spectrum. The naming of the input arguments for
    cosmological parameters follow
    `baccoemu <https://baccoemu.readthedocs.io/en/latest/>`_ .

    Note that everything is **not in h unit** unless explicitly specified in name
    (of course except sigma_8 which follows the definition of 8 Mpc/h).

    Further note that, baccoemu is trained on `CLASS <https://github.com/lesgourg/class_public>`_ .
    Therefore, in the usual range of parameters in the LCDM,
    you should see the <1% difference between these two
    backends as differences between the Boltzmann solver codes (although this
    is not well tested on our end). Use it with precaution if you want to do
    precision cosmology type of forecasts and sims.
    """

    def __init__(
        self,
        ps_type="linear",
        kmin=1e-3,
        kmax=3.0,
        omega_cold=Planck18.Om0,
        As=np.exp(3.047) / 1e10,
        # sigma8_cold=Planck18.meta['sigma8'],
        omega_baryon=Planck18.Ob0,
        ns=Planck18.meta["n"],
        hubble=Planck18.h,
        neutrino_mass=Planck18.m_nu.sum().value,
        w0=-1.0,
        wa=0.0,
        expfactor=1.0,
        cold=True,
        backend="camb",
        num_kpoints=200,
        omega_de=None,
        **params,
    ):
        self.ps_type = ps_type
        self.kmin = kmin
        self.kmax = kmax
        self.omega_cold = omega_cold
        # self.sigma8_cold = sigma8_cold
        self.As = As
        self.omega_baryon = omega_baryon
        self.ns = ns
        self.h = hubble
        self.neutrino_mass = neutrino_mass
        self.w0 = w0
        self.wa = wa
        self.expfactor = expfactor
        self.cold = cold
        self.backend = backend
        # hard coded no curvature for now
        self.Ok0 = 0
        # CMB related, not needed
        self.Neff = 3.046
        # self.Neff = 2.0
        self.Tcmb0 = 2.7255
        self.tau = 0.0561
        self.hubble = self.h
        self.camb_dark_energy_model = "ppf"
        self.num_kpoints = num_kpoints
        self.karr_in_h = np.geomspace(
            self.kmin / self.h, self.kmax / self.h, self.num_kpoints
        )
        self.omega_de = omega_de

    def set_astropy_cosmo(self):
        """
        Generate a :class:`astropy.cosmology.w0waCDM` set by the input cosmology.
        Note that the dark energy density ``Ode0`` is a derived quantity which is
        calculated using ``camb`` if not put in.
        """
        cosmo = w0waCDM(
            H0=self.h * 100,
            Om0=self.omega_cold,
            Ode0=self.omega_de,
            w0=self.w0,
            wa=self.wa,
            Tcmb0=self.Tcmb0,
            Neff=self.Neff,
            m_nu=[0, 0, self.neutrino_mass],
            Ob0=self.omega_baryon,
        )
        self.cosmo = cosmo
        return cosmo

    def get_camb_pars(self):
        """
        Generate a :class:`camb.model.CAMBparams` set by the input cosmology.
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.h * 100,
            ombh2=self.omega_baryon * self.h**2,
            omch2=(self.omega_cold - self.omega_baryon) * self.h**2,
            omk=self.Ok0,
            mnu=self.neutrino_mass,
            # these should not affect matter ps?
            nnu=self.Neff,
            TCMB=self.Tcmb0,
            tau=self.tau,
        )
        pars.InitPower.set_params(As=self.As, ns=self.ns)

        if self.ps_type == "linear":
            instr = "none"
        else:
            instr = "both"
        pars.NonLinear = getattr(camb.model, "NonLinear_" + instr)
        pars.set_dark_energy(
            w=self.w0, wa=self.wa, dark_energy_model=self.camb_dark_energy_model
        )
        pars.set_matter_power(
            redshifts=np.unique([1 / self.expfactor - 1, 0.0]), kmax=2.0
        )
        return pars

    def get_derived_Ode(self):
        """
        Use camb to calculate the Ode0 given input parameters.
        If Ode0 is given as an input, it will be skipped.
        """
        if self.omega_de is None:
            camb_pars = self.get_camb_pars()
            results = camb.get_background(camb_pars)
            tot, de = results.get_background_densities(1.0, ["tot", "de"]).values()
            self.omega_de = (de / tot)[0]
        return self.omega_de

    def get_bacco_pars(self):
        """
        Generate a dictionary that can be used as input for the
        ``bacco`` emulator. Currently only support non-baryonic
        matter power.
        """
        params = {
            "omega_cold": self.omega_cold,
            #'sigma8_cold'   :  self.sigma8_cold,
            "A_s": self.As,
            "omega_baryon": self.omega_baryon,
            "ns": self.ns,
            "hubble": self.h,
            "neutrino_mass": self.neutrino_mass,
            "w0": self.w0,
            "wa": self.wa,
            "expfactor": self.expfactor,
        }
        return params

    # def get_matter_power_spectrum(self):
    #    return getattr(self,f'get_matter_power_spectrum_{self.backend}')()

    def get_matter_power_spectrum_camb(self):
        """
        Compute the CDM power spectrum using camb.
        """
        camb_pars = self.get_camb_pars()
        camb_pars.set_matter_power(
            redshifts=np.unique([1 / self.expfactor - 1, 0.0]),
            kmax=self.kmax / self.h,
        )
        results = camb.get_results(camb_pars)
        # get sigma8
        s8_fid = results.get_sigma8_0()
        self.sigma_8_0 = s8_fid
        self.f_growth = results.get_fsigma8()[0] / results.get_sigma8()[0]
        self.sigma_8_z = results.get_sigma8()[0]
        kh, z, pk_camb = results.get_matter_power_spectrum(
            minkh=self.kmin / self.h,
            maxkh=self.kmax / self.h,
            npoints=self.num_kpoints,
            var1=7 - 5 * int(self.cold),
            var2=7 - 5 * int(self.cold),
        )
        return pk_camb[0]

    def get_matter_power_spectrum_bacco(self):
        """
        Emulate the CDM power spectrum using bacco.
        """
        emulator = baccoemu.Matter_powerspectrum()
        bacco_pars = self.get_bacco_pars()
        _, baccopk = getattr(emulator, f"get_{self.ps_type}_pk")(
            k=self.karr_in_h, cold=self.cold, **bacco_pars
        )
        return baccopk


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
        return self._weights_1

    @property
    def weights_2(self):
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
