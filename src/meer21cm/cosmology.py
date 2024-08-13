import numpy as np
import camb
import astropy
from meer21cm import Specification


class CosmologyCalculator(Specification):
    """
    The class for storing the cosmological model used for calculation.

    The underlying cosmological model is defined via :class:`astropy.cosmology.LambdaCDM` with all the background
    properties calculated via ``astropy``.

    The matter density fluctuation is calculated using ``camb``.
    """

    def __init__(
        self,
        **params,
    ):
        super().__init__(**params)

    @property
    def camb_pars(self):
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
        return pars
