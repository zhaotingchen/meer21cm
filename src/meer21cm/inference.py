import numpy as np
from meer21cm.power import PowerSpectrum
from meer21cm.transfer import required_attrs
import emcee
import os
import nautilus
from multiprocessing import Pool
from scipy.stats import norm


def extract_model_fitting_inputs(
    ps: PowerSpectrum,
):
    """
    Extract a dictionary of model fitting inputs from a
    :class:`meer21cm.power.PowerSpectrum` instance.

    Parameters
    ----------
    ps : :class:`meer21cm.power.PowerSpectrum`
        A :class:`meer21cm.power.PowerSpectrum` instance.

    Returns
    -------
    ps_dict : dict
        A dictionary of model fitting inputs.
    """
    attr_dict = {}
    for attr in required_attrs:
        attr_dict[attr] = getattr(ps, attr)
    attr_dict["box_len"] = ps.box_len
    attr_dict["box_ndim"] = ps.box_ndim
    return attr_dict


class SamplerBase:
    """
    Base class for all samplers.
    """

    def __init__(
        self,
        ps_dict: dict,
        data_vector: np.ndarray,
        data_covariance: np.ndarray,
        params_name: list[str],
        params_prior: list[tuple[str, float, float]],
        observables: list[str] = ["cross"],
        save: bool = False,
        save_filename: str | None = None,
        save_model_blobs: bool = False,
    ):
        self.ps_dict = ps_dict
        self.params_name = params_name
        self.params_prior = params_prior
        self.save = save
        self.save_filename = save_filename
        self.observables = observables
        self.data_vector = data_vector
        self.data_covariance = data_covariance
        self._inverse_covariance = None
        self.save_model_blobs = save_model_blobs
        self.validate_input()

    @property
    def inverse_covariance(self) -> np.ndarray:
        if self._inverse_covariance is None:
            self._inverse_covariance = np.linalg.inv(self.data_covariance)
        return self._inverse_covariance

    @property
    def data_covariance(self) -> np.ndarray:
        return self._data_covariance

    @data_covariance.setter
    def data_covariance(self, data_covariance: np.ndarray):
        self._data_covariance = data_covariance
        self._inverse_covariance = None

    def validate_input(self):
        if not len(self.params_name) == len(self.params_prior):
            raise ValueError("params_names and params_prior must have the same length")
        data_len = self.data_vector.size
        if self.data_covariance.shape != (data_len, data_len):
            raise ValueError(
                "data_covariance must be a square matrix with the same length as data_vector"
            )
        if not all(obs in ["cross", "hi-auto", "gg-auto"] for obs in self.observables):
            raise ValueError(
                "observables must be a list of 'cross', 'hi-auto', or 'gg-auto'"
            )

    def get_model_instance(self, params: list[float] | np.ndarray) -> PowerSpectrum:
        model_instance = PowerSpectrum(**self.ps_dict)
        model_instance._box_len = self.ps_dict["box_len"]
        model_instance._box_ndim = self.ps_dict["box_ndim"]
        model_instance.propagate_field_k_to_model()
        for param_name, param_value in zip(self.params_name, params):
            setattr(model_instance, param_name, param_value)
        return model_instance

    def get_model_vector(self, params: list[float] | np.ndarray) -> np.ndarray:
        model_instance = self.get_model_instance(params)
        model_vector = np.zeros(
            (len(self.observables), len(self.ps_dict["k1dbins"]) - 1)
        )
        i = 0
        for obs in self.observables:
            if obs == "cross":
                model_vector[i], _, _ = model_instance.get_1d_power(
                    model_instance.cross_power_tracer_model,
                    k1dweights=model_instance.k1dweights,
                )
            elif obs == "hi-auto":
                model_vector[i], _, _ = model_instance.get_1d_power(
                    model_instance.auto_power_tracer_1_model,
                    k1dweights=model_instance.k1dweights,
                )
            elif obs == "gg-auto":
                model_vector[i], _, _ = model_instance.get_1d_power(
                    model_instance.auto_power_tracer_2_model,
                    k1dweights=model_instance.k1dweights,
                )
            i += 1
        return model_vector

    def compute_log_likelihood(
        self, params: list[float] | np.ndarray
    ) -> float | tuple[float, np.ndarray]:
        model_vector = self.get_model_vector(params)
        model_minus_data = model_vector.ravel() - self.data_vector.ravel()
        log_likelihood = (
            -0.5 * model_minus_data @ self.inverse_covariance @ model_minus_data
        )
        if self.save_model_blobs:
            return log_likelihood, model_vector
        return log_likelihood


class SamplerEmcee(SamplerBase):
    def __init__(
        self,
        ps_dict: dict,
        data_vector: np.ndarray,
        data_covariance: np.ndarray,
        params_name: list[str],
        params_prior: list[tuple[str, float, float]],
        nwalkers: int,
        nsteps: int,
        nthreads: int,
        observables: list[str] = ["cross"],
        mp_backend: str = "multiprocessing",
        save: bool = False,
        save_filename: str | None = None,
        save_model_blobs: bool = False,
    ):
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nthreads = nthreads
        self.mp_backend = mp_backend
        super().__init__(
            ps_dict,
            data_vector,
            data_covariance,
            params_name,
            params_prior,
            observables,
            save,
            save_filename,
            save_model_blobs,
        )

    def log_prior_gaussian(self, value, mean, sigma):
        return -0.5 * (value - mean) ** 2 / sigma**2

    def log_prior_uniform(self, value, low, high):
        if value < low or value > high:
            return -np.inf
        else:
            return 0.0

    def log_prior(self, params_values):
        log_prior = 0.0
        for i, param_name in enumerate(self.params_name):
            prior_func = getattr(self, f"log_prior_{self.params_prior[i][0]}")
            log_prior_i = prior_func(
                params_values[i],
                self.params_prior[i][1],
                self.params_prior[i][2],
            )
            log_prior += log_prior_i
        return log_prior

    def log_likelihood(self, params_values):
        lp = self.log_prior(params_values)
        if not np.isfinite(lp):
            if self.save_model_blobs:
                return lp, np.zeros(
                    (len(self.observables), (len(self.ps_dict["k1dbins"]) - 1))
                )
            return lp
        else:
            if self.save_model_blobs:
                ll, model_vector = self.compute_log_likelihood(params_values)
                return lp + ll, model_vector
            return lp + self.compute_log_likelihood(params_values)

    @property
    def ndim(self):
        return len(self.params_name)

    def run(
        self, resume: bool, progress: bool = True, start_coord: np.ndarray | None = None
    ):
        if resume and self.save:
            start_coord = None
        else:
            init_pos = np.array(
                [self.ps_dict[param_name] for param_name in self.params_name]
            )
            # TODO: need smarter auto init position
            if start_coord is None:
                start_coord = (
                    1 + np.random.uniform(-1e-2, 1e-2, size=(self.nwalkers, self.ndim))
                ) * init_pos[None, :] + np.random.uniform(
                    0, 1e-2, size=(self.nwalkers, self.ndim)
                ) * (
                    np.abs(init_pos[None, :]) < 1e-2
                )
        nsteps = self.nsteps
        if self.save:
            backend = emcee.backends.HDFBackend(self.save_filename)
            if os.path.isfile(self.save_filename):
                if not resume and backend.iteration > 0:
                    backend.reset(self.nwalkers, self.ndim)
                nsteps = nsteps - backend.iteration
        else:
            backend = None
        pool = None
        if self.mp_backend == "multiprocessing":
            pool_func = Pool
        elif self.mp_backend == "mpi":
            from mpi4py.futures import MPIPoolExecutor

            pool_func = MPIPoolExecutor
        with pool_func(self.nthreads) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                self.log_likelihood,
                backend=backend,
                pool=pool,
            )
            sampler.run_mcmc(start_coord, nsteps, progress=progress)
        return sampler

    def get_backend(self) -> emcee.backends.HDFBackend:
        if self.save:
            return emcee.backends.HDFBackend(self.save_filename)
        else:
            raise ValueError("No save filename provided")

    def get_points(self, sampler: emcee.EnsembleSampler | None = None) -> np.ndarray:
        if sampler is None:
            return self.get_backend().get_chain()
        return sampler.get_chain()

    def get_blobs(self, sampler: emcee.EnsembleSampler | None = None) -> np.ndarray:
        if sampler is None:
            return self.get_backend().get_blobs()
        return sampler.get_blobs()

    def get_log_prob(self, sampler: emcee.EnsembleSampler | None = None) -> np.ndarray:
        if sampler is None:
            return self.get_backend().get_log_prob()
        return sampler.get_log_prob()


class SamplerNautilus(SamplerBase):
    def __init__(
        self,
        ps_dict: dict,
        data_vector: np.ndarray,
        data_covariance: np.ndarray,
        params_name: list[str],
        params_prior: list[tuple[str, float, float]],
        observables: list[str] = ["cross"],
        n_live_points: int = 2000,
        f_live: float = 0.01,
        n_shell: int = 1,
        n_eff: int = 10000,
        save: bool = False,
        save_filename: str | None = None,
        mp_backend: str = "multiprocessing",
        nthreads: int = 1,
        timeout: float = np.inf,
        save_model_blobs: bool = False,
    ):
        self.n_live_points = n_live_points
        self.f_live = f_live
        self.n_shell = n_shell
        self.n_eff = n_eff
        self.mp_backend = mp_backend
        self.nthreads = nthreads
        self.timeout = timeout
        super().__init__(
            ps_dict,
            data_vector,
            data_covariance,
            params_name,
            params_prior,
            observables,
            save,
            save_filename,
            save_model_blobs,
        )

    def get_nautilus_prior(self):
        prior = nautilus.Prior()
        for i, param_name in enumerate(self.params_name):
            if self.params_prior[i][0] == "uniform":
                dist = (self.params_prior[i][1], self.params_prior[i][2])
            elif self.params_prior[i][0] == "gaussian":
                dist = norm(loc=self.params_prior[i][1], scale=self.params_prior[i][2])
            prior.add_parameter(param_name, dist=dist)
        return prior

    def run(self, resume: bool, progress: bool = True):
        if not self.save:
            resume = False
        if self.mp_backend == "multiprocessing":
            pool_func = Pool
        elif self.mp_backend == "mpi":
            from mpi4py.futures import MPIPoolExecutor

            pool_func = MPIPoolExecutor
        with pool_func(self.nthreads) as pool:
            sampler = nautilus.Sampler(
                self.get_nautilus_prior(),
                self.compute_log_likelihood,
                pass_dict=False,
                pool=pool,
                n_live=self.n_live_points,
                resume=resume,
                filepath=self.save_filename,
            )
            sampler.run(
                f_live=self.f_live,
                n_shell=self.n_shell,
                n_eff=self.n_eff,
                discard_exploration=True,
                verbose=progress,
                timeout=self.timeout,
            )
        return sampler

    def get_posterior(self, sampler: nautilus.Sampler | None = None):
        if sampler is None:
            sampler = nautilus.Sampler(
                self.get_nautilus_prior(),
                self.compute_log_likelihood,
                pass_dict=False,
                filepath=self.save_filename,
            )
        return sampler.posterior(return_blobs=self.save_model_blobs)
