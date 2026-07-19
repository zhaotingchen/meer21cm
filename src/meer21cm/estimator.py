"""
Power spectrum estimation from already-gridded 3D fields.

The class :py:class:`FieldPowerSpectrum` can be used standalone given a
pre-gridded ``field_1`` and ``box_len``. Sky↔box gridding lives on
:class:`meer21cm.grid.LightconeGriddingMixin` / :class:`meer21cm.power.PowerSpectrum`.
"""

import inspect
import logging

import numpy as np

from .dataanalysis import Specification
from .power_ops import (
    get_fourier_density,
    get_k_vector,
    get_power_spectrum,
    get_vec_mode,
    get_x_vector,
    power_weights_renorm,
)
from .util import tagging

logger = logging.getLogger(__name__)


class FieldPowerSpectrum(Specification):
    """
    The class for computing the power spectrum of a gridded field from LSS data.

    Parameters
    ----------
    field_1: np.ndarray.
        The density field of the first tracer.
    field_2: np.ndarray, default None
        The density field of the second tracer.
        If None, calculation of the second tracer and the cross-correlation will be skipped.
    box_len: list of 3 floats.
        The length of the box along each axis.
    weights_1: np.ndarray, default None
        The weights of the first tracer. Default is uniform weights.
    mean_center_1: bool, default False
        Whether to mean-center the first tracer field.
    unitless_1: bool, default False
        Whether to divide the first tracer field by its mean.
    weights_2: np.ndarray, default None
        The weights of the second tracer. Default is uniform weights.
    mean_center_2: bool, default False
        Whether to mean-center the second tracer field.
    unitless_2: bool, default False
        Whether to divide the second tracer field by its mean.
    **params: dict
        Additional parameters to be passed to the base class :class:`meer21cm.dataanalysis.Specification`.
    """

    def __init__(
        self,
        field_1,
        box_len,
        weights_1=None,
        mean_center_1=False,
        unitless_1=False,
        field_2=None,
        weights_2=None,
        mean_center_2=False,
        unitless_2=False,
        _skip_specification=False,
        **params,
    ):
        if not _skip_specification:
            Specification.__init__(self, **params)
        self.field_1 = field_1
        self.field_2 = field_2
        self.weights_1 = weights_1
        self.weights_2 = weights_2
        self.box_len = np.array(box_len)
        self.box_ndim = np.array(field_1.shape)
        self.mean_center_1 = mean_center_1
        self.unitless_1 = unitless_1
        self.mean_center_2 = mean_center_2
        self.unitless_2 = unitless_2
        if field_2 is not None:
            error_message = "field_1 and field_2 must have same dimensions"
            assert np.allclose(field_2.shape, field_1.shape), error_message
        self._fourier_field_1 = None
        self._fourier_field_2 = None

    @property
    def box_len(self):
        """
        The length of all sides of the box in Mpc.
        """
        return self._box_len

    @box_len.setter
    def box_len(self, value):
        self._box_len = value
        if "box_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.box_dep_attr} due to resetting box_len"
            )
            self.clean_cache(self.box_dep_attr)

    @property
    def box_resol(self):
        """
        The grid length of each side of the enclosing box in Mpc.
        """
        return self.box_len / self.box_ndim

    @property
    def box_ndim(self):
        """
        The number of grids along each side of the enclosing box.
        To ensure even sampling of +k and -k modes, the number of grids along every axis needs to be odd.
        """
        return self._box_ndim

    @box_ndim.setter
    def box_ndim(self, value):
        self._box_ndim = value
        if "box_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.box_dep_attr} due to resetting box_ndim"
            )
            self.clean_cache(self.box_dep_attr)

    def set_corr_type(self, corr_type, tracer_indx):
        """
        A utility function to help decide whether a tracer field
        needs to be mean centred, renormalised by its mean, and shot noise removed.
        Currently only two types are supported, "Gal" and "HI" (case-insensitive).
        If the tracer is galaxy (number counts),
        the auto power spectrum is mean centred, renormalised, and then
        shot noise removed. If HI, none of the above will be performed.

        Parameters
        ----------
        corr_type: str
            The tracer type.
        tracer_indx: int
            Either 1 or 2.
        """
        logger.debug("setting corr_type: %s for tracer %s", corr_type, tracer_indx)
        if corr_type[:3].lower() == "gal":
            mean_center = True
            unitless = True
            mean_amp = 1.0
        elif corr_type[:2].lower() == "hi":
            mean_center = False
            unitless = False
            mean_amp = "average_hi_temp"
        else:
            raise ValueError("unknown corr_type")
        if not tracer_indx in [1, 2]:
            raise ValueError("tracer_indx should be either 1 or 2")
        logger.debug("setting mean_center_%s: %s", tracer_indx, mean_center)
        logger.debug("setting unitless_%s: %s", tracer_indx, unitless)
        logger.debug("setting mean_amp_%s: %s", tracer_indx, mean_amp)
        setattr(self, "mean_center_" + str(tracer_indx), mean_center)
        setattr(self, "unitless_" + str(tracer_indx), unitless)
        setattr(self, "mean_amp_" + str(tracer_indx), mean_amp)

    @property
    def x_vec(self):
        """
        The 3D x-vector of the box.
        """
        return get_x_vector(
            self.box_ndim,
            self.box_resol,
        )

    @property
    def x_mode(self):
        """
        The mode of the 3D x-vector.
        """
        return get_vec_mode(self.x_vec)

    @property
    def k_vec(self):
        """
        The 3D k-vector of the box.
        """
        return get_k_vector(
            self.box_ndim,
            self.box_resol,
        )

    @property
    def k_nyquist(self):
        """
        The Nyquist frequency of the 3D box along each axis.
        """
        k_max = np.array([np.abs(self.k_vec[i]).max() for i in range(len(self.k_vec))])
        return k_max

    @property
    def k_perp(self):
        """
        The **fiducial** perpendicular k-vector of the 3D box.
        """
        return get_vec_mode(self.k_vec[:-1])

    @property
    def k_para(self):
        """
        The **fiducial** parallel k-mode of the 3D box.
        """
        return self.k_vec[-1]

    @property
    def k_mode(self):
        """
        The **fiducial** (observed) mode of the 3D k-vector.
        """
        return get_vec_mode(self.k_vec)

    @property
    def mu_mode(self):
        """
        The **fiducial** (observed) mu values of each k-mode so that :math:`k_\parallel = k \times \mu`.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(self.k_para[None, None, :] / self.k_mode)

    @property
    def field_1(self):
        """
        The density field of the first tracer.
        """
        return self._field_1

    @property
    def field_2(self):
        """
        The density field of the second tracer.
        """
        return self._field_2

    @field_1.setter
    def field_1(self, value):
        # if field is updated, clear fourier field
        self._field_1 = value
        if "field_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_1_dep_attr} due to resetting field_1"
            )
            self.clean_cache(self.field_1_dep_attr)

    @field_2.setter
    def field_2(self, value):
        # if field is updated, clear fourier field
        self._field_2 = value
        if "field_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_2_dep_attr} due to resetting field_2"
            )
            self.clean_cache(self.field_2_dep_attr)

    @property
    def mean_center_1(self):
        """
        Whether field_1 needs to be mean centered
        """
        return self._mean_center_1

    @property
    def mean_center_2(self):
        """
        Whether field_2 needs to be mean centered
        """
        return self._mean_center_2

    @mean_center_1.setter
    def mean_center_1(self, value):
        # if weight is updated, clear fourier field
        self._mean_center_1 = value
        if "field_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_1_dep_attr} due to resetting mean_center_1"
            )
            self.clean_cache(self.field_1_dep_attr)

    @mean_center_2.setter
    def mean_center_2(self, value):
        # if weight is updated, clear fourier field
        self._mean_center_2 = value
        if "field_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_2_dep_attr} due to resetting mean_center_2"
            )
            self.clean_cache(self.field_2_dep_attr)

    @property
    def unitless_1(self):
        """
        Whether field_1 needs to be divided by its mean
        """
        return self._unitless_1

    @property
    def unitless_2(self):
        """
        Whether field_2 needs to be divided by its mean
        """
        return self._unitless_2

    @unitless_1.setter
    def unitless_1(self, value):
        # if weight is updated, clear fourier field
        self._unitless_1 = value
        if "field_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_1_dep_attr} due to resetting unitless_1"
            )
            self.clean_cache(self.field_1_dep_attr)

    @unitless_2.setter
    def unitless_2(self, value):
        # if weight is updated, clear fourier field
        self._unitless_2 = value
        if "field_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_2_dep_attr} due to resetting unitless_2"
            )
            self.clean_cache(self.field_2_dep_attr)

    @property
    @tagging("box", "field_1")
    def fourier_field_1(self):
        """
        The Fourier transform of the density field of the first tracer.
        """
        if self._fourier_field_1 is None:
            self.get_fourier_field_1()
        return self._fourier_field_1

    def get_fourier_field_1(self):
        """
        Calculate the Fourier transform of the density field of the first tracer.
        """
        result = get_fourier_density(
            self.field_1,
            weights=self.weights_1,
            mean_center=self.mean_center_1,
            unitless=self.unitless_1,
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting self._fourier_field_1"
        )
        self._fourier_field_1 = result

    @property
    @tagging("box", "field_2")
    def fourier_field_2(self):
        """
        The Fourier transform of the density field of the second tracer.
        """
        if self._fourier_field_2 is None:
            self.get_fourier_field_2()
        return self._fourier_field_2

    def get_fourier_field_2(self):
        """
        Calculate the Fourier transform of the density field of the second tracer.
        """
        if self.field_2 is None:
            logger.info("field_2 is None, returning None")
            return None
        result = get_fourier_density(
            self.field_2,
            weights=self.weights_2,
            mean_center=self.mean_center_2,
            unitless=self.unitless_2,
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting self._fourier_field_2"
        )
        self._fourier_field_2 = result

    # the calculation of this is not heavy, simply on the fly
    @property
    def auto_power_3d_1(self):
        """
        The 3D power spectrum of the first tracer.
        """
        power_spectrum = get_power_spectrum(
            self.fourier_field_1,
            self.box_len,
            weights=self.weights_1,
            renorm=False,
        )
        return power_spectrum * self.renorm_ps_1

    @property
    def auto_power_3d_2(self):
        """
        The 3D power spectrum of the second tracer.
        """
        if self.field_2 is None:
            return None
        power_spectrum = get_power_spectrum(
            self.fourier_field_2,
            self.box_len,
            weights=self.weights_2,
            renorm=False,
        )
        return power_spectrum * self.renorm_ps_2

    @property
    def cross_power_3d(self):
        """
        The 3D cross power spectrum between the two tracers.
        """
        if self.field_2 is None:
            return None
        weights_2 = self.weights_2
        # if none, the default for get_power_spectrum is
        # to use weights_1, here we want separate weights_2
        if weights_2 is None:
            weights_2 = np.ones(self.field_2.shape)
        power_spectrum = get_power_spectrum(
            self.fourier_field_1,
            self.box_len,
            weights=self.weights_1,
            field_2=self.fourier_field_2,
            weights_2=weights_2,
            renorm=False,
        )
        return power_spectrum * self.renorm_ps_cross

    @property
    def renorm_ps_1(self):
        """
        The renormalization factor of the power spectrum of the first tracer.
        """
        grid_w = self.get_weights_none_to_one("weights_1")
        field_w = 1.0
        mean_renorm = 1.0
        if hasattr(self, "weights_field_1"):
            field_w = self.get_weights_none_to_one("weights_field_1")
            if self.unitless_1:
                mean_renorm = (field_w * grid_w).sum() / (grid_w).sum()
        return (
            power_weights_renorm(grid_w * field_w, grid_w * field_w) * mean_renorm**2
        )

    @property
    def renorm_ps_2(self):
        """
        The renormalization factor of the power spectrum of the second tracer.
        """
        grid_w = self.get_weights_none_to_one("weights_2")
        field_w = 1.0
        mean_renorm = 1.0
        if hasattr(self, "weights_field_2"):
            field_w = self.get_weights_none_to_one("weights_field_2")
            if self.unitless_2:
                mean_renorm = (field_w * grid_w).sum() / (grid_w).sum()
        return (
            power_weights_renorm(grid_w * field_w, grid_w * field_w) * mean_renorm**2
        )

    @property
    def renorm_ps_cross(self):
        """
        The renormalization factor of the cross power spectrum.
        """
        grid_w_1 = self.get_weights_none_to_one("weights_1")
        field_w_1 = 1.0
        mean_renorm_1 = 1.0
        if hasattr(self, "weights_field_1"):
            field_w_1 = self.get_weights_none_to_one("weights_field_1")
            if self.unitless_1:
                mean_renorm_1 = (field_w_1 * grid_w_1).sum() / (grid_w_1).sum()
        grid_w_2 = self.get_weights_none_to_one("weights_2")
        field_w_2 = 1.0
        mean_renorm_2 = 1.0
        if hasattr(self, "weights_field_2"):
            field_w_2 = self.get_weights_none_to_one("weights_field_2")
            if self.unitless_2:
                mean_renorm_2 = (field_w_2 * grid_w_2).sum() / (grid_w_2).sum()
        return (
            power_weights_renorm(grid_w_1 * field_w_1, grid_w_2 * field_w_2)
            * mean_renorm_1
            * mean_renorm_2
        )
