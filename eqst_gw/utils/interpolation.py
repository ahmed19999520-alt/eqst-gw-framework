import numpy as np
from scipy.interpolate import (interp1d, CubicSpline, RectBivariateSpline,
                                 RegularGridInterpolator, LinearNDInterpolator,
                                 NearestNDInterpolator)
from scipy.ndimage import map_coordinates
from typing import Tuple, Optional, Callable, Union


class AdaptiveInterpolator1D:
    def __init__(self,
                  x: np.ndarray,
                  y: np.ndarray,
                  method: str = 'cubic',
                  log_x: bool = False,
                  log_y: bool = False,
                  extrapolate: bool = True):

        self.x_raw = np.atleast_1d(x).copy()
        self.y_raw = np.atleast_1d(y).copy()
        self.log_x = log_x
        self.log_y = log_y
        self.method = method
        self.extrapolate = extrapolate

        idx_sort = np.argsort(self.x_raw)
        self.x_raw = self.x_raw[idx_sort]
        self.y_raw = self.y_raw[idx_sort]

        self.x_fit = np.log10(self.x_raw) if log_x else self.x_raw
        self.y_fit = np.log10(np.abs(self.y_raw)) if log_y else self.y_raw
        self.y_sign = np.sign(self.y_raw) if log_y else None

        fill_value = 'extrapolate' if extrapolate else (self.y_fit[0], self.y_fit[-1])

        if method == 'cubic':
            self._interp = CubicSpline(self.x_fit, self.y_fit,
                                        extrapolate=extrapolate)
        elif method == 'linear':
            self._interp = interp1d(self.x_fit, self.y_fit,
                                     kind='linear', bounds_error=False,
                                     fill_value=fill_value)
        elif method == 'pchip':
            from scipy.interpolate import PchipInterpolator
            self._interp = PchipInterpolator(self.x_fit, self.y_fit,
                                              extrapolate=extrapolate)
        elif method == 'akima':
            from scipy.interpolate import Akima1DInterpolator
            self._interp = Akima1DInterpolator(self.x_fit, self.y_fit)
        else:
            self._interp = CubicSpline(self.x_fit, self.y_fit,
                                        extrapolate=extrapolate)

    def __call__(self, x_new: Union[float, np.ndarray]) -> np.ndarray:
        x_arr = np.atleast_1d(x_new)

        x_eval = np.log10(x_arr) if self.log_x else x_arr

        y_eval = self._interp(x_eval)

        if self.log_y:
            y_out = 10.0**y_eval
        else:
            y_out = y_eval

        return y_out if len(x_arr) > 1 else y_out[0]

    def derivative(self, x_new: np.ndarray, order: int = 1) -> np.ndarray:
        x_arr = np.atleast_1d(x_new)
        x_eval = np.log10(x_arr) if self.log_x else x_arr

        if isinstance(self._interp, CubicSpline):
            dy = self._interp(x_eval, nu=order)
        else:
            dy = np.gradient(self._interp(x_eval), x_eval)

        if self.log_x:
            dy = dy / (x_arr * np.log(10.0))

        return dy if len(x_arr) > 1 else dy[0]


class BilinearInterpolator2D:
    def __init__(self,
                  x: np.ndarray,
                  y: np.ndarray,
                  z: np.ndarray,
                  log_x: bool = False,
                  log_y: bool = False,
                  log_z: bool = False):

        self.log_x = log_x
        self.log_y = log_y
        self.log_z = log_z

        x_fit = np.log10(x) if log_x else x
        y_fit = np.log10(y) if log_y else y
        z_fit = np.log10(np.abs(z)) if log_z else z

        self._interp = RectBivariateSpline(x_fit, y_fit, z_fit, kx=3, ky=3)

    def __call__(self, x_new: np.ndarray, y_new: np.ndarray) -> np.ndarray:
        x_eval = np.log10(x_new) if self.log_x else x_new
        y_eval = np.log10(y_new) if self.log_y else y_new

        z_eval = self._interp(x_eval, y_eval)

        if self.log_z:
            return 10.0**z_eval
        return z_eval


class PowerLawExtrapolator:
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray,
                  n_fit_low: int = 5, n_fit_high: int = 5):

        self.x_data = np.atleast_1d(x_data)
        self.y_data = np.atleast_1d(y_data)

        mask_pos = (self.x_data > 0) & (self.y_data > 0)

        log_x = np.log10(self.x_data[mask_pos])
        log_y = np.log10(self.y_data[mask_pos])

        coeffs_low = np.polyfit(log_x[:n_fit_low], log_y[:n_fit_low], 1)
        self.slope_low = coeffs_low[0]
        self.amp_low = 10.0**coeffs_low[1]

        coeffs_high = np.polyfit(log_x[-n_fit_high:], log_y[-n_fit_high:], 1)
        self.slope_high = coeffs_high[0]
        self.amp_high = 10.0**coeffs_high[1]

        self._core_interp = AdaptiveInterpolator1D(
            self.x_data[mask_pos], self.y_data[mask_pos],
            method='cubic', log_x=True, log_y=True, extrapolate=False
        )

        self.x_min = self.x_data[mask_pos][0]
        self.x_max = self.x_data[mask_pos][-1]

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_1d(x_new)
        y_out = np.zeros_like(x_arr, dtype=float)

        mask_low = x_arr < self.x_min
        mask_high = x_arr > self.x_max
        mask_core = (~mask_low) & (~mask_high)

        if np.any(mask_low):
            y_out[mask_low] = self.amp_low * x_arr[mask_low]**self.slope_low

        if np.any(mask_high):
            y_out[mask_high] = self.amp_high * x_arr[mask_high]**self.slope_high

        if np.any(mask_core):
            y_out[mask_core] = self._core_interp(x_arr[mask_core])

        return y_out if x_arr.size > 1 else y_out[0]


def interpolate_gw_spectrum_to_detector_frequencies(f_theory: np.ndarray,
                                                      Omega_theory: np.ndarray,
                                                      f_detector: np.ndarray) -> np.ndarray:
    mask_pos = (f_theory > 0) & (Omega_theory > 0)

    if np.sum(mask_pos) < 4:
        return np.zeros_like(f_detector)

    interp = PowerLawExtrapolator(f_theory[mask_pos], Omega_theory[mask_pos])

    return interp(f_detector)


def build_gw_template_bank(f_array: np.ndarray,
                             alpha_grid: np.ndarray,
                             beta_H_grid: np.ndarray,
                             v_w_grid: np.ndarray,
                             eqst_params,
                             constants) -> np.ndarray:
    n_alpha = len(alpha_grid)
    n_beta = len(beta_H_grid)
    n_vw = len(v_w_grid)
    n_freq = len(f_array)

    template_bank = np.zeros((n_alpha, n_beta, n_vw, n_freq))

    from ..gravitational_waves.spectrum import GravitationalWaveSpectrum

    for i, alpha in enumerate(alpha_grid):
        for j, beta_H in enumerate(beta_H_grid):
            for k, v_w in enumerate(v_w_grid):
                ep_temp = type(eqst_params)()
                ep_temp.alpha_PT = alpha
                ep_temp.beta_over_H = beta_H
                ep_temp.v_w = v_w

                gw = GravitationalWaveSpectrum(ep_temp, constants)
                template_bank[i, j, k, :] = gw.total_spectrum(f_array)

    return template_bank