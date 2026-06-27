import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import brentq
from typing import Tuple, Optional, Callable, List
import warnings

def adaptive_integration(func: Callable, a: float, b: float, tol: float = 1.0e-8, max_subdivisions: int = 100) -> Tuple[float, float]:
    result, error = quad(func, a, b, limit=max_subdivisions, epsabs=tol, epsrel=tol)
    return result, error


def logarithmic_integration(func: Callable, log_a: float, log_b: float, n_points: int = 1000) -> float:
    log_x = np.linspace(log_a, log_b, n_points)
    x = np.exp(log_x)
    y = func(x) * x
    return np.trapz(y, log_x)


def five_point_derivative(func: Callable, x: float, h: float = None) -> float:
    if h is None:
        h = 1.0e-5 * abs(x) if x != 0 else 1.0e-8
    
    f1 = func(x + h)
    f2 = func(x + 2.0 * h)
    fm1 = func(x - h)
    fm2 = func(x - 2.0 * h)
    
    deriv = (-f2 + 8.0 * f1 - 8.0 * fm1 + fm2) / (12.0 * h)
    
    return deriv


def richardson_extrapolation(func: Callable, x: float, h0: float = 0.01, n_levels: int = 4) -> Tuple[float, float]:
    h = h0
    
    D = np.zeros((n_levels, n_levels))
    
    for i in range(n_levels):
        D[i, 0] = (func(x + h) - func(x - h)) / (2.0 * h)
        h /= 2.0
    
    for j in range(1, n_levels):
        for i in range(n_levels - j):
            D[i, j] = D[i+1, j-1] + (D[i+1, j-1] - D[i, j-1]) / (4.0**j - 1.0)
    
    best_estimate = D[0, n_levels - 1]
    error_estimate = abs(D[0, n_levels-1] - D[0, n_levels-2])
    
    return best_estimate, error_estimate


def runge_kutta_45(func: Callable, y0: np.ndarray, t_span: Tuple[float, float], t_eval: np.ndarray = None, rtol: float = 1.0e-6, atol: float = 1.0e-9) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.integrate import solve_ivp
    
    sol = solve_ivp(func, t_span, y0, method='RK45', t_eval=t_eval, rtol=rtol, atol=atol, dense_output=True)
    
    return sol.t, sol.y


def leapfrog_integrator(positions: np.ndarray, velocities: np.ndarray, accelerations_func: Callable, dt: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = positions.copy()
    vel = velocities.copy()
    
    history_pos = [pos.copy()]
    history_vel = [vel.copy()]
    
    for step in range(n_steps):
        acc = accelerations_func(pos)
        
        vel_half = vel + 0.5 * dt * acc
        
        pos = pos + dt * vel_half
        
        acc_new = accelerations_func(pos)
        
        vel = vel_half + 0.5 * dt * acc_new
        
        history_pos.append(pos.copy())
        history_vel.append(vel.copy())
    
    return np.array(history_pos), np.array(history_vel)


def power_law_fit(x: np.ndarray, y: np.ndarray, x_min: float = None, x_max: float = None) -> Tuple[float, float, float, float]:
    if x_min is not None and x_max is not None:
        mask = (x >= x_min) & (x <= x_max)
        x_fit = x[mask]
        y_fit = y[mask]
    else:
        x_fit = x
        y_fit = y
    
    mask_valid = (x_fit > 0) & (y_fit > 0)
    x_log = np.log10(x_fit[mask_valid])
    y_log = np.log10(y_fit[mask_valid])
    
    coeffs = np.polyfit(x_log, y_log, 1)
    
    spectral_index = coeffs[0]
    log_amplitude = coeffs[1]
    amplitude = 10.0**log_amplitude
    
    y_pred = amplitude * x_fit[mask_valid]**spectral_index
    residuals = y_log - np.log10(y_pred)
    chi2 = np.sum(residuals**2)
    
    return amplitude, spectral_index, chi2, len(x_fit[mask_valid])

