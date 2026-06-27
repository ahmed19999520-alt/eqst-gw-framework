import numpy as np
from typing import Any, Optional, List, Dict, Tuple
import warnings


def validate_frequency_array(f: np.ndarray, f_min: float = 0.0, f_max: float = np.inf, name: str = 'f') -> np.ndarray:
    f_array = np.atleast_1d(np.asarray(f, dtype=float))
    
    if np.any(f_array <= 0):
        raise ValueError(f"Frequency array '{name}' must contain only positive values.")
    
    if f_min > 0 and np.any(f_array < f_min):
        warnings.warn(f"Some frequencies in '{name}' are below f_min={f_min:.2e} Hz.")
    
    if np.isfinite(f_max) and np.any(f_array > f_max):
        warnings.warn(f"Some frequencies in '{name}' are above f_max={f_max:.2e} Hz.")
    
    if np.any(np.diff(f_array) < 0):
        warnings.warn(f"Frequency array '{name}' is not sorted in ascending order. Sorting...")
        f_array = np.sort(f_array)
    
    return f_array


def validate_eqst_parameters(params) -> bool:
    errors = []
    
    if params.alpha_PT <= 0 or params.alpha_PT >= 10:
        errors.append(f"alpha_PT={params.alpha_PT:.3f} out of physical range (0, 10)")
    
    if params.beta_over_H <= 0 or params.beta_over_H >= 10000:
        errors.append(f"beta_over_H={params.beta_over_H:.1f} out of range (0, 10000)")
    
    if params.v_w <= 0 or params.v_w >= 1.0:
        errors.append(f"v_w={params.v_w:.3f} out of physical range (0, 1)")
    
    if params.T_n <= 0 or params.T_n >= 1.0e20:
        errors.append(f"T_n={params.T_n:.2e} GeV out of physical range")
    
    if params.g_star <= 0 or params.g_star >= 10000:
        errors.append(f"g_star={params.g_star:.1f} out of physical range")
    
    if errors:
        for err in errors:
            warnings.warn(f"Parameter validation warning: {err}")
        return False
    
    return True


def validate_covariance_matrix(cov: np.ndarray, name: str = 'covariance matrix') -> bool:
    if cov.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be square.")
    
    symmetry_error = np.max(np.abs(cov - cov.T))
    if symmetry_error > 1.0e-10 * np.max(np.abs(cov)):
        warnings.warn(f"{name} is not symmetric (max asymmetry = {symmetry_error:.2e}). Symmetrizing...")
    
    eigenvalues = np.linalg.eigvalsh(cov)
    if np.any(eigenvalues < -1.0e-10 * np.max(eigenvalues)):
        warnings.warn(f"{name} has negative eigenvalues (min = {np.min(eigenvalues):.2e}). May not be positive semi-definite.")
        return False
    
    return True


def check_numerical_stability(arr: np.ndarray, name: str = 'array') -> bool:
    if np.any(np.isnan(arr)):
        warnings.warn(f"{name} contains NaN values.")
        return False
    
    if np.any(np.isinf(arr)):
        warnings.warn(f"{name} contains Inf values.")
        return False
    
    if np.any(np.abs(arr) > 1.0e300):
        warnings.warn(f"{name} contains extremely large values.")
        return False
    
    return True