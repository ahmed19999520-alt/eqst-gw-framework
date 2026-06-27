import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable, Dict
from ..core.constants import FundamentalConstants

class BaseDetector(ABC):
    def __init__(self, constants: Optional[FundamentalConstants] = None):
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        self.name = "BaseDetector"
        self.f_min = 0.0
        self.f_max = np.inf
        self.T_obs = 1.0
    
    @abstractmethod
    def noise_psd(self, f: np.ndarray) -> np.ndarray:
        pass
    
    def characteristic_strain_noise(self, f: np.ndarray) -> np.ndarray:
        S_n = self.noise_psd(f)
        h_n = np.sqrt(f * S_n)
        return h_n
    
    def omega_sensitivity(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        h_c_n = self.characteristic_strain_noise(f_array)
        
        H0_SI = self.const.H0_Planck2018 * 1000.0 / self.const.Mpc_to_m
        
        Omega_sens = (2.0 * np.pi**2 / (3.0 * H0_SI**2)) * f_array**2 * h_c_n**2 / np.sqrt(self.T_obs)
        
        return Omega_sens if f_array.size > 1 else Omega_sens[0]
    
    def compute_snr(self, f: np.ndarray, Omega_signal: np.ndarray) -> float:
        f_array = np.atleast_1d(f)
        Omega_sig = np.atleast_1d(Omega_signal)
        
        Omega_sens = self.omega_sensitivity(f_array)
        
        integrand = (Omega_sig / Omega_sens)**2
        
        mask = (f_array >= self.f_min) & (f_array <= self.f_max)
        
        SNR_squared = self.T_obs * np.trapz(integrand[mask], np.log(f_array[mask]))
        
        return np.sqrt(SNR_squared)
    
    def optimal_matched_filter_snr(self, f: np.ndarray, h_signal: np.ndarray) -> float:
        f_array = np.atleast_1d(f)
        h_sig = np.atleast_1d(h_signal)
        
        S_n = self.noise_psd(f_array)
        
        integrand = np.abs(h_sig)**2 / S_n
        
        mask = (f_array >= self.f_min) & (f_array <= self.f_max)
        
        SNR_squared = 4.0 * np.trapz(integrand[mask], f_array[mask])
        
        return np.sqrt(SNR_squared)
    
    def detection_threshold(self, false_alarm_probability: float = 1.0e-3) -> float:
        from scipy.special import erfcinv
        
        SNR_threshold = np.sqrt(2.0) * erfcinv(2.0 * false_alarm_probability)
        
        return SNR_threshold
    
    def parameter_fisher_matrix(self, f: np.ndarray, template_derivatives: Dict[str, np.ndarray]) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        param_names = list(template_derivatives.keys())
        N_params = len(param_names)
        
        Fisher = np.zeros((N_params, N_params))
        
        S_n = self.noise_psd(f_array)
        
        for i, param_i in enumerate(param_names):
            for j, param_j in enumerate(param_names):
                dh_di = template_derivatives[param_i]
                dh_dj = template_derivatives[param_j]
                
                integrand = (dh_di * np.conj(dh_dj)) / S_n
                
                mask = (f_array >= self.f_min) & (f_array <= self.f_max)
                
                Fisher[i, j] = 4.0 * np.real(np.trapz(integrand[mask], f_array[mask]))
        
        return Fisher

