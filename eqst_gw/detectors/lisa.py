import numpy as np
from typing import Optional
from .base_detector import BaseDetector
from ..core.constants import FundamentalConstants

class LISADetector(BaseDetector):
    def __init__(self,
                 mission_duration_years: float = 4.0,
                 constants: Optional[FundamentalConstants] = None):
        super().__init__(constants)
        
        self.name = "LISA"
        self.L_arm = 2.5e9
        self.f_star = self.const.c / (2.0 * np.pi * self.L_arm)
        
        self.S_acc = 3.0e-15
        self.S_IMS = 15.0e-12
        
        self.T_obs = mission_duration_years * self.const.year_to_s
        
        self.f_min = 1.0e-5
        self.f_max = 1.0e-1
    
    def acceleration_noise(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        S_a = self.S_acc**2 * (1.0 + (4.0e-4 / f_array)**2)
        
        return S_a if f_array.size > 1 else S_a[0]
    
    def optical_metrology_noise(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        S_oms = self.S_IMS**2 * (1.0 + (2.0e-3 / f_array)**4)
        
        return S_oms if f_array.size > 1 else S_oms[0]
    
    def noise_psd(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        x = f_array / self.f_star
        
        P_acc = self.acceleration_noise(f_array) / ((2.0 * np.pi * f_array)**4)
        
        P_oms = self.optical_metrology_noise(f_array)
        
        transfer_function = (3.0 + 2.0 * np.cos(2.0 * x) + np.cos(4.0 * x)) / 5.0
        
        S_n = (10.0 / (3.0 * self.L_arm**2)) * (P_oms + 4.0 * P_acc) * transfer_function * (1.0 + 0.6 * x**2)
        
        return S_n if f_array.size > 1 else S_n[0]
    
    def galactic_confusion_noise(self, f: np.ndarray, include_confusion: bool = True) -> np.ndarray:
        if not include_confusion:
            return np.zeros_like(f)
        
        f_array = np.atleast_1d(f)
        
        f_knee = 2.0e-3
        A_conf = 9.0e-45
        
        S_conf = A_conf * (f_array / f_knee)**(-7.0/3.0) * np.exp(-(f_array / f_knee)**2)
        
        return S_conf if f_array.size > 1 else S_conf[0]
    
    def total_noise_psd(self, f: np.ndarray, include_confusion: bool = True) -> np.ndarray:
        S_instrumental = self.noise_psd(f)
        S_astrophysical = self.galactic_confusion_noise(f, include_confusion)
        
        return S_instrumental + S_astrophysical
    
    def response_function(self, f: np.ndarray, theta: float = 0.0, phi: float = 0.0, psi: float = 0.0) -> complex:
        f_array = np.atleast_1d(f)
        
        x = f_array / self.f_star
        
        F_plus = (1.0 + np.cos(theta)**2) / 2.0 * np.cos(2.0 * phi) * np.cos(2.0 * psi)
        F_cross = np.cos(theta) * np.sin(2.0 * phi) * np.sin(2.0 * psi)
        
        R_f = np.sqrt(3.0 / 20.0) * (np.sin(x) / x) * (F_plus + 1j * F_cross)
        
        return R_f if f_array.size > 1 else R_f[0]

