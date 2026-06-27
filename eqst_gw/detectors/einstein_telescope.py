import numpy as np
from typing import Optional
from .base_detector import BaseDetector
from ..core.constants import FundamentalConstants

class EinsteinTelescopeDetector(BaseDetector):
    def __init__(self,
                 design: str = 'ET-D',
                 constants: Optional[FundamentalConstants] = None):
        super().__init__(constants)
        
        self.name = "Einstein Telescope"
        self.design = design
        
        self.f_min = 1.0
        self.f_max = 10000.0
        
        self.T_obs = 1.0 * self.const.year_to_s
    
    def noise_psd(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        S_n = np.zeros_like(f_array)
        
        mask = (f_array >= self.f_min) & (f_array <= self.f_max)
        f_masked = f_array[mask]
        
        if self.design == 'ET-D':
            S_0 = 1.449e-52
            x = f_masked / 100.0
            
            seismic = x**(-4.05)
            gravity_gradient = 0.017 * x**(-0.69)
            suspension = 0.0018 * x**(1.59)
            coating_brownian = 0.26 * x**(2.8)
            
            S_n[mask] = S_0 * (seismic + gravity_gradient + suspension + coating_brownian)
            
        elif self.design == 'ET-B':
            S_0 = 2.0e-52
            x = f_masked / 100.0
            
            S_n[mask] = S_0 * (x**(-4.0) + 0.02 * x**(-1.0) + 0.001 * x**(2.0) + 0.3 * x**(3.0))
        
        S_n[~mask] = 1.0e-48
        
        return S_n if f_array.size > 1 else S_n[0]
