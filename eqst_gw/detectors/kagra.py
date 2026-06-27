import numpy as np
from typing import Optional
from .base_detector import BaseDetector
from ..core.constants import FundamentalConstants

class KAGRADetector(BaseDetector):
    def __init__(self,
                 design: str = 'O4',
                 constants: Optional[FundamentalConstants] = None):
        super().__init__(constants)
        
        self.name = "KAGRA"
        self.design = design
        
        self.f_min = 10.0
        self.f_max = 5000.0
        
        self.T_obs = 1.0 * self.const.year_to_s
    
    def noise_psd(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        S_n = np.zeros_like(f_array)
        
        mask = (f_array >= self.f_min) & (f_array <= self.f_max)
        f_masked = f_array[mask]
        
        if self.design == 'O4':
            S_0 = 1.0e-45
            x = f_masked / 100.0
            
            S_n[mask] = S_0 * (x**(-4.2) + 3.0 * x**(-1.5) + 1.0 + 0.3 * x**2)
            
        elif self.design == 'design':
            S_0 = 2.0e-47
            x = f_masked / 150.0
            
            S_n[mask] = S_0 * (x**(-4.5) + 2.0 * x**(-1.5) + 0.7)
        
        S_n[~mask] = 1.0e-40
        
        return S_n if f_array.size > 1 else S_n[0]
