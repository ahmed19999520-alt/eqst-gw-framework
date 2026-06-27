import numpy as np
from typing import Optional
from .base_detector import BaseDetector
from ..core.constants import FundamentalConstants

class VirgoDetector(BaseDetector):
    def __init__(self,
                 design: str = 'O4',
                 constants: Optional[FundamentalConstants] = None):
        super().__init__(constants)
        
        self.name = "Virgo"
        self.design = design
        
        self.f_min = 10.0
        self.f_max = 6000.0
        
        self.T_obs = 1.0 * self.const.year_to_s
    
    def noise_psd(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        S_n = np.zeros_like(f_array)
        
        mask = (f_array >= self.f_min) & (f_array <= self.f_max)
        f_masked = f_array[mask]
        
        if self.design == 'O4':
            S_0 = 3.2e-46
            x = f_masked / 100.0
            
            S_n[mask] = S_0 * (x**(-4.05) + 2.0 * x**(-1.0) + 0.5 + 0.2 * x**2)
            
        elif self.design == 'design':
            S_0 = 1.0e-47
            x = f_masked / 150.0
            
            S_n[mask] = S_0 * (x**(-4.5) + 2.5 * x**(-1.5) + 0.6)
        
        S_n[~mask] = 1.0e-40
        
        return S_n if f_array.size > 1 else S_n[0]
