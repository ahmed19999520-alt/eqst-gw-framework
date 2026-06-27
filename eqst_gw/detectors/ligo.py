import numpy as np
from typing import Optional
from .base_detector import BaseDetector
from ..core.constants import FundamentalConstants

class LIGODetector(BaseDetector):
    def __init__(self,
                 design: str = 'O4',
                 detector_name: str = 'H1',
                 constants: Optional[FundamentalConstants] = None):
        super().__init__(constants)
        
        self.name = f"LIGO-{detector_name}"
        self.design = design
        self.detector = detector_name
        
        self.f_min = 10.0
        self.f_max = 5000.0
        
        self.T_obs = 1.0 * self.const.year_to_s
    
    def noise_psd(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        S_n = np.zeros_like(f_array)
        
        mask = (f_array >= self.f_min) & (f_array <= self.f_max)
        f_masked = f_array[mask]
        
        if self.design == 'O4':
            S_0 = 1.0e-47
            x = f_masked / 150.0
            
            seismic = x**(-4.14)
            suspension = -5.0 * x**(-2.0)
            mirror_thermal = 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2)
            
            S_n[mask] = S_0 * (seismic + suspension + mirror_thermal)
            
        elif self.design == 'design':
            S_0 = 1.5e-49
            x = f_masked / 215.0
            
            S_n[mask] = S_0 * (x**(-4.5) + 3.0 * x**(-1.5) + 0.8)
            
        elif self.design == 'A+':
            S_0 = 5.0e-48
            x = f_masked / 150.0
            
            S_n[mask] = S_0 * (x**(-4.0) + 2.0 * x**(-1.0) + 0.5)
        
        S_n[~mask] = 1.0e-40
        
        return S_n if f_array.size > 1 else S_n[0]
    
    def antenna_pattern(self, theta: float, phi: float, psi: float) -> Tuple[float, float]:
        F_plus = 0.5 * (1.0 + np.cos(theta)**2) * np.cos(2.0 * phi) * np.cos(2.0 * psi) - np.cos(theta) * np.sin(2.0 * phi) * np.sin(2.0 * psi)
        
        F_cross = 0.5 * (1.0 + np.cos(theta)**2) * np.cos(2.0 * phi) * np.sin(2.0 * psi) + np.cos(theta) * np.sin(2.0 * phi) * np.cos(2.0 * psi)
        
        return F_plus, F_cross