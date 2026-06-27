import numpy as np
from typing import Tuple, Optional, Dict
from .sources import BubbleCollisionSource, SoundWaveSource, TurbulenceSource
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class GravitationalWaveSpectrum:
    def __init__(self,
                 eqst_params: Optional[EQSTGPParameters] = None,
                 constants: Optional[FundamentalConstants] = None):
        
        if eqst_params is None:
            self.ep = EQSTGPParameters()
        else:
            self.ep = eqst_params
        
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        self.bubble_source = BubbleCollisionSource(self.ep, self.const)
        self.sound_source = SoundWaveSource(self.ep, self.const)
        self.turb_source = TurbulenceSource(self.ep, self.const)
    
    def total_spectrum(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        Omega_bubble = self.bubble_source.spectrum(f_array)
        Omega_sound = self.sound_source.spectrum(f_array)
        Omega_turb = self.turb_source.spectrum(f_array)
        
        Omega_total = Omega_bubble + Omega_sound + Omega_turb
        
        return Omega_total if f_array.size > 1 else Omega_total[0]
    
    def spectrum_components(self, f: np.ndarray) -> Dict[str, np.ndarray]:
        f_array = np.atleast_1d(f)
        
        components = {
            'bubble_collisions': self.bubble_source.spectrum(f_array),
            'sound_waves': self.sound_source.spectrum(f_array),
            'turbulence': self.turb_source.spectrum(f_array),
            'total': self.total_spectrum(f_array)
        }
        
        return components
    
    def peak_properties(self) -> Dict[str, float]:
        f_test = np.logspace(-5, 0, 10000)
        
        Omega_sound = self.sound_source.spectrum(f_test)
        idx_sound_peak = np.argmax(Omega_sound)
        
        Omega_turb = self.turb_source.spectrum(f_test)
        idx_turb_peak = np.argmax(Omega_turb)
        
        properties = {
            'f_sound_peak_Hz': f_test[idx_sound_peak],
            'Omega_sound_peak_h2': Omega_sound[idx_sound_peak],
            'f_turb_peak_Hz': f_test[idx_turb_peak],
            'Omega_turb_peak_h2': Omega_turb[idx_turb_peak],
            'f_bubble_peak_Hz': self.bubble_source.peak_frequency_today(self.ep.beta_over_H, self.ep.v_w),
            'Omega_bubble_peak_h2': self.bubble_source.peak_amplitude()
        }
        
        return properties
    
    def characteristic_strain(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        
        Omega_gw = self.total_spectrum(f_array)
        
        H0_SI = self.const.H0_Planck2018 * 1000.0 / self.const.Mpc_to_m
        
        h_c = np.sqrt((3.0 * H0_SI**2) / (2.0 * np.pi**2 * f_array**2) * Omega_gw)
        
        return h_c if f_array.size > 1 else h_c[0]
    
    def energy_density_parameter(self, f_min: float, f_max: float, n_points: int = 1000) -> float:
        f_range = np.logspace(np.log10(f_min), np.log10(f_max), n_points)
        
        Omega_spectrum = self.total_spectrum(f_range)
        
        Omega_total = np.trapz(Omega_spectrum / f_range, np.log(f_range))
        
        return Omega_total
