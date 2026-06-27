import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Optional, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class SoundWaveEvolution:
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
        
        self.c_s = 1.0 / np.sqrt(3.0)
    
    def fluid_velocity_profile(self, r: np.ndarray, R_bubble: float, v_w: float) -> np.ndarray:
        r_array = np.atleast_1d(r)
        v_fluid = np.zeros_like(r_array)
        
        alpha = self.ep.alpha_PT
        
        for i, r_val in enumerate(r_array):
            if r_val < R_bubble:
                xi = r_val / R_bubble
                if v_w < self.c_s:
                    v_fluid[i] = (self.c_s / np.sqrt(3.0)) * (alpha / (1.0 + alpha)) * xi
                else:
                    v_fluid[i] = v_w * xi
            else:
                decay_length = R_bubble * 0.5
                v_fluid[i] = (self.c_s / np.sqrt(3.0)) * (alpha / (1.0 + alpha)) * np.exp(-(r_val - R_bubble) / decay_length)
        
        return v_fluid if r_array.size > 1 else v_fluid[0]
    
    def velocity_divergence(self, r: np.ndarray, R_bubble: float, v_w: float) -> np.ndarray:
        dr = 1.0e-18
        v_plus = self.fluid_velocity_profile(r + dr, R_bubble, v_w)
        v_minus = self.fluid_velocity_profile(r - dr, R_bubble, v_w)
        
        div_v = (v_plus - v_minus) / (2.0 * dr) + 2.0 * self.fluid_velocity_profile(r, R_bubble, v_w) / r
        
        return div_v
    
    def acoustic_power_spectrum(self, k: np.ndarray, R_bubble: float, v_w: float) -> np.ndarray:
        k_array = np.atleast_1d(k)
        P_acoustic = np.zeros_like(k_array)
        
        r_max = 10.0 * R_bubble
        r_grid = np.linspace(0.01 * R_bubble, r_max, 1000)
        
        v_fluid = self.fluid_velocity_profile(r_grid, R_bubble, v_w)
        div_v = self.velocity_divergence(r_grid, R_bubble, v_w)
        
        for i, k_val in enumerate(k_array):
            integrand = r_grid**2 * div_v**2 * np.sin(k_val * r_grid) / (k_val * r_grid)
            P_acoustic[i] = 4.0 * np.pi * np.trapz(integrand, r_grid)
        
        return P_acoustic if k_array.size > 1 else P_acoustic[0]
    
    def sound_wave_gw_source(self, k: np.ndarray, t: float, R_bubble: float, v_w: float) -> np.ndarray:
        P_k = self.acoustic_power_spectrum(k, R_bubble, v_w)
        
        omega_k = self.c_s * k
        
        source = P_k * np.sin(omega_k * t)**2
        
        return source
    
    def time_averaged_gw_spectrum(self, k: np.ndarray, R_bubble: float, v_w: float, t_duration: float) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        P_k = self.acoustic_power_spectrum(k_array, R_bubble, v_w)
        
        omega_k = self.c_s * k_array
        
        time_factor = 0.5
        
        Omega_gw_k = (16.0 * np.pi / 3.0) * (k_array / (self.hubble_rate()))**2 * P_k * time_factor
        
        return Omega_gw_k if k_array.size > 1 else Omega_gw_k[0]
    
    def hubble_rate(self) -> float:
        T_star = self.ep.T_n
        g_star = self.ep.g_star
        M_pl = self.const.M_pl_GeV * 1.0e9
        
        H_star = 1.66 * np.sqrt(g_star) * T_star**2 / M_pl
        
        return H_star
    
    def map_k_to_frequency(self, k: np.ndarray) -> np.ndarray:
        a_ratio = self.redshift_factor()
        
        f = (k / (2.0 * np.pi)) * self.c_s * self.const.c * a_ratio
        
        return f
    
    def redshift_factor(self) -> float:
        g_star_S_today = 3.91
        g_star_S_PT = self.ep.g_star
        T_CMB = 2.7255
        T_PT = self.ep.T_n / (8.617333262e-5 * 1.0e9)
        
        a_ratio = (g_star_S_today / g_star_S_PT)**(1.0/3.0) * (T_CMB / T_PT)
        
        return a_ratio
