import numpy as np
from scipy.integrate import quad
from typing import Tuple, Dict, Optional
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class GravitationalWaveSource:
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
    
    def redshift_factor(self) -> float:
        g_star_S_today = 3.91
        g_star_S_PT = self.ep.g_star
        return (g_star_S_today / g_star_S_PT)**(1.0/3.0)
    
    def hubble_rate_at_transition(self) -> float:
        H_star_GeV = 1.66 * np.sqrt(self.ep.g_star) * (self.ep.T_n / 1.0e2)**2 / (self.const.M_pl_GeV)
        return H_star_GeV
    
    def peak_frequency_today(self, beta_over_H: float, v_w: float) -> float:
        H_star_GeV = self.hubble_rate_at_transition()
        H_star_Hz = H_star_GeV * self.const.eV_to_J / self.const.hbar
        
        f_star_Hz = beta_over_H * H_star_Hz / (2.0 * np.pi * v_w)
        
        T_CMB_eV = self.const.T_CMB_K * 8.617333262e-5
        T_PT_eV = self.ep.T_n
        
        f_today = f_star_Hz * (T_CMB_eV / T_PT_eV) * self.redshift_factor()
        
        return f_today


class BubbleCollisionSource(GravitationalWaveSource):
    def __init__(self, eqst_params=None, constants=None):
        super().__init__(eqst_params, constants)
    
    def efficiency_factor(self) -> float:
        v_w = self.ep.v_w
        c_s = 1.0 / np.sqrt(3.0)
        
        if v_w < c_s:
            delta_v = np.sqrt(c_s**2 - v_w**2)
            kappa_phi = 4.9e-3 * (0.135 + delta_v)**2
        else:
            kappa_phi = 4.9e-3
        
        return kappa_phi
    
    def peak_amplitude(self) -> float:
        kappa_phi = self.efficiency_factor()
        alpha = self.ep.alpha_PT
        beta_H = self.ep.beta_over_H
        g_star = self.ep.g_star
        v_w = self.ep.v_w
        
        Omega_peak = 1.67e-5 * (1.0 / beta_H)**2 * (kappa_phi * alpha / (1.0 + alpha))**2 * (100.0 / g_star)**(1.0/3.0) * (0.11 * v_w**3 / (0.42 + v_w**2))
        
        return Omega_peak * self.const.h_Planck2018**2
    
    def spectral_shape(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        f_peak = self.peak_frequency_today(self.ep.beta_over_H, self.ep.v_w)
        
        x = f_array / f_peak
        
        S_phi = x**3 * (7.0 / (4.0 + 3.0 * x**2))**(7.0/2.0)
        
        return S_phi if f_array.size > 1 else S_phi[0]
    
    def spectrum(self, f: np.ndarray) -> np.ndarray:
        Omega_peak = self.peak_amplitude()
        S_phi = self.spectral_shape(f)
        return Omega_peak * S_phi


class SoundWaveSource(GravitationalWaveSource):
    def __init__(self, eqst_params=None, constants=None):
        super().__init__(eqst_params, constants)
    
    def efficiency_factor(self) -> float:
        alpha = self.ep.alpha_PT
        kappa_v = alpha / (0.73 + 0.083 * np.sqrt(alpha) + alpha)
        return kappa_v
    
    def peak_amplitude(self) -> float:
        kappa_v = self.efficiency_factor()
        alpha = self.ep.alpha_PT
        beta_H = self.ep.beta_over_H
        g_star = self.ep.g_star
        v_w = self.ep.v_w
        
        Omega_peak = 2.65e-6 * (1.0 / beta_H) * (kappa_v * alpha / (1.0 + alpha))**2 * (100.0 / g_star)**(1.0/3.0) * v_w
        
        return Omega_peak * self.const.h_Planck2018**2
    
    def peak_frequency_today(self) -> float:
        return super().peak_frequency_today(self.ep.beta_over_H, self.ep.v_w)
    
    def spectral_shape(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        f_peak = self.peak_frequency_today()
        
        x = f_array / f_peak
        
        S_sw = x**3 * (7.0 / (4.0 + 3.0 * x**2))**(7.0/2.0)
        
        return S_sw if f_array.size > 1 else S_sw[0]
    
    def spectrum(self, f: np.ndarray) -> np.ndarray:
        Omega_peak = self.peak_amplitude()
        S_sw = self.spectral_shape(f)
        return Omega_peak * S_sw


class TurbulenceSource(GravitationalWaveSource):
    def __init__(self, eqst_params=None, constants=None):
        super().__init__(eqst_params, constants)
    
    def efficiency_factor(self) -> float:
        epsilon_turb = 0.05
        kappa_v = SoundWaveSource(self.ep, self.const).efficiency_factor()
        kappa_turb = epsilon_turb * kappa_v
        return kappa_turb
    
    def peak_amplitude(self) -> float:
        kappa_turb = self.efficiency_factor()
        alpha = self.ep.alpha_PT
        beta_H = self.ep.beta_over_H
        g_star = self.ep.g_star
        v_w = self.ep.v_w
        
        Omega_peak = 3.35e-4 * (1.0 / beta_H) * (kappa_turb * alpha / (1.0 + alpha))**(3.0/2.0) * (100.0 / g_star)**(1.0/3.0) * v_w
        
        return Omega_peak * self.const.h_Planck2018**2
    
    def peak_frequency_today(self) -> float:
        f_sw = SoundWaveSource(self.ep, self.const).peak_frequency_today()
        return 1.71 * f_sw
    
    def spectral_shape(self, f: np.ndarray) -> np.ndarray:
        f_array = np.atleast_1d(f)
        f_peak = self.peak_frequency_today()
        
        H_star_GeV = self.hubble_rate_at_transition()
        H_star_Hz = H_star_GeV * self.const.eV_to_J / self.const.hbar
        
        T_CMB_eV = self.const.T_CMB_K * 8.617333262e-5
        T_PT_eV = self.ep.T_n
        h_star = H_star_Hz * (T_CMB_eV / T_PT_eV) * self.redshift_factor()
        
        x = f_array / f_peak
        
        S_turb = x**3 / ((1.0 + x)**(11.0/3.0) * (1.0 + 8.0 * np.pi * f_array / h_star))
        
        return S_turb if f_array.size > 1 else S_turb[0]
    
    def spectrum(self, f: np.ndarray) -> np.ndarray:
        Omega_peak = self.peak_amplitude()
        S_turb = self.spectral_shape(f)
        return Omega_peak * S_turb