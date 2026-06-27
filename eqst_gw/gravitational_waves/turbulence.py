import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as gamma_func
from typing import Tuple, Optional
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class MHDTurbulence:
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
    
    def kolmogorov_spectrum(self, k: np.ndarray, k_integral: float, epsilon_turb: float) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        C_K = 1.6
        
        E_k = C_K * epsilon_turb**(2.0/3.0) * k_array**(-5.0/3.0)
        
        k_dissipation = 100.0 * k_integral
        
        cutoff = np.exp(-(k_array / k_dissipation)**2)
        
        E_k *= cutoff
        
        return E_k if k_array.size > 1 else E_k[0]
    
    def integral_scale(self, v_w: float, beta: float) -> float:
        H_star = self.hubble_rate()
        
        L_integral = v_w / beta
        
        return L_integral
    
    def dissipation_scale(self, L_integral: float, Reynolds_number: float = 1.0e6) -> float:
        L_dissipation = L_integral * Reynolds_number**(-3.0/4.0)
        
        return L_dissipation
    
    def turbulent_kinetic_energy_density(self) -> float:
        alpha = self.ep.alpha_PT
        epsilon_fraction = 0.05
        
        kappa_v = alpha / (0.73 + 0.083 * np.sqrt(alpha) + alpha)
        
        rho_rad = (np.pi**2 / 30.0) * self.ep.g_star * self.ep.T_n**4
        
        rho_turb = epsilon_fraction * kappa_v * alpha * rho_rad / (1.0 + alpha)
        
        return rho_turb
    
    def turbulent_velocity_rms(self) -> float:
        rho_turb = self.turbulent_kinetic_energy_density()
        rho_rad = (np.pi**2 / 30.0) * self.ep.g_star * self.ep.T_n**4
        
        v_rms_over_c = np.sqrt(rho_turb / rho_rad)
        
        return v_rms_over_c * self.const.c
    
    def anisotropic_stress_tensor_spectrum(self, k: np.ndarray) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        L_integral = self.integral_scale(self.ep.v_w, self.ep.beta_over_H * self.hubble_rate())
        k_integral = 2.0 * np.pi / L_integral
        
        epsilon_turb = self.turbulent_kinetic_energy_density()
        
        E_k = self.kolmogorov_spectrum(k_array, k_integral, epsilon_turb)
        
        Pi_ij_k = (4.0 / 3.0) * E_k
        
        return Pi_ij_k if k_array.size > 1 else Pi_ij_k[0]
    
    def gw_power_from_turbulence(self, k: np.ndarray) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        Pi_k = self.anisotropic_stress_tensor_spectrum(k_array)
        
        H_star = self.hubble_rate()
        
        P_gw = (16.0 * np.pi * self.const.G**2 / (self.const.c**4)) * (k_array / H_star)**2 * Pi_k**2
        
        return P_gw if k_array.size > 1 else P_gw[0]
    
    def turbulence_lifetime(self) -> float:
        H_star = self.hubble_rate()
        
        tau_turb = 1.0 / H_star
        
        return tau_turb
    
    def hubble_rate(self) -> float:
        T_star = self.ep.T_n
        g_star = self.ep.g_star
        M_pl = self.const.M_pl_GeV * 1.0e9
        
        H_star = 1.66 * np.sqrt(g_star) * T_star**2 / M_pl
        
        return H_star
