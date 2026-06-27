import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import spherical_jn
from typing import Tuple, Optional, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class BubbleCollisionDynamics:
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
    
    def bubble_wall_profile_fourier(self, k: np.ndarray, R_bubble: float, L_wall: float) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        profile_k = np.zeros_like(k_array, dtype=complex)
        
        for i, k_val in enumerate(k_array):
            if k_val < 1.0e-20:
                profile_k[i] = (4.0 * np.pi / 3.0) * R_bubble**3
            else:
                kR = k_val * R_bubble
                kL = k_val * L_wall
                
                profile_k[i] = 4.0 * np.pi * R_bubble**2 * (np.sin(kR) - kR * np.cos(kR)) / k_val**3
                
                profile_k[i] *= 1.0 / (1.0 + (kL)**2)
        
        return profile_k if k_array.size > 1 else profile_k[0]
    
    def collision_rate_per_volume(self, t: float) -> float:
        beta = self.ep.beta_over_H * self.hubble_rate()
        
        I_nucleation = (beta * t)**3 / 6.0
        
        P_no_collision = np.exp(-I_nucleation)
        
        Gamma_collision = beta * I_nucleation * P_no_collision
        
        return Gamma_collision
    
    def uncorrelated_collision_spectrum(self, k: np.ndarray, t_collision: float) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        v_w = self.ep.v_w
        R_typical = v_w * t_collision
        
        L_wall = 1.0 / (self.ep.mu_GeV * 1.0e9)
        
        profile_k = self.bubble_wall_profile_fourier(k_array, R_typical, L_wall)
        
        kappa_phi = 4.9e-3
        alpha = self.ep.alpha_PT
        rho_rad = (np.pi**2 / 30.0) * self.ep.g_star * self.ep.T_n**4
        
        epsilon_phi = kappa_phi * alpha * rho_rad / (1.0 + alpha)
        
        S_phi_k = (epsilon_phi / R_typical**3) * np.abs(profile_k)**2
        
        return S_phi_k if k_array.size > 1 else S_phi_k[0]
    
    def envelope_approximation_spectrum(self, k: np.ndarray) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        beta = self.ep.beta_over_H * self.hubble_rate()
        v_w = self.ep.v_w
        
        k_peak = beta / v_w
        
        kappa_phi = 4.9e-3
        alpha = self.ep.alpha_PT
        rho_rad = (np.pi**2 / 30.0) * self.ep.g_star * self.ep.T_n**4
        
        epsilon_phi = kappa_phi * alpha * rho_rad / (1.0 + alpha)
        
        H_star = self.hubble_rate()
        
        normalization = (epsilon_phi / rho_rad) * (H_star / beta)**2
        
        x = k_array / k_peak
        
        shape_function = x**3 / (1.0 + x**2)**(7.0/2.0)
        
        Omega_phi_k = normalization * shape_function
        
        return Omega_phi_k if k_array.size > 1 else Omega_phi_k[0]
    
    def bubble_correlation_function(self, r: np.ndarray, t: float) -> np.ndarray:
        r_array = np.atleast_1d(r)
        
        beta = self.ep.beta_over_H * self.hubble_rate()
        v_w = self.ep.v_w
        
        R_typical = v_w * t
        
        xi = np.zeros_like(r_array)
        
        for i, r_val in enumerate(r_array):
            if r_val < 2.0 * R_typical:
                overlap_volume = (4.0 * np.pi / 3.0) * (R_typical - r_val / 2.0)**3
                xi[i] = overlap_volume / ((4.0 * np.pi / 3.0) * R_typical**3)
            else:
                xi[i] = 0.0
        
        return xi if r_array.size > 1 else xi[0]
    
    def hubble_rate(self) -> float:
        T_star = self.ep.T_n
        g_star = self.ep.g_star
        M_pl = self.const.M_pl_GeV * 1.0e9
        
        H_star = 1.66 * np.sqrt(g_star) * T_star**2 / M_pl
        
        return H_star