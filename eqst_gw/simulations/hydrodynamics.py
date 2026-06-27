import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Optional, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class RelativisticHydrodynamics:
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
        self.c = 1.0
    
    def lorentz_factor(self, v: np.ndarray) -> np.ndarray:
        v_array = np.atleast_1d(v)
        gamma = 1.0 / np.sqrt(1.0 - v_array**2)
        return gamma if v_array.size > 1 else gamma[0]
    
    def jouguet_velocity(self, alpha: float) -> float:
        v_J = (1.0 / (1.0 + alpha)) * (self.c_s + np.sqrt(alpha**2 + 2.0 * alpha / 3.0))
        return v_J
    
    def deflagration_velocity(self, alpha: float) -> float:
        xi_d = (self.c_s / (1.0 + alpha)) * (1.0 + np.sqrt(3.0 * alpha * (2.0 + 3.0 * alpha)))
        return min(xi_d, self.c_s)
    
    def fluid_shell_profiles(self, v_w: float, alpha: float, xi_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(xi_grid)
        
        v_fluid = np.zeros(N)
        T_fluid = np.ones(N)
        rho_fluid = np.ones(N)
        
        xi_sh = self._shock_front_position(v_w, alpha)
        
        for i, xi in enumerate(xi_grid):
            if xi > xi_sh:
                v_fluid[i] = 0.0
                T_fluid[i] = 1.0
                
            elif xi > v_w:
                mu_sh = (xi - xi_sh) / (1.0 - xi * xi_sh)
                
                v1_sh = mu_sh
                
                if v1_sh < self.c_s:
                    e_ratio = (1.0 + alpha) * (self.c_s**2 - v1_sh**2) / (1.0 - v1_sh**2)
                    T_ratio = (e_ratio)**(1.0/4.0)
                    T_fluid[i] = T_ratio
                    
                    v_fluid[i] = (xi_sh - xi) / (1.0 - xi_sh * xi) * 0.5
                    
            else:
                mu_w = (v_w - xi) / (1.0 - v_w * xi)
                
                if abs(mu_w) < self.c_s:
                    v_fluid[i] = v_w * 0.3
                    T_fluid[i] = (1.0 + alpha)**0.25
                else:
                    v_fluid[i] = 0.0
                    T_fluid[i] = (1.0 + alpha)**0.25
        
        return v_fluid, T_fluid, rho_fluid
    
    def _shock_front_position(self, v_w: float, alpha: float) -> float:
        if v_w < self.c_s:
            v_J = self.jouguet_velocity(alpha)
            xi_sh = v_w * (1.0 + alpha) / (1.0 + alpha * v_w**2 / self.c_s**2)
            xi_sh = max(xi_sh, v_w * 1.05)
            xi_sh = min(xi_sh, 0.99)
        else:
            xi_sh = v_w
        
        return xi_sh
    
    def compute_kinetic_energy_efficiency(self, v_w: float, alpha: float, n_grid: int = 1000) -> float:
        xi_grid = np.linspace(0.01, 0.99, n_grid)
        
        v_fluid, T_fluid, rho_fluid = self.fluid_shell_profiles(v_w, alpha, xi_grid)
        
        gamma_v = self.lorentz_factor(v_fluid)
        
        e_0 = (np.pi**2 / 30.0) * self.ep.g_star
        
        rho_background = e_0
        p_background = e_0 / 3.0
        
        T_sq = (T_fluid - 1.0)**2
        
        e_kin_integrand = rho_background * gamma_v**2 * v_fluid**2 * xi_grid**2
        
        kappa_v = 3.0 * np.trapz(e_kin_integrand, xi_grid) / (alpha * rho_background)
        
        return kappa_v
    
    def mhd_turbulence_spectrum(self, k: np.ndarray, v_rms: float, L_integral: float, nu: float = 1.0e-10) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        k_integral = 2.0 * np.pi / L_integral
        k_dissipation = k_integral * (L_integral * v_rms / nu)**(3.0/4.0)
        
        E_k = np.zeros_like(k_array)
        
        for i, k_val in enumerate(k_array):
            if k_val < k_integral:
                E_k[i] = (k_val / k_integral)**4 * v_rms**2 / k_integral
            elif k_val < k_dissipation:
                E_k[i] = v_rms**2 / k_integral * (k_val / k_integral)**(-5.0/3.0)
            else:
                E_k[i] = v_rms**2 / k_integral * (k_dissipation / k_integral)**(-5.0/3.0) * np.exp(-(k_val - k_dissipation) / k_dissipation)
        
        return E_k if k_array.size > 1 else E_k[0]