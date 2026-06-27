import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp, quad
from typing import Tuple, Dict, Optional, Callable
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class PhaseTransitionDynamics:
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
    
    def effective_potential(self, phi: np.ndarray, T: float) -> np.ndarray:
        phi_array = np.atleast_1d(phi)
        
        mu_squared = (self.ep.mu_GeV * 1.0e9)**2
        kappa = self.ep.kappa_thermal
        gamma = self.ep.gamma_thermal
        lambda_q = self.ep.lambda_quartic
        
        T_GeV = T / 1.0e9
        
        V_eff = (0.5 * (-mu_squared + kappa * T**2) * phi_array**2 
                 - gamma * T * phi_array**3 
                 + 0.25 * lambda_q * phi_array**4)
        
        return V_eff if phi_array.size > 1 else V_eff[0]
    
    def potential_derivative(self, phi: np.ndarray, T: float) -> np.ndarray:
        phi_array = np.atleast_1d(phi)
        
        mu_squared = (self.ep.mu_GeV * 1.0e9)**2
        kappa = self.ep.kappa_thermal
        gamma = self.ep.gamma_thermal
        lambda_q = self.ep.lambda_quartic
        
        dV = ((-mu_squared + kappa * T**2) * phi_array 
              - 3.0 * gamma * T * phi_array**2 
              + lambda_q * phi_array**3)
        
        return dV if phi_array.size > 1 else dV[0]
    
    def find_minima(self, T: float) -> Tuple[float, float]:
        phi_range = np.linspace(0, 1.0e20, 1000)
        V_range = self.effective_potential(phi_range, T)
        
        local_minima = []
        for i in range(1, len(phi_range) - 1):
            if V_range[i] < V_range[i-1] and V_range[i] < V_range[i+1]:
                local_minima.append(phi_range[i])
        
        phi_false = 0.0
        V_false = self.effective_potential(0.0, T)
        
        if len(local_minima) > 0:
            phi_true_candidates = local_minima
            V_true_candidates = [self.effective_potential(phi, T) for phi in phi_true_candidates]
            idx_min = np.argmin(V_true_candidates)
            phi_true = phi_true_candidates[idx_min]
        else:
            phi_true = 0.0
        
        return phi_false, phi_true
    
    def critical_temperature(self) -> float:
        mu_squared = (self.ep.mu_GeV * 1.0e9)**2
        kappa = self.ep.kappa_thermal
        gamma = self.ep.gamma_thermal
        lambda_q = self.ep.lambda_quartic
        
        T_c_squared = mu_squared / (kappa - 9.0 * gamma**2 / (2.0 * lambda_q))
        T_c = np.sqrt(T_c_squared)
        
        return T_c
    
    def latent_heat(self, T: float) -> float:
        phi_false, phi_true = self.find_minima(T)
        
        V_false = self.effective_potential(phi_false, T)
        V_true = self.effective_potential(phi_true, T)
        
        Delta_V = V_false - V_true
        
        dT = 0.001 * T
        T_plus = T + dT
        T_minus = T - dT
        
        _, phi_true_plus = self.find_minima(T_plus)
        _, phi_true_minus = self.find_minima(T_minus)
        
        V_false_plus = self.effective_potential(0.0, T_plus)
        V_true_plus = self.effective_potential(phi_true_plus, T_plus)
        Delta_V_plus = V_false_plus - V_true_plus
        
        V_false_minus = self.effective_potential(0.0, T_minus)
        V_true_minus = self.effective_potential(phi_true_minus, T_minus)
        Delta_V_minus = V_false_minus - V_true_minus
        
        dDelta_V_dT = (Delta_V_plus - Delta_V_minus) / (2.0 * dT)
        
        epsilon = Delta_V - (T / 4.0) * dDelta_V_dT
        
        return epsilon
    
    def transition_strength_alpha(self, T: float) -> float:
        epsilon = self.latent_heat(T)
        
        g_star = self.ep.g_star
        rho_rad = (np.pi**2 / 30.0) * g_star * T**4
        
        alpha = epsilon / rho_rad
        
        return alpha
    
    def radiation_energy_density(self, T: float) -> float:
        g_star = self.ep.g_star
        rho_rad = (np.pi**2 / 30.0) * g_star * T**4
        return rho_rad