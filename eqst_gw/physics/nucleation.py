import numpy as np
from scipy.integrate import solve_bvp, odeint
from scipy.optimize import brentq, minimize
from typing import Tuple, Dict, Optional, Callable
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class BubbleNucleation:
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
    
    def bounce_equation_ode(self, rho: float, y: np.ndarray, T: float) -> np.ndarray:
        phi, phi_prime = y
        
        from ..physics.phase_transition import PhaseTransitionDynamics
        pt = PhaseTransitionDynamics(self.ep, self.const)
        
        dV_dphi = pt.potential_derivative(phi, T)
        
        if rho < 1.0e-30:
            phi_double_prime = 0.0
        else:
            phi_double_prime = -2.0 * phi_prime / rho + dV_dphi
        
        return np.array([phi_prime, phi_double_prime])
    
    def solve_bounce_shooting(self, T: float, phi_center_guess: float = None) -> Tuple[np.ndarray, np.ndarray, float]:
        if phi_center_guess is None:
            phi_center_guess = 3.0 * self.ep.gamma_thermal * T / self.ep.lambda_quartic
        
        from ..physics.phase_transition import PhaseTransitionDynamics
        pt = PhaseTransitionDynamics(self.ep, self.const)
        
        rho_max = 100.0 / T
        rho_grid = np.logspace(-6, np.log10(rho_max), 5000)
        rho_grid[0] = 1.0e-30
        
        def residual(phi_c):
            y0 = [phi_c, 0.0]
            
            sol = odeint(lambda y, r: self.bounce_equation_ode(r, y, T), y0, rho_grid)
            
            return sol[-1, 0]
        
        phi_c_min = 0.0
        phi_c_max = phi_center_guess * 2.0
        
        try:
            phi_center_optimal = brentq(residual, phi_c_min, phi_c_max, xtol=1e-8)
        except:
            phi_center_optimal = phi_center_guess
        
        y0 = [phi_center_optimal, 0.0]
        solution = odeint(lambda y, r: self.bounce_equation_ode(r, y, T), y0, rho_grid)
        
        phi_bounce = solution[:, 0]
        
        return rho_grid, phi_bounce, phi_center_optimal
    
    def compute_euclidean_action(self, rho: np.ndarray, phi: np.ndarray, T: float) -> float:
        from ..physics.phase_transition import PhaseTransitionDynamics
        pt = PhaseTransitionDynamics(self.ep, self.const)
        
        phi_prime = np.gradient(phi, rho)
        
        V_eff = pt.effective_potential(phi, T)
        V_false = pt.effective_potential(0.0, T)
        
        integrand = 0.5 * phi_prime**2 + (V_eff - V_false)
        
        S3 = 4.0 * np.pi * np.trapz(rho**2 * integrand, rho)
        
        return S3
    
    def nucleation_rate(self, T: float, S3: float) -> float:
        prefactor = T**4 * (S3 / (2.0 * np.pi * T))**(3.0/2.0)
        
        Gamma = prefactor * np.exp(-S3 / T)
        
        return Gamma
    
    def find_nucleation_temperature(self, T_c: float, S3_threshold: float = 140.0) -> Tuple[float, float]:
        def objective(T):
            try:
                rho, phi, phi_c = self.solve_bounce_shooting(T)
                S3 = self.compute_euclidean_action(rho, phi, T)
                return (S3 / T) - S3_threshold
            except:
                return 1.0e10
        
        T_min = 0.8 * T_c
        T_max = T_c
        
        try:
            T_n = brentq(objective, T_min, T_max, xtol=1.0e-6)
            rho_n, phi_n, phi_c_n = self.solve_bounce_shooting(T_n)
            S3_n = self.compute_euclidean_action(rho_n, phi_n, T_n)
        except:
            T_n = 0.93 * T_c
            S3_n = S3_threshold * T_n
        
        return T_n, S3_n
    
    def compute_beta_parameter(self, T_n: float) -> float:
        dT = 0.001 * T_n
        
        T_plus = T_n + dT
        T_minus = T_n - dT
        
        try:
            rho_plus, phi_plus, _ = self.solve_bounce_shooting(T_plus)
            S3_plus = self.compute_euclidean_action(rho_plus, phi_plus, T_plus)
            ratio_plus = S3_plus / T_plus
        except:
            ratio_plus = 140.0
        
        try:
            rho_minus, phi_minus, _ = self.solve_bounce_shooting(T_minus)
            S3_minus = self.compute_euclidean_action(rho_minus, phi_minus, T_minus)
            ratio_minus = S3_minus / T_minus
        except:
            ratio_minus = 140.0
        
        d_ratio_dT = (ratio_plus - ratio_minus) / (2.0 * dT)
        
        H_star = 1.66 * np.sqrt(self.ep.g_star) * T_n**2 / (self.const.M_pl_GeV * 1.0e9)
        
        beta = -H_star * T_n * d_ratio_dT
        
        return beta