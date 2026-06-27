import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import brentq
from typing import Tuple, Dict, Optional
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class BubbleDynamics:
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
    
    def wall_velocity_from_friction(self, alpha: float, T: float) -> float:
        epsilon = alpha * (np.pi**2 / 30.0) * self.ep.g_star * T**4
        
        eta_shear = (1.0 / (4.0 * np.pi)) * ((np.pi**2 / 30.0) * self.ep.g_star * T**4) / T
        
        friction_coefficient = 3.0 * eta_shear / T
        
        v_w_squared = epsilon / (epsilon + friction_coefficient * T)
        v_w = np.sqrt(v_w_squared)
        
        c_s = 1.0 / np.sqrt(3.0)
        if v_w > c_s:
            v_w = c_s * 0.95
        
        return v_w
    
    def bubble_wall_profile(self, r: np.ndarray, R_bubble: float, L_w: float, phi_false: float, phi_true: float) -> np.ndarray:
        r_array = np.atleast_1d(r)
        phi = np.zeros_like(r_array)
        
        for i, r_val in enumerate(r_array):
            if r_val < R_bubble - L_w:
                phi[i] = phi_true
            elif r_val > R_bubble + L_w:
                phi[i] = phi_false
            else:
                xi = (r_val - R_bubble) / L_w
                phi[i] = phi_false + (phi_true - phi_false) * 0.5 * (1.0 - np.tanh(xi))
        
        return phi if r_array.size > 1 else phi[0]
    
    def wall_thickness(self, T: float) -> float:
        mu_squared = (self.ep.mu_GeV * 1.0e9)**2
        kappa = self.ep.kappa_thermal
        
        m_eff_squared = -mu_squared + kappa * T**2
        
        if m_eff_squared > 0:
            m_eff = np.sqrt(m_eff_squared)
            L_w = 1.0 / m_eff
        else:
            L_w = 1.0e-15
        
        return L_w
    
    def surface_tension(self, T: float, phi_false: float, phi_true: float) -> float:
        L_w = self.wall_thickness(T)
        
        phi_range = np.linspace(phi_false, phi_true, 1000)
        
        from ..physics.phase_transition import PhaseTransitionDynamics
        pt = PhaseTransitionDynamics(self.ep, self.const)
        
        V_range = pt.effective_potential(phi_range, T)
        V_false = pt.effective_potential(phi_false, T)
        
        Delta_V = V_range - V_false
        
        gradient_energy = 0.5 * ((phi_true - phi_false) / L_w)**2
        
        sigma = np.sqrt(2.0 * gradient_energy * np.trapz(Delta_V, phi_range))
        
        return sigma
    
    def bubble_expansion_equation(self, t: float, y: Tuple[float, float], alpha: float, T: float) -> Tuple[float, float]:
        R, R_dot = y
        
        epsilon = alpha * (np.pi**2 / 30.0) * self.ep.g_star * T**4
        
        H = 1.66 * np.sqrt(self.ep.g_star) * T**2 / (self.const.M_pl_GeV * 1.0e9)
        
        rho_rad = (np.pi**2 / 30.0) * self.ep.g_star * T**4
        
        driving_pressure = epsilon
        
        friction_pressure = 3.0 * H * rho_rad * R_dot
        
        R_ddot = (driving_pressure - friction_pressure) / rho_rad - 3.0 * H * R_dot
        
        return [R_dot, R_ddot]
    
    def simulate_bubble_expansion(self, alpha: float, T: float, t_max: float = 1.0e-20, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, n_points)
        
        L_w = self.wall_thickness(T)
        R_0 = L_w
        R_dot_0 = 0.0
        
        y0 = [R_0, R_dot_0]
        
        sol = solve_ivp(lambda t, y: self.bubble_expansion_equation(t, y, alpha, T), 
                       t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
        
        return sol.t, sol.y[0], sol.y[1]