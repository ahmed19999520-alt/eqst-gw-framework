import numpy as np
from scipy.special import zeta
from scipy.integrate import quad
from typing import Tuple, Dict, Optional
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class EffectivePotential:
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
    
    def thermal_integral_boson(self, m_squared_over_T_squared: float) -> float:
        if m_squared_over_T_squared < 0:
            return 0.0
        
        x = m_squared_over_T_squared
        
        if x < 1.0e-6:
            J_B = -np.pi**4 / 45.0 + (np.pi**2 / 12.0) * x - (np.pi / 6.0) * x**(3.0/2.0)
        elif x < 10.0:
            J_B = -np.pi**4 / 45.0 + (np.pi**2 / 12.0) * x - (np.pi / 6.0) * x**(3.0/2.0) + x**2 * np.log(x / (2.0 * np.pi**2))
        else:
            J_B = x**2 * (np.log(x) - 2.5)
        
        return J_B
    
    def thermal_integral_fermion(self, m_squared_over_T_squared: float) -> float:
        if m_squared_over_T_squared < 0:
            return 0.0
        
        x = m_squared_over_T_squared
        
        if x < 1.0e-6:
            J_F = 7.0 * np.pi**4 / 360.0
        elif x < 10.0:
            J_F = 7.0 * np.pi**4 / 360.0 + (np.pi**2 / 24.0) * x + x**2 * np.log(x / (2.0 * np.pi**2))
        else:
            J_F = x**2 * (np.log(x) - 2.5)
        
        return J_F
    
    def one_loop_correction(self, phi: np.ndarray, T: float, include_daisy: bool = True) -> np.ndarray:
        phi_array = np.atleast_1d(phi)
        
        g = np.sqrt(self.ep.kappa_thermal * 12.0 / 4.0)
        
        m_gauge_squared = g**2 * phi_array**2
        
        N_gauge_bosons = 15
        
        V_1loop = np.zeros_like(phi_array)
        
        for i, phi_val in enumerate(phi_array):
            m_sq = m_gauge_squared[i]
            x = m_sq / T**2
            
            J_B = self.thermal_integral_boson(x)
            
            V_1loop[i] += N_gauge_bosons * (T**4 / (2.0 * np.pi**2)) * J_B
        
        if include_daisy:
            Pi_T_squared = (self.ep.kappa_thermal / 3.0) * T**2
            
            m_daisy_squared = m_gauge_squared + Pi_T_squared
            
            for i, phi_val in enumerate(phi_array):
                m_sq_daisy = m_daisy_squared[i]
                x_daisy = m_sq_daisy / T**2
                
                correction = -(T / (12.0 * np.pi)) * (m_sq_daisy**(3.0/2.0) - m_gauge_squared[i]**(3.0/2.0))
                
                V_1loop[i] += N_gauge_bosons * correction
        
        return V_1loop if phi_array.size > 1 else V_1loop[0]
    
    def coleman_weinberg_potential(self, phi: np.ndarray) -> np.ndarray:
        phi_array = np.atleast_1d(phi)
        
        g = np.sqrt(self.ep.kappa_thermal * 12.0 / 4.0)
        m_gauge_squared = g**2 * phi_array**2
        
        Lambda_cutoff = self.const.M_pl_GeV * 1.0e9
        
        N_gauge = 15
        
        V_CW = (N_gauge / (64.0 * np.pi**2)) * m_gauge_squared**2 * (np.log(m_gauge_squared / Lambda_cutoff**2) - 25.0/6.0)
        
        return V_CW if phi_array.size > 1 else V_CW[0]
    
    def full_effective_potential(self, phi: np.ndarray, T: float, include_CW: bool = True, include_thermal: bool = True) -> np.ndarray:
        phi_array = np.atleast_1d(phi)
        
        mu_squared = (self.ep.mu_GeV * 1.0e9)**2
        kappa = self.ep.kappa_thermal
        gamma = self.ep.gamma_thermal
        lambda_q = self.ep.lambda_quartic
        
        V_tree = 0.5 * (-mu_squared + kappa * T**2) * phi_array**2 - gamma * T * phi_array**3 + 0.25 * lambda_q * phi_array**4
        
        V_total = V_tree
        
        if include_CW:
            V_CW = self.coleman_weinberg_potential(phi_array)
            V_total += V_CW
        
        if include_thermal and T > 0:
            V_thermal = self.one_loop_correction(phi_array, T, include_daisy=True)
            V_total += V_thermal
        
        return V_total if phi_array.size > 1 else V_total[0]