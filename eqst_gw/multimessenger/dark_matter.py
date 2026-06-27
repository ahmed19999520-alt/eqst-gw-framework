import numpy as np
from typing import Tuple, Dict, Optional, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class MajoranaGluonDarkMatter:
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
    
    def relic_density(self, T_n: float, beta_over_H: float) -> float:
        H_star = 1.66 * np.sqrt(self.ep.g_star) * T_n**2 / (self.const.M_pl_GeV * 1.0e9)
        
        beta = beta_over_H * H_star
        
        L_corr = 1.0 / beta
        
        n_DM = (1.0 / L_corr)**3
        
        m_DM_GeV = self.ep.m_DM_GeV
        
        rho_DM = n_DM * m_DM_GeV
        
        rho_crit_today = 3.0 * (self.const.H0_Planck2018 * 1000.0 / self.const.Mpc_to_m)**2 / (8.0 * np.pi * self.const.G)
        
        g_star_S_ratio = (3.91 / self.ep.g_star)**(4.0/3.0)
        
        rho_DM_today = rho_DM * g_star_S_ratio
        
        Omega_DM_h2 = rho_DM_today / rho_crit_today * self.const.h_Planck2018**2
        
        return Omega_DM_h2
    
    def direct_detection_rate(self, rho_DM_local: float = 0.4) -> float:
        sigma_DM_SM = self.ep.sigma_DM_SM_cm2 * 1.0e-4
        
        m_DM_kg = self.ep.m_DM_GeV * self.const.GeV_to_kg
        
        n_DM = rho_DM_local * 1.989e30 / (3.086e19**3) / m_DM_kg
        
        v_DM = 220.0e3
        
        R = n_DM * v_DM * sigma_DM_SM
        
        return R
    
    def indirect_detection_flux(self, d_source_Mpc: float, rho_DM_source: float) -> float:
        sigma_v_ann = self.ep.sigma_DM_SM_cm2 * 1.0e-4
        
        m_DM_kg = self.ep.m_DM_GeV * self.const.GeV_to_kg
        
        n_DM = rho_DM_source / m_DM_kg
        
        d_source_m = d_source_Mpc * self.const.Mpc_to_m
        
        Gamma_ann = 0.5 * n_DM**2 * sigma_v_ann * 220.0e3
        
        E_DM = self.ep.m_DM_GeV * 1.0e9 * self.const.eV_to_J
        
        Phi = E_DM * Gamma_ann / (4.0 * np.pi * d_source_m**2)
        
        return Phi
    
    def topological_stability(self) -> Dict:
        m_DM_GeV = self.ep.m_DM_GeV
        
        m_top_threshold = 1.0e14
        
        is_stable = m_DM_GeV > m_top_threshold
        
        topological_charge = 1
        
        lifetime_years = np.inf if is_stable else 1.0e40
        
        return {
            'is_topologically_stable': is_stable,
            'topological_charge': topological_charge,
            'mass_GeV': m_DM_GeV,
            'stability_threshold_GeV': m_top_threshold,
            'estimated_lifetime_years': lifetime_years,
            'formation_mechanism': 'Spontaneous symmetry breaking SU(4) -> SU(3)_C x U(1)_DM'
        }
    
    def phase_space_distribution(self, v: np.ndarray, v_0: float = 220.0e3) -> np.ndarray:
        v_array = np.atleast_1d(v)
        
        v_esc = 544.0e3
        
        N_esc = erf(v_esc / v_0) - 2.0 / np.sqrt(np.pi) * (v_esc / v_0) * np.exp(-(v_esc / v_0)**2)
        
        f_v = np.zeros_like(v_array)
        
        mask = v_array < v_esc
        
        from scipy.special import erf as scipy_erf
        f_v[mask] = (1.0 / N_esc) * (4.0 * np.pi * v_array[mask]**2 / (np.pi * v_0**2)**(3.0/2.0)) * np.exp(-v_array[mask]**2 / v_0**2)
        
        return f_v if v_array.size > 1 else f_v[0]
