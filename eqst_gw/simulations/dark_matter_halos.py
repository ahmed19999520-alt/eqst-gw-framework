import numpy as np
from scipy.integrate import quad, odeint
from scipy.optimize import brentq, minimize
from scipy.special import erf
from typing import Tuple, Dict, Optional
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class DMHaloProfile:
    def __init__(self,
                 M_vir_solar: float,
                 z: float = 0.0,
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
        
        self.M_vir_solar = M_vir_solar
        self.M_vir_kg = M_vir_solar * 1.989e30
        self.z = z
        
        self.rho_crit = self._critical_density()
        self.r_vir = self._virial_radius()
        self.concentration = self._concentration_parameter()
        self.r_s = self.r_vir / self.concentration
        self.rho_s = self._scale_density()
    
    def _critical_density(self) -> float:
        H0_SI = self.const.H0_Planck2018 * 1000.0 / self.const.Mpc_to_m
        Omega_m = self.const.Omega_m_Planck2018
        Omega_Lambda = self.const.Omega_Lambda_Planck2018
        
        E_z = np.sqrt(Omega_m * (1.0 + self.z)**3 + Omega_Lambda)
        H_z = H0_SI * E_z
        
        rho_crit = 3.0 * H_z**2 / (8.0 * np.pi * self.const.G)
        
        return rho_crit
    
    def _virial_radius(self) -> float:
        Delta_vir = 200.0
        
        r_vir = (3.0 * self.M_vir_kg / (4.0 * np.pi * Delta_vir * self.rho_crit))**(1.0/3.0)
        
        return r_vir
    
    def _concentration_parameter(self) -> float:
        M_pivot = 1.0e12 * 1.989e30
        c_0 = 9.0
        alpha = -0.13
        beta = -0.7
        
        c = c_0 * (self.M_vir_kg / M_pivot)**alpha * (1.0 + self.z)**beta
        
        return max(c, 1.5)
    
    def _scale_density(self) -> float:
        c = self.concentration
        
        Delta_c = (200.0 / 3.0) * c**3 / (np.log(1.0 + c) - c / (1.0 + c))
        
        rho_s = Delta_c * self.rho_crit
        
        return rho_s
    
    def nfw_density(self, r: np.ndarray) -> np.ndarray:
        r_array = np.atleast_1d(r)
        
        x = r_array / self.r_s
        
        rho = self.rho_s / (x * (1.0 + x)**2)
        
        return rho if r_array.size > 1 else rho[0]
    
    def nfw_mass_enclosed(self, r: np.ndarray) -> np.ndarray:
        r_array = np.atleast_1d(r)
        
        x = r_array / self.r_s
        
        M = 4.0 * np.pi * self.rho_s * self.r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))
        
        return M if r_array.size > 1 else M[0]
    
    def eqst_gp_density(self, r: np.ndarray) -> np.ndarray:
        r_array = np.atleast_1d(r)
        
        rho_nfw = self.nfw_density(r_array)
        
        m_DM_kg = self.ep.m_DM_GeV * self.const.GeV_to_kg
        
        n_DM_nfw = rho_nfw / m_DM_kg
        
        r_core = 0.05 * self.r_s
        core_suppression = 1.0 - np.exp(-(r_array / r_core)**2)
        
        sigma_v = 220.0e3
        v_rms = np.sqrt(self.const.G * self.M_vir_kg / self.r_vir) / np.sqrt(3.0)
        
        rho_eqst = rho_nfw * core_suppression
        
        return rho_eqst if r_array.size > 1 else rho_eqst[0]
    
    def eqst_gp_mass_enclosed(self, r: np.ndarray) -> np.ndarray:
        r_array = np.atleast_1d(r)
        
        M_enc = np.zeros_like(r_array)
        
        for i, r_val in enumerate(r_array):
            r_integral = np.linspace(1.0e-6 * self.r_s, r_val, 2000)
            rho_integral = self.eqst_gp_density(r_integral)
            integrand = 4.0 * np.pi * r_integral**2 * rho_integral
            M_enc[i] = np.trapz(integrand, r_integral)
        
        return M_enc if r_array.size > 1 else M_enc[0]
    
    def circular_velocity_eqst_gp(self, r: np.ndarray) -> np.ndarray:
        M_enc = self.eqst_gp_mass_enclosed(r)
        
        v_circ = np.sqrt(self.const.G * M_enc / r)
        
        return v_circ
    
    def velocity_dispersion_profile(self, r: np.ndarray) -> np.ndarray:
        r_array = np.atleast_1d(r)
        
        sigma_v_sq = np.zeros_like(r_array)
        
        for i, r_val in enumerate(r_array):
            integrand = lambda r_prime: self.nfw_density(r_prime) * self.const.G * self.nfw_mass_enclosed(r_prime) / r_prime**2
            
            result, _ = quad(integrand, r_val, 10.0 * self.r_vir, limit=50)
            
            sigma_v_sq[i] = result / self.nfw_density(r_val)
        
        return np.sqrt(sigma_v_sq) if r_array.size > 1 else np.sqrt(sigma_v_sq[0])
    
    def gravitational_potential(self, r: np.ndarray) -> np.ndarray:
        r_array = np.atleast_1d(r)
        
        x = r_array / self.r_s
        
        Phi = -4.0 * np.pi * self.const.G * self.rho_s * self.r_s**3 * np.log(1.0 + x) / r_array
        
        return Phi if r_array.size > 1 else Phi[0]
    
    def eqst_gp_dm_annihilation_rate(self, r: np.ndarray) -> np.ndarray:
        rho_eqst = self.eqst_gp_density(r)
        
        m_DM_kg = self.ep.m_DM_GeV * self.const.GeV_to_kg
        n_DM = rho_eqst / m_DM_kg
        
        sigma_v_ann = self.ep.sigma_DM_SM_cm2 * 1.0e-4
        
        v_typical = 220.0e3
        
        Gamma_ann = 0.5 * n_DM**2 * sigma_v_ann * v_typical
        
        return Gamma_ann
    
    def density_profile_summary(self) -> Dict:
        r_test = np.logspace(np.log10(0.001 * self.r_vir), np.log10(self.r_vir), 100)
        
        rho_nfw = self.nfw_density(r_test)
        rho_eqst = self.eqst_gp_density(r_test)
        
        v_circ = self.circular_velocity_eqst_gp(r_test)
        
        return {
            'r_vir_m': self.r_vir,
            'r_vir_kpc': self.r_vir / 3.086e19,
            'r_s_kpc': self.r_s / 3.086e19,
            'concentration': self.concentration,
            'rho_s': self.rho_s,
            'M_vir_solar': self.M_vir_solar,
            'v_max_km_s': np.max(v_circ) / 1000.0,
            'r_vmax_kpc': r_test[np.argmax(v_circ)] / 3.086e19,
            'ratio_eqst_nfw_core': rho_eqst[0] / rho_nfw[0]
        }

