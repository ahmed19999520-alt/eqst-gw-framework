import numpy as np
from scipy.integrate import odeint, quad
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Optional, List
from ..core.constants import FundamentalConstants
from ..core.cosmology import LambdaEffectiveCosmology

class StructureFormationAnalysis:
    def __init__(self,
                 constants: Optional[FundamentalConstants] = None,
                 cosmology: Optional[LambdaEffectiveCosmology] = None):
        
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        if cosmology is None:
            self.cosmo = LambdaEffectiveCosmology(self.const)
        else:
            self.cosmo = cosmology
    
    def linear_growth_factor(self, z: np.ndarray) -> np.ndarray:
        z_array = np.atleast_1d(z)
        D = np.zeros_like(z_array)
        
        Omega_m = self.const.Omega_m_Planck2018
        Omega_Lambda_0 = self.const.Omega_Lambda_Planck2018
        
        for i, z_val in enumerate(z_array):
            a = 1.0 / (1.0 + z_val)
            
            Omega_m_z = Omega_m * (1.0 + z_val)**3 / (Omega_m * (1.0 + z_val)**3 + self.cosmo.Lambda_eff(z_val))
            
            integrand = lambda z_prime: (1.0 + z_prime) / (Omega_m * (1.0 + z_prime)**3 + self.cosmo.Lambda_eff(z_prime))**(3.0/2.0)
            
            integral, _ = quad(integrand, z_val, 1000.0, limit=100)
            
            H_z = self.cosmo.H_eff(z_val, Omega_m)
            
            D[i] = 5.0 * Omega_m * H_z / (2.0 * self.const.H0_Planck2018) * integral
        
        D_0 = D[0] if len(z_array) > 1 else D[0]
        
        return D / D_0 if z_array.size > 1 else (D / D_0)[0]
    
    def growth_rate(self, z: np.ndarray) -> np.ndarray:
        z_array = np.atleast_1d(z)
        
        dz = 0.001
        D_plus = self.linear_growth_factor(z_array + dz)
        D_minus = self.linear_growth_factor(z_array - dz)
        
        dD_dz = (D_plus - D_minus) / (2.0 * dz)
        D = self.linear_growth_factor(z_array)
        
        a = 1.0 / (1.0 + z_array)
        
        f = -a * dD_dz / D
        
        return f if z_array.size > 1 else f[0]
    
    def matter_power_spectrum(self, k: np.ndarray, z: float = 0.0) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        n_s = self.const.n_s_Planck2018
        A_s = 2.1e-9
        k_pivot = 0.05
        
        P_primordial = A_s * (k_array / k_pivot)**(n_s - 1.0)
        
        T_k = self.transfer_function_BBKS(k_array)
        
        D_z = self.linear_growth_factor(z)
        
        P_k = 2.0 * np.pi**2 * k_array * P_primordial * T_k**2 * D_z**2
        
        return P_k if k_array.size > 1 else P_k[0]
    
    def transfer_function_BBKS(self, k: np.ndarray) -> np.ndarray:
        k_array = np.atleast_1d(k)
        
        h = self.const.h_Planck2018
        Omega_m = self.const.Omega_m_Planck2018
        Omega_b = self.const.Omega_b_h2_Planck2018 / h**2
        
        Gamma = Omega_m * h * np.exp(-Omega_b * (1.0 + np.sqrt(2.0 * h) / Omega_m))
        
        q = k_array / (Gamma * h)
        
        T_k = np.log(1.0 + 2.34 * q) / (2.34 * q)
        T_k *= (1.0 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**(-0.25)
        
        return T_k if k_array.size > 1 else T_k[0]
    
    def sigma8_eqst_gp(self, R_filter: float = 8.0) -> float:
        k_min = 1.0e-4
        k_max = 1.0e4
        k_grid = np.logspace(np.log10(k_min), np.log10(k_max), 1000)
        
        P_k = self.matter_power_spectrum(k_grid, z=0.0)
        
        kR = k_grid * R_filter / self.const.h_Planck2018
        W_kR = 3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR**3
        
        integrand = k_grid**2 * P_k * W_kR**2 / (2.0 * np.pi**2)
        
        sigma8_sq = np.trapz(integrand, k_grid)
        
        return np.sqrt(sigma8_sq)
    
    def halo_mass_function_Press_Schechter(self, M: np.ndarray, z: float = 0.0) -> np.ndarray:
        M_array = np.atleast_1d(M)
        
        rho_m = (np.pi**2 / 30.0) * 2.0 * (self.const.T_CMB_K * 8.617333262e-5)**4 * self.const.Omega_m_Planck2018
        
        delta_c = 1.686
        
        R = (3.0 * M_array / (4.0 * np.pi * rho_m))**(1.0/3.0)
        
        sigma_R = np.array([self._sigma_R(r) for r in R])
        
        D_z = self.linear_growth_factor(z)
        sigma_Rz = sigma_R * D_z
        
        nu = delta_c / sigma_Rz
        
        d_ln_sigma_inv_d_ln_M = np.gradient(np.log(1.0 / sigma_Rz), np.log(M_array))
        
        dn_dlnM = np.sqrt(2.0 / np.pi) * (rho_m / M_array) * nu * np.exp(-0.5 * nu**2) * np.abs(d_ln_sigma_inv_d_ln_M)
        
        return dn_dlnM if M_array.size > 1 else dn_dlnM[0]
    
    def _sigma_R(self, R: float) -> float:
        k_min = 1.0e-4
        k_max = 1.0e4
        k_grid = np.logspace(np.log10(k_min), np.log10(k_max), 500)
        
        P_k = self.matter_power_spectrum(k_grid)
        
        kR = k_grid * R
        W_kR = 3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR**3
        
        integrand = k_grid**2 * P_k * W_kR**2 / (2.0 * np.pi**2)
        
        sigma_sq = np.trapz(integrand, k_grid)
        
        return np.sqrt(sigma_sq)
    
    def two_point_correlation_function(self, r: np.ndarray, z: float = 0.0) -> np.ndarray:
        r_array = np.atleast_1d(r)
        
        k_min = 1.0e-4
        k_max = 1.0e4
        k_grid = np.logspace(np.log10(k_min), np.log10(k_max), 1000)
        
        P_k = self.matter_power_spectrum(k_grid, z=z)
        
        xi_r = np.zeros_like(r_array)
        
        for i, r_val in enumerate(r_array):
            kr = k_grid * r_val
            integrand = k_grid**2 * P_k * np.sin(kr) / (kr) / (2.0 * np.pi**2)
            xi_r[i] = np.trapz(integrand, k_grid)
        
        return xi_r if r_array.size > 1 else xi_r[0]

