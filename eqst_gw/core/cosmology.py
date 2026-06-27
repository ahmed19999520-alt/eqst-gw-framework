import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Callable
from .constants import FundamentalConstants

class LambdaEffectiveCosmology:
    def __init__(self, constants: Optional[FundamentalConstants] = None):
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        self.Lambda_0 = self.const.Omega_Lambda_Planck2018
        self.R_exp = -3.01
        self.R_exp_err = 0.15
        self.z_pivot = 1.09
        self.z_pivot_err = 0.05
        self.F_QCD_amp = 0.41
        self.F_QCD_amp_err = 0.04
        self.F_QCD_exp = 0.68
        self.F_QCD_exp_err = 0.07
        self.T_c_QCD_MeV = 154.0
        self.delta_T_QCD_MeV = 20.0
        
        self.Omega_m = self.const.Omega_m_Planck2018
        self.H0 = self.const.H0_Planck2018
        self.h = self.const.h_Planck2018
        
        self._z_cache = None
        self._Lambda_eff_cache = None
        
    def R_factor(self, z: np.ndarray) -> np.ndarray:
        return np.power((1.0 + z) / self.z_pivot, self.R_exp)
    
    def F_QCD_factor(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(self.F_QCD_amp * np.power(1.0 + z, self.F_QCD_exp))
    
    def M_factor(self, T_GeV: np.ndarray) -> np.ndarray:
        T_MeV = T_GeV * 1.0e3
        return 1.0 / (1.0 + np.exp((T_MeV - self.T_c_QCD_MeV) / self.delta_T_QCD_MeV))
    
    def temperature_from_redshift(self, z: np.ndarray) -> np.ndarray:
        T_CMB_eV = self.const.T_CMB_K * 8.617333262e-5
        return T_CMB_eV * (1.0 + z) * 1.0e-9
    
    def Lambda_eff(self, z: np.ndarray, T_GeV: Optional[np.ndarray] = None) -> np.ndarray:
        z_array = np.atleast_1d(z)
        if T_GeV is None:
            T_GeV = self.temperature_from_redshift(z_array)
        else:
            T_GeV = np.atleast_1d(T_GeV)
        
        Lambda_eff = self.Lambda_0 * self.R_factor(z_array) * self.F_QCD_factor(z_array) * self.M_factor(T_GeV)
        return Lambda_eff if z_array.size > 1 else Lambda_eff[0]
    
    def H_eff(self, z: np.ndarray, Omega_m: Optional[float] = None) -> np.ndarray:
        if Omega_m is None:
            Omega_m = self.Omega_m
        
        z_array = np.atleast_1d(z)
        Omega_Lambda_eff = self.Lambda_eff(z_array)
        Omega_r = 2.469e-5 / self.h**2 * (1.0 + 0.2271 * 3.046)
        
        H = self.H0 * np.sqrt(Omega_m * (1.0 + z_array)**3 + Omega_r * (1.0 + z_array)**4 + Omega_Lambda_eff)
        return H if z_array.size > 1 else H[0]
    
    def comoving_distance(self, z: float) -> float:
        c_km_s = self.const.c / 1000.0
        integrand = lambda zp: c_km_s / self.H_eff(zp)
        result, _ = quad(integrand, 0, z, limit=100)
        return result
    
    def luminosity_distance(self, z: np.ndarray) -> np.ndarray:
        z_array = np.atleast_1d(z)
        d_L = np.zeros_like(z_array)
        for i, z_val in enumerate(z_array):
            d_C = self.comoving_distance(z_val)
            d_L[i] = (1.0 + z_val) * d_C
        return d_L if z_array.size > 1 else d_L[0]
    
    def angular_diameter_distance(self, z: np.ndarray) -> np.ndarray:
        z_array = np.atleast_1d(z)
        d_A = np.zeros_like(z_array)
        for i, z_val in enumerate(z_array):
            d_C = self.comoving_distance(z_val)
            d_A[i] = d_C / (1.0 + z_val)
        return d_A if z_array.size > 1 else d_A[0]
    
    def age_of_universe(self, z: float = 0.0) -> float:
        integrand = lambda zp: 1.0 / ((1.0 + zp) * self.H_eff(zp))
        result, _ = quad(integrand, z, 1000.0, limit=100)
        return result * (self.const.Mpc_to_m / 1000.0) / self.const.year_to_s / 1.0e9
    
    def compute_derived_parameters(self) -> dict:
        H0_SI = self.H0 * 1000.0 / self.const.Mpc_to_m
        rho_crit = 3.0 * H0_SI**2 / (8.0 * np.pi * self.const.G)
        
        t_0 = self.age_of_universe(0.0)
        
        z_eq = (self.const.Omega_m_Planck2018 / (2.469e-5 / self.h**2 * (1.0 + 0.2271 * 3.046))) - 1.0
        
        return {
            'rho_crit_kg_m3': rho_crit,
            'age_Gyr': t_0,
            'z_eq': z_eq,
            'Omega_Lambda_eff_today': self.Lambda_eff(0.0)
        }