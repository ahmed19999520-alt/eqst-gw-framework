import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Optional, Callable, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants
from ..core.cosmology import LambdaEffectiveCosmology

class ParameterEstimator:
    def __init__(self,
                 eqst_params: Optional[EQSTGPParameters] = None,
                 constants: Optional[FundamentalConstants] = None,
                 cosmology: Optional[LambdaEffectiveCosmology] = None):
        
        if eqst_params is None:
            self.ep = EQSTGPParameters()
        else:
            self.ep = eqst_params
        
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        if cosmology is None:
            self.cosmo = LambdaEffectiveCosmology(self.const)
        else:
            self.cosmo = cosmology
    
    def gw_spectrum_model(self, f: np.ndarray, params: np.ndarray) -> np.ndarray:
        alpha, beta_H, v_w, T_n = params
        
        ep_temp = EQSTGPParameters()
        ep_temp.alpha_PT = alpha
        ep_temp.beta_over_H = beta_H
        ep_temp.v_w = v_w
        ep_temp.T_n = T_n
        
        from ..gravitational_waves.spectrum import GravitationalWaveSpectrum
        gw = GravitationalWaveSpectrum(ep_temp, self.const)
        
        return gw.total_spectrum(f)
    
    def chi_squared_gw(self, params: np.ndarray, f_data: np.ndarray, Omega_data: np.ndarray, sigma_data: np.ndarray) -> float:
        alpha, beta_H, v_w, T_n = params
        
        if alpha <= 0 or beta_H <= 0 or v_w <= 0 or v_w >= 1.0 or T_n <= 0:
            return 1.0e20
        
        Omega_model = self.gw_spectrum_model(f_data, params)
        
        residuals = (Omega_data - Omega_model) / sigma_data
        
        chi2 = np.sum(residuals**2)
        
        return chi2
    
    def fit_gw_spectrum(self, f_data: np.ndarray, Omega_data: np.ndarray, sigma_data: np.ndarray) -> Dict:
        initial_guess = [self.ep.alpha_PT, self.ep.beta_over_H, self.ep.v_w, self.ep.T_n]
        
        bounds = [(0.01, 3.0), (1.0, 1000.0), (0.05, 0.99), (1.0e14, 1.0e18)]
        
        result_local = minimize(
            lambda p: self.chi_squared_gw(p, f_data, Omega_data, sigma_data),
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1.0e-12}
        )
        
        result_global = differential_evolution(
            lambda p: self.chi_squared_gw(p, f_data, Omega_data, sigma_data),
            bounds,
            maxiter=500,
            tol=1.0e-8,
            seed=42
        )
        
        if result_global.fun < result_local.fun:
            best_result = result_global
        else:
            best_result = result_local
        
        alpha_best, beta_H_best, v_w_best, T_n_best = best_result.x
        
        chi2_min = best_result.fun
        dof = len(f_data) - 4
        
        return {
            'alpha': alpha_best,
            'alpha_err': self.ep.alpha_PT_err,
            'beta_over_H': beta_H_best,
            'beta_over_H_err': self.ep.beta_over_H_err,
            'v_w': v_w_best,
            'v_w_err': self.ep.v_w_err,
            'T_n_GeV': T_n_best,
            'chi2_min': chi2_min,
            'dof': dof,
            'chi2_reduced': chi2_min / dof
        }
    
    def bao_chi_squared(self, params: np.ndarray, z_eff: np.ndarray, DM_obs: np.ndarray, DH_obs: np.ndarray, cov_matrix: np.ndarray) -> float:
        Omega_m, h = params
        
        if Omega_m <= 0 or Omega_m >= 1 or h <= 0.4 or h >= 1.0:
            return 1.0e20
        
        self.cosmo.Omega_m = Omega_m
        self.cosmo.H0 = h * 100.0
        self.cosmo.h = h
        
        r_d = self.sound_horizon_rs(Omega_m, h)
        
        DM_theory = np.zeros_like(z_eff)
        DH_theory = np.zeros_like(z_eff)
        
        for i, z in enumerate(z_eff):
            from scipy.integrate import quad
            c_km_s = self.const.c / 1000.0
            d_C, _ = quad(lambda zp: c_km_s / self.cosmo.H_eff(zp, Omega_m), 0, z)
            DM_theory[i] = d_C / r_d
            DH_theory[i] = (c_km_s / self.cosmo.H_eff(z, Omega_m)) / r_d
        
        data_vec = np.concatenate([DM_obs, DH_obs])
        theory_vec = np.concatenate([DM_theory, DH_theory])
        
        residual = data_vec - theory_vec
        
        chi2 = residual @ np.linalg.inv(cov_matrix) @ residual
        
        return chi2
    
    def sound_horizon_rs(self, Omega_m: float, h: float) -> float:
        Omega_b_h2 = self.const.Omega_b_h2_Planck2018
        Omega_m_h2 = Omega_m * h**2
        
        z_drag = 1291.0 * Omega_m_h2**0.251 / (1.0 + 0.659 * Omega_m_h2**0.828) * (1.0 + 0.395 * Omega_m_h2**(-0.569))
        
        a_drag = 1.0 / (1.0 + z_drag)
        
        R_eq = 31500.0 * Omega_b_h2 * (2.7255 / 2.7)**(-4) / z_drag
        
        c_km_s = self.const.c / 1000.0
        
        from scipy.integrate import quad
        def integrand(a):
            z = 1.0 / a - 1.0
            R_a = 31500.0 * Omega_b_h2 * (2.7255 / 2.7)**(-4) / (1.0 + z)
            c_s = c_km_s / np.sqrt(3.0 * (1.0 + R_a))
            H_a = self.cosmo.H_eff(z, Omega_m)
            return c_s / (a**2 * H_a)
        
        r_s, _ = quad(integrand, 1.0e-5, a_drag)
        
        return r_s
    
    def fit_bao(self, z_eff: np.ndarray, DM_obs: np.ndarray, DH_obs: np.ndarray, cov_matrix: np.ndarray) -> Dict:
        initial_guess = [self.const.Omega_m_Planck2018, self.const.h_Planck2018]
        
        bounds = [(0.15, 0.55), (0.55, 0.85)]
        
        result = minimize(
            lambda p: self.bao_chi_squared(p, z_eff, DM_obs, DH_obs, cov_matrix),
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1.0e-12}
        )
        
        Omega_m_best, h_best = result.x
        chi2_min = result.fun
        dof = len(z_eff) * 2 - 2
        
        hess_inv = result.hess_inv if hasattr(result, 'hess_inv') else np.eye(2) * 0.01
        
        return {
            'Omega_m': Omega_m_best,
            'Omega_m_err': np.sqrt(hess_inv[0, 0]) if hasattr(hess_inv, '__getitem__') else 0.01,
            'h': h_best,
            'h_err': np.sqrt(hess_inv[1, 1]) if hasattr(hess_inv, '__getitem__') else 0.005,
            'chi2_min': chi2_min,
            'dof': dof,
            'chi2_reduced': chi2_min / dof
        }
    
    def sn_distance_modulus(self, z: np.ndarray, Omega_m: float, h: float, w0: float = -1.0, wa: float = 0.0) -> np.ndarray:
        z_array = np.atleast_1d(z)
        H0 = h * 100.0
        c_km_s = self.const.c / 1000.0
        
        d_L = np.zeros_like(z_array)
        
        for i, z_val in enumerate(z_array):
            z_grid = np.linspace(0, z_val, 500)
            a_grid = 1.0 / (1.0 + z_grid)
            
            Omega_de = 1.0 - Omega_m
            w_a_integral = np.zeros(len(z_grid))
            for j in range(1, len(z_grid)):
                dz = z_grid[j] - z_grid[j-1]
                w_z = w0 + wa * z_grid[j] / (1.0 + z_grid[j])
                w_a_integral[j] = w_a_integral[j-1] + 3.0 * (1.0 + w_z) * dz / (1.0 + z_grid[j])
            
            Omega_Lambda_eff = self.cosmo.Lambda_eff(z_grid)
            
            E_z = np.sqrt(Omega_m * (1.0 + z_grid)**3 + Omega_Lambda_eff * np.exp(w_a_integral))
            
            integrand = 1.0 / E_z
            d_C = c_km_s / H0 * np.trapz(integrand, z_grid)
            d_L[i] = (1.0 + z_val) * d_C
        
        mu = 5.0 * np.log10(d_L * 1.0e6) + 25.0
        
        return mu if z_array.size > 1 else mu[0]
    
    def fit_sn(self, z_SN: np.ndarray, mu_obs: np.ndarray, cov_SN: np.ndarray) -> Dict:
        def chi2_sn(params):
            Omega_m, h, w0, wa = params
            if Omega_m <= 0 or Omega_m >= 1 or h <= 0.4 or h >= 1.0:
                return 1.0e20
            if w0 < -3.0 or w0 > 0.0:
                return 1.0e20
            if wa < -3.0 or wa > 3.0:
                return 1.0e20
            
            mu_theory = self.sn_distance_modulus(z_SN, Omega_m, h, w0, wa)
            residual = mu_obs - mu_theory
            chi2 = residual @ np.linalg.inv(cov_SN) @ residual
            return chi2
        
        initial_guess = [0.315, 0.674, -1.0, 0.0]
        bounds = [(0.1, 0.7), (0.5, 0.9), (-3.0, 0.0), (-3.0, 3.0)]
        
        result = minimize(chi2_sn, initial_guess, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
        
        Omega_m_best, h_best, w0_best, wa_best = result.x
        chi2_min = result.fun
        dof = len(z_SN) - 4
        
        return {
            'Omega_m': Omega_m_best,
            'h': h_best,
            'w0': w0_best,
            'wa': wa_best,
            'chi2_min': chi2_min,
            'dof': dof,
            'chi2_reduced': chi2_min / dof
        }

