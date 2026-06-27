import numpy as np
from typing import Tuple, Dict, Optional, Callable, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class FisherMatrixAnalysis:
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
    
    def compute_fisher_gw(self, f: np.ndarray, noise_psd: np.ndarray, param_names: List[str], fiducial_params: np.ndarray, delta_rel: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_params = len(param_names)
        
        F = np.zeros((n_params, n_params))
        
        derivatives = {}
        
        for i, name in enumerate(param_names):
            params_plus = fiducial_params.copy()
            params_minus = fiducial_params.copy()
            
            delta = delta_rel * np.abs(fiducial_params[i])
            if delta == 0:
                delta = 1.0e-5
            
            params_plus[i] += delta
            params_minus[i] -= delta
            
            spectrum_plus = self._compute_spectrum(f, params_plus)
            spectrum_minus = self._compute_spectrum(f, params_minus)
            
            derivatives[name] = (spectrum_plus - spectrum_minus) / (2.0 * delta)
        
        H0_SI = self.const.H0_Planck2018 * 1000.0 / self.const.Mpc_to_m
        
        Omega_sens_sq = noise_psd**2
        
        for i, name_i in enumerate(param_names):
            for j, name_j in enumerate(param_names):
                integrand = derivatives[name_i] * derivatives[name_j] / Omega_sens_sq
                
                F[i, j] = np.trapz(integrand, np.log(f))
        
        covariance = np.linalg.inv(F)
        
        errors = np.sqrt(np.diag(covariance))
        
        correlation = covariance / np.outer(errors, errors)
        
        return F, covariance, errors
    
    def _compute_spectrum(self, f: np.ndarray, params: np.ndarray) -> np.ndarray:
        alpha, beta_H, v_w, T_n = params
        
        ep_temp = EQSTGPParameters()
        ep_temp.alpha_PT = np.abs(alpha)
        ep_temp.beta_over_H = np.abs(beta_H)
        ep_temp.v_w = np.clip(np.abs(v_w), 0.05, 0.99)
        ep_temp.T_n = np.abs(T_n)
        
        from ..gravitational_waves.spectrum import GravitationalWaveSpectrum
        gw = GravitationalWaveSpectrum(ep_temp, self.const)
        
        return gw.total_spectrum(f)
    
    def forecast_constraints(self, f: np.ndarray, noise_psd: np.ndarray, T_obs_years: float = 4.0) -> Dict:
        param_names = ['alpha', 'beta_over_H', 'v_w', 'T_n']
        
        fiducial_params = np.array([
            self.ep.alpha_PT,
            self.ep.beta_over_H,
            self.ep.v_w,
            self.ep.T_n
        ])
        
        noise_psd_scaled = noise_psd / np.sqrt(T_obs_years * self.const.year_to_s)
        
        F, covariance, errors = self.compute_fisher_gw(f, noise_psd_scaled, param_names, fiducial_params)
        
        return {
            'Fisher_matrix': F,
            'covariance_matrix': covariance,
            'parameter_errors': dict(zip(param_names, errors)),
            'relative_errors': dict(zip(param_names, errors / fiducial_params)),
            'correlations': covariance / np.outer(errors, errors)
        }
    
    def marginalized_error(self, Fisher: np.ndarray, param_index: int) -> float:
        covariance = np.linalg.inv(Fisher)
        return np.sqrt(covariance[param_index, param_index])
    
    def conditional_error(self, Fisher: np.ndarray, param_index: int) -> float:
        return 1.0 / np.sqrt(Fisher[param_index, param_index])
    
    def figure_of_merit(self, covariance: np.ndarray, indices: Tuple[int, int] = (0, 1)) -> float:
        i, j = indices
        
        sub_cov = covariance[np.ix_(list(indices), list(indices))]
        
        fom = 1.0 / np.sqrt(np.linalg.det(sub_cov))
        
        return fom