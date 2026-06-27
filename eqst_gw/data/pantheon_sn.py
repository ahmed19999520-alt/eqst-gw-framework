import numpy as np
import os
import requests
from typing import Tuple, Dict, Optional
from .loaders import load_pantheon_sn, _generate_mock_pantheon_sn


class PantheonSNDataManager:
    PANTHEON_PLUS_URL = "https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"

    SH0ES_H0_VALUE = 73.04
    SH0ES_H0_ERR = 1.04

    PANTHEON_PLUS_SUMMARY = {
        'n_SN': 1701,
        'z_range': (0.001, 2.26),
        'H0_SH0ES': 73.04,
        'H0_SH0ES_err': 1.04,
        'Omega_m': 0.334,
        'Omega_m_err': 0.018,
        'w': -1.026,
        'w_err': 0.064,
        'reference': 'Brout et al. (2022), ApJ 938, 110',
        'doi': '10.3847/1538-4357/ac8e04',
    }

    def __init__(self, data_dir: str = './data/observational/pantheon_plus/'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_data(self, use_mock: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_pantheon_sn(self.data_dir, use_mock=use_mock)

    def get_summary(self) -> Dict:
        return self.PANTHEON_PLUS_SUMMARY

    def magnitude_system_correction(self, mu_obs: np.ndarray, M_B: float = -19.253) -> np.ndarray:
        return mu_obs + M_B

    def distance_modulus_theory_lcdm(self, z: np.ndarray, H0: float = 70.0, Omega_m: float = 0.315) -> np.ndarray:
        from scipy.integrate import quad
        c_km_s = 299792.458
        Omega_Lambda = 1.0 - Omega_m
        z_arr = np.atleast_1d(z)
        mu = np.zeros_like(z_arr)
        for i, z_val in enumerate(z_arr):
            d_C, _ = quad(lambda zp: c_km_s / (H0 * np.sqrt(Omega_m * (1+zp)**3 + Omega_Lambda)), 0, z_val)
            d_L = (1.0 + z_val) * d_C
            mu[i] = 5.0 * np.log10(d_L * 1.0e6) + 25.0
        return mu if z_arr.size > 1 else mu[0]

    def distance_modulus_theory_eqst_gp(self, z: np.ndarray, cosmo, H0: float = 67.4, Omega_m: float = 0.315) -> np.ndarray:
        from scipy.integrate import quad
        c_km_s = 299792.458
        z_arr = np.atleast_1d(z)
        mu = np.zeros_like(z_arr)
        for i, z_val in enumerate(z_arr):
            d_C, _ = quad(lambda zp: c_km_s / cosmo.H_eff(zp, Omega_m), 0, z_val)
            d_L = (1.0 + z_val) * d_C
            mu[i] = 5.0 * np.log10(d_L * 1.0e6) + 25.0
        return mu if z_arr.size > 1 else mu[0]

    def chi2_lcdm(self, z: np.ndarray, mu_obs: np.ndarray, sigma_mu: np.ndarray, H0: float = 70.0, Omega_m: float = 0.315) -> float:
        mu_theory = self.distance_modulus_theory_lcdm(z, H0, Omega_m)
        residuals = mu_obs - mu_theory
        return np.sum((residuals / sigma_mu)**2)

    def chi2_eqst_gp(self, z: np.ndarray, mu_obs: np.ndarray, sigma_mu: np.ndarray, cosmo, H0: float = 67.4, Omega_m: float = 0.315) -> float:
        mu_theory = self.distance_modulus_theory_eqst_gp(z, cosmo, H0, Omega_m)
        residuals = mu_obs - mu_theory
        return np.sum((residuals / sigma_mu)**2)

    def hubble_residuals(self, z: np.ndarray, mu_obs: np.ndarray, H0: float = 70.0, Omega_m: float = 0.315) -> np.ndarray:
        mu_lcdm = self.distance_modulus_theory_lcdm(z, H0, Omega_m)
        return mu_obs - mu_lcdm

    def generate_mock_data_with_eqst_signal(self, n_SN: int = 200, sigma_mu: float = 0.12, cosmo=None, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.random.seed(seed)
        z = np.sort(np.random.uniform(0.01, 2.0, n_SN))
        if cosmo is not None:
            mu_theory = self.distance_modulus_theory_eqst_gp(z, cosmo)
        else:
            mu_theory = self.distance_modulus_theory_lcdm(z)
        sigma_arr = sigma_mu * np.ones(n_SN)
        mu_obs = mu_theory + np.random.normal(0, sigma_arr)
        return z, mu_obs, sigma_arr