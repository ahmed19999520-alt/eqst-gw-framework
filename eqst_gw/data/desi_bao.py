import numpy as np
import os
import requests
import json
from typing import Tuple, Dict, Optional, List
from .loaders import load_desi_bao, _generate_mock_desi_bao


class DESIBAODataManager:
    DESI_PUBLIC_URL = "https://data.desi.lbl.gov/public/dr1/science/bao/"

    DESI_DR1_MEASUREMENTS = {
        'BGS': {'z_eff': 0.295, 'DM_over_rd': 7.93, 'DH_over_rd': 20.08, 'sigma_DM': 0.15, 'sigma_DH': 0.60},
        'LRG1': {'z_eff': 0.510, 'DM_over_rd': 13.62, 'DH_over_rd': 20.98, 'sigma_DM': 0.25, 'sigma_DH': 0.61},
        'LRG2': {'z_eff': 0.706, 'DM_over_rd': 16.85, 'DH_over_rd': 19.33, 'sigma_DM': 0.32, 'sigma_DH': 0.53},
        'LRG3+ELG1': {'z_eff': 0.930, 'DM_over_rd': 21.71, 'DH_over_rd': 17.88, 'sigma_DM': 0.28, 'sigma_DH': 0.35},
        'ELG2': {'z_eff': 1.317, 'DM_over_rd': 27.79, 'DH_over_rd': 13.82, 'sigma_DM': 0.69, 'sigma_DH': 0.42},
        'QSO': {'z_eff': 1.491, 'DM_over_rd': 30.21, 'DH_over_rd': 13.23, 'sigma_DM': 0.79, 'sigma_DH': 0.59},
        'Lya_QSO': {'z_eff': 2.330, 'DM_over_rd': 39.71, 'DH_over_rd': 8.52, 'sigma_DM': 0.94, 'sigma_DH': 0.17},
    }

    def __init__(self, data_dir: str = './data/observational/desi/'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_dr1_data(self, use_mock: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return load_desi_bao(self.data_dir, release='DR1', use_mock=use_mock)

    def get_dr1_table(self) -> Dict:
        return self.DESI_DR1_MEASUREMENTS

    def sound_horizon_fiducial(self) -> float:
        return 147.09

    def compute_DM_DH_ratio(self) -> np.ndarray:
        ratios = []
        for tracer, vals in self.DESI_DR1_MEASUREMENTS.items():
            ratio = vals['DM_over_rd'] / vals['DH_over_rd']
            ratios.append({'tracer': tracer, 'z_eff': vals['z_eff'], 'DM_DH_ratio': ratio})
        return ratios

    def chi2_lcdm(self, H0: float = 67.4, Omega_m: float = 0.315) -> float:
        from scipy.integrate import quad

        r_d = self.sound_horizon_fiducial()
        c_km_s = 299792.458
        Omega_Lambda = 1.0 - Omega_m

        chi2 = 0.0
        for tracer, vals in self.DESI_DR1_MEASUREMENTS.items():
            z = vals['z_eff']
            H_z = H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)
            d_C, _ = quad(lambda zp: c_km_s / (H0 * np.sqrt(Omega_m * (1+zp)**3 + Omega_Lambda)), 0, z)
            DM_th = d_C / r_d
            DH_th = c_km_s / (H_z * r_d)
            chi2 += ((vals['DM_over_rd'] - DM_th) / vals['sigma_DM'])**2
            chi2 += ((vals['DH_over_rd'] - DH_th) / vals['sigma_DH'])**2

        return chi2

    def chi2_eqst_gp(self, cosmo, H0: float = 67.4, Omega_m: float = 0.315) -> float:
        from scipy.integrate import quad

        r_d = self.sound_horizon_fiducial()
        c_km_s = 299792.458

        chi2 = 0.0
        for tracer, vals in self.DESI_DR1_MEASUREMENTS.items():
            z = vals['z_eff']
            H_z = cosmo.H_eff(z, Omega_m)
            d_C, _ = quad(lambda zp: c_km_s / cosmo.H_eff(zp, Omega_m), 0, z)
            DM_th = d_C / r_d
            DH_th = c_km_s / (H_z * r_d)
            chi2 += ((vals['DM_over_rd'] - DM_th) / vals['sigma_DM'])**2
            chi2 += ((vals['DH_over_rd'] - DH_th) / vals['sigma_DH'])**2

        return chi2

    def save_dr1_csv(self) -> str:
        import pandas as pd
        rows = []
        for tracer, vals in self.DESI_DR1_MEASUREMENTS.items():
            row = {'tracer': tracer}
            row.update(vals)
            rows.append(row)
        df = pd.DataFrame(rows)
        filepath = os.path.join(self.data_dir, 'DESI_BAO_2024_DR1.csv')
        df.to_csv(filepath, index=False)
        print(f"DESI DR1 BAO data saved to {filepath}")
        return filepath

    def build_full_covariance_matrix(self) -> np.ndarray:
        n = len(self.DESI_DR1_MEASUREMENTS)
        tracers = list(self.DESI_DR1_MEASUREMENTS.keys())
        dim = 2 * n
        cov = np.zeros((dim, dim))

        for i, t in enumerate(tracers):
            vals = self.DESI_DR1_MEASUREMENTS[t]
            cov[i, i] = vals['sigma_DM']**2
            cov[n + i, n + i] = vals['sigma_DH']**2
            cov[i, n + i] = -0.40 * vals['sigma_DM'] * vals['sigma_DH']
            cov[n + i, i] = cov[i, n + i]

        filepath = os.path.join(self.data_dir, 'covariance_matrix.npy')
        np.save(filepath, cov)
        print(f"DESI DR1 covariance matrix saved to {filepath}")
        return cov