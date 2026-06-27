import numpy as np
import os
import requests
from typing import Tuple, Dict, Optional, List
from .loaders import load_jwst_galaxies, _generate_mock_jwst_galaxies


class JWSTDataManager:
    CEERS_FIELD = {'RA_center': 214.825, 'Dec_center': 52.825, 'area_arcmin2': 100.0}
    COSMOS_WEB_FIELD = {'RA_center': 150.119, 'Dec_center': 2.206, 'area_arcmin2': 1700.0}
    PRIMER_FIELD = {'RA_center': 34.500, 'Dec_center': -5.200, 'area_arcmin2': 144.0}

    HIGH_Z_CANDIDATES = [
        {'name': 'JADES-GS-z14-0', 'z_spec': 14.32, 'M_UV': -20.81, 'reference': 'Carniani et al. 2024'},
        {'name': 'JADES-GS-z13-0', 'z_spec': 13.20, 'M_UV': -18.60, 'reference': 'Curtis-Lake et al. 2023'},
        {'name': 'JADES-GS-z12-0', 'z_spec': 12.63, 'M_UV': -18.60, 'reference': 'Curtis-Lake et al. 2023'},
        {'name': 'JADES-GS-z11-0', 'z_spec': 11.58, 'M_UV': -18.60, 'reference': 'Curtis-Lake et al. 2023'},
        {'name': 'GN-z11', 'z_spec': 10.60, 'M_UV': -21.50, 'reference': 'Bunker et al. 2023'},
        {'name': 'MACS0647-JD', 'z_spec': 10.17, 'M_UV': -20.10, 'reference': 'Hsiao et al. 2023'},
        {'name': 'Maisie Galaxy', 'z_spec': 11.44, 'M_UV': -20.50, 'reference': 'Finkelstein et al. 2022'},
    ]

    def __init__(self, data_dir: str = './data/observational/jwst/'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_high_z_galaxies(self, field: str = 'CEERS', z_min: float = 8.0, z_max: float = 15.0, use_mock: bool = False) -> Dict[str, np.ndarray]:
        return load_jwst_galaxies(self.data_dir, field=field, z_min=z_min, z_max=z_max, use_mock=use_mock)

    def get_confirmed_high_z_candidates(self) -> List[Dict]:
        return self.HIGH_Z_CANDIDATES

    def stellar_mass_function_schechter(self, log_M: np.ndarray, z: float) -> np.ndarray:
        log_M_star_z = 10.9 - 0.1 * (z - 3.0)
        alpha_z = -1.5 - 0.05 * (z - 3.0)
        phi_star_z = 10**(-3.5 - 0.3 * (z - 3.0))
        M_ratio = 10.0**(log_M - log_M_star_z)
        phi = phi_star_z * M_ratio**(alpha_z + 1.0) * np.exp(-M_ratio) * np.log(10.0)
        return phi

    def uv_luminosity_function(self, M_UV: np.ndarray, z: float) -> np.ndarray:
        M_UV_star_z = -21.0 + 0.5 * (z - 6.0) * 0.3
        alpha_z = -2.0 - 0.05 * (z - 6.0)
        phi_star_z = 10**(-3.8 - 0.3 * (z - 6.0))
        ratio = 10.0**(-0.4 * (M_UV - M_UV_star_z))
        phi = 0.4 * np.log(10.0) * phi_star_z * ratio**(alpha_z + 1.0) * np.exp(-ratio)
        return phi

    def sfr_density_evolution(self, z: np.ndarray) -> np.ndarray:
        z_arr = np.atleast_1d(z)
        log_sfrd = -0.997 + 3.241 * z_arr / (1.0 + (z_arr / 3.241)**1.741)
        return 10.0**log_sfrd

    def eqst_gp_galaxy_formation_modification(self, z: np.ndarray, Lambda_eff_func) -> np.ndarray:
        z_arr = np.atleast_1d(z)
        Lambda_eff = Lambda_eff_func(z_arr)
        Lambda_lcdm = 0.685
        modification = 1.0 + 0.15 * (Lambda_eff - Lambda_lcdm) / Lambda_lcdm * np.exp(-z_arr / 10.0)
        sfrd_standard = self.sfr_density_evolution(z_arr)
        sfrd_eqst = sfrd_standard * modification
        return sfrd_eqst

    def generate_photometric_catalog(self, n_galaxies: int = 500, z_min: float = 8.0, z_max: float = 15.0, seed: int = 42) -> Dict[str, np.ndarray]:
        np.random.seed(seed)
        z_phot = np.random.uniform(z_min, z_max, n_galaxies)
        log_M_mean = 8.0 + 0.5 * (z_phot - 10.0) / 5.0
        log_M = log_M_mean + np.random.normal(0, 0.4, n_galaxies)
        M_stellar = 10.0**log_M
        SFR = 10.0**(0.9 * log_M - 8.5 + np.random.normal(0, 0.3, n_galaxies))
        M_UV = -2.5 * np.log10(SFR) - 17.5 + np.random.normal(0, 0.3, n_galaxies)
        beta_UV = -2.0 - 0.2 * (log_M - 9.0) + np.random.normal(0, 0.3, n_galaxies)
        RA = np.random.uniform(214.6, 215.1, n_galaxies)
        Dec = np.random.uniform(52.7, 52.9, n_galaxies)
        z_phot_err = 0.3 * np.ones(n_galaxies) + 0.1 * np.random.rand(n_galaxies)

        catalog = {
            'z_phot': z_phot,
            'z_phot_err': z_phot_err,
            'M_stellar_solar': M_stellar,
            'SFR_solar_per_yr': SFR,
            'M_UV': M_UV,
            'UV_slope': beta_UV,
            'RA_deg': RA,
            'Dec_deg': Dec,
        }
        return catalog

    def save_catalog_csv(self, catalog: Dict[str, np.ndarray], filename: str = 'high_z_galaxies_catalog.csv') -> str:
        import pandas as pd
        df = pd.DataFrame(catalog)
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Galaxy catalog saved to {filepath}")
        return filepath