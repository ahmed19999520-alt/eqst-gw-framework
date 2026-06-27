import numpy as np
import os
import requests
from typing import Tuple, Dict, Optional
from .loaders import load_planck_data, _generate_mock_planck_spectrum


class PlanckDataManager:
    BASE_URL = "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/"

    FILE_MAP = {
        'TT': 'COM_PowerSpect_CMB-TT-full_R3.01.txt',
        'TE': 'COM_PowerSpect_CMB-TE-full_R3.01.txt',
        'EE': 'COM_PowerSpect_CMB-EE-full_R3.01.txt',
        'BB': 'COM_PowerSpect_CMB-BB-full_R3.01.txt',
    }

    COSMO_PARAMS_2018 = {
        'H0': 67.4,
        'H0_err': 0.5,
        'Omega_b_h2': 0.02237,
        'Omega_b_h2_err': 0.00015,
        'Omega_cdm_h2': 0.1200,
        'Omega_cdm_h2_err': 0.0012,
        'Omega_Lambda': 0.6847,
        'Omega_Lambda_err': 0.0073,
        'n_s': 0.9649,
        'n_s_err': 0.0042,
        'ln_A_s_1e10': 3.044,
        'ln_A_s_1e10_err': 0.014,
        'tau_reio': 0.0544,
        'tau_reio_err': 0.0073,
        'sigma_8': 0.8101,
        'sigma_8_err': 0.0060,
        'Omega_m': 0.3153,
        'Omega_m_err': 0.0073,
        'z_eq': 3402.0,
        'r_drag_Mpc': 147.09,
        'r_drag_err_Mpc': 0.26,
        'theta_star': 0.0104109,
        'theta_star_err': 0.0000030,
        'reference': 'Planck Collaboration (2020), A&A 641, A6',
        'doi': '10.1051/0004-6361/201833910',
    }

    def __init__(self, data_dir: str = './data/observational/planck/'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_spectrum(self, spectrum_type: str = 'TT', force: bool = False) -> str:
        if spectrum_type not in self.FILE_MAP:
            raise ValueError(f"Unknown spectrum type '{spectrum_type}'. Choose from {list(self.FILE_MAP.keys())}")

        filename = self.FILE_MAP[spectrum_type]
        filepath = os.path.join(self.data_dir, filename)

        if os.path.exists(filepath) and not force:
            print(f"File already exists: {filepath}")
            return filepath

        url = self.BASE_URL + filename
        print(f"Downloading {filename} from Planck Legacy Archive...")

        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded to {filepath}")
                return filepath
            else:
                print(f"Download failed (HTTP {response.status_code}). Using mock data.")
                return ""
        except Exception as e:
            print(f"Download error: {e}. Using mock data.")
            return ""

    def download_all_spectra(self, force: bool = False) -> Dict[str, str]:
        results = {}
        for spectrum_type in self.FILE_MAP:
            path = self.download_spectrum(spectrum_type, force=force)
            results[spectrum_type] = path
        return results

    def load_spectrum(self, spectrum_type: str = 'TT') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return load_planck_data(self.data_dir, spectrum_type)

    def get_cosmological_parameters(self) -> Dict:
        params_file = os.path.join(self.data_dir, 'planck_2018_cosmo_params.json')
        if os.path.exists(params_file):
            import json
            with open(params_file, 'r') as f:
                return json.load(f)
        return self.COSMO_PARAMS_2018

    def save_cosmological_parameters(self) -> str:
        import json
        params_file = os.path.join(self.data_dir, 'planck_2018_cosmo_params.json')
        with open(params_file, 'w') as f:
            json.dump(self.COSMO_PARAMS_2018, f, indent=4)
        print(f"Planck 2018 cosmological parameters saved to {params_file}")
        return params_file

    def compute_angular_power_spectrum_lcdm(self, ell: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
        if params is None:
            params = self.COSMO_PARAMS_2018

        A_s = np.exp(params['ln_A_s_1e10']) * 1.0e-10
        n_s = params['n_s']
        tau = params['tau_reio']
        ell_pivot = 50.0

        prefactor = ell * (ell + 1.0) / (2.0 * np.pi)
        C_ell_primordial = A_s * (ell / ell_pivot)**(n_s - 1.0) * np.exp(-2.0 * tau)

        for ell_peak, amp, width in [(220, 5700, 75), (540, 2600, 105), (810, 1550, 85), (1120, 1050, 90)]:
            C_ell_primordial += amp / 1.0e12 * np.exp(-0.5 * ((ell - ell_peak) / width)**2)

        silk = np.exp(-(ell / 1500.0)**2.15)
        D_ell = prefactor * C_ell_primordial * silk * 1.0e12

        return D_ell

    def residuals_eqst_gp_vs_planck(self, ell: np.ndarray, D_ell_planck: np.ndarray, Lambda_eff_0: float = 0.685) -> np.ndarray:
        D_ell_lcdm = self.compute_angular_power_spectrum_lcdm(ell)
        delta_Lambda = Lambda_eff_0 - self.COSMO_PARAMS_2018['Omega_Lambda']
        scaling = 1.0 + 0.5 * delta_Lambda * np.exp(-(ell / 500.0)**2)
        D_ell_eqst = D_ell_lcdm * scaling
        residuals = D_ell_planck - D_ell_eqst
        return residuals