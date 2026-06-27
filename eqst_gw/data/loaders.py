import numpy as np
import pandas as pd
import requests
import os
from typing import Tuple, Dict, Optional, List
from astropy.io import fits
import h5py
import json

def load_planck_data(data_dir: str = './data/observational/planck/',
                     spectrum_type: str = 'TT',
                     use_mock: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    filename_map = {
        'TT': 'COM_PowerSpect_CMB-TT-full_R3.01.txt',
        'TE': 'COM_PowerSpect_CMB-TE-full_R3.01.txt',
        'EE': 'COM_PowerSpect_CMB-EE-full_R3.01.txt',
        'BB': 'COM_PowerSpect_CMB-BB-full_R3.01.txt'
    }
    
    filepath = os.path.join(data_dir, filename_map.get(spectrum_type, filename_map['TT']))
    
    if os.path.exists(filepath) and not use_mock:
        data = np.loadtxt(filepath, skiprows=1)
        ell = data[:, 0]
        D_ell = data[:, 1]
        sigma_D_ell = data[:, 2]
        return ell, D_ell, sigma_D_ell
    else:
        return _generate_mock_planck_spectrum(spectrum_type)


def _generate_mock_planck_spectrum(spectrum_type: str = 'TT') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ell = np.arange(2, 2509, dtype=float)
    
    A_s = 2.1e-9
    n_s = 0.9649
    tau = 0.054
    H0 = 67.4
    Omega_m = 0.315
    
    if spectrum_type == 'TT':
        l_pivot = 200.0
        D_ell = (A_s * (ell / l_pivot)**(n_s - 1.0) * np.exp(-2.0 * tau) *
                 ell * (ell + 1.0) / (2.0 * np.pi) * 5.765e12)
        
        for l_peak, amplitude, width in [(200, 5700, 70), (540, 2650, 100), (810, 1600, 80), (1120, 1100, 90)]:
            D_ell += amplitude * np.exp(-0.5 * ((ell - l_peak) / width)**2)
        
        silk_damping = np.exp(-(ell / 1500.0)**2.1)
        D_ell *= silk_damping
        
        sigma = 0.008 * D_ell + 5.0
        
    elif spectrum_type == 'EE':
        D_ell = (0.1 * A_s * (ell / 200.0)**(n_s - 1.0) * np.exp(-2.0 * tau) *
                 ell * (ell + 1.0) / (2.0 * np.pi) * 5.765e12)
        
        for l_peak, amplitude, width in [(140, 35, 40), (400, 22, 60), (700, 18, 70)]:
            D_ell += amplitude * np.exp(-0.5 * ((ell - l_peak) / width)**2)
        
        silk_damping = np.exp(-(ell / 1600.0)**2.2)
        D_ell *= silk_damping
        
        sigma = 0.015 * D_ell + 0.5
        
    elif spectrum_type == 'TE':
        D_ell = (0.05 * A_s * (ell / 200.0)**(n_s - 1.0) * np.exp(-2.0 * tau) *
                 ell * (ell + 1.0) / (2.0 * np.pi) * 5.765e12)
        
        D_ell *= np.sin(np.pi * ell / 300.0)
        
        silk_damping = np.exp(-(ell / 1550.0)**2.1)
        D_ell *= silk_damping
        
        sigma = 0.02 * np.abs(D_ell) + 1.0
    
    else:
        D_ell = np.zeros_like(ell)
        sigma = np.ones_like(ell) * 0.01
    
    noise = np.random.normal(0, sigma)
    D_ell = D_ell + noise
    
    return ell, D_ell, sigma


def load_desi_bao(data_dir: str = './data/observational/desi/',
                  release: str = 'DR1',
                  use_mock: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    filepath = os.path.join(data_dir, f'DESI_BAO_2024_{release}.csv')
    cov_filepath = os.path.join(data_dir, 'covariance_matrix.npy')
    
    if os.path.exists(filepath) and not use_mock:
        data = pd.read_csv(filepath)
        z_eff = data['z_eff'].values
        DM_over_rd = data['DM_over_rd'].values
        DH_over_rd = data['DH_over_rd'].values
        
        if os.path.exists(cov_filepath):
            cov_matrix = np.load(cov_filepath)
        else:
            sigma_DM = data['sigma_DM'].values
            sigma_DH = data['sigma_DH'].values
            cov_matrix = np.diag(np.concatenate([sigma_DM**2, sigma_DH**2]))
        
        return z_eff, DM_over_rd, DH_over_rd, cov_matrix
    else:
        return _generate_mock_desi_bao()


def _generate_mock_desi_bao() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_eff = np.array([0.51, 0.71, 0.93, 1.32, 1.49, 2.33])
    
    rd_fid = 147.09
    
    DM_over_rd = np.array([13.62, 16.85, 19.77, 24.05, 26.07, 37.79])
    DH_over_rd = np.array([20.98, 19.33, 17.88, 13.82, 13.23, 8.52])
    
    sigma_DM = np.array([0.27, 0.34, 0.40, 0.48, 0.52, 0.76])
    sigma_DH = np.array([0.42, 0.51, 0.61, 0.72, 0.78, 1.14])
    
    corr_DM_DH = np.array([-0.45, -0.42, -0.40, -0.38, -0.37, -0.35])
    
    N = len(z_eff)
    cov_matrix = np.zeros((2 * N, 2 * N))
    
    for i in range(N):
        cov_matrix[i, i] = sigma_DM[i]**2
        cov_matrix[N + i, N + i] = sigma_DH[i]**2
        cov_matrix[i, N + i] = corr_DM_DH[i] * sigma_DM[i] * sigma_DH[i]
        cov_matrix[N + i, i] = cov_matrix[i, N + i]
    
    noise_DM = np.random.multivariate_normal(np.zeros(N), np.diag(sigma_DM**2))
    noise_DH = np.random.multivariate_normal(np.zeros(N), np.diag(sigma_DH**2))
    
    DM_over_rd = DM_over_rd + noise_DM
    DH_over_rd = DH_over_rd + noise_DH
    
    return z_eff, DM_over_rd, DH_over_rd, cov_matrix


def load_pantheon_sn(data_dir: str = './data/observational/pantheon_plus/',
                     use_mock: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    filepath = os.path.join(data_dir, 'Pantheon+SH0ES.dat')
    cov_filepath = os.path.join(data_dir, 'systematics_covariance.fits')
    
    if os.path.exists(filepath) and not use_mock:
        data = np.loadtxt(filepath, skiprows=1)
        z_SN = data[:, 0]
        mu_obs = data[:, 1]
        
        if os.path.exists(cov_filepath):
            with fits.open(cov_filepath) as hdul:
                cov_matrix = hdul[1].data
        else:
            sigma_mu = data[:, 2]
            cov_matrix = np.diag(sigma_mu**2)
        
        return z_SN, mu_obs, cov_matrix
    else:
        return _generate_mock_pantheon_sn()


def _generate_mock_pantheon_sn() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N_SN = 1701
    
    z_SN = np.sort(np.random.uniform(0.01, 2.3, N_SN))
    
    H0 = 73.04
    Omega_m = 0.315
    Omega_Lambda = 0.685
    c_km_s = 299792.458
    
    d_L = np.zeros(N_SN)
    
    for i, z in enumerate(z_SN):
        z_grid = np.linspace(0, z, 200)
        E_z = np.sqrt(Omega_m * (1.0 + z_grid)**3 + Omega_Lambda)
        integrand = 1.0 / E_z
        d_C = c_km_s / H0 * np.trapz(integrand, z_grid)
        d_L[i] = (1.0 + z) * d_C
    
    mu_theory = 5.0 * np.log10(d_L * 1.0e6) + 25.0
    
    sigma_stat = 0.12
    sigma_sys = 0.05
    sigma_mu = np.sqrt(sigma_stat**2 + sigma_sys**2) * np.ones(N_SN)
    
    mu_obs = mu_theory + np.random.normal(0, sigma_mu)
    
    cov_matrix = np.diag(sigma_mu**2)
    
    f_sys = 0.01
    for i in range(N_SN):
        for j in range(max(0, i - 5), min(N_SN, i + 5)):
            if i != j:
                cov_matrix[i, j] = f_sys * sigma_mu[i] * sigma_mu[j]
    
    return z_SN, mu_obs, cov_matrix


def load_jwst_galaxies(data_dir: str = './data/observational/jwst/',
                       field: str = 'CEERS',
                       z_min: float = 8.0,
                       z_max: float = 15.0,
                       use_mock: bool = False) -> Dict[str, np.ndarray]:
    
    filepath = os.path.join(data_dir, f'{field}_high_z_galaxies.csv')
    
    if os.path.exists(filepath) and not use_mock:
        data = pd.read_csv(filepath)
        
        mask = (data['z_phot'] >= z_min) & (data['z_phot'] <= z_max)
        data_filtered = data[mask]
        
        return {
            'z_phot': data_filtered['z_phot'].values,
            'z_phot_err': data_filtered['z_phot_err'].values,
            'M_stellar_solar': 10.0**data_filtered['log_M_stellar'].values,
            'SFR_solar_per_yr': data_filtered['SFR'].values,
            'RA_deg': data_filtered['RA'].values,
            'Dec_deg': data_filtered['Dec'].values,
            'UV_slope': data_filtered['beta_UV'].values
        }
    else:
        return _generate_mock_jwst_galaxies(z_min, z_max)


def _generate_mock_jwst_galaxies(z_min: float = 8.0, z_max: float = 15.0) -> Dict[str, np.ndarray]:
    N_gal = 200
    
    z_phot = np.random.uniform(z_min, z_max, N_gal)
    z_phot_err = np.random.uniform(0.2, 0.8, N_gal)
    
    log_M_star_mean = 8.5 - 0.1 * (z_phot - 10.0)
    log_M_star = log_M_star_mean + np.random.normal(0, 0.4, N_gal)
    M_stellar = 10.0**log_M_star
    
    log_SFR_mean = 0.7 * log_M_star - 5.5
    SFR = 10.0**(log_SFR_mean + np.random.normal(0, 0.3, N_gal))
    
    RA = np.random.uniform(214.6, 215.1, N_gal)
    Dec = np.random.uniform(52.8, 53.1, N_gal)
    
    beta_UV = np.random.normal(-2.1, 0.4, N_gal)
    
    return {
        'z_phot': z_phot,
        'z_phot_err': z_phot_err,
        'M_stellar_solar': M_stellar,
        'SFR_solar_per_yr': SFR,
        'RA_deg': RA,
        'Dec_deg': Dec,
        'UV_slope': beta_UV
    }


def load_ligo_gwosc_catalog(data_dir: str = './data/observational/ligo_gwosc/',
                             catalog: str = 'GWTC-3',
                             use_mock: bool = False) -> List[Dict]:
    
    filepath = os.path.join(data_dir, 'gwtc3_confident.json')
    
    if os.path.exists(filepath) and not use_mock:
        with open(filepath, 'r') as f:
            catalog_data = json.load(f)
        return catalog_data['events']
    else:
        return _generate_mock_gwosc_catalog()


def _generate_mock_gwosc_catalog() -> List[Dict]:
    N_events = 90
    
    events = []
    
    for i in range(N_events):
        event = {
            'name': f'GW{200000 + i * 100:06d}',
            'm1_source_solar': np.random.uniform(5.0, 80.0),
            'm2_source_solar': np.random.uniform(5.0, 50.0),
            'z_luminosity': np.random.uniform(0.01, 1.5),
            'chi_eff': np.random.uniform(-0.5, 0.9),
            'chirp_mass_solar': np.random.uniform(10.0, 50.0),
            'network_snr': np.random.uniform(8.0, 40.0),
            'luminosity_distance_Mpc': np.random.uniform(100.0, 5000.0),
            'far_per_year': np.random.exponential(1.0e-4),
            'event_type': np.random.choice(['BBH', 'BNS', 'NSBH'], p=[0.80, 0.12, 0.08])
        }
        events.append(event)
    
    events.sort(key=lambda x: x['z_luminosity'])
    
    return events


def load_nanograv_pta_data(data_dir: str = './data/observational/',
                           release: str = '15yr',
                           use_mock: bool = False) -> Dict[str, np.ndarray]:
    
    filepath = os.path.join(data_dir, f'nanograv_{release}_timing_residuals.h5')
    
    if os.path.exists(filepath) and not use_mock:
        with h5py.File(filepath, 'r') as hf:
            f_gw = hf['frequencies_Hz'][:]
            Omega_gw = hf['Omega_gw_h2'][:]
            sigma_Omega = hf['sigma_Omega_gw_h2'][:]
            pulsars = list(hf['pulsars'].keys())
        return {
            'f_gw_Hz': f_gw,
            'Omega_gw_h2': Omega_gw,
            'sigma_Omega_gw_h2': sigma_Omega,
            'pulsars': pulsars
        }
    else:
        return _generate_mock_pta_data()


def _generate_mock_pta_data() -> Dict[str, np.ndarray]:
    f_gw = np.logspace(-9, -7, 30)
    
    gamma_gw = 13.0 / 3.0
    A_gw = 2.4e-15
    
    Omega_gw = (2.0 * np.pi**2 / 3.0) * (A_gw * (f_gw / 1.0e-8)**(2.0 - gamma_gw))**2 * f_gw**2 / (100.0 / 3.086e22)**2
    
    sigma_Omega = 0.3 * Omega_gw
    
    Omega_gw = Omega_gw * (1.0 + np.random.normal(0, 0.2, len(f_gw)))
    
    pulsars = [f'PSR_J{1000 + i:04d}+{100 + i:04d}' for i in range(68)]
    
    return {
        'f_gw_Hz': f_gw,
        'Omega_gw_h2': Omega_gw,
        'sigma_Omega_gw_h2': sigma_Omega,
        'pulsars': pulsars
    }

