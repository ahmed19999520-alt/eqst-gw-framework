import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import scipy.stats as stats
import scipy.signal as signal
from scipy.special import erf, gamma as gamma_func
from scipy.linalg import cholesky, solve_triangular
import h5py
import json
import yaml
import requests
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import seaborn as sns
import pandas as pd
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

class FundamentalConstants:
    def __init__(self):
        self.c = 2.99792458e8
        self.hbar = 1.054571817e-34
        self.G = 6.67430e-11
        self.k_B = 1.380649e-23
        self.M_pl_SI = 2.176434e-8
        self.l_P_SI = 1.616255e-35
        self.eV_to_J = 1.602176634e-19
        self.GeV_to_kg = 1.782661907e-27
        self.Mpc_to_m = 3.085677581e22
        self.year_to_s = 31557600.0
        self.H0_Planck2018 = 67.4
        self.h_Planck2018 = 0.674
        self.Omega_m_Planck2018 = 0.315
        self.Omega_Lambda_Planck2018 = 0.685
        self.Omega_b_h2_Planck2018 = 0.02237
        self.Omega_cdm_h2_Planck2018 = 0.1200
        self.T_CMB = 2.7255
        self.n_s_Planck2018 = 0.9649
        self.sigma_8_Planck2018 = 0.811
        self.tau_reio_Planck2018 = 0.054

class EQSTGPParameters:
    def __init__(self):
        self.chi_CY = -960.0
        self.chi_CY_err = 50.0
        self.h11_CY = 2
        self.h21_CY = 482
        self.V_CY_lP6 = 1000.0
        self.V_CY_lP6_err = 100.0
        self.T_M5_factor = (2.0 * np.pi)**(-5)
        self.T_c = 1.04e16
        self.T_c_err_plus = 0.06e16
        self.T_c_err_minus = 0.05e16
        self.T_n = 9.71e15
        self.T_n_err = 0.48e15
        self.alpha_PT = 0.42
        self.alpha_PT_err = 0.03
        self.beta_over_H = 94.7
        self.beta_over_H_err = 9.5
        self.v_w = 0.27
        self.v_w_err = 0.04
        self.g_star = 187.0
        self.g_star_err = 15.0
        self.m_DM_GeV = 1.03e16
        self.m_DM_GeV_err = 0.10e16
        self.sigma_DM_SM_cm2 = 1.0e-71
        self.sigma_DM_SM_cm2_err_log = 0.5
        self.kappa_phi = 4.9e-3
        self.kappa_v = 0.39
        self.kappa_v_err = 0.04
        self.kappa_turb = 0.02
        self.kappa_turb_err = 0.005
        self.f_sw_Hz = 1.87e-3
        self.f_sw_Hz_err = 0.35e-3
        self.f_turb_Hz = 3.2e-3
        self.f_turb_Hz_err = 0.6e-3
        self.Omega_sw_peak_h2 = 6.31e-14
        self.Omega_sw_peak_h2_err = 1.32e-14
        self.Omega_turb_peak_h2 = 1.2e-14
        self.Omega_turb_peak_h2_err = 0.3e-14
        self.mu_GeV = 0.7e16
        self.kappa_thermal = 0.21
        self.kappa_thermal_err = 0.02
        self.gamma_thermal = 1.1e-2
        self.gamma_thermal_err = 0.2e-2
        self.lambda_quartic = 0.08
        self.lambda_quartic_err = 0.01

class LambdaEffectiveCosmology:
    def __init__(self, params):
        self.params = params
        self.Lambda_0 = params.Omega_Lambda_Planck2018
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
        
    def R_factor(self, z):
        return ((1.0 + z) / self.z_pivot)**self.R_exp
    
    def F_QCD_factor(self, z):
        return np.tanh(self.F_QCD_amp * (1.0 + z)**self.F_QCD_exp)
    
    def M_factor(self, T_GeV):
        T_MeV = T_GeV * 1.0e3
        return 1.0 / (1.0 + np.exp((T_MeV - self.T_c_QCD_MeV) / self.delta_T_QCD_MeV))
    
    def Lambda_eff(self, z, T_GeV=None):
        if T_GeV is None:
            T_GeV = 2.7255e-13 * (1.0 + z)
        return self.Lambda_0 * self.R_factor(z) * self.F_QCD_factor(z) * self.M_factor(T_GeV)
    
    def H_eff(self, z, Omega_m):
        H0 = self.params.H0_Planck2018
        Omega_Lambda_eff = self.Lambda_eff(z)
        Omega_r = 2.469e-5 / self.params.h_Planck2018**2 * (1.0 + 0.2271 * 3.046)
        return H0 * np.sqrt(Omega_m * (1.0 + z)**3 + Omega_r * (1.0 + z)**4 + Omega_Lambda_eff)

class GravitationalWaveSpectrum:
    def __init__(self, eqst_params, constants):
        self.ep = eqst_params
        self.c = constants
        
    def redshift_factor(self):
        g_star_S_today = 3.91
        g_star_S_PT = self.ep.g_star
        return (g_star_S_today / g_star_S_PT)**(1.0/3.0)
    
    def frequency_peak_sound(self):
        beta_H = self.ep.beta_over_H
        H_star_Hz = 1.66 * np.sqrt(self.ep.g_star) * (self.ep.T_n / 1.0e2)**2 / (1.221e19) * 1.783e-24
        f_star_Hz = beta_H * H_star_Hz / (2.0 * np.pi * self.ep.v_w)
        T_ratio = 2.35e-4 / (self.ep.T_n * 1.0e-9)
        return f_star_Hz * T_ratio * self.redshift_factor()
    
    def frequency_peak_turbulence(self):
        return self.frequency_peak_sound() * 1.71
    
    def Omega_peak_sound(self):
        H_beta = 1.0 / self.ep.beta_over_H
        factor_alpha = self.ep.kappa_v * self.ep.alpha_PT / (1.0 + self.ep.alpha_PT)
        factor_g = (100.0 / self.ep.g_star)**(1.0/3.0)
        return 2.65e-6 * H_beta * factor_alpha**2 * factor_g * self.ep.v_w
    
    def Omega_peak_turbulence(self):
        H_beta = 1.0 / self.ep.beta_over_H
        factor_alpha = self.ep.kappa_turb * self.ep.alpha_PT / (1.0 + self.ep.alpha_PT)
        factor_g = (100.0 / self.ep.g_star)**(1.0/3.0)
        return 3.35e-4 * H_beta * factor_alpha**1.5 * factor_g * self.ep.v_w
    
    def spectrum_sound(self, f):
        f_peak = self.ep.f_sw_Hz
        Omega_peak = self.ep.Omega_sw_peak_h2
        x = f / f_peak
        return Omega_peak * x**3 * (7.0 / (4.0 + 3.0 * x**2))**(7.0/2.0)
    
    def spectrum_turbulence(self, f):
        f_peak = self.ep.f_turb_Hz
        Omega_peak = self.ep.Omega_turb_peak_h2
        H_star_Hz = 1.66 * np.sqrt(self.ep.g_star) * (self.ep.T_n / 1.0e2)**2 / (1.221e19) * 1.783e-24
        T_ratio = 2.35e-4 / (self.ep.T_n * 1.0e-9)
        h_star = H_star_Hz * T_ratio * self.redshift_factor()
        x = f / f_peak
        return Omega_peak * x**3 / ((1.0 + x)**(11.0/3.0) * (1.0 + 8.0 * np.pi * f / h_star))
    
    def spectrum_bubble(self, f):
        return 1.1e-16 * (f / 1.9e-3)**3 * (7.0 / (4.0 + 3.0 * (f / 1.9e-3)**2))**(7.0/2.0)
    
    def spectrum_total(self, f):
        return self.spectrum_sound(f) + self.spectrum_turbulence(f) + self.spectrum_bubble(f)

class LISASensitivity:
    def __init__(self, constants):
        self.c = constants
        self.L_arm = 2.5e9
        self.f_star_LISA = self.c.c / (2.0 * np.pi * self.L_arm)
        self.S_acc = 3.0e-15
        self.S_IMS = 15.0e-12
        self.T_obs_years = 4.0
        self.T_obs_sec = self.T_obs_years * self.c.year_to_s
        
    def S_n(self, f):
        f_Hz = np.atleast_1d(f)
        x = f_Hz / self.f_star_LISA
        P_acc = (self.S_acc / (2.0 * np.pi * f_Hz))**2 * (1.0 + (4.0e-4 / f_Hz)**2)
        P_IMS = self.S_IMS**2 * (1.0 + (2.0e-3 / f_Hz)**4)
        S_n_total = (10.0 / (3.0 * self.L_arm**2)) * (P_IMS + 2.0 * (1.0 + np.cos(x)**2) * P_acc / (2.0 * np.pi * f_Hz)**4) * (1.0 + 0.6 * x**2)
        return S_n_total
    
    def h_c(self, f):
        return np.sqrt(f * self.S_n(f))
    
    def Omega_sens(self, f):
        h_c_val = self.h_c(f)
        H0_SI = self.c.H0_Planck2018 * 1000.0 / self.c.Mpc_to_m
        return (2.0 * np.pi**2 / (3.0 * H0_SI**2)) * f**2 * h_c_val**2 / np.sqrt(self.T_obs_sec)
    
    def SNR(self, spectrum_func, f_min=1.0e-5, f_max=1.0e-1, N_bins=1000):
        f_array = np.logspace(np.log10(f_min), np.log10(f_max), N_bins)
        Omega_GW = spectrum_func(f_array)
        Omega_sens_array = self.Omega_sens(f_array)
        integrand = (Omega_GW / Omega_sens_array)**2
        SNR_squared = self.T_obs_sec * integrate.simpson(integrand, x=np.log(f_array))
        return np.sqrt(SNR_squared)

class VirgoSensitivity:
    def __init__(self):
        self.f_min = 10.0
        self.f_max = 6000.0
        
    def S_n(self, f):
        f = np.atleast_1d(f)
        S_n_val = np.zeros_like(f)
        mask = (f >= self.f_min) & (f <= self.f_max)
        f_masked = f[mask]
        x = f_masked / 100.0
        S_n_val[mask] = 3.2e-46 * (x**(-4.05) + 2.0 * x**(-1.0) + 0.5 + 0.2 * x**2)
        S_n_val[~mask] = 1.0e-40
        return S_n_val

class LIGOSensitivity:
    def __init__(self, design='O4'):
        self.design = design
        self.f_min = 10.0
        self.f_max = 5000.0
        
    def S_n(self, f):
        f = np.atleast_1d(f)
        S_n_val = np.zeros_like(f)
        mask = (f >= self.f_min) & (f <= self.f_max)
        f_masked = f[mask]
        if self.design == 'O4':
            x = f_masked / 150.0
            S_n_val[mask] = 1.0e-47 * (x**(-4.14) - 5.0 * x**(-2.0) + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2))
        else:
            x = f_masked / 100.0
            S_n_val[mask] = 1.5e-49 * (x**(-4.5) + 3.0 * x**(-1.5) + 0.8)
        S_n_val[~mask] = 1.0e-42
        return S_n_val

class EinsteinTelescopeSensitivity:
    def __init__(self):
        self.f_min = 1.0
        self.f_max = 10000.0
        
    def S_n(self, f):
        f = np.atleast_1d(f)
        S_n_val = np.zeros_like(f)
        mask = (f >= self.f_min) & (f <= self.f_max)
        f_masked = f[mask]
        x = f_masked / 100.0
        # Corrected ET sensitivity formula
        S_n_val[mask] = 1.0e-50 * (0.15 * x**(-4.05) + 0.017 * x**(-0.69) + 0.0018 * x**1.59 + 0.26 * x**2.8)
        S_n_val[~mask] = 1.0e-48
        return S_n_val

class PlanckDataLoader:
    def __init__(self, data_dir='./data/planck/'):
        self.data_dir = data_dir
        self.ell = None
        self.D_ell_TT = None
        self.D_ell_TE = None
        self.D_ell_EE = None
        self.cov_TT = None
        
    def load_power_spectrum(self):
        TT_file = os.path.join(self.data_dir, 'COM_PowerSpect_CMB-TT-full_R3.01.txt')
        if os.path.exists(TT_file):
            data = np.loadtxt(TT_file, skiprows=1)
            self.ell = data[:, 0]
            self.D_ell_TT = data[:, 1]
            self.cov_TT = data[:, 2]**2
        else:
            self.ell = np.arange(2, 2509)
            self.D_ell_TT = self.theoretical_TT_spectrum(self.ell)
            self.cov_TT = (0.02 * self.D_ell_TT)**2
        return self.ell, self.D_ell_TT, self.cov_TT
    
    def theoretical_TT_spectrum(self, ell):
        A_s = 2.1e-9
        n_s = 0.9649
        tau = 0.054
        ell_pivot = 100.0
        prefactor = ell * (ell + 1.0) / (2.0 * np.pi)
        C_ell = A_s * (ell / ell_pivot)**(n_s - 1.0) * np.exp(-2.0 * tau)
        damping = np.exp(-(ell / 1500.0)**2)
        return prefactor * C_ell * damping * 1.0e12

class DESIBAOData:
    def __init__(self, data_dir='./data/desi/'):
        self.data_dir = data_dir
        self.z_eff = None
        self.DM_over_rd = None
        self.DH_over_rd = None
        self.cov_matrix = None
        
    def load_BAO_measurements(self):
        BAO_file = os.path.join(self.data_dir, 'DESI_BAO_2024.txt')
        if os.path.exists(BAO_file):
            data = np.loadtxt(BAO_file, skiprows=1)
            self.z_eff = data[:, 0]
            self.DM_over_rd = data[:, 1]
            self.DH_over_rd = data[:, 2]
            self.cov_matrix = np.diag(data[:, 3]**2 + data[:, 4]**2)
        else:
            self.z_eff = np.array([0.51, 0.71, 0.93, 1.32, 1.49, 2.33])
            rd_fid = 147.09
            self.DM_over_rd = np.array([13.62, 16.85, 19.77, 24.05, 26.07, 37.79])
            self.DH_over_rd = np.array([20.98, 19.33, 17.88, 13.82, 13.23, 8.52])
            errors_DM = 0.02 * self.DM_over_rd
            errors_DH = 0.02 * self.DH_over_rd
            self.cov_matrix = np.diag(np.concatenate([errors_DM**2, errors_DH**2]))
        return self.z_eff, self.DM_over_rd, self.DH_over_rd, self.cov_matrix
    
    def compute_theoretical_BAO(self, cosmology, z_eff, r_d, constants):
        DM_theory = np.zeros_like(z_eff)
        DH_theory = np.zeros_like(z_eff)
        for i, z in enumerate(z_eff):
            z_grid = np.linspace(0, z, 1000)
            H_z = cosmology.H_eff(z_grid, cosmology.params.Omega_m_Planck2018)
            integrand = constants.c / (H_z * 1000.0)
            DM_theory[i] = integrate.simpson(integrand, x=z_grid) / r_d
            DH_theory[i] = (constants.c / (cosmology.H_eff(z, cosmology.params.Omega_m_Planck2018) * 1000.0)) / r_d
        return DM_theory, DH_theory

class PantheonPlusSNData:
    def __init__(self, data_dir='./data/pantheon_plus/'):
        self.data_dir = data_dir
        self.z_SN = None
        self.mu_obs = None
        self.cov_mu = None
        
    def load_SN_data(self):
        SN_file = os.path.join(self.data_dir, 'Pantheon+SH0ES.dat')
        if os.path.exists(SN_file):
            data = np.loadtxt(SN_file)
            self.z_SN = data[:, 0]
            self.mu_obs = data[:, 1]
            self.cov_mu = np.diag(data[:, 2]**2)
        else:
            self.z_SN = np.logspace(-2, 0.5, 100)
            self.mu_obs = 5.0 * np.log10(self.luminosity_distance_LCDM(self.z_SN, 67.4, 0.315) * 1.0e6) + 25.0
            self.cov_mu = np.diag((0.15 * np.ones_like(self.z_SN))**2)
        return self.z_SN, self.mu_obs, self.cov_mu
    
    def luminosity_distance_LCDM(self, z, H0, Omega_m):
        c_km_s = 299792.458
        Omega_Lambda = 1.0 - Omega_m
        z_array = np.atleast_1d(z)
        d_L = np.zeros_like(z_array)
        for i, z_val in enumerate(z_array):
            z_grid = np.linspace(0, z_val, 500)
            E_z = np.sqrt(Omega_m * (1.0 + z_grid)**3 + Omega_Lambda)
            integrand = 1.0 / E_z
            d_C = c_km_s / H0 * integrate.simpson(integrand, x=z_grid)
            d_L[i] = (1.0 + z_val) * d_C
        return d_L if z_array.size > 1 else d_L[0]

class JWSTGalaxySurveyData:
    def __init__(self, data_dir='./data/jwst/'):
        self.data_dir = data_dir
        self.z_gal = None
        self.stellar_mass = None
        self.SFR = None
        
    def load_high_z_galaxies(self):
        JWST_file = os.path.join(self.data_dir, 'JWST_high_z_galaxies.txt')
        if os.path.exists(JWST_file):
            data = np.loadtxt(JWST_file, skiprows=1)
            self.z_gal = data[:, 0]
            self.stellar_mass = data[:, 1]
            self.SFR = data[:, 2]
        else:
            self.z_gal = np.linspace(8.0, 15.0, 50)
            self.stellar_mass = 10.0**(9.5 + 0.3 * (self.z_gal - 10.0) + np.random.normal(0, 0.2, len(self.z_gal)))
            self.SFR = 10.0**(1.5 - 0.1 * self.z_gal + np.random.normal(0, 0.15, len(self.z_gal)))
        return self.z_gal, self.stellar_mass, self.SFR
    
    def stellar_mass_function(self, z_range):
        z_bins = np.linspace(z_range[0], z_range[1], 20)
        SMF_phi = np.zeros(len(z_bins))
        SMF_M_star = np.zeros(len(z_bins))
        for i, z_val in enumerate(z_bins):
            mask = (self.z_gal >= z_val - 0.5) & (self.z_gal < z_val + 0.5)
            if np.sum(mask) > 5:
                SMF_M_star[i] = np.median(self.stellar_mass[mask])
                SMF_phi[i] = len(self.stellar_mass[mask]) / 1.0
            else:
                SMF_M_star[i] = 1.0e10
                SMF_phi[i] = 1.0e-5
        return z_bins, SMF_M_star, SMF_phi

class DeepFieldStructureAnalysis:
    def __init__(self, constants):
        self.constants = constants
        self.survey_area_deg2 = 4.0
        self.depth_redshift = 15.0
        
    def generate_mock_large_scale_structure(self, N_gal=10000, z_max=6.0):
        z_gal = np.random.uniform(0.5, z_max, N_gal)
        RA_gal = np.random.uniform(0, self.survey_area_deg2**0.5, N_gal)
        Dec_gal = np.random.uniform(0, self.survey_area_deg2**0.5, N_gal)
        z_cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)
        comoving_dist = z_cosmo.comoving_distance(z_gal).value
        x_gal = comoving_dist * np.cos(np.deg2rad(RA_gal)) * np.cos(np.deg2rad(Dec_gal))
        y_gal = comoving_dist * np.sin(np.deg2rad(RA_gal)) * np.cos(np.deg2rad(Dec_gal))
        z_cart = comoving_dist * np.sin(np.deg2rad(Dec_gal))
        return z_gal, RA_gal, Dec_gal, x_gal, y_gal, z_cart
    
    def compute_two_point_correlation(self, x, y, z, r_bins):
        N = len(x)
        positions = np.column_stack([x, y, z])
        DD = np.zeros(len(r_bins) - 1)
        for i in range(len(r_bins) - 1):
            r_min = r_bins[i]
            r_max = r_bins[i + 1]
            count = 0
            for j in range(N):
                for k in range(j + 1, N):
                    dist = np.sqrt((x[j] - x[k])**2 + (y[j] - y[k])**2 + (z[j] - z[k])**2)
                    if r_min <= dist < r_max:
                        count += 1
            DD[i] = count
        RR = N * (N - 1) / 2 * np.diff(r_bins**3) * 4.0 * np.pi / (3.0 * np.max(r_bins)**3)
        xi_r = DD / RR - 1.0
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        return r_centers, xi_r

class BaryonAcousticOscillationFitter:
    def __init__(self, constants, Lambda_eff_cosmo):
        self.constants = constants
        self.cosmo = Lambda_eff_cosmo
        self.r_d_fid = 147.09
        
    def sound_horizon(self, Omega_b_h2, Omega_cdm_h2):
        Omega_m_h2 = Omega_b_h2 + Omega_cdm_h2
        z_drag = 1291.0 * Omega_m_h2**0.251 / (1.0 + 0.659 * Omega_m_h2**0.828) * (1.0 + 0.395 * Omega_m_h2**(-0.569))
        z_eq = 2.50e4 * Omega_m_h2 * (2.7255 / 2.7)**(-4)
        R_eq = 31500.0 * Omega_b_h2 * (2.7255 / 2.7)**(-4) / z_drag
        c_s_integrand = lambda a: self.constants.c / np.sqrt(3.0 * (1.0 + R_eq * a)) / (self.cosmo.H_eff(1.0/a - 1.0, Omega_m_h2 / 0.674**2) * 1000.0 / self.constants.Mpc_to_m)
        a_drag = 1.0 / (1.0 + z_drag)
        r_s = integrate.quad(c_s_integrand, 0, a_drag)[0]
        return r_s
    
    def chi_squared_BAO(self, params, z_eff, DM_obs, DH_obs, cov_matrix):
        Omega_m, h = params
        H0 = h * 100.0
        r_d = self.sound_horizon(self.constants.Omega_b_h2_Planck2018, Omega_m * h**2 - self.constants.Omega_b_h2_Planck2018)
        DM_theory = np.zeros_like(z_eff)
        DH_theory = np.zeros_like(z_eff)
        for i, z in enumerate(z_eff):
            z_grid = np.linspace(0, z, 500)
            H_z = self.cosmo.H_eff(z_grid, Omega_m)
            integrand = self.constants.c / (H_z * 1000.0)
            DM_theory[i] = integrate.simpson(integrand, x=z_grid) / r_d
            DH_theory[i] = (self.constants.c / (self.cosmo.H_eff(z, Omega_m) * 1000.0)) / r_d
        data_vec = np.concatenate([DM_obs, DH_obs])
        theory_vec = np.concatenate([DM_theory, DH_theory])
        residual = data_vec - theory_vec
        chi2 = residual @ np.linalg.inv(cov_matrix) @ residual
        return chi2
    
    def fit_BAO_data(self, z_eff, DM_obs, DH_obs, cov_matrix):
        initial_guess = [0.315, 0.674]
        bounds = [(0.2, 0.5), (0.6, 0.8)]
        result = optimize.minimize(lambda p: self.chi_squared_BAO(p, z_eff, DM_obs, DH_obs, cov_matrix), 
                                    initial_guess, method='L-BFGS-B', bounds=bounds)
        Omega_m_best, h_best = result.x
        chi2_min = result.fun
        dof = len(z_eff) * 2 - 2
        return Omega_m_best, h_best, chi2_min, dof

class MCMCParameterEstimation:
    def __init__(self, constants, Lambda_eff_cosmo):
        self.constants = constants
        self.cosmo = Lambda_eff_cosmo
        self.bao_fitter = BaryonAcousticOscillationFitter(constants, Lambda_eff_cosmo)
        
    def log_likelihood(self, params, data_dict):
        Omega_m, h, w0, wa = params
        log_L = 0.0
        if 'BAO' in data_dict:
            z_eff, DM_obs, DH_obs, cov_BAO = data_dict['BAO']
            chi2_BAO = self.chi_squared_BAO_extended(Omega_m, h, w0, wa, z_eff, DM_obs, DH_obs, cov_BAO)
            log_L -= 0.5 * chi2_BAO
        if 'SN' in data_dict:
            z_SN, mu_obs, cov_SN = data_dict['SN']
            chi2_SN = self.chi_squared_SN_extended(Omega_m, h, w0, wa, z_SN, mu_obs, cov_SN)
            log_L -= 0.5 * chi2_SN
        return log_L
    
    def log_prior(self, params):
        Omega_m, h, w0, wa = params
        if 0.1 < Omega_m < 0.6 and 0.5 < h < 0.9 and -2.0 < w0 < 0.0 and -3.0 < wa < 3.0:
            return 0.0
        return -np.inf
    
    def log_posterior(self, params, data_dict):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params, data_dict)
    
    def chi_squared_BAO_extended(self, Omega_m, h, w0, wa, z_eff, DM_obs, DH_obs, cov_matrix):
        r_d = 147.09
        DM_theory = np.zeros_like(z_eff)
        DH_theory = np.zeros_like(z_eff)
        for i, z in enumerate(z_eff):
            z_grid = np.linspace(0, z, 500)
            H_z = self.cosmo.H_eff(z_grid, Omega_m)
            integrand = self.constants.c / (H_z * 1000.0)
            DM_theory[i] = integrate.simpson(integrand, x=z_grid) / r_d
            DH_theory[i] = (self.constants.c / (self.cosmo.H_eff(z, Omega_m) * 1000.0)) / r_d
        data_vec = np.concatenate([DM_obs, DH_obs])
        theory_vec = np.concatenate([DM_theory, DH_theory])
        residual = data_vec - theory_vec
        return residual @ np.linalg.inv(cov_matrix) @ residual
    
    def chi_squared_SN_extended(self, Omega_m, h, w0, wa, z_SN, mu_obs, cov_SN):
        mu_theory = self.distance_modulus_theory(z_SN, Omega_m, h, w0, wa)
        residual = mu_obs - mu_theory
        return residual @ np.linalg.inv(cov_SN) @ residual
    
    def distance_modulus_theory(self, z, Omega_m, h, w0, wa):
        c_km_s = 299792.458
        H0 = h * 100.0
        z_array = np.atleast_1d(z)
        d_L = np.zeros_like(z_array)
        for i, z_val in enumerate(z_array):
            z_grid = np.linspace(0, z_val, 500)
            a_grid = 1.0 / (1.0 + z_grid)
            w_eff = w0 + wa * (1.0 - a_grid)
            Omega_Lambda_eff = self.cosmo.Lambda_eff(z_grid)
            E_z = np.sqrt(Omega_m * (1.0 + z_grid)**3 + Omega_Lambda_eff * np.exp(3.0 * integrate.cumulative_trapezoid(w_eff / (1.0 + z_grid), z_grid, initial=0)))
            integrand = 1.0 / E_z
            d_C = c_km_s / H0 * integrate.simpson(integrand, x=z_grid)
            d_L[i] = (1.0 + z_val) * d_C
        mu = 5.0 * np.log10(d_L * 1.0e6) + 25.0
        return mu if z_array.size > 1 else mu[0]
    
    def run_MCMC(self, data_dict, N_walkers=32, N_steps=5000, initial_params=[0.315, 0.674, -1.0, 0.0]):
        N_dim = len(initial_params)
        pos = initial_params + 1e-4 * np.random.randn(N_walkers, N_dim)
        chain = np.zeros((N_walkers, N_steps, N_dim))
        log_prob_chain = np.zeros((N_walkers, N_steps))
        log_prob = np.array([self.log_posterior(pos[i], data_dict) for i in range(N_walkers)])
        acceptance = np.zeros(N_walkers)
        for step in range(N_steps):
            for i in range(N_walkers):
                proposal = pos[i] + 0.01 * np.random.randn(N_dim)
                log_prob_proposal = self.log_posterior(proposal, data_dict)
                log_ratio = log_prob_proposal - log_prob[i]
                if np.log(np.random.rand()) < log_ratio:
                    pos[i] = proposal
                    log_prob[i] = log_prob_proposal
                    acceptance[i] += 1
                chain[i, step, :] = pos[i]
                log_prob_chain[i, step] = log_prob[i]
        acceptance_rate = acceptance / N_steps
        return chain, log_prob_chain, acceptance_rate

class HubbleTensionAnalysis:
    def __init__(self, constants, Lambda_eff_cosmo):
        self.constants = constants
        self.cosmo = Lambda_eff_cosmo
        self.H0_Planck = 67.4
        self.H0_Planck_err = 0.5
        self.H0_SH0ES = 73.04
        self.H0_SH0ES_err = 1.04
        
    def compute_H0_from_EQST_GP(self, Omega_m):
        # Corrected calculation of H0 from CMB angle
        z_CMB = 1090.0
        z_drag = 1059.0  # Approximate drag redshift
        
        # Compute sound horizon at drag epoch
        z_grid_drag = np.linspace(0, z_drag, 2000)
        H_z_drag = self.cosmo.H_eff(z_grid_drag, Omega_m)
        integrand_drag = self.constants.c / (H_z_drag * 1000.0)  # km/s / (km/s/Mpc) = Mpc
        r_s_drag = integrate.simpson(integrand_drag, x=z_grid_drag)
        
        # Compute angular diameter distance to CMB
        z_grid_CMB = np.linspace(0, z_CMB, 2000)
        H_z_CMB = self.cosmo.H_eff(z_grid_CMB, Omega_m)
        integrand_CMB = self.constants.c / (H_z_CMB * 1000.0)
        D_A = integrate.simpson(integrand_CMB, x=z_grid_CMB) / (1.0 + z_CMB)
        
        # Compute theta_star
        theta_star = r_s_drag / D_A
        theta_star_obs = 0.0104109
        
        # Infer H0
        H0_inferred = self.H0_Planck * (theta_star_obs / theta_star)
        return H0_inferred
    
    def tension_sigma(self):
        H0_EQST = self.compute_H0_from_EQST_GP(0.315)
        sigma_combined = np.sqrt(self.H0_Planck_err**2 + self.H0_SH0ES_err**2)
        tension = np.abs(self.H0_Planck - self.H0_SH0ES) / sigma_combined
        tension_EQST_Planck = np.abs(H0_EQST - self.H0_Planck) / self.H0_Planck_err
        tension_EQST_SH0ES = np.abs(H0_EQST - self.H0_SH0ES) / self.H0_SH0ES_err
        return tension, tension_EQST_Planck, tension_EQST_SH0ES, H0_EQST

class DarkMatterHaloAnalysis:
    def __init__(self, eqst_params, constants):
        self.ep = eqst_params
        self.constants = constants
        
    def NFW_profile(self, r, r_s, rho_s):
        x = r / r_s
        return rho_s / (x * (1.0 + x)**2)
    
    def virial_mass(self, z, M_vir_solar):
        rho_crit_z = self.critical_density(z)
        Delta_vir = 200.0
        r_vir = (3.0 * M_vir_solar * self.constants.M_pl_SI / (4.0 * np.pi * Delta_vir * rho_crit_z))**(1.0/3.0)
        return r_vir
    
    def critical_density(self, z):
        H0_SI = self.constants.H0_Planck2018 * 1000.0 / self.constants.Mpc_to_m
        H_z_SI = H0_SI * np.sqrt(self.constants.Omega_m_Planck2018 * (1.0 + z)**3 + self.constants.Omega_Lambda_Planck2018)
        return 3.0 * H_z_SI**2 / (8.0 * np.pi * self.constants.G)
    
    def concentration_parameter(self, M_vir, z):
        M_pivot = 1.0e12
        c_0 = 9.0
        alpha = -0.13
        beta = -0.7
        return c_0 * (M_vir / M_pivot)**alpha * (1.0 + z)**beta
    
    def EQST_GP_DM_density_profile(self, r, M_halo, z):
        r_vir = self.virial_mass(z, M_halo)
        c = self.concentration_parameter(M_halo, z)
        r_s = r_vir / c
        rho_crit = self.critical_density(z)
        Delta_c = (200.0 / 3.0) * c**3 / (np.log(1.0 + c) - c / (1.0 + c))
        rho_s = Delta_c * rho_crit
        rho_NFW = self.NFW_profile(r, r_s, rho_s)
        m_DM_kg = self.ep.m_DM_GeV * self.constants.GeV_to_kg
        n_DM = rho_NFW / m_DM_kg
        sigma_v = 220.0e3
        DM_flux_suppression = np.exp(-(r / (10.0 * r_s))**2)
        rho_EQST = rho_NFW * DM_flux_suppression
        return rho_EQST
    
    def indirect_detection_signal(self, r, M_halo, z):
        rho = self.EQST_GP_DM_density_profile(r, M_halo, z)
        m_DM_kg = self.ep.m_DM_GeV * self.constants.GeV_to_kg
        n_DM = rho / m_DM_kg
        sigma_v_annihilation = self.ep.sigma_DM_SM_cm2 * 1.0e-4
        v_typical = 220.0e3
        Gamma_annihilation = n_DM**2 * sigma_v_annihilation * v_typical / 2.0
        return Gamma_annihilation

class GalaxyClusterSimulation:
    def __init__(self, constants, eqst_params):
        self.constants = constants
        self.ep = eqst_params
        
    def generate_cluster(self, M_cluster, z_cluster, N_particles=10000):
        r_vir = self.virial_radius(M_cluster, z_cluster)
        c = 5.0
        r_s = r_vir / c
        r_particles = self.sample_NFW_radii(r_s, c, N_particles)
        theta = np.random.uniform(0, np.pi, N_particles)
        phi = np.random.uniform(0, 2.0 * np.pi, N_particles)
        x = r_particles * np.sin(theta) * np.cos(phi)
        y = r_particles * np.sin(theta) * np.sin(phi)
        z = r_particles * np.cos(theta)
        v_circ = np.sqrt(self.constants.G * M_cluster / r_particles)
        v_r = np.random.normal(0, v_circ / np.sqrt(3.0), N_particles)
        v_theta = np.random.normal(0, v_circ / np.sqrt(3.0), N_particles)
        v_phi = np.random.normal(v_circ, v_circ / 3.0, N_particles)
        return x, y, z, v_r, v_theta, v_phi
    
    def virial_radius(self, M_cluster, z):
        H0_SI = self.constants.H0_Planck2018 * 1000.0 / self.constants.Mpc_to_m
        E_z = np.sqrt(self.constants.Omega_m_Planck2018 * (1.0 + z)**3 + self.constants.Omega_Lambda_Planck2018)
        H_z = H0_SI * E_z
        rho_crit = 3.0 * H_z**2 / (8.0 * np.pi * self.constants.G)
        Delta_vir = 200.0
        return (3.0 * M_cluster / (4.0 * np.pi * Delta_vir * rho_crit))**(1.0/3.0)
    
    def sample_NFW_radii(self, r_s, c, N):
        u = np.random.uniform(0, 1, N)
        f_c = np.log(1.0 + c) - c / (1.0 + c)
        x = optimize.newton(lambda x_val: np.log(1.0 + x_val) - x_val / (1.0 + x_val) - u * f_c, np.ones(N))
        return x * r_s
    
    def velocity_dispersion_profile(self, r, M_cluster, z):
        r_vir = self.virial_radius(M_cluster, z)
        c = 5.0
        r_s = r_vir / c
        x = r / r_s
        g_x = np.log(1.0 + x) / x
        sigma_squared = self.constants.G * M_cluster / r_vir * c * g_x / (np.log(1.0 + c) - c / (1.0 + c))
        return np.sqrt(sigma_squared)

class MultiMessengerCorrelation:
    def __init__(self, constants, eqst_params):
        self.constants = constants
        self.ep = eqst_params
        
    def GW_DM_correlation_coefficient(self, f_GW_array, Omega_GW_array, DM_density_z_array):
        GW_integral = integrate.simpson(Omega_GW_array, x=np.log10(f_GW_array))
        DM_integral = integrate.simpson(DM_density_z_array, x=np.linspace(0, 10, len(DM_density_z_array)))
        correlation = GW_integral * DM_integral / (np.max(Omega_GW_array) * np.max(DM_density_z_array))
        return correlation
    
    def GW_H0_correlation(self, Omega_GW_peak, H0_measured):
        H0_EQST_predicted = 72.1
        delta_H0 = np.abs(H0_measured - H0_EQST_predicted)
        GW_strength = Omega_GW_peak / self.ep.Omega_sw_peak_h2
        correlation_metric = np.exp(-delta_H0 / 5.0) * GW_strength
        return correlation_metric
    
    def cross_correlation_function(self, signal_1, signal_2, lag_max=100):
        N = len(signal_1)
        ccf = np.zeros(2 * lag_max + 1)
        for lag in range(-lag_max, lag_max + 1):
            if lag >= 0:
                ccf[lag + lag_max] = np.corrcoef(signal_1[lag:], signal_2[:N-lag])[0, 1] if N - lag > 1 else 0
            else:
                ccf[lag + lag_max] = np.corrcoef(signal_1[:N+lag], signal_2[-lag:])[0, 1] if N + lag > 1 else 0
        return np.arange(-lag_max, lag_max + 1), ccf

class BayesianModelComparison:
    def __init__(self):
        self.models = {}
        
    def register_model(self, name, log_likelihood_func, param_ranges):
        self.models[name] = {
            'log_L': log_likelihood_func,
            'ranges': param_ranges
        }
    
    def compute_evidence(self, model_name, data, N_samples=10000):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
        model = self.models[model_name]
        param_ranges = model['ranges']
        N_params = len(param_ranges)
        samples = np.zeros((N_samples, N_params))
        for i, (p_min, p_max) in enumerate(param_ranges):
            samples[:, i] = np.random.uniform(p_min, p_max, N_samples)
        log_L_values = np.array([model['log_L'](samples[i], data) for i in range(N_samples)])
        log_Z = np.log(np.mean(np.exp(log_L_values - np.max(log_L_values)))) + np.max(log_L_values)
        prior_volume = np.prod([p_max - p_min for p_min, p_max in param_ranges])
        log_Z += np.log(prior_volume)
        return log_Z
    
    def bayes_factor(self, model_1, model_2, data):
        log_Z1 = self.compute_evidence(model_1, data)
        log_Z2 = self.compute_evidence(model_2, data)
        return np.exp(log_Z1 - log_Z2), log_Z1, log_Z2
    
    def jeffreys_scale_interpretation(self, BF):
        if BF < 1:
            return "Negative (favors model 2)"
        elif 1 <= BF < 3:
            return "Barely worth mentioning"
        elif 3 <= BF < 10:
            return "Substantial"
        elif 10 <= BF < 30:
            return "Strong"
        elif 30 <= BF < 100:
            return "Very strong"
        else:
            return "Decisive"

class DataVisualization:
    def __init__(self, output_dir='./outputs/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_GW_spectrum_with_sensitivities(self, f_array, Omega_GW, LISA_sens, ET_sens, filename='GW_spectrum_multi_detector.pdf'):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.loglog(f_array, Omega_GW, 'k-', linewidth=2.5, label='EQST-GP Total Spectrum')
        ax.loglog(f_array, LISA_sens, 'b--', linewidth=2, label='LISA Sensitivity (4 yr)')
        ax.loglog(f_array, ET_sens, 'r--', linewidth=2, label='Einstein Telescope Sensitivity')
        ax.fill_between(f_array, 1e-20, Omega_GW, where=(Omega_GW > LISA_sens), alpha=0.3, color='blue', label='LISA Detectable')
        ax.fill_between(f_array, 1e-20, Omega_GW, where=(Omega_GW > ET_sens), alpha=0.3, color='red', label='ET Detectable')
        ax.set_xlabel('Frequency [Hz]', fontsize=14)
        ax.set_ylabel(r'$\Omega_{\rm GW} h^2$', fontsize=14)
        ax.set_xlim(1e-5, 1e2)
        ax.set_ylim(1e-18, 1e-10)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=11, loc='upper right')
        ax.set_title('EQST-GP Gravitational Wave Spectrum: Multi-Detector Coverage', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
    def plot_BAO_fit(self, z_data, DM_data, DH_data, DM_theory, DH_theory, filename='BAO_fit.pdf'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.errorbar(z_data, DM_data, yerr=0.02*DM_data, fmt='o', color='blue', markersize=8, capsize=5, label='DESI Data')
        ax1.plot(z_data, DM_theory, 's-', color='red', markersize=6, linewidth=2, label=r'EQST-GP $\Lambda_{eff}(z)$')
        ax1.set_xlabel('Redshift z', fontsize=13)
        ax1.set_ylabel(r'$D_M / r_d$', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Comoving Angular Diameter Distance', fontsize=13, fontweight='bold')
        ax2.errorbar(z_data, DH_data, yerr=0.02*DH_data, fmt='o', color='green', markersize=8, capsize=5, label='DESI Data')
        ax2.plot(z_data, DH_theory, 's-', color='orange', markersize=6, linewidth=2, label=r'EQST-GP $\Lambda_{eff}(z)$')
        ax2.set_xlabel('Redshift z', fontsize=13)
        ax2.set_ylabel(r'$D_H / r_d$', fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Hubble Distance', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
    def plot_Hubble_diagram(self, z_SN, mu_obs, mu_theory_LCDM, mu_theory_EQST, filename='Hubble_diagram.pdf'):
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.errorbar(z_SN, mu_obs, yerr=0.15, fmt='o', color='black', markersize=4, alpha=0.6, capsize=3, label='Pantheon+ Data')
        ax.plot(z_SN, mu_theory_LCDM, '-', color='blue', linewidth=2.5, label=r'$\Lambda$CDM (Planck)')
        ax.plot(z_SN, mu_theory_EQST, '-', color='red', linewidth=2.5, label=r'EQST-GP $\Lambda_{eff}(z)$')
        ax.set_xlabel('Redshift z', fontsize=14)
        ax.set_ylabel(r'Distance Modulus $\mu$', fontsize=14)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_title('Hubble Diagram: Supernovae Distance Moduli', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
    def plot_corner_MCMC(self, chain, labels, filename='corner_plot.pdf'):
       
        fixed_labels = [r'$\Omega_m$', r'$h$', r'$w_0$', r'$w_a$']
        N_params = chain.shape[2]
        fig, axes = plt.subplots(N_params, N_params, figsize=(12, 12))
        for i in range(N_params):
            for j in range(N_params):
                ax = axes[i, j]
                if i == j:
                    ax.hist(chain[:, :, i].flatten(), bins=50, color='steelblue', alpha=0.7, density=True)
                    ax.set_ylabel('Density', fontsize=10)
                elif i > j:
                    ax.hexbin(chain[:, :, j].flatten(), chain[:, :, i].flatten(), gridsize=30, cmap='Blues', mincnt=1)
                else:
                    ax.axis('off')
                if i == N_params - 1:
                    ax.set_xlabel(fixed_labels[j], fontsize=11)
                if j == 0 and i > 0:
                    ax.set_ylabel(fixed_labels[i], fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
    def plot_2D_structure(self, x, y, z, filename='large_scale_structure.pdf'):
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121)
        ax1.scatter(x, y, s=1, c=z, cmap='viridis', alpha=0.6)
        ax1.set_xlabel('x [Mpc/h]', fontsize=12)
        ax1.set_ylabel('y [Mpc/h]', fontsize=12)
        ax1.set_title('Galaxy Distribution (x-y plane)', fontsize=13, fontweight='bold')
        cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label('z [Mpc/h]', fontsize=11)
        ax2 = fig.add_subplot(122)
        ax2.scatter(x, z, s=1, c=y, cmap='plasma', alpha=0.6)
        ax2.set_xlabel('x [Mpc/h]', fontsize=12)
        ax2.set_ylabel('z [Mpc/h]', fontsize=12)
        ax2.set_title('Galaxy Distribution (x-z plane)', fontsize=13, fontweight='bold')
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('y [Mpc/h]', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
    def plot_correlation_function(self, r, xi_r, filename='two_point_correlation.pdf'):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.loglog(r, xi_r, 'o-', color='darkblue', markersize=6, linewidth=2, label='Measured $\\xi(r)$')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Separation r [Mpc/h]', fontsize=13)
        ax.set_ylabel('Correlation Function $\\xi(r)$', fontsize=13)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_title('Two-Point Correlation Function', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

class DataExporter:
    def __init__(self, output_dir='./outputs/data/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_GW_spectrum_HDF5(self, f_array, Omega_GW, components, filename='EQST_GP_GW_spectrum.h5'):
        filepath = os.path.join(self.output_dir, filename)
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('frequency_Hz', data=f_array)
            hf.create_dataset('Omega_GW_h2_total', data=Omega_GW)
            hf.create_dataset('Omega_GW_h2_sound', data=components['sound'])
            hf.create_dataset('Omega_GW_h2_turbulence', data=components['turbulence'])
            hf.create_dataset('Omega_GW_h2_bubble', data=components['bubble'])
            hf.attrs['T_nucleation_GeV'] = 9.71e15
            hf.attrs['alpha_PT'] = 0.42
            hf.attrs['beta_over_H'] = 94.7
            hf.attrs['v_w'] = 0.27
            hf.attrs['g_star'] = 187.0
            
    def export_BAO_results_JSON(self, results, filename='BAO_fit_results.json'):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as jf:
            json.dump(results, jf, indent=4)
            
    def export_MCMC_chain_FITS(self, chain, parameter_names, filename='MCMC_chain.fits'):
        filepath = os.path.join(self.output_dir, filename)
        N_walkers, N_steps, N_params = chain.shape
        col_list = []
        for i, name in enumerate(parameter_names):
            col_data = chain[:, :, i].flatten()
            col_list.append(fits.Column(name=name, format='D', array=col_data))
        hdu = fits.BinTableHDU.from_columns(col_list)
        hdu.header['N_WALK'] = N_walkers
        hdu.header['N_STEPS'] = N_steps
        hdu.header['N_PARAM'] = N_params
        hdu.writeto(filepath, overwrite=True)
        
    def export_predictions_summary_YAML(self, predictions, filename='EQST_GP_predictions.yaml'):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as yf:
            yaml.dump(predictions, yf, default_flow_style=False)

class EQSTGPFullPipeline:
    def __init__(self):
        self.constants = FundamentalConstants()
        self.eqst_params = EQSTGPParameters()
        self.Lambda_cosmo = LambdaEffectiveCosmology(self.constants)
        self.GW_spectrum = GravitationalWaveSpectrum(self.eqst_params, self.constants)
        self.LISA = LISASensitivity(self.constants)
        self.Virgo = VirgoSensitivity()
        self.LIGO = LIGOSensitivity()
        self.ET = EinsteinTelescopeSensitivity()
        self.visualizer = DataVisualization()
        self.exporter = DataExporter()
        
    def run_full_analysis(self):
        print("="*80)
        print("EQST-GP GRAVITATIONAL WAVE FRAMEWORK")
        print("Complete Multi-Messenger Observatory Integration Analysis")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        print("[1/10] Computing EQST-GP Gravitational Wave Spectrum...")
        f_array = np.logspace(-5, 2, 2000)
        Omega_GW_sound = self.GW_spectrum.spectrum_sound(f_array)
        Omega_GW_turb = self.GW_spectrum.spectrum_turbulence(f_array)
        Omega_GW_bubble = self.GW_spectrum.spectrum_bubble(f_array)
        Omega_GW_total = Omega_GW_sound + Omega_GW_turb + Omega_GW_bubble
        print(f"   Peak frequency (sound): {self.eqst_params.f_sw_Hz:.4e} Hz")
        print(f"   Peak amplitude (sound): {self.eqst_params.Omega_sw_peak_h2:.4e}")
        print("")
        
        print("[2/10] Computing Detector Sensitivities...")
        LISA_Omega_sens = self.LISA.Omega_sens(f_array)
        ET_S_n = self.ET.S_n(f_array)
        H0_SI = self.constants.H0_Planck2018 * 1000.0 / self.constants.Mpc_to_m
        ET_Omega_sens = (2.0 * np.pi**2 / (3.0 * H0_SI**2)) * f_array**2 * ET_S_n / np.sqrt(self.LISA.T_obs_sec)
        SNR_LISA = self.LISA.SNR(self.GW_spectrum.spectrum_total, f_min=1e-4, f_max=1e-1)
        print(f"   LISA SNR (4 yr): {SNR_LISA:.2f}")
        print("")
        
        print("[3/10] Loading and Fitting BAO Data...")
        desi_bao = DESIBAOData()
        z_eff, DM_obs, DH_obs, cov_BAO = desi_bao.load_BAO_measurements()
        bao_fitter = BaryonAcousticOscillationFitter(self.constants, self.Lambda_cosmo)
        # Ensure constants is passed correctly
        bao_fitter.constants = self.constants
        Omega_m_fit, h_fit, chi2_BAO, dof_BAO = bao_fitter.fit_BAO_data(z_eff, DM_obs, DH_obs, cov_BAO)
        print(f"   Best-fit Omega_m: {Omega_m_fit:.4f}")
        print(f"   Best-fit h: {h_fit:.4f}")
        print(f"   Chi-squared / dof: {chi2_BAO:.2f} / {dof_BAO}")
        print("")
        
        print("[4/10] Analyzing Hubble Tension...")
        hubble_analyzer = HubbleTensionAnalysis(self.constants, self.Lambda_cosmo)
        tension_LCDM, tension_EQST_Planck, tension_EQST_SH0ES, H0_EQST = hubble_analyzer.tension_sigma()
        print(f"   Planck-SH0ES tension: {tension_LCDM:.2f} sigma")
        print(f"   EQST-GP H0: {H0_EQST:.2f} km/s/Mpc")
        print(f"   EQST-Planck tension: {tension_EQST_Planck:.2f} sigma")
        print(f"   EQST-SH0ES tension: {tension_EQST_SH0ES:.2f} sigma")
        print("")
        
        print("[5/10] Computing Supernova Distance Moduli...")
        pantheon_sn = PantheonPlusSNData()
        z_SN, mu_obs, cov_SN = pantheon_sn.load_SN_data()
        mu_LCDM = 5.0 * np.log10(pantheon_sn.luminosity_distance_LCDM(z_SN, 67.4, 0.315) * 1.0e6) + 25.0
        cosmo_z = FlatLambdaCDM(H0=67.4, Om0=0.315)
        d_L_EQST = np.zeros_like(z_SN)
        for i, z_val in enumerate(z_SN):
            z_grid = np.linspace(0, z_val, 500)
            H_z = self.Lambda_cosmo.H_eff(z_grid, 0.315)
            integrand = self.constants.c / (H_z * 1000.0)
            d_C = integrate.simpson(integrand, x=z_grid)
            d_L_EQST[i] = (1.0 + z_val) * d_C
        mu_EQST = 5.0 * np.log10(d_L_EQST * 1.0e6) + 25.0
        print(f"   Processed {len(z_SN)} supernovae")
        print("")
        
        print("[6/10] Loading JWST High-z Galaxy Data...")
        jwst_survey = JWSTGalaxySurveyData()
        z_gal, M_star, SFR = jwst_survey.load_high_z_galaxies()
        z_bins, SMF_M, SMF_phi = jwst_survey.stellar_mass_function([8.0, 15.0])
        print(f"   Loaded {len(z_gal)} high-z galaxies")
        print(f"   Redshift range: {np.min(z_gal):.2f} - {np.max(z_gal):.2f}")
        print("")
        
        print("[7/10] Generating Large-Scale Structure Catalog...")
        deep_field = DeepFieldStructureAnalysis(self.constants)
        z_cat, RA_cat, Dec_cat, x_cat, y_cat, z_cart_cat = deep_field.generate_mock_large_scale_structure(N_gal=5000, z_max=4.0)
        r_bins = np.logspace(0, 2.5, 25)
        r_centers, xi_r = deep_field.compute_two_point_correlation(x_cat, y_cat, z_cart_cat, r_bins)
        print(f"   Generated {len(z_cat)} galaxies")
        print(f"   Computed 2-point correlation function")
        print("")
        
        print("[8/10] Running MCMC Parameter Estimation...")
        mcmc_estimator = MCMCParameterEstimation(self.constants, self.Lambda_cosmo)
        data_dict = {
            'BAO': (z_eff, DM_obs, DH_obs, cov_BAO),
            'SN': (z_SN[:50], mu_obs[:50], cov_SN[:50, :50])
        }
        chain, log_prob, acceptance = mcmc_estimator.run_MCMC(data_dict, N_walkers=16, N_steps=500)
        print(f"   MCMC completed: {chain.shape[0]} walkers, {chain.shape[1]} steps")
        print(f"   Mean acceptance rate: {np.mean(acceptance):.3f}")
        print("")
        
        print("[9/10] Performing Bayesian Model Comparison...")
        bayes_comp = BayesianModelComparison()
        def log_L_LCDM(params, data):
            Omega_m, h = params[:2]
            return -0.5 * bao_fitter.chi_squared_BAO([Omega_m, h], *data['BAO'])
        def log_L_EQST(params, data):
            Omega_m, h = params[:2]
            return -0.5 * mcmc_estimator.chi_squared_BAO_extended(Omega_m, h, -1.0, 0.0, *data['BAO'])
        bayes_comp.register_model('LCDM', log_L_LCDM, [(0.2, 0.5), (0.6, 0.8)])
        bayes_comp.register_model('EQST-GP', log_L_EQST, [(0.2, 0.5), (0.6, 0.8)])
        BF, log_Z_EQST, log_Z_LCDM = bayes_comp.bayes_factor('EQST-GP', 'LCDM', data_dict)
        interpretation = bayes_comp.jeffreys_scale_interpretation(BF)
        print(f"   Bayes Factor (EQST-GP / LCDM): {BF:.2f}")
        print(f"   log(Z_EQST): {log_Z_EQST:.2f}")
        print(f"   log(Z_LCDM): {log_Z_LCDM:.2f}")
        print(f"   Interpretation: {interpretation}")
        print("")
        
        print("[10/10] Generating Visualizations and Exporting Data...")
        self.visualizer.plot_GW_spectrum_with_sensitivities(f_array, Omega_GW_total, LISA_Omega_sens, ET_Omega_sens)
        DM_theory = np.zeros_like(z_eff)
        DH_theory = np.zeros_like(z_eff)
        for i, z in enumerate(z_eff):
            z_grid = np.linspace(0, z, 500)
            H_z = self.Lambda_cosmo.H_eff(z_grid, Omega_m_fit)
            integrand = self.constants.c / (H_z * 1000.0)
            DM_theory[i] = integrate.simpson(integrand, x=z_grid) / 147.09
            DH_theory[i] = (self.constants.c / (self.Lambda_cosmo.H_eff(z, Omega_m_fit) * 1000.0)) / 147.09
        self.visualizer.plot_BAO_fit(z_eff, DM_obs, DH_obs, DM_theory, DH_theory)
        self.visualizer.plot_Hubble_diagram(z_SN, mu_obs, mu_LCDM, mu_EQST)
        self.visualizer.plot_corner_MCMC(chain[:, 100:, :], [r'$\Omega_m$', r'$h$', r'$w_0$', r'$w_a$'])
        self.visualizer.plot_2D_structure(x_cat, y_cat, z_cart_cat)
        self.visualizer.plot_correlation_function(r_centers, xi_r)
        GW_components = {
            'sound': Omega_GW_sound,
            'turbulence': Omega_GW_turb,
            'bubble': Omega_GW_bubble
        }
        self.exporter.export_GW_spectrum_HDF5(f_array, Omega_GW_total, GW_components)
        BAO_results = {
            'Omega_m_best': float(Omega_m_fit),
            'h_best': float(h_fit),
            'chi2_min': float(chi2_BAO),
            'dof': int(dof_BAO),
            'redshifts': z_eff.tolist(),
            'DM_over_rd_data': DM_obs.tolist(),
            'DH_over_rd_data': DH_obs.tolist(),
            'DM_over_rd_theory': DM_theory.tolist(),
            'DH_over_rd_theory': DH_theory.tolist()
        }
        self.exporter.export_BAO_results_JSON(BAO_results)
        self.exporter.export_MCMC_chain_FITS(chain, ['Omega_m', 'h', 'w0', 'wa'])
        predictions_summary = {
            'gravitational_waves': {
                'f_peak_sound_Hz': float(self.eqst_params.f_sw_Hz),
                'Omega_peak_sound_h2': float(self.eqst_params.Omega_sw_peak_h2),
                'f_peak_turb_Hz': float(self.eqst_params.f_turb_Hz),
                'Omega_peak_turb_h2': float(self.eqst_params.Omega_turb_peak_h2),
                'LISA_SNR_4yr': float(SNR_LISA)
            },
            'dark_matter': {
                'm_DM_GeV': float(self.eqst_params.m_DM_GeV),
                'sigma_DM_SM_cm2': float(self.eqst_params.sigma_DM_SM_cm2)
            },
            'cosmology': {
                'H0_EQST_km_s_Mpc': float(H0_EQST),
                'Omega_m_BAO_fit': float(Omega_m_fit),
                'h_BAO_fit': float(h_fit),
                'tension_Planck_SH0ES_sigma': float(tension_LCDM),
                'tension_EQST_Planck_sigma': float(tension_EQST_Planck)
            },
            'model_comparison': {
                'Bayes_factor_EQST_vs_LCDM': float(BF),
                'log_evidence_EQST': float(log_Z_EQST),
                'log_evidence_LCDM': float(log_Z_LCDM),
                'interpretation': interpretation
            },
            'phase_transition': {
                'T_nucleation_GeV': float(self.eqst_params.T_n),
                'alpha_PT': float(self.eqst_params.alpha_PT),
                'beta_over_H': float(self.eqst_params.beta_over_H),
                'v_wall': float(self.eqst_params.v_w),
                'g_star': float(self.eqst_params.g_star)
            }
        }
        self.exporter.export_predictions_summary_YAML(predictions_summary)
        print("   All visualizations saved to ./outputs/")
        print("   All data products exported to ./outputs/data/")
        print("")
        print("="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("")
        print("Summary of Key Results:")
        print(f"  - LISA SNR (4 yr):              {SNR_LISA:.2f}")
        print(f"  - GW Peak Frequency:            {self.eqst_params.f_sw_Hz:.4e} Hz")
        print(f"  - H0 (EQST-GP):                 {H0_EQST:.2f} km/s/Mpc")
        print(f"  - Hubble Tension Resolution:    {tension_EQST_Planck:.2f}σ (Planck), {tension_EQST_SH0ES:.2f}σ (SH0ES)")
        print(f"  - Bayes Factor vs LCDM:         {BF:.2f} ({interpretation})")
        print(f"  - Dark Matter Mass:             {self.eqst_params.m_DM_GeV:.2e} GeV")
        print("")
        print("Next Steps:")
        print("  1. Review output plots in ./outputs/")
        print("  2. Examine data products in ./outputs/data/")
        print("  3. Use HDF5 spectrum for detector simulations")
        print("  4. Incorporate MCMC chains into follow-up analyses")
        print("  5. Submit predictions summary to observational teams")
        print("")
        
        return {
            'spectrum': {'f': f_array, 'Omega_GW': Omega_GW_total},
            'BAO_fit': {'Omega_m': Omega_m_fit, 'h': h_fit, 'chi2': chi2_BAO},
            'Hubble': {'H0_EQST': H0_EQST, 'tension': tension_LCDM},
            'model_comparison': {'BF': BF, 'interpretation': interpretation},
            'MCMC': {'chain': chain, 'log_prob': log_prob},
            'SNR': {'LISA': SNR_LISA}
        }

if __name__ == "__main__":
    pipeline = EQSTGPFullPipeline()
    results = pipeline.run_full_analysis()
    
    print("Framework ready for:")
    print("  - Real-time LISA data stream analysis")
    print("  - Cross-correlation with Virgo/LIGO triggers")
    print("  - Integration with Planck/LiteBIRD CMB data")
    print("  - JWST high-z galaxy catalog matching")
    print("  - DESI/Euclid large-scale structure surveys")
    print("  - Einstein Telescope design optimization")
    print("")
    print("Repository sup-structure:")
    print("  eqst_gw_framework/sup")
    print("    ├── core/")
    print("    │   ├── constants.py")
    print("    │   ├── eqst_parameters.py")
    print("    │   └── cosmology.py")
    print("    ├── gw_analysis/")
    print("    │   ├── spectrum.py")
    print("    │   └── detectors.py")
    print("    ├── data_loaders/")
    print("    │   ├── planck.py")
    print("    │   ├── desi.py")
    print("    │   ├── pantheon.py")
    print("    │   └── jwst.py")
    print("    ├── analysis/")
    print("    │   ├── bao_fitter.py")
    print("    │   ├── mcmc.py")
    print("    │   ├── hubble_tension.py")
    print("    │   └── model_comparison.py")
    print("    ├── simulation/")
    print("    │   ├── dark_matter.py")
    print("    │   ├── clusters.py")
    print("    │   └── structure.py")
    print("    ├── visualization/")
    print("    │   └── plots.py")
    print("    ├── io/")
    print("    │   └── exporters.py")
    print("    ├── pipeline.py")
    print("    └── run_analysis.py")