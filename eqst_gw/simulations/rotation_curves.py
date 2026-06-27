import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List

class GalaxyRotationCurve:
    def __init__(self, 
                 M_vir: float,
                 z: float = 0.0,
                 m_DM_GeV: float = 1.03e16,
                 constants=None,
                 eqst_params=None):
        self.M_vir_solar = M_vir
        self.z = z
        self.m_DM_GeV = m_DM_GeV
        
        if constants is None:
            from ..core.constants import FundamentalConstants
            self.c = FundamentalConstants()
        else:
            self.c = constants
            
        if eqst_params is None:
            from ..core.parameters import EQSTGPParameters
            self.ep = EQSTGPParameters()
        else:
            self.ep = eqst_params
        
        self.M_vir_kg = M_vir * 1.989e30
        self.compute_halo_parameters()
    
    def compute_halo_parameters(self):
        H0_SI = self.c.H0_Planck2018 * 1000.0 / self.c.Mpc_to_m
        E_z = np.sqrt(self.c.Omega_m_Planck2018 * (1.0 + self.z)**3 + self.c.Omega_Lambda_Planck2018)
        H_z = H0_SI * E_z
        rho_crit = 3.0 * H_z**2 / (8.0 * np.pi * self.c.G)
        
        Delta_vir = 200.0
        self.r_vir = (3.0 * self.M_vir_kg / (4.0 * np.pi * Delta_vir * rho_crit))**(1.0/3.0)
        
        M_pivot = 1.0e12 * 1.989e30
        c_0 = 9.0
        alpha_c = -0.13
        beta_c = -0.7
        self.concentration = c_0 * (self.M_vir_kg / M_pivot)**alpha_c * (1.0 + self.z)**beta_c
        
        self.r_s = self.r_vir / self.concentration
        
        Delta_c = (200.0 / 3.0) * self.concentration**3 / (np.log(1.0 + self.concentration) - self.concentration / (1.0 + self.concentration))
        self.rho_s = Delta_c * rho_crit
    
    def nfw_density(self, r: np.ndarray) -> np.ndarray:
        x = r / self.r_s
        return self.rho_s / (x * (1.0 + x)**2)
    
    def nfw_mass_enclosed(self, r: np.ndarray) -> np.ndarray:
        x = r / self.r_s
        return 4.0 * np.pi * self.rho_s * self.r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))
    
    def eqst_gp_density_modification(self, r: np.ndarray) -> np.ndarray:
        m_DM_kg = self.m_DM_GeV * self.c.GeV_to_kg
        n_DM_nfw = self.nfw_density(r) / m_DM_kg
        
        r_core = 0.1 * self.r_s
        suppression_factor = 1.0 - np.exp(-(r / r_core)**2)
        
        lambda_interaction = 1.0e-15
        interaction_enhancement = 1.0 + lambda_interaction * (self.r_s / r)**0.5
        
        rho_eqst = self.nfw_density(r) * suppression_factor * interaction_enhancement
        return rho_eqst
    
    def eqst_gp_mass_enclosed(self, r: np.ndarray) -> np.ndarray:
        r_array = np.atleast_1d(r)
        M_enc = np.zeros_like(r_array)
        
        for i, r_val in enumerate(r_array):
            r_integral = np.linspace(0, r_val, 1000)
            rho_integral = self.eqst_gp_density_modification(r_integral)
            integrand = 4.0 * np.pi * r_integral**2 * rho_integral
            M_enc[i] = np.trapz(integrand, r_integral)
        
        return M_enc if r_array.size > 1 else M_enc[0]
    
    def compute_rotation_curve_nfw(self, r: np.ndarray) -> np.ndarray:
        M_enc = self.nfw_mass_enclosed(r)
        v_circ = np.sqrt(self.c.G * M_enc / r)
        return v_circ
    
    def compute_rotation_curve_eqst_gp(self, r: np.ndarray) -> np.ndarray:
        M_enc = self.eqst_gp_mass_enclosed(r)
        v_circ = np.sqrt(self.c.G * M_enc / r)
        return v_circ
    
    def baryonic_component(self, r: np.ndarray, M_disk: float, R_d: float, M_bulge: float = 0.0, a_bulge: float = 1.0) -> np.ndarray:
        M_disk_kg = M_disk * 1.989e30
        v_disk_sq = self.c.G * M_disk_kg * r**2 / (2.0 * R_d * (r**2 + R_d**2)**(3.0/2.0))
        
        if M_bulge > 0:
            M_bulge_kg = M_bulge * 1.989e30
            v_bulge_sq = self.c.G * M_bulge_kg * r / (r + a_bulge)**2
        else:
            v_bulge_sq = 0.0
        
        return np.sqrt(v_disk_sq + v_bulge_sq)
    
    def total_rotation_curve(self, r: np.ndarray, M_disk: float, R_d: float, model: str = 'eqst_gp') -> np.ndarray:
        if model == 'eqst_gp':
            v_dm = self.compute_rotation_curve_eqst_gp(r)
        elif model == 'nfw':
            v_dm = self.compute_rotation_curve_nfw(r)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        v_baryon = self.baryonic_component(r, M_disk, R_d)
        v_total = np.sqrt(v_dm**2 + v_baryon**2)
        return v_total
    
    def fit_to_observed_data(self, r_obs: np.ndarray, v_obs: np.ndarray, v_err: np.ndarray, model: str = 'eqst_gp') -> Dict:
        def model_func(r, M_disk, R_d):
            return self.total_rotation_curve(r, M_disk, R_d, model=model)
        
        p0 = [1.0e10, 5.0e3]
        bounds = ([1.0e8, 1.0e3], [1.0e12, 5.0e4])
        
        popt, pcov = curve_fit(model_func, r_obs, v_obs, p0=p0, sigma=v_err, bounds=bounds, maxfev=5000)
        
        M_disk_best, R_d_best = popt
        M_disk_err, R_d_err = np.sqrt(np.diag(pcov))
        
        v_model = model_func(r_obs, *popt)
        chi2 = np.sum(((v_obs - v_model) / v_err)**2)
        dof = len(r_obs) - len(popt)
        chi2_reduced = chi2 / dof
        
        results = {
            'M_disk_solar': M_disk_best,
            'M_disk_err_solar': M_disk_err,
            'R_d_kpc': R_d_best / 3.086e19,
            'R_d_err_kpc': R_d_err / 3.086e19,
            'chi2': chi2,
            'dof': dof,
            'chi2_reduced': chi2_reduced,
            'model': model
        }
        
        return results
    
    def plot_rotation_curve_comparison(self, r_obs: np.ndarray, v_obs: np.ndarray, v_err: np.ndarray, M_disk: float, R_d: float, filename: str = 'rotation_curve_comparison.pdf'):
        r_theory = np.linspace(0.1 * np.min(r_obs), 1.5 * np.max(r_obs), 200)
        
        v_eqst = self.total_rotation_curve(r_theory, M_disk, R_d, model='eqst_gp')
        v_nfw = self.total_rotation_curve(r_theory, M_disk, R_d, model='nfw')
        
        v_dm_eqst = self.compute_rotation_curve_eqst_gp(r_theory)
        v_dm_nfw = self.compute_rotation_curve_nfw(r_theory)
        v_baryon = self.baryonic_component(r_theory, M_disk, R_d)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.errorbar(r_obs / 3.086e19, v_obs / 1000.0, yerr=v_err / 1000.0, fmt='o', color='black', markersize=6, capsize=4, label='Observed Data', zorder=3)
        ax1.plot(r_theory / 3.086e19, v_eqst / 1000.0, '-', color='red', linewidth=2.5, label='EQST-GP Total', zorder=2)
        ax1.plot(r_theory / 3.086e19, v_nfw / 1000.0, '--', color='blue', linewidth=2.5, label='NFW (ΛCDM) Total', zorder=2)
        ax1.plot(r_theory / 3.086e19, v_baryon / 1000.0, ':', color='green', linewidth=2, label='Baryonic Component', zorder=1)
        ax1.set_xlabel('Radius [kpc]', fontsize=13)
        ax1.set_ylabel('Circular Velocity [km/s]', fontsize=13)
        ax1.legend(fontsize=11, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Rotation Curve: Data vs Models', fontsize=14, fontweight='bold')
        
        ax2.plot(r_theory / 3.086e19, v_dm_eqst / 1000.0, '-', color='darkred', linewidth=2.5, label='EQST-GP Dark Matter', zorder=2)
        ax2.plot(r_theory / 3.086e19, v_dm_nfw / 1000.0, '--', color='darkblue', linewidth=2.5, label='NFW Dark Matter', zorder=2)
        ax2.plot(r_theory / 3.086e19, v_baryon / 1000.0, ':', color='green', linewidth=2, label='Baryonic', zorder=1)
        ax2.set_xlabel('Radius [kpc]', fontsize=13)
        ax2.set_ylabel('Circular Velocity [km/s]', fontsize=13)
        ax2.legend(fontsize=11, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Component Breakdown', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Rotation curve comparison plot saved to {filename}")