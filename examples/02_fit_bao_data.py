import numpy as np
import matplotlib.pyplot as plt
import os
from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.cosmology import LambdaEffectiveCosmology
from eqst_gw.data.loaders import load_desi_bao
from eqst_gw.data.desi_bao import DESIBAODataManager
from eqst_gw.analysis.parameter_estimation import ParameterEstimator

os.makedirs('./outputs/plots', exist_ok=True)

const = FundamentalConstants()
ep = EQSTGPParameters()
cosmo = LambdaEffectiveCosmology(const)

desi_manager = DESIBAODataManager()
desi_manager.save_dr1_csv()
desi_manager.build_full_covariance_matrix()

z_eff, DM_obs, DH_obs, cov_matrix = load_desi_bao(use_mock=False)

pe = ParameterEstimator(ep, const, cosmo)

print("Fitting BAO data to EQST-GP cosmology...")
results_eqst = pe.fit_bao(z_eff, DM_obs, DH_obs, cov_matrix)

print("Fitting BAO data to standard LCDM...")
cosmo_lcdm = LambdaEffectiveCosmology(const)
cosmo_lcdm.Lambda_0 = const.Omega_Lambda_Planck2018
pe_lcdm = ParameterEstimator(ep, const, cosmo_lcdm)
results_lcdm = pe_lcdm.fit_bao(z_eff, DM_obs, DH_obs, cov_matrix)

print("\n" + "="*65)
print("BAO Fitting Results")
print("="*65)
print(f"EQST-GP: Omega_m = {results_eqst['Omega_m']:.4f} +/- {results_eqst['Omega_m_err']:.4f}")
print(f"EQST-GP: h       = {results_eqst['h']:.4f} +/- {results_eqst['h_err']:.4f}")
print(f"EQST-GP: chi2/dof = {results_eqst['chi2_min']:.2f} / {results_eqst['dof']}")
print(f"LCDM:    Omega_m = {results_lcdm['Omega_m']:.4f} +/- {results_lcdm['Omega_m_err']:.4f}")
print(f"LCDM:    h       = {results_lcdm['h']:.4f} +/- {results_lcdm['h_err']:.4f}")
print(f"LCDM:    chi2/dof = {results_lcdm['chi2_min']:.2f} / {results_lcdm['dof']}")
print("="*65)

Omega_m_eqst = results_eqst['Omega_m']
h_eqst = results_eqst['h']
r_d = 147.09
c_km_s = const.c / 1000.0

DM_theory_eqst = np.zeros_like(z_eff)
DH_theory_eqst = np.zeros_like(z_eff)
DM_theory_lcdm = np.zeros_like(z_eff)
DH_theory_lcdm = np.zeros_like(z_eff)

from scipy.integrate import quad
for i, z in enumerate(z_eff):
    H_eqst_z = cosmo.H_eff(z, Omega_m_eqst)
    d_C_eqst, _ = quad(lambda zp: c_km_s / cosmo.H_eff(zp, Omega_m_eqst), 0, z)
    DM_theory_eqst[i] = d_C_eqst / r_d
    DH_theory_eqst[i] = c_km_s / (H_eqst_z * r_d)

    Omega_m_lcdm = results_lcdm['Omega_m']
    H_lcdm_z = cosmo_lcdm.H_eff(z, Omega_m_lcdm)
    d_C_lcdm, _ = quad(lambda zp: c_km_s / cosmo_lcdm.H_eff(zp, Omega_m_lcdm), 0, z)
    DM_theory_lcdm[i] = d_C_lcdm / r_d
    DH_theory_lcdm[i] = c_km_s / (H_lcdm_z * r_d)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
ax1.errorbar(z_eff, DM_obs, yerr=0.02 * DM_obs, fmt='o', color='black', ms=7, capsize=5, label='DESI DR1')
ax1.plot(z_eff, DM_theory_eqst, 'rs-', ms=6, lw=2, label=f'EQST-GP ($\\chi^2/dof={results_eqst["chi2_min"]:.1f}/{results_eqst["dof"]}$)')
ax1.plot(z_eff, DM_theory_lcdm, 'b^--', ms=6, lw=2, alpha=0.75, label=f'$\\Lambda$CDM ($\\chi^2/dof={results_lcdm["chi2_min"]:.1f}/{results_lcdm["dof"]}$)')
ax1.set_xlabel('Effective Redshift $z_{\\rm eff}$', fontsize=13)
ax1.set_ylabel('$D_M / r_d$', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Transverse Comoving Distance', fontsize=13, fontweight='bold')

ax2 = axes[1]
ax2.errorbar(z_eff, DH_obs, yerr=0.02 * DH_obs, fmt='o', color='black', ms=7, capsize=5, label='DESI DR1')
ax2.plot(z_eff, DH_theory_eqst, 'rs-', ms=6, lw=2, label='EQST-GP')
ax2.plot(z_eff, DH_theory_lcdm, 'b^--', ms=6, lw=2, alpha=0.75, label='$\\Lambda$CDM')
ax2.set_xlabel('Effective Redshift $z_{\\rm eff}$', fontsize=13)
ax2.set_ylabel('$D_H / r_d$', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('Hubble Distance', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('./outputs/plots/02_bao_fit.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: ./outputs/plots/02_bao_fit.pdf")