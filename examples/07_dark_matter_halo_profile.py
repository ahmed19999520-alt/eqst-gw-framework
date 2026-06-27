import numpy as np
import matplotlib.pyplot as plt
import os
from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.simulations.dark_matter_halos import DMHaloProfile
from eqst_gw.multimessenger.dark_matter import MajoranaGluonDarkMatter

os.makedirs('./outputs/plots', exist_ok=True)

const = FundamentalConstants()
ep = EQSTGPParameters()

print("="*65)
print("EQST-GP Dark Matter Halo Profile Analysis")
print("="*65)

masses = [1.0e10, 1.0e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15]
colors = ['purple', 'blue', 'green', 'orange', 'red', 'brown']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

kpc = 3.086e19

for M_vir, col in zip(masses, colors):
    halo = DMHaloProfile(M_vir_solar=M_vir, z=0.0, eqst_params=ep, constants=const)
    r_array = np.logspace(np.log10(0.001 * halo.r_vir), np.log10(halo.r_vir), 200)

    rho_nfw = halo.nfw_density(r_array)
    rho_eqst = halo.eqst_gp_density(r_array)
    v_circ_eqst = halo.circular_velocity_eqst_gp(r_array)
    v_circ_nfw = halo.compute_rotation_curve_nfw(r_array) if hasattr(halo, 'compute_rotation_curve_nfw') else np.sqrt(const.G * halo.nfw_mass_enclosed(r_array) / r_array)

    label = f'$M_{{\\rm vir}} = 10^{{{int(np.log10(M_vir))}}} M_\\odot$'

    axes[0].loglog(r_array / kpc, rho_nfw, '--', color=col, lw=1.5, alpha=0.7)
    axes[0].loglog(r_array / kpc, rho_eqst, '-', color=col, lw=2.0, label=label)

    axes[1].semilogx(r_array / kpc, v_circ_eqst / 1000.0, '-', color=col, lw=2.0)
    axes[1].semilogx(r_array / kpc, v_circ_nfw / 1000.0, '--', color=col, lw=1.5, alpha=0.7)

    gamma_ann = halo.eqst_gp_dm_annihilation_rate(r_array)
    axes[2].loglog(r_array / kpc, gamma_ann, '-', color=col, lw=2.0)

axes[0].set_xlabel('Radius [kpc]', fontsize=13)
axes[0].set_ylabel(r'$\rho_{\rm DM}$ [kg/m$^3$]', fontsize=13)
axes[0].legend(fontsize=8, loc='upper right')
axes[0].grid(True, which='both', alpha=0.3)
axes[0].set_title('DM Density: EQST-GP (solid) vs NFW (dashed)', fontsize=12, fontweight='bold')

axes[1].set_xlabel('Radius [kpc]', fontsize=13)
axes[1].set_ylabel('Circular Velocity [km/s]', fontsize=13)
axes[1].grid(True, which='both', alpha=0.3)
axes[1].set_title('Rotation Curves: EQST-GP (solid) vs NFW (dashed)', fontsize=12, fontweight='bold')

axes[2].set_xlabel('Radius [kpc]', fontsize=13)
axes[2].set_ylabel(r'$\Gamma_{\rm ann}$ [s$^{-1}$ m$^{-3}$]', fontsize=13)
axes[2].grid(True, which='both', alpha=0.3)
axes[2].set_title('Majorana Gluon DM Annihilation Rate', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('./outputs/plots/07_dark_matter_halo_profile.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: ./outputs/plots/07_dark_matter_halo_profile.pdf")

dm = MajoranaGluonDarkMatter(ep, const)
Omega_DM_h2 = dm.relic_density(ep.T_n, ep.beta_over_H)
stability = dm.topological_stability()

print(f"\nMajorana Gluon Dark Matter Properties:")
print(f"  Mass:                     {ep.m_DM_GeV:.3e} GeV")
print(f"  Cross-section (DM-SM):    {ep.sigma_DM_SM_cm2:.3e} cm^2")
print(f"  Relic density Omega h^2:  {Omega_DM_h2:.4f} (observed: 0.120)")
print(f"  Topologically stable:     {stability['is_topologically_stable']}")
print(f"  Topological charge:       {stability['topological_charge']}")
print(f"  Formation mechanism:      {stability['formation_mechanism']}")