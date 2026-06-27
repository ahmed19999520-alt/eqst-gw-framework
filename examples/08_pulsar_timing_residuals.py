import numpy as np
import matplotlib.pyplot as plt
import os
from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.gravitational_waves.spectrum import GravitationalWaveSpectrum
from eqst_gw.data.loaders import load_nanograv_pta_data
from eqst_gw.multimessenger.correlation import GWMultiMessengerCorrelator
from eqst_gw.multimessenger.cross_checks import MultiMessengerCrossChecks

os.makedirs('./outputs/plots', exist_ok=True)

const = FundamentalConstants()
ep = EQSTGPParameters()

print("="*65)
print("Pulsar Timing Array Analysis: EQST-GP vs NANOGrav 15yr")
print("="*65)

pta_data = load_nanograv_pta_data(use_mock=False)
f_pta = pta_data['f_gw_Hz']
Omega_pta = pta_data['Omega_gw_h2']
sigma_pta = pta_data['sigma_Omega_gw_h2']
n_pulsars = len(pta_data['pulsars'])

print(f"  PTA frequency bins: {len(f_pta)}")
print(f"  Number of pulsars:  {n_pulsars}")
print(f"  Frequency range:    {f_pta[0]:.2e} - {f_pta[-1]:.2e} Hz")

gw = GravitationalWaveSpectrum(ep, const)
f_full = np.logspace(-10, 0, 5000)
Omega_eqst_full = gw.total_spectrum(f_full)
Omega_pta_extrapolated = np.interp(f_pta, f_full, Omega_eqst_full)

mm_checks = MultiMessengerCrossChecks(ep, const)
pta_consistency = mm_checks.pta_consistency_check(f_pta, Omega_pta, sigma_pta)

print(f"\nPTA Consistency Check:")
print(f"  EQST-GP chi2_red:    {pta_consistency['chi2_reduced_EQST']:.4f}")
print(f"  Power Law chi2_red:  {pta_consistency['chi2_reduced_cosmic_strings']:.4f}")
print(f"  Preferred model:     {pta_consistency['preferred_model']}")
print(f"  Spectral tilt (PTA):        {pta_consistency['spectral_tilt_PTA']:.3f}")
print(f"  Spectral tilt (EQST extrapolated): {pta_consistency['spectral_tilt_EQST_at_PTA']:.3f}")

correlator = GWMultiMessengerCorrelator(ep, const)
f_band_for_pta = np.logspace(-10, -6, 1000)
Omega_band = gw.total_spectrum(f_band_for_pta)
pta_gw_result = correlator.pta_gw_background_correlation(f_pta, Omega_pta, sigma_pta, f_band_for_pta, Omega_band)

print(f"\nPTA-GW Background Correlation Analysis:")
print(f"  Spectral tilt match: {pta_gw_result['spectral_tilt_consistent']}")
print(f"  Freq extrapolation ratio: {pta_gw_result['frequency_extrapolation_ratio']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax1 = axes[0]
ax1.errorbar(f_pta, Omega_pta, yerr=sigma_pta, fmt='o', color='black', ms=7, capsize=5, label='NANOGrav 15yr', zorder=5)
ax1.loglog(f_pta, Omega_pta_extrapolated, 'r-', lw=2.5, label='EQST-GP (extrapolated to nHz)', zorder=4)

gamma_pl = 13.0 / 3.0
A_pl = Omega_pta[len(Omega_pta)//2]
Omega_pl = A_pl * (f_pta / f_pta[len(f_pta)//2])**(2.0 - gamma_pl) * (f_pta / f_pta[len(f_pta)//2])**(3.0 - gamma_pl)
ax1.loglog(f_pta, np.abs(Omega_pl), 'b--', lw=2.0, alpha=0.75, label='Power Law (SMBHBs)', zorder=3)

ax1.set_xlabel('Frequency [Hz]', fontsize=13)
ax1.set_ylabel(r'$\Omega_{\rm GW}\, h^2$', fontsize=13)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, which='both', alpha=0.3)
ax1.set_title('PTA GW Background: Data vs Models', fontsize=13, fontweight='bold')

ax2 = axes[1]
residuals_eqst = Omega_pta - Omega_pta_extrapolated
residuals_pl = Omega_pta - np.abs(Omega_pl)
ax2.errorbar(f_pta, residuals_eqst / sigma_pta, fmt='rs', ms=7, capsize=5, label='EQST-GP residuals')
ax2.errorbar(f_pta * 1.2, residuals_pl / sigma_pta, fmt='b^', ms=7, capsize=5, alpha=0.7, label='Power Law residuals')
ax2.axhline(0, color='black', lw=1.5, ls='--')
ax2.axhline(1, color='gray', lw=1, ls=':', alpha=0.5)
ax2.axhline(-1, color='gray', lw=1, ls=':', alpha=0.5)
ax2.set_xscale('log')
ax2.set_xlabel('Frequency [Hz]', fontsize=13)
ax2.set_ylabel('Normalized Residuals $(O-E)/\\sigma$', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_title('Normalized Residuals', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('./outputs/plots/08_pulsar_timing_residuals.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("\nPlot saved: ./outputs/plots/08_pulsar_timing_residuals.pdf")