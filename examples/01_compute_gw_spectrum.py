import numpy as np
import matplotlib.pyplot as plt
import os
from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.gravitational_waves.spectrum import GravitationalWaveSpectrum
from eqst_gw.detectors.lisa import LISADetector
from eqst_gw.detectors.einstein_telescope import EinsteinTelescopeDetector

os.makedirs('./outputs/plots', exist_ok=True)

const = FundamentalConstants()
ep = EQSTGPParameters()

gw = GravitationalWaveSpectrum(ep, const)
lisa = LISADetector(mission_duration_years=4.0, constants=const)
et = EinsteinTelescopeDetector(design='ET-D', constants=const)

f = np.logspace(-5, 2, 2000)

components = gw.spectrum_components(f)
props = gw.peak_properties()
snr_lisa = lisa.compute_snr(f, components['total'])
snr_et = et.compute_snr(f, components['total'])

print("="*65)
print("EQST-GP Gravitational Wave Spectrum - Key Results")
print("="*65)
print(f"Sound wave peak frequency:  {props['f_sound_peak_Hz']:.4e} Hz")
print(f"Sound wave peak amplitude:  {props['Omega_sound_peak_h2']:.4e}")
print(f"Turbulence peak frequency:  {props['f_turb_peak_Hz']:.4e} Hz")
print(f"Turbulence peak amplitude:  {props['Omega_turb_peak_h2']:.4e}")
print(f"LISA SNR (4 yr):            {snr_lisa:.2f}")
print(f"Einstein Telescope SNR:     {snr_et:.4f}")
print("="*65)

LISA_sens = lisa.omega_sensitivity(f)
ET_sens = et.omega_sensitivity(f)

fig, ax = plt.subplots(figsize=(13, 8))
ax.loglog(f, components['total'], 'k-', lw=2.5, label='EQST-GP Total')
ax.loglog(f, components['sound_waves'], 'b-', lw=2.0, alpha=0.85, label='Sound Waves')
ax.loglog(f, components['turbulence'], 'r-', lw=2.0, alpha=0.85, label='MHD Turbulence')
ax.loglog(f, components['bubble_collisions'], 'g--', lw=1.5, alpha=0.75, label='Bubble Collisions')
ax.loglog(f, LISA_sens, 'b--', lw=1.8, alpha=0.6, label='LISA Sensitivity (4 yr)')
ax.loglog(f, ET_sens, 'purple', lw=1.8, linestyle='-.', alpha=0.6, label='ET-D Sensitivity')
ax.fill_between(f, components['total'], LISA_sens, where=(components['total'] > LISA_sens), alpha=0.15, color='blue', label='LISA Detectable')
ax.axvline(x=ep.f_sw_Hz, color='navy', lw=1.5, ls=':', alpha=0.9, label=f'$f_{{\\rm sw}}$ = {ep.f_sw_Hz:.2e} Hz')
ax.axvline(x=ep.f_turb_Hz, color='darkred', lw=1.5, ls=':', alpha=0.9, label=f'$f_{{\\rm turb}}$ = {ep.f_turb_Hz:.2e} Hz')
ax.set_xlabel('Frequency [Hz]', fontsize=14)
ax.set_ylabel(r'$\Omega_{\rm GW}\, h^2$', fontsize=14)
ax.set_xlim(1.0e-5, 1.0e2)
ax.set_ylim(1.0e-18, 1.0e-10)
ax.legend(fontsize=10, loc='upper right', ncol=2)
ax.grid(True, which='both', alpha=0.3)
ax.set_title('EQST-GP Gravitational Wave Background Spectrum', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('./outputs/plots/01_gw_spectrum.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved: ./outputs/plots/01_gw_spectrum.pdf")