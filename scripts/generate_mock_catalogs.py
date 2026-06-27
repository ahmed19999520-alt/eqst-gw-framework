import numpy as np
import os
import json
import h5py
from datetime import datetime

os.makedirs('./data/observational/planck', exist_ok=True)
os.makedirs('./data/observational/desi', exist_ok=True)
os.makedirs('./data/observational/pantheon_plus', exist_ok=True)
os.makedirs('./data/observational/jwst', exist_ok=True)
os.makedirs('./data/observational/ligo_gwosc', exist_ok=True)
os.makedirs('./data/simulations/cluster_merger_snapshots', exist_ok=True)
os.makedirs('./data/simulations/rotation_curves_samples', exist_ok=True)
os.makedirs('./data/simulations/stellar_structure_models', exist_ok=True)
os.makedirs('./data/templates/gw_waveforms', exist_ok=True)
os.makedirs('./data/templates/nfw_profiles', exist_ok=True)
os.makedirs('./data/templates/phase_transition_templates', exist_ok=True)

print("="*65)
print("EQST-GP Framework: Generating Mock Observational Data Catalogs")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*65)

print("\n[1/8] Generating Planck CMB mock data...")
ell = np.arange(2, 2509, dtype=float)
A_s = 2.1e-9
n_s = 0.9649
tau = 0.054
D_ell_TT = A_s * (ell / 100.0)**(n_s - 1.0) * np.exp(-2.0 * tau) * ell * (ell + 1.0) / (2.0 * np.pi) * 5.765e12
for l_pk, amp, wid in [(220, 5700, 75), (540, 2600, 105), (810, 1550, 85), (1120, 1050, 90)]:
    D_ell_TT += amp * np.exp(-0.5 * ((ell - l_pk) / wid)**2)
D_ell_TT *= np.exp(-(ell / 1500.0)**2.15)
sigma_TT = 0.008 * D_ell_TT + 5.0
data_TT = np.column_stack([ell, D_ell_TT, sigma_TT])
np.savetxt('./data/observational/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt', data_TT,
           header='ell    D_ell_TT [uK^2]    sigma', comments='#')
planck_params = {
    'H0': 67.4, 'H0_err': 0.5, 'Omega_m': 0.315, 'Omega_m_err': 0.007,
    'Omega_b_h2': 0.02237, 'sigma_8': 0.811, 'n_s': 0.9649, 'tau': 0.054,
    'reference': 'Planck Collaboration (2020)', 'doi': '10.1051/0004-6361/201833910'
}
with open('./data/observational/planck/planck_2018_cosmo_params.json', 'w') as f:
    json.dump(planck_params, f, indent=4)
print("  Planck CMB TT spectrum and parameters saved.")

print("\n[2/8] Generating DESI BAO DR1 mock data...")
z_eff = np.array([0.295, 0.510, 0.706, 0.930, 1.317, 1.491, 2.330])
DM_rd = np.array([7.93, 13.62, 16.85, 21.71, 27.79, 30.21, 39.71])
DH_rd = np.array([20.08, 20.98, 19.33, 17.88, 13.82, 13.23, 8.52])
sigma_DM = np.array([0.15, 0.25, 0.32, 0.28, 0.69, 0.79, 0.94])
sigma_DH = np.array([0.60, 0.61, 0.53, 0.35, 0.42, 0.59, 0.17])
tracers = ['BGS', 'LRG1', 'LRG2', 'LRG3+ELG1', 'ELG2', 'QSO', 'Lya_QSO']
import pandas as pd
df_desi = pd.DataFrame({'tracer': tracers, 'z_eff': z_eff, 'DM_over_rd': DM_rd, 'DH_over_rd': DH_rd, 'sigma_DM': sigma_DM, 'sigma_DH': sigma_DH})
df_desi.to_csv('./data/observational/desi/DESI_BAO_2024_DR1.csv', index=False)
N = len(z_eff)
cov = np.zeros((2*N, 2*N))
for i in range(N):
    cov[i, i] = sigma_DM[i]**2
    cov[N+i, N+i] = sigma_DH[i]**2
    cov[i, N+i] = -0.40 * sigma_DM[i] * sigma_DH[i]
    cov[N+i, i] = cov[i, N+i]
np.save('./data/observational/desi/covariance_matrix.npy', cov)
print("  DESI BAO DR1 data and covariance matrix saved.")

print("\n[3/8] Generating Pantheon+ supernovae mock data...")
np.random.seed(42)
N_SN = 1701
z_sn = np.sort(np.random.uniform(0.001, 2.26, N_SN))
H0_true = 73.04
Om_true = 0.315
OL_true = 0.685
c_kms = 299792.458
from scipy.integrate import quad
mu_th = np.zeros(N_SN)
for i, z in enumerate(z_sn):
    dC, _ = quad(lambda zp: c_kms / (H0_true * np.sqrt(Om_true * (1+zp)**3 + OL_true)), 0, z)
    dL = (1 + z) * dC
    mu_th[i] = 5.0 * np.log10(dL * 1.0e6) + 25.0
sigma_mu = 0.12 * np.ones(N_SN)
mu_obs = mu_th + np.random.normal(0, sigma_mu)
header = "z  mu_obs  sigma_mu"
np.savetxt('./data/observational/pantheon_plus/Pantheon+SH0ES.dat', np.column_stack([z_sn, mu_obs, sigma_mu]), header=header)
print(f"  Pantheon+ {N_SN} supernovae saved.")

print("\n[4/8] Generating JWST high-z galaxy mock catalog...")
N_gal = 500
z_gal = np.random.uniform(8.0, 15.0, N_gal)
log_M = 8.5 + 0.3 * (z_gal - 10.0) / 5.0 + np.random.normal(0, 0.4, N_gal)
SFR = 10.0**(0.9 * log_M - 8.5 + np.random.normal(0, 0.3, N_gal))
beta_UV = -2.0 + np.random.normal(0, 0.35, N_gal)
RA_gal = np.random.uniform(214.6, 215.1, N_gal)
Dec_gal = np.random.uniform(52.7, 52.9, N_gal)
z_err = 0.3 + 0.1 * np.random.rand(N_gal)
df_jwst = pd.DataFrame({'z_phot': z_gal, 'z_phot_err': z_err, 'log_M_stellar': log_M, 'SFR': SFR, 'beta_UV': beta_UV, 'RA': RA_gal, 'Dec': Dec_gal})
df_jwst.to_csv('./data/observational/jwst/high_z_galaxies_catalog.csv', index=False)
smf_mass = np.logspace(7, 12, 100)
smf_phi = 1.0e-3 * (smf_mass / 1.0e10)**(-1.5) * np.exp(-smf_mass / 1.0e11)
np.save('./data/observational/jwst/stellar_mass_function.npy', np.column_stack([smf_mass, smf_phi]))
print(f"  JWST {N_gal} high-z galaxies and SMF saved.")

print("\n[5/8] Generating LIGO/GWOSC mock event catalog...")
N_events = 90
events = []
for i in range(N_events):
    m1 = np.random.uniform(5.0, 80.0)
    m2 = np.random.uniform(5.0, min(m1, 50.0))
    Mc = (m1 * m2)**(3.0/5.0) / (m1 + m2)**(1.0/5.0)
    z = np.random.uniform(0.01, 1.5)
    events.append({'name': f'GW{200000 + i * 100:06d}', 'm1_source_solar': round(m1, 2), 'm2_source_solar': round(m2, 2), 'chirp_mass_solar': round(Mc, 2), 'z_luminosity': round(z, 4), 'network_snr': round(np.random.uniform(8, 40), 1), 'far_per_year': float(np.random.exponential(1.0e-4)), 'event_type': np.random.choice(['BBH', 'BNS', 'NSBH'], p=[0.80, 0.12, 0.08])})
catalog = {'events': events, 'n_events': N_events, 'catalog': 'GWTC-3', 'reference': 'LIGO-Virgo-KAGRA (2021)'}
with open('./data/observational/ligo_gwosc/gwtc3_confident.json', 'w') as f:
    json.dump(catalog, f, indent=2)
with h5py.File('./data/observational/ligo_gwosc/event_parameters.h5', 'w') as hf:
    hf.create_dataset('event_names', data=np.array([e['name'] for e in events], dtype='S20'))
    hf.create_dataset('m1_source', data=np.array([e['m1_source_solar'] for e in events]))
    hf.create_dataset('m2_source', data=np.array([e['m2_source_solar'] for e in events]))
    hf.create_dataset('z_luminosity', data=np.array([e['z_luminosity'] for e in events]))
    hf.create_dataset('network_snr', data=np.array([e['network_snr'] for e in events]))
print(f"  LIGO GWOSC {N_events} events saved.")

print("\n[6/8] Generating bubble nucleation simulation grid...")
alpha_grid = np.array([0.1, 0.2, 0.42, 0.8, 1.5])
beta_H_grid = np.array([10.0, 50.0, 94.7, 200.0, 500.0])
v_w_grid = np.array([0.1, 0.27, 0.5, 0.8])
f_grid = np.logspace(-5, 0, 200)
with h5py.File('./data/simulations/bubble_nucleation_grid.h5', 'w') as hf:
    hf.create_dataset('alpha_grid', data=alpha_grid)
    hf.create_dataset('beta_H_grid', data=beta_H_grid)
    hf.create_dataset('v_w_grid', data=v_w_grid)
    hf.create_dataset('frequency_Hz', data=f_grid)
    spectra = np.zeros((len(alpha_grid), len(beta_H_grid), len(v_w_grid), len(f_grid)))
    for i, alpha in enumerate(alpha_grid):
        for j, beta_H in enumerate(beta_H_grid):
            for k, v_w in enumerate(v_w_grid):
                f_pk = 1.87e-3 * (94.7 / beta_H) * (0.27 / v_w)
                Om_pk = 6.31e-14 * (alpha / 0.42)**2 * (94.7 / beta_H)
                x = f_grid / f_pk
                spectra[i, j, k, :] = Om_pk * x**3 * (7.0 / (4.0 + 3.0 * x**2))**(7.0/2.0)
    hf.create_dataset('Omega_GW_h2_grid', data=spectra, compression='gzip')
    hf.attrs['description'] = 'EQST-GP GW Spectrum Parameter Grid'
    hf.attrs['created_at'] = datetime.now().isoformat()
print("  Bubble nucleation grid saved.")

print("\n[7/8] Generating rotation curve samples...")
galaxy_types = ['dwarf', 'spiral_small', 'spiral_large', 'elliptical']
for gtype in galaxy_types:
    M_map = {'dwarf': 1.0e9, 'spiral_small': 1.0e10, 'spiral_large': 1.0e11, 'elliptical': 1.0e12}
    M_vir = M_map[gtype]
    r_kpc = np.linspace(0.5, 50.0, 100)
    v_obs = 100.0 * (M_vir / 1.0e11)**0.25 * (1.0 - np.exp(-r_kpc / 5.0)) + np.random.normal(0, 3.0, 100)
    v_err = 5.0 * np.ones(100)
    np.savetxt(f'./data/simulations/rotation_curves_samples/{gtype}_rotation_curve.txt',
               np.column_stack([r_kpc, v_obs, v_err]), header='r_kpc  v_obs_km_s  v_err_km_s')
print("  Rotation curve samples saved.")

print("\n[8/8] Generating GW waveform templates...")
f_templates = np.logspace(-5, 2, 2000)
for name, (f_pk, Om_pk) in [('eqst_gp_fiducial', (1.87e-3, 6.31e-14)), ('strong_transition', (2.0e-3, 2.0e-13)), ('weak_transition', (5.0e-4, 1.0e-15))]:
    x = f_templates / f_pk
    Omega = Om_pk * x**3 * (7.0 / (4.0 + 3.0 * x**2))**(7.0/2.0)
    np.savetxt(f'./data/templates/gw_waveforms/{name}_template.txt',
               np.column_stack([f_templates, Omega]), header='frequency_Hz  Omega_GW_h2')

print("  GW waveform templates saved.")

print("\n" + "="*65)
print("All mock catalogs generated successfully.")
print("="*65)