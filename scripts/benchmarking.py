import numpy as np
import time
import os
import json
from datetime import datetime

os.makedirs('./outputs/reports', exist_ok=True)

print("="*65)
print("EQST-GP Framework Benchmarking Suite")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*65)

from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.cosmology import LambdaEffectiveCosmology
from eqst_gw.gravitational_waves.spectrum import GravitationalWaveSpectrum
from eqst_gw.detectors.lisa import LISADetector
from eqst_gw.simulations.dark_matter_halos import DMHaloProfile
from eqst_gw.simulations.structure_formation import StructureFormationAnalysis
from eqst_gw.physics.phase_transition import PhaseTransitionDynamics

const = FundamentalConstants()
ep = EQSTGPParameters()
cosmo = LambdaEffectiveCosmology(const)

benchmarks = {}

print("\n[BENCH 1] GW Spectrum Computation (2000 frequency points)...")
f = np.logspace(-5, 2, 2000)
gw = GravitationalWaveSpectrum(ep, const)
n_trials = 20
t_start = time.perf_counter()
for _ in range(n_trials):
    Omega = gw.total_spectrum(f)
t_end = time.perf_counter()
t_per_call = (t_end - t_start) / n_trials * 1000.0
benchmarks['gw_spectrum_2000pts_ms'] = t_per_call
print(f"  Time per call: {t_per_call:.3f} ms")

print("\n[BENCH 2] LISA SNR Computation...")
lisa = LISADetector(mission_duration_years=4.0, constants=const)
n_trials = 10
t_start = time.perf_counter()
for _ in range(n_trials):
    snr = lisa.compute_snr(f, Omega)
t_end = time.perf_counter()
t_per_call = (t_end - t_start) / n_trials * 1000.0
benchmarks['lisa_snr_ms'] = t_per_call
print(f"  Time per call: {t_per_call:.3f} ms  (SNR = {snr:.2f})")

print("\n[BENCH 3] Phase Transition Critical Temperature...")
pt = PhaseTransitionDynamics(ep, const)
n_trials = 100
t_start = time.perf_counter()
for _ in range(n_trials):
    T_c = pt.critical_temperature()
t_end = time.perf_counter()
t_per_call = (t_end - t_start) / n_trials * 1000.0
benchmarks['critical_temperature_ms'] = t_per_call
print(f"  Time per call: {t_per_call:.4f} ms  (T_c = {T_c:.3e} GeV)")

print("\n[BENCH 4] DM Halo Profile (100 radial bins)...")
halo = DMHaloProfile(M_vir_solar=1.0e12, z=0.0, eqst_params=ep, constants=const)
r_test = np.logspace(18, 22, 100)
n_trials = 50
t_start = time.perf_counter()
for _ in range(n_trials):
    rho = halo.eqst_gp_density(r_test)
t_end = time.perf_counter()
t_per_call = (t_end - t_start) / n_trials * 1000.0
benchmarks['dm_halo_density_100bins_ms'] = t_per_call
print(f"  Time per call: {t_per_call:.3f} ms")

print("\n[BENCH 5] H_eff(z) Cosmological Function (100 z values)...")
z_test = np.linspace(0, 5.0, 100)
n_trials = 100
t_start = time.perf_counter()
for _ in range(n_trials):
    H = cosmo.H_eff(z_test, const.Omega_m_Planck2018)
t_end = time.perf_counter()
t_per_call = (t_end - t_start) / n_trials * 1000.0
benchmarks['H_eff_100pts_ms'] = t_per_call
print(f"  Time per call: {t_per_call:.4f} ms")

print("\n[BENCH 6] Comoving Distance Integration...")
n_trials = 20
t_start = time.perf_counter()
for _ in range(n_trials):
    d = cosmo.comoving_distance(1.0)
t_end = time.perf_counter()
t_per_call = (t_end - t_start) / n_trials * 1000.0
benchmarks['comoving_distance_ms'] = t_per_call
print(f"  Time per call: {t_per_call:.3f} ms  (d_C(z=1) = {d:.2f} Mpc)")

print("\n[BENCH 7] Matter Power Spectrum (100 k values)...")
sfa = StructureFormationAnalysis(const, cosmo)
k_test = np.logspace(-3, 1, 100)
n_trials = 20
t_start = time.perf_counter()
for _ in range(n_trials):
    P_k = sfa.matter_power_spectrum(k_test, z=0.0)
t_end = time.perf_counter()
t_per_call = (t_end - t_start) / n_trials * 1000.0
benchmarks['matter_power_spectrum_ms'] = t_per_call
print(f"  Time per call: {t_per_call:.3f} ms")

print("\n[BENCH 8] GW Spectrum Component Decomposition...")
n_trials = 20
t_start = time.perf_counter()
for _ in range(n_trials):
    components = gw.spectrum_components(f)
t_end = time.perf_counter()
t_per_call = (t_end - t_start) / n_trials * 1000.0
benchmarks['spectrum_components_ms'] = t_per_call
print(f"  Time per call: {t_per_call:.3f} ms")

print("\n" + "="*65)
print("BENCHMARK SUMMARY")
print("="*65)
for name, time_ms in sorted(benchmarks.items(), key=lambda x: x[1]):
    print(f"  {name:<45s}: {time_ms:10.4f} ms")

total_pipeline_time = sum(benchmarks.values())
print(f"\n  Total (sum of all benchmarks):                 : {total_pipeline_time:.2f} ms")
print(f"  Estimated full analysis pipeline (all steps)   : ~{total_pipeline_time * 5:.0f} ms")

report = {
    'timestamp': datetime.now().isoformat(),
    'python_version': __import__('sys').version,
    'numpy_version': np.__version__,
    'benchmarks_ms': benchmarks,
    'total_core_ms': total_pipeline_time,
    'platform': __import__('platform').platform()
}

with open('./outputs/reports/benchmark_results.json', 'w') as f_out:
    json.dump(report, f_out, indent=4)

print("\nBenchmark report saved: ./outputs/reports/benchmark_results.json")