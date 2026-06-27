import numpy as np
import os
import json
import h5py
from datetime import datetime
from itertools import product

os.makedirs('./outputs/simulation_results/clusters', exist_ok=True)

print("="*65)
print("EQST-GP: Batch Galaxy Cluster Merger Simulations")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*65)

from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.simulations.dark_matter_halos import DMHaloProfile

const = FundamentalConstants()
ep = EQSTGPParameters()

CLUSTER_MASS_RATIOS = [1.0, 2.0, 4.0]
IMPACT_PARAMETERS_KPC = [0.0, 300.0, 700.0]
RELATIVE_VELOCITIES_KM_S = [1000.0, 2000.0, 3000.0]
M_CLUSTER1_SOLAR = 1.0e15

simulation_grid = list(product(CLUSTER_MASS_RATIOS, IMPACT_PARAMETERS_KPC, RELATIVE_VELOCITIES_KM_S))

print(f"Total simulations planned: {len(simulation_grid)}")
print(f"(Batch mode: density profiles and DM halos only, no full N-body)")
print()

results_summary = []

for sim_idx, (mass_ratio, b_kpc, v_rel) in enumerate(simulation_grid):
    M1 = M_CLUSTER1_SOLAR
    M2 = M_CLUSTER1_SOLAR / mass_ratio

    print(f"Simulation {sim_idx+1:02d}/{len(simulation_grid)}: M1={M1:.1e}, M2={M2:.1e}, b={b_kpc:.0f} kpc, v_rel={v_rel:.0f} km/s")

    halo1 = DMHaloProfile(M_vir_solar=M1, z=0.0, eqst_params=ep, constants=const)
    halo2 = DMHaloProfile(M_vir_solar=M2, z=0.0, eqst_params=ep, constants=const)

    r_bins = np.logspace(np.log10(0.01 * halo1.r_vir), np.log10(3.0 * halo1.r_vir), 100)

    rho1_nfw = halo1.nfw_density(r_bins)
    rho1_eqst = halo1.eqst_gp_density(r_bins)
    rho2_nfw = halo2.nfw_density(r_bins)
    rho2_eqst = halo2.eqst_gp_density(r_bins)

    v_circ1 = halo1.circular_velocity_eqst_gp(r_bins)
    v_circ2 = halo2.circular_velocity_eqst_gp(r_bins)

    sim_result = {
        'sim_id': sim_idx,
        'M1_solar': M1,
        'M2_solar': M2,
        'mass_ratio': mass_ratio,
        'impact_parameter_kpc': b_kpc,
        'relative_velocity_km_s': v_rel,
        'r_vir1_kpc': halo1.r_vir / 3.086e19,
        'r_vir2_kpc': halo2.r_vir / 3.086e19,
        'concentration1': halo1.concentration,
        'concentration2': halo2.concentration,
        'v_max1_km_s': np.max(v_circ1) / 1000.0,
        'v_max2_km_s': np.max(v_circ2) / 1000.0,
        'status': 'completed'
    }
    results_summary.append(sim_result)

    sim_filename = f'./outputs/simulation_results/clusters/cluster_sim_{sim_idx:03d}.h5'
    with h5py.File(sim_filename, 'w') as hf:
        hf.attrs['sim_id'] = sim_idx
        hf.attrs['M1_solar'] = M1
        hf.attrs['M2_solar'] = M2
        hf.attrs['mass_ratio'] = mass_ratio
        hf.attrs['impact_parameter_kpc'] = b_kpc
        hf.attrs['relative_velocity_km_s'] = v_rel
        hf.attrs['DM_model'] = 'EQST-GP Majorana Gluon'
        hf.attrs['created_at'] = datetime.now().isoformat()
        hf.create_dataset('r_bins_m', data=r_bins)
        hf.create_dataset('rho1_nfw', data=rho1_nfw, compression='gzip')
        hf.create_dataset('rho1_eqst', data=rho1_eqst, compression='gzip')
        hf.create_dataset('rho2_nfw', data=rho2_nfw, compression='gzip')
        hf.create_dataset('rho2_eqst', data=rho2_eqst, compression='gzip')
        hf.create_dataset('v_circ1_eqst', data=v_circ1, compression='gzip')
        hf.create_dataset('v_circ2_eqst', data=v_circ2, compression='gzip')

print(f"\nAll {len(simulation_grid)} simulations completed.")

with open('./outputs/simulation_results/clusters/batch_summary.json', 'w') as f:
    json.dump({'n_simulations': len(simulation_grid), 'timestamp': datetime.now().isoformat(), 'results': results_summary}, f, indent=4)

print("Batch summary saved: ./outputs/simulation_results/clusters/batch_summary.json")