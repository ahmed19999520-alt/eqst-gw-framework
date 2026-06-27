import numpy as np
import h5py
import os
from typing import Dict, Optional, Any
from datetime import datetime

class HDF5Handler:
    def __init__(self, output_dir: str = './outputs/data/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_gw_spectrum(self, f: np.ndarray, components: Dict[str, np.ndarray], eqst_params, filename: str = 'gw_spectrum.h5'):
        filepath = os.path.join(self.output_dir, filename)
        
        with h5py.File(filepath, 'w') as hf:
            hf.attrs['creation_date'] = datetime.now().isoformat()
            hf.attrs['framework_version'] = '1.0.0'
            hf.attrs['author'] = 'Ahmed Ali'
            hf.attrs['description'] = 'EQST-GP Gravitational Wave Spectrum'
            
            freq_ds = hf.create_dataset('frequency_Hz', data=f, compression='gzip')
            freq_ds.attrs['units'] = 'Hz'
            freq_ds.attrs['description'] = 'Gravitational wave frequency'
            
            for comp_name, comp_data in components.items():
                ds = hf.create_dataset(f'Omega_GW_h2_{comp_name}', data=comp_data, compression='gzip')
                ds.attrs['units'] = 'dimensionless'
                ds.attrs['description'] = f'GW energy density from {comp_name}'
            
            params_grp = hf.create_group('eqst_gp_parameters')
            params_grp.attrs['T_nucleation_GeV'] = eqst_params.T_n
            params_grp.attrs['alpha_PT'] = eqst_params.alpha_PT
            params_grp.attrs['beta_over_H'] = eqst_params.beta_over_H
            params_grp.attrs['v_w'] = eqst_params.v_w
            params_grp.attrs['g_star'] = eqst_params.g_star
            params_grp.attrs['m_DM_GeV'] = eqst_params.m_DM_GeV
            params_grp.attrs['f_sw_peak_Hz'] = eqst_params.f_sw_Hz
            params_grp.attrs['Omega_sw_peak_h2'] = eqst_params.Omega_sw_peak_h2
        
        print(f"GW spectrum saved to {filepath}")
        return filepath
    
    def load_gw_spectrum(self, filepath: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
        with h5py.File(filepath, 'r') as hf:
            f = hf['frequency_Hz'][:]
            
            components = {}
            for key in hf.keys():
                if key.startswith('Omega_GW_h2_'):
                    comp_name = key.replace('Omega_GW_h2_', '')
                    components[comp_name] = hf[key][:]
            
            params = {}
            if 'eqst_gp_parameters' in hf:
                for attr_name, attr_val in hf['eqst_gp_parameters'].attrs.items():
                    params[attr_name] = attr_val
        
        return f, components, params
    
    def save_mcmc_chains(self, chain: np.ndarray, log_prob: np.ndarray, param_names: List[str], filename: str = 'mcmc_chains.h5'):
        filepath = os.path.join(self.output_dir, filename)
        
        with h5py.File(filepath, 'w') as hf:
            hf.attrs['creation_date'] = datetime.now().isoformat()
            hf.attrs['n_walkers'] = chain.shape[0]
            hf.attrs['n_steps'] = chain.shape[1]
            hf.attrs['n_params'] = chain.shape[2]
            hf.attrs['param_names'] = json.dumps(param_names)
            
            hf.create_dataset('chain', data=chain, compression='gzip')
            hf.create_dataset('log_prob', data=log_prob, compression='gzip')
            
            for i, name in enumerate(param_names):
                flat_samples = chain[:, :, i].flatten()
                hf.create_dataset(f'param_{name}', data=flat_samples, compression='gzip')
        
        print(f"MCMC chains saved to {filepath}")
        return filepath
    
    def save_simulation_snapshot(self, snapshot_data: Dict, filename: str = 'simulation_snapshot.h5'):
        filepath = os.path.join(self.output_dir, filename)
        
        with h5py.File(filepath, 'w') as hf:
            hf.attrs['creation_date'] = datetime.now().isoformat()
            hf.attrs['time_Gyr'] = snapshot_data.get('time_Gyr', 0.0)
            
            if 'positions' in snapshot_data:
                pos_ds = hf.create_dataset('positions', data=snapshot_data['positions'], compression='gzip')
                pos_ds.attrs['units'] = 'meters'
            
            if 'velocities' in snapshot_data:
                vel_ds = hf.create_dataset('velocities', data=snapshot_data['velocities'], compression='gzip')
                vel_ds.attrs['units'] = 'm/s'
            
            if 'masses' in snapshot_data:
                mass_ds = hf.create_dataset('masses', data=snapshot_data['masses'], compression='gzip')
                mass_ds.attrs['units'] = 'kg'
            
            if 'density_profile' in snapshot_data:
                r_bins, rho_bins = snapshot_data['density_profile']
                hf.create_dataset('density_profile_r', data=r_bins, compression='gzip')
                hf.create_dataset('density_profile_rho', data=rho_bins, compression='gzip')
        
        print(f"Simulation snapshot saved to {filepath}")
        return filepath
