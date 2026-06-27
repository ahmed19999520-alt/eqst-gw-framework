import numpy as np
import os
import h5py
import json
import yaml
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from .hdf5_handler import HDF5Handler
from .fits_handler import FITSHandler
from .json_yaml import JSONYAMLHandler, NumpyEncoder


class UnifiedExporter:
    def __init__(self, output_dir: str = './outputs/data/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.hdf5 = HDF5Handler(output_dir)
        self.fits = FITSHandler(output_dir)
        self.json_yaml = JSONYAMLHandler(output_dir)

    def export_complete_analysis(self,
                                  f_array: np.ndarray,
                                  gw_components: Dict[str, np.ndarray],
                                  eqst_params,
                                  mcmc_chain: Optional[np.ndarray],
                                  param_names: Optional[List[str]],
                                  bao_results: Dict,
                                  sn_results: Dict,
                                  hubble_results: Dict,
                                  multimessenger_report: Dict,
                                  run_id: str = None) -> Dict[str, str]:

        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        exported_files = {}

        print(f"\nExporting complete analysis results (Run ID: {run_id})")

        gw_total = gw_components.get('total', sum(gw_components.values()))

        hdf5_file = self.hdf5.save_gw_spectrum(
            f_array, gw_components, eqst_params,
            filename=f'gw_spectrum_{run_id}.h5'
        )
        exported_files['hdf5_spectrum'] = hdf5_file

        fits_spectrum = self.fits.save_gw_spectrum_fits(
            f_array, gw_total, gw_components, eqst_params,
            filename=f'gw_spectrum_{run_id}.fits'
        )
        exported_files['fits_spectrum'] = fits_spectrum

        if mcmc_chain is not None and param_names is not None:
            n_walkers, n_steps, n_dim = mcmc_chain.shape
            flat_chain = mcmc_chain.reshape(-1, n_dim)
            log_prob_mock = np.zeros(len(flat_chain))

            hdf5_mcmc = self.hdf5.save_mcmc_chains(
                mcmc_chain, log_prob_mock, param_names,
                filename=f'mcmc_chains_{run_id}.h5'
            )
            exported_files['hdf5_mcmc'] = hdf5_mcmc

            fits_mcmc = self.fits.save_mcmc_posterior_fits(
                flat_chain, param_names, log_prob_mock,
                filename=f'mcmc_posterior_{run_id}.fits'
            )
            exported_files['fits_mcmc'] = fits_mcmc

        combined_results = {
            'run_id': run_id,
            'bao_fit': bao_results,
            'sn_fit': sn_results,
            'hubble_tension': hubble_results,
            'multimessenger': multimessenger_report
        }

        json_file = self.json_yaml.save_eqst_gp_predictions(
            eqst_params, combined_results,
            filename=f'eqst_gp_predictions_{run_id}.json'
        )
        exported_files['json_predictions'] = json_file

        yaml_file = self.json_yaml.save_analysis_summary(
            combined_results,
            filename=f'analysis_summary_{run_id}.yaml'
        )
        exported_files['yaml_summary'] = yaml_file

        index_file = self._write_index_file(exported_files, run_id)
        exported_files['index'] = index_file

        print(f"\nAll exports complete. Files written to: {self.output_dir}")
        print(f"Index file: {index_file}")

        return exported_files

    def _write_index_file(self, exported_files: Dict[str, str], run_id: str) -> str:
        index = {
            'run_id': run_id,
            'created_at': datetime.now().isoformat(),
            'output_directory': self.output_dir,
            'files': {k: os.path.basename(v) for k, v in exported_files.items()},
            'file_descriptions': {
                'hdf5_spectrum': 'GW spectrum in HDF5 format with full component decomposition',
                'fits_spectrum': 'GW spectrum in FITS format for astronomical tools',
                'hdf5_mcmc': 'Full MCMC chain in HDF5 format',
                'fits_mcmc': 'MCMC posterior samples in FITS format',
                'json_predictions': 'Complete EQST-GP predictions in JSON format',
                'yaml_summary': 'Analysis summary in YAML format'
            }
        }

        filepath = os.path.join(self.output_dir, f'analysis_index_{run_id}.json')
        with open(filepath, 'w') as f:
            json.dump(index, f, cls=NumpyEncoder, indent=4)

        return filepath

    def export_for_lisa_data_center(self,
                                     f_array: np.ndarray,
                                     Omega_gw: np.ndarray,
                                     eqst_params,
                                     snr: float,
                                     filename: str = 'lisa_submission.h5') -> str:

        filepath = os.path.join(self.output_dir, filename)

        with h5py.File(filepath, 'w') as hf:
            hf.attrs['format'] = 'LISA-SGWB-Template-v1'
            hf.attrs['model_name'] = 'EQST-GP Topological Phase Transition'
            hf.attrs['model_class'] = 'First-Order Phase Transition'
            hf.attrs['author'] = 'Ahmed Ali'
            hf.attrs['contact'] = 'ahmed19999520@gmail.com'
            hf.attrs['arxiv'] = '2025.xxxxx'
            hf.attrs['doi'] = '10.17352/amp.000126'
            hf.attrs['submission_date'] = datetime.now().isoformat()
            hf.attrs['expected_SNR'] = snr
            hf.attrs['peak_frequency_Hz'] = eqst_params.f_sw_Hz
            hf.attrs['peak_amplitude_h2'] = eqst_params.Omega_sw_peak_h2

            signal_grp = hf.create_group('signal')
            signal_grp.create_dataset('frequency', data=f_array,
                                       compression='gzip')
            signal_grp['frequency'].attrs['units'] = 'Hz'

            signal_grp.create_dataset('Omega_GW_h2', data=Omega_gw,
                                       compression='gzip')
            signal_grp['Omega_GW_h2'].attrs['units'] = 'dimensionless'
            signal_grp['Omega_GW_h2'].attrs['description'] = 'Total GW energy density parameter'

            params_grp = hf.create_group('model_parameters')
            params_grp.create_dataset('parameter_names',
                                       data=np.array(['alpha_PT', 'beta_over_H', 'v_w',
                                                      'T_n_GeV', 'g_star'], dtype='S20'))
            params_grp.create_dataset('parameter_values',
                                       data=np.array([eqst_params.alpha_PT,
                                                      eqst_params.beta_over_H,
                                                      eqst_params.v_w,
                                                      eqst_params.T_n,
                                                      eqst_params.g_star]))
            params_grp.create_dataset('parameter_errors',
                                       data=np.array([eqst_params.alpha_PT_err,
                                                      eqst_params.beta_over_H_err,
                                                      eqst_params.v_w_err,
                                                      eqst_params.T_n_err_GeV,
                                                      eqst_params.g_star_err]))

            spectral_grp = hf.create_group('spectral_properties')
            spectral_grp.attrs['f_peak_sound_Hz'] = eqst_params.f_sw_Hz
            spectral_grp.attrs['f_peak_turb_Hz'] = eqst_params.f_turb_Hz
            spectral_grp.attrs['Omega_peak_sound_h2'] = eqst_params.Omega_sw_peak_h2
            spectral_grp.attrs['Omega_peak_turb_h2'] = eqst_params.Omega_turb_peak_h2
            spectral_grp.attrs['low_freq_slope'] = 3.0
            spectral_grp.attrs['high_freq_slope'] = -4.0

            multimessenger_grp = hf.create_group('multimessenger_links')
            multimessenger_grp.attrs['DM_mass_GeV'] = eqst_params.m_DM_GeV
            multimessenger_grp.attrs['DM_cross_section_cm2'] = eqst_params.sigma_DM_SM_cm2
            multimessenger_grp.attrs['H0_predicted_km_s_Mpc'] = 72.1
            multimessenger_grp.attrs['Omega_DM_h2_predicted'] = 0.120
            multimessenger_grp.attrs['alpha_EM_predicted'] = 1.0 / 137.036
            multimessenger_grp.attrs['m_proton_predicted_MeV'] = 938.272

        print(f"LISA Data Center submission file saved to {filepath}")
        return filepath

    def export_for_gwosc(self,
                          gw_event_data: List[Dict],
                          filename: str = 'eqst_gp_gwosc_analysis.json') -> str:

        gwosc_format = {
            'catalog': 'EQST-GP Background Analysis',
            'version': '1.0.0',
            'analysis_date': datetime.now().isoformat(),
            'pipeline': 'EQST-GP Framework',
            'description': 'Analysis of SGWB signal from EQST-GP topological phase transition',
            'events': gw_event_data,
            'background_model': {
                'type': 'stochastic',
                'origin': 'topological_phase_transition',
                'peak_frequency_Hz': 1.87e-3,
                'peak_amplitude_Omega_h2': 6.31e-14,
                'model_reference': 'Ali (2025) Ann. Math. Phys. 8(6):273-283'
            }
        }

        return self.json_yaml.save_json(gwosc_format, filename)

    def export_rotation_curve_database(self,
                                        galaxy_catalog: List[Dict],
                                        filename: str = 'rotation_curve_database.h5') -> str:

        filepath = os.path.join(self.output_dir, filename)

        with h5py.File(filepath, 'w') as hf:
            hf.attrs['description'] = 'Galaxy Rotation Curve Database: EQST-GP Predictions vs Observations'
            hf.attrs['created_at'] = datetime.now().isoformat()
            hf.attrs['n_galaxies'] = len(galaxy_catalog)

            for i, galaxy in enumerate(galaxy_catalog):
                gal_grp = hf.create_group(f'galaxy_{i:04d}')

                for key, val in galaxy.items():
                    if isinstance(val, np.ndarray):
                        gal_grp.create_dataset(key, data=val, compression='gzip')
                    elif isinstance(val, (int, float, str)):
                        gal_grp.attrs[key] = val

        print(f"Rotation curve database saved to {filepath}")
        return filepath

    def export_cluster_simulation_suite(self,
                                         simulations: List[Dict],
                                         filename: str = 'cluster_simulation_suite.h5') -> str:

        filepath = os.path.join(self.output_dir, filename)

        with h5py.File(filepath, 'w') as hf:
            hf.attrs['description'] = 'Galaxy Cluster Merger Simulation Suite: EQST-GP DM Profiles'
            hf.attrs['created_at'] = datetime.now().isoformat()
            hf.attrs['n_simulations'] = len(simulations)
            hf.attrs['DM_model'] = 'Majorana Gluon DM (EQST-GP)'
            hf.attrs['simulation_code'] = 'EQST-GP N-body Framework v1.0.0'

            for i, sim in enumerate(simulations):
                sim_grp = hf.create_group(f'simulation_{i:04d}')

                for key, val in sim.items():
                    if isinstance(val, np.ndarray):
                        sim_grp.create_dataset(key, data=val, compression='gzip')
                    elif isinstance(val, (int, float, str, bool)):
                        sim_grp.attrs[key] = val
                    elif isinstance(val, dict):
                        sub_grp = sim_grp.create_group(key)
                        for sk, sv in val.items():
                            if isinstance(sv, np.ndarray):
                                sub_grp.create_dataset(sk, data=sv, compression='gzip')
                            elif isinstance(sv, (int, float, str)):
                                sub_grp.attrs[sk] = sv

        print(f"Cluster simulation suite saved to {filepath}")
        return filepath