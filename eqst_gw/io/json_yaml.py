import json
import yaml
import numpy as np
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        return super().default(obj)


class JSONYAMLHandler:
    def __init__(self, output_dir: str = './outputs/data/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_json(self,
                   data: Dict[str, Any],
                   filename: str,
                   indent: int = 4,
                   add_metadata: bool = True) -> str:

        if add_metadata:
            data_with_meta = {
                '_metadata': {
                    'created_at': datetime.now().isoformat(),
                    'framework': 'EQST-GP v1.0.0',
                    'author': 'Ahmed Ali',
                    'contact': 'ahmed19999520@gmail.com'
                }
            }
            data_with_meta.update(data)
        else:
            data_with_meta = data

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_with_meta, f, cls=NumpyEncoder, indent=indent, ensure_ascii=False)

        print(f"JSON file saved to {filepath}")
        return filepath

    def load_json(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def save_yaml(self,
                   data: Dict[str, Any],
                   filename: str,
                   add_metadata: bool = True) -> str:

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        data_converted = convert_numpy(data)

        if add_metadata:
            data_with_meta = {
                '_metadata': {
                    'created_at': datetime.now().isoformat(),
                    'framework': 'EQST-GP v1.0.0',
                    'author': 'Ahmed Ali',
                    'contact': 'ahmed19999520@gmail.com'
                }
            }
            data_with_meta.update(data_converted)
        else:
            data_with_meta = data_converted

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data_with_meta, f, default_flow_style=False,
                       allow_unicode=True, sort_keys=False, width=120)

        print(f"YAML file saved to {filepath}")
        return filepath

    def load_yaml(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data

    def save_eqst_gp_predictions(self, eqst_params, results_dict: Dict, filename: str = 'eqst_gp_predictions.json') -> str:
        predictions = {
            'eqst_gp_parameters': {
                'T_nucleation_GeV': eqst_params.T_n,
                'T_nucleation_err_GeV': eqst_params.T_n_err_GeV,
                'alpha_PT': eqst_params.alpha_PT,
                'alpha_PT_err': eqst_params.alpha_PT_err,
                'beta_over_H': eqst_params.beta_over_H,
                'beta_over_H_err': eqst_params.beta_over_H_err,
                'v_w': eqst_params.v_w,
                'v_w_err': eqst_params.v_w_err,
                'g_star': eqst_params.g_star,
                'm_DM_GeV': eqst_params.m_DM_GeV,
                'sigma_DM_SM_cm2': eqst_params.sigma_DM_SM_cm2,
                'chi_CY': eqst_params.chi_CY,
                'h11_CY': eqst_params.h11_CY,
                'h21_CY': eqst_params.h21_CY
            },
            'gravitational_wave_predictions': {
                'f_sw_peak_Hz': eqst_params.f_sw_Hz,
                'f_sw_peak_err_Hz': eqst_params.f_sw_Hz_err,
                'Omega_sw_peak_h2': eqst_params.Omega_sw_peak_h2,
                'Omega_sw_peak_err_h2': eqst_params.Omega_sw_peak_h2_err,
                'f_turb_peak_Hz': eqst_params.f_turb_Hz,
                'f_turb_peak_err_Hz': eqst_params.f_turb_Hz_err,
                'Omega_turb_peak_h2': eqst_params.Omega_turb_peak_h2,
                'Omega_turb_peak_err_h2': eqst_params.Omega_turb_peak_h2_err,
                'kappa_phi': eqst_params.kappa_phi,
                'kappa_v': eqst_params.kappa_v,
                'kappa_turb': eqst_params.kappa_turb
            },
            'observational_predictions': results_dict,
            'falsifiability': {
                'LISA_detection': 'Signal detectable if SNR > 5 at f ~ 1.87e-3 Hz',
                'dark_matter_null': 'DM direct detection: sigma < 1e-71 cm^2',
                'hubble_tension': 'H0 ~ 72.1 km/s/Mpc from Lambda_eff(z)',
                'alpha_EM': '1/137.036 from compactification geometry',
                'm_proton': '938.272 MeV from QCD confinement scale'
            },
            'reference': {
                'paper': 'Ali, A. (2025). Ann. Math. Phys. 8(6):273-283',
                'doi': '10.17352/amp.000126',
                'preprint': 'DOI:10.20944/preprints202601.0003.v1'
            }
        }

        return self.save_json(predictions, filename)

    def save_analysis_summary(self,
                               analysis_results: Dict,
                               filename: str = 'analysis_summary.yaml') -> str:

        summary = {
            'analysis_date': datetime.now().isoformat(),
            'framework': 'EQST-GP Gravitational Wave Analysis Framework v1.0.0',
            'key_results': {},
            'data_sources': {},
            'model_comparison': {},
            'status': {}
        }

        for key, val in analysis_results.items():
            if isinstance(val, (int, float, str, bool)):
                summary['key_results'][key] = val
            elif isinstance(val, dict):
                summary['key_results'][key] = {
                    k: v for k, v in val.items()
                    if isinstance(v, (int, float, str, bool, np.floating, np.integer))
                }

        return self.save_yaml(summary, filename)

    def merge_results_files(self,
                             file_list: List[str],
                             output_filename: str = 'merged_results.json') -> str:

        merged = {}

        for filepath in file_list:
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue

            ext = os.path.splitext(filepath)[1].lower()

            if ext == '.json':
                data = self.load_json(filepath)
            elif ext in ('.yaml', '.yml'):
                data = self.load_yaml(filepath)
            else:
                print(f"Warning: Unsupported file format: {ext}")
                continue

            key = os.path.splitext(os.path.basename(filepath))[0]
            merged[key] = data

        merged['_merge_info'] = {
            'merged_at': datetime.now().isoformat(),
            'source_files': file_list,
            'n_files_merged': len(file_list)
        }

        return self.save_json(merged, output_filename)