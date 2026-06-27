import numpy as np
from typing import Tuple, Dict, Optional, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class SystematicErrorBudget:
    def __init__(self,
                 eqst_params: Optional[EQSTGPParameters] = None,
                 constants: Optional[FundamentalConstants] = None):
        
        if eqst_params is None:
            self.ep = EQSTGPParameters()
        else:
            self.ep = eqst_params
        
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        self.systematic_budget = {}
    
    def propagate_phase_transition_uncertainties(self) -> Dict[str, Dict[str, float]]:
        from ..gravitational_waves.spectrum import GravitationalWaveSpectrum
        
        gw_fiducial = GravitationalWaveSpectrum(self.ep, self.const)
        
        f_test = np.array([self.ep.f_sw_Hz])
        Omega_fiducial = gw_fiducial.total_spectrum(f_test)[0]
        
        variations = {
            'alpha_PT': {'value': self.ep.alpha_PT, 'error': self.ep.alpha_PT_err},
            'beta_over_H': {'value': self.ep.beta_over_H, 'error': self.ep.beta_over_H_err},
            'v_w': {'value': self.ep.v_w, 'error': self.ep.v_w_err},
            'T_n': {'value': self.ep.T_n, 'error': self.ep.T_n_err}
        }
        
        sensitivities = {}
        
        for param_name, param_info in variations.items():
            ep_plus = EQSTGPParameters()
            ep_minus = EQSTGPParameters()
            
            delta = param_info['error']
            
            setattr(ep_plus, param_name, param_info['value'] + delta)
            setattr(ep_minus, param_name, param_info['value'] - delta)
            
            gw_plus = GravitationalWaveSpectrum(ep_plus, self.const)
            gw_minus = GravitationalWaveSpectrum(ep_minus, self.const)
            
            Omega_plus = gw_plus.total_spectrum(f_test)[0]
            Omega_minus = gw_minus.total_spectrum(f_test)[0]
            
            d_Omega_d_param = (Omega_plus - Omega_minus) / (2.0 * delta)
            
            sensitivity = d_Omega_d_param * param_info['error'] / Omega_fiducial
            
            sensitivities[param_name] = {
                'sensitivity': sensitivity,
                'absolute_error': abs(d_Omega_d_param * param_info['error']),
                'relative_error': abs(sensitivity)
            }
        
        total_relative_error = np.sqrt(sum([v['relative_error']**2 for v in sensitivities.values()]))
        
        sensitivities['total'] = {
            'relative_error': total_relative_error,
            'absolute_error': total_relative_error * Omega_fiducial
        }
        
        self.systematic_budget['phase_transition'] = sensitivities
        
        return sensitivities
    
    def detector_calibration_systematics(self, calibration_uncertainty_percent: float = 5.0) -> Dict:
        delta_cal = calibration_uncertainty_percent / 100.0
        
        systematics = {
            'amplitude_calibration': {
                'relative_error_Omega': 2.0 * delta_cal,
                'source': 'Detector strain amplitude calibration'
            },
            'timing': {
                'relative_error_f_peak': 0.001,
                'source': 'GPS timing accuracy'
            },
            'arm_length': {
                'relative_error_f_peak': 0.001,
                'source': 'Laser ranging accuracy'
            }
        }
        
        self.systematic_budget['detector'] = systematics
        
        return systematics
    
    def astrophysical_foreground_contamination(self) -> Dict:
        foregrounds = {
            'galactic_binaries': {
                'contamination_level': 1.0e-11,
                'frequency_range': (1.0e-4, 3.0e-3),
                'mitigation': 'Spectral fitting and subtraction'
            },
            'extragalactic_binaries': {
                'contamination_level': 1.0e-12,
                'frequency_range': (1.0e-4, 1.0e-1),
                'mitigation': 'Statistical population model subtraction'
            },
            'massive_black_hole_mergers': {
                'contamination_level': 1.0e-13,
                'frequency_range': (1.0e-5, 1.0e-2),
                'mitigation': 'Matched filter identification and removal'
            }
        }
        
        self.systematic_budget['astrophysical_foregrounds'] = foregrounds
        
        return foregrounds
    
    def cosmological_parameter_degeneracies(self) -> Dict:
        degeneracies = {
            'H0_f_peak_degeneracy': {
                'description': 'Peak frequency scales linearly with T_n/H0',
                'delta_f_peak_per_percent_H0': 0.01,
                'mitigation': 'Independent H0 constraints from BAO/SN'
            },
            'g_star_amplitude_degeneracy': {
                'description': 'Peak amplitude depends on g_star^(-1/3)',
                'delta_Omega_per_percent_g_star': -1.0/3.0 * 0.01,
                'mitigation': 'Particle physics models constrain g_star'
            }
        }
        
        self.systematic_budget['cosmological_degeneracies'] = degeneracies
        
        return degeneracies
    
    def total_systematic_budget(self) -> Dict:
        if not self.systematic_budget:
            self.propagate_phase_transition_uncertainties()
            self.detector_calibration_systematics()
            self.astrophysical_foreground_contamination()
            self.cosmological_parameter_degeneracies()
        
        pt_error = self.systematic_budget.get('phase_transition', {}).get('total', {}).get('relative_error', 0.21)
        cal_error = self.systematic_budget.get('detector', {}).get('amplitude_calibration', {}).get('relative_error_Omega', 0.10)
        
        total_systematic = np.sqrt(pt_error**2 + cal_error**2)
        
        return {
            'phase_transition_uncertainty': pt_error,
            'detector_calibration_uncertainty': cal_error,
            'total_systematic_error': total_systematic,
            'systematic_budget': self.systematic_budget
        }

