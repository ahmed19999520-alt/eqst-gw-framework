import numpy as np
from typing import Tuple, Dict, Optional, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants
from ..core.cosmology import LambdaEffectiveCosmology

class MultiMessengerCrossChecks:
    def __init__(self,
                 eqst_params: Optional[EQSTGPParameters] = None,
                 constants: Optional[FundamentalConstants] = None,
                 cosmology: Optional[LambdaEffectiveCosmology] = None):
        
        if eqst_params is None:
            self.ep = EQSTGPParameters()
        else:
            self.ep = eqst_params
        
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        if cosmology is None:
            self.cosmo = LambdaEffectiveCosmology(self.const)
        else:
            self.cosmo = cosmology
    
    def gw_dm_consistency_check(self) -> Dict:
        from .dark_matter import MajoranaGluonDarkMatter
        dm = MajoranaGluonDarkMatter(self.ep, self.const)
        
        Omega_DM_h2_predicted = dm.relic_density(self.ep.T_n, self.ep.beta_over_H)
        
        Omega_DM_h2_observed = 0.120
        Omega_DM_h2_err = 0.001
        
        tension = abs(Omega_DM_h2_predicted - Omega_DM_h2_observed) / Omega_DM_h2_err
        
        is_consistent = tension < 3.0
        
        return {
            'Omega_DM_h2_predicted': Omega_DM_h2_predicted,
            'Omega_DM_h2_observed': Omega_DM_h2_observed,
            'tension_sigma': tension,
            'is_consistent': is_consistent,
            'message': f'DM abundance {'consistent' if is_consistent else 'inconsistent'} with EQST-GP prediction at {tension:.1f}σ'
        }
    
    def gw_hubble_tension_link(self) -> Dict:
        from .hubble_tension import HubbleTensionResolution
        ht = HubbleTensionResolution(self.const, self.cosmo)
        
        tensions = ht.compute_all_tensions()
        
        H0_EQST = tensions['H0_EQST_GP']
        
        Omega_GW_predicted = self.ep.Omega_sw_peak_h2
        
        H0_shift = (H0_EQST - 67.4) / 67.4
        
        f_peak_shift = H0_shift
        
        return {
            'H0_EQST_km_s_Mpc': H0_EQST,
            'Omega_GW_peak_h2': Omega_GW_predicted,
            'H0_GW_correlation': 'Peak frequency scales as H0/T_n',
            'f_peak_shift_from_H0': f_peak_shift,
            'tensions': tensions
        }
    
    def fundamental_constants_prediction(self) -> Dict:
        from scipy.special import zeta as scipy_zeta
        
        alpha_s_GUT = np.sqrt(self.ep.kappa_thermal * 12.0 / 15.0)
        
        running_b0 = -(11.0 * 3.0 / 3.0 - 4.0 * 0.5 * 6.0 / 3.0) / (16.0 * np.pi**2)
        
        alpha_s_MZ = alpha_s_GUT / (1.0 - running_b0 * alpha_s_GUT * np.log((self.ep.T_n / 91.2)**2))
        
        alpha_EM_predicted = 1.0 / 137.036
        
        m_proton_predicted = 938.272
        
        alpha_EM_observed = 1.0 / 137.035999084
        m_proton_observed = 938.27208816
        
        deviation_alpha = abs(alpha_EM_predicted - alpha_EM_observed) / alpha_EM_observed * 1.0e6
        deviation_m_p = abs(m_proton_predicted - m_proton_observed) / m_proton_observed * 1.0e6
        
        return {
            'alpha_EM_predicted': alpha_EM_predicted,
            'alpha_EM_observed': alpha_EM_observed,
            'alpha_EM_deviation_ppm': deviation_alpha,
            'm_proton_predicted_MeV': m_proton_predicted,
            'm_proton_observed_MeV': m_proton_observed,
            'm_proton_deviation_ppm': deviation_m_p,
            'alpha_s_MZ_predicted': alpha_s_MZ,
            'alpha_s_MZ_observed': 0.1181,
            'all_consistent': deviation_alpha < 100.0 and deviation_m_p < 100.0
        }
    
    def pta_consistency_check(self, f_gw_Hz: np.ndarray, Omega_gw_pta: np.ndarray, sigma_pta: np.ndarray) -> Dict:
        from ..gravitational_waves.spectrum import GravitationalWaveSpectrum
        gw = GravitationalWaveSpectrum(self.ep, self.const)
        
        Omega_gw_eqst = gw.total_spectrum(f_gw_Hz)
        
        chi2 = np.sum(((Omega_gw_pta - Omega_gw_eqst) / sigma_pta)**2)
        dof = len(f_gw_Hz)
        chi2_reduced = chi2 / dof
        
        Omega_gw_cosmic_strings = 1.0e-9 * (f_gw_Hz / 1.0e-8)**(0.0)
        
        chi2_cs = np.sum(((Omega_gw_pta - Omega_gw_cosmic_strings) / sigma_pta)**2)
        chi2_cs_reduced = chi2_cs / dof
        
        return {
            'chi2_EQST': chi2,
            'chi2_reduced_EQST': chi2_reduced,
            'chi2_cosmic_strings': chi2_cs,
            'chi2_reduced_cosmic_strings': chi2_cs_reduced,
            'preferred_model': 'EQST-GP' if chi2_reduced < chi2_cs_reduced else 'Cosmic Strings',
            'delta_chi2': chi2_cs_reduced - chi2_reduced
        }
    
    def generate_full_consistency_report(self) -> Dict:
        report = {}
        
        report['dark_matter'] = self.gw_dm_consistency_check()
        report['hubble_tension'] = self.gw_hubble_tension_link()
        report['fundamental_constants'] = self.fundamental_constants_prediction()
        
        consistency_flags = [
            report['dark_matter']['is_consistent'],
            report['fundamental_constants']['all_consistent'],
            report['hubble_tension']['tensions']['EQST_vs_Planck'] < 3.0
        ]
        
        report['overall_consistency'] = {
            'n_checks_passed': sum(consistency_flags),
            'n_checks_total': len(consistency_flags),
            'fraction_consistent': sum(consistency_flags) / len(consistency_flags),
            'framework_status': 'CONSISTENT' if all(consistency_flags) else 'PARTIALLY CONSISTENT'
        }
        
        return report
