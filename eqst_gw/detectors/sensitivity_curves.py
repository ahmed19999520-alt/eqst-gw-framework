import numpy as np
from typing import Dict, List, Optional
from .lisa import LISADetector
from .ligo import LIGODetector
from .virgo import VirgoDetector
from .kagra import KAGRADetector
from .einstein_telescope import EinsteinTelescopeDetector
from ..core.constants import FundamentalConstants

class MultiDetectorNetwork:
    def __init__(self, constants: Optional[FundamentalConstants] = None):
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        self.detectors = {}
    
    def add_detector(self, detector_name: str, detector_instance):
        self.detectors[detector_name] = detector_instance
    
    def initialize_standard_network(self):
        self.add_detector('LISA', LISADetector(mission_duration_years=4.0, constants=self.const))
        self.add_detector('LIGO-H1', LIGODetector(design='O4', detector_name='H1', constants=self.const))
        self.add_detector('LIGO-L1', LIGODetector(design='O4', detector_name='L1', constants=self.const))
        self.add_detector('Virgo', VirgoDetector(design='O4', constants=self.const))
        self.add_detector('KAGRA', KAGRADetector(design='O4', constants=self.const))
        self.add_detector('ET', EinsteinTelescopeDetector(design='ET-D', constants=self.const))
    
    def compute_network_snr(self, f: np.ndarray, h_signal: np.ndarray, detector_list: Optional[List[str]] = None) -> Dict[str, float]:
        if detector_list is None:
            detector_list = list(self.detectors.keys())
        
        snr_dict = {}
        
        for det_name in detector_list:
            if det_name in self.detectors:
                detector = self.detectors[det_name]
                snr = detector.optimal_matched_filter_snr(f, h_signal)
                snr_dict[det_name] = snr
        
        network_snr = np.sqrt(sum([snr**2 for snr in snr_dict.values()]))
        snr_dict['network'] = network_snr
        
        return snr_dict
    
    def optimal_frequency_coverage(self) -> Dict[str, Tuple[float, float]]:
        coverage = {}
        
        for det_name, detector in self.detectors.items():
            coverage[det_name] = (detector.f_min, detector.f_max)
        
        return coverage
    
    def combined_sensitivity_curve(self, f: np.ndarray, detector_list: Optional[List[str]] = None) -> np.ndarray:
        if detector_list is None:
            detector_list = list(self.detectors.keys())
        
        f_array = np.atleast_1d(f)
        
        combined_inv_variance = np.zeros_like(f_array)
        
        for det_name in detector_list:
            if det_name in self.detectors:
                detector = self.detectors[det_name]
                S_n = detector.noise_psd(f_array)
                
                mask = (f_array >= detector.f_min) & (f_array <= detector.f_max)
                combined_inv_variance[mask] += 1.0 / S_n[mask]
        
        S_combined = 1.0 / combined_inv_variance
        S_combined[combined_inv_variance == 0] = np.inf
        
        return S_combined if f_array.size > 1 else S_combined[0]
