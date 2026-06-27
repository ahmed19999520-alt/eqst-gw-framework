import numpy as np
import pytest
from eqst_gw.detectors.lisa import LISADetector
from eqst_gw.detectors.ligo import LIGODetector
from eqst_gw.detectors.virgo import VirgoDetector
from eqst_gw.detectors.kagra import KAGRADetector
from eqst_gw.detectors.einstein_telescope import EinsteinTelescopeDetector
from eqst_gw.detectors.sensitivity_curves import MultiDetectorNetwork
from eqst_gw.core.constants import FundamentalConstants


class TestDetectors:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.lisa = LISADetector(mission_duration_years=4.0, constants=self.const)
        self.ligo = LIGODetector(design='O4', constants=self.const)
        self.virgo = VirgoDetector(design='O4', constants=self.const)
        self.kagra = KAGRADetector(design='O4', constants=self.const)
        self.et = EinsteinTelescopeDetector(design='ET-D', constants=self.const)

    def test_lisa_noise_psd_positive(self):
        f = np.logspace(-4, -1, 50)
        S_n = self.lisa.noise_psd(f)
        assert np.all(S_n > 0)

    def test_ligo_noise_psd_positive_in_band(self):
        f = np.logspace(1, 3, 50)
        S_n = self.ligo.noise_psd(f)
        assert np.all(S_n > 0)

    def test_et_noise_psd_lower_than_ligo(self):
        f = np.array([100.0])
        S_lisa_100 = self.ligo.noise_psd(f)[0]
        S_et_100 = self.et.noise_psd(f)[0]
        assert S_et_100 < S_lisa_100

    def test_lisa_sensitivity_finite(self):
        f = np.logspace(-4, -1, 50)
        Omega_sens = self.lisa.omega_sensitivity(f)
        assert np.all(np.isfinite(Omega_sens))

    def test_snr_positive(self):
        f = np.logspace(-4, -1, 200)
        Omega_signal = 1.0e-11 * np.ones_like(f)
        snr = self.lisa.compute_snr(f, Omega_signal)
        assert snr >= 0

    def test_detection_threshold_range(self):
        threshold = self.lisa.detection_threshold(false_alarm_probability=1.0e-3)
        assert 2.0 < threshold < 10.0

    def test_multi_detector_network_initialization(self):
        network = MultiDetectorNetwork(self.const)
        network.initialize_standard_network()
        assert 'LISA' in network.detectors
        assert 'ET' in network.detectors

    def test_combined_sensitivity_finite(self):
        network = MultiDetectorNetwork(self.const)
        network.initialize_standard_network()
        f = np.logspace(-4, 3, 200)
        S_combined = network.combined_sensitivity_curve(f)
        assert np.all(np.isfinite(S_combined))

    def test_frequency_coverage_correct(self):
        network = MultiDetectorNetwork(self.const)
        network.initialize_standard_network()
        coverage = network.optimal_frequency_coverage()
        assert coverage['LISA'][0] < coverage['LISA'][1]
        assert coverage['ET'][0] < coverage['ET'][1]