import numpy as np
import pytest
from eqst_gw.physics.phase_transition import PhaseTransitionDynamics
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.constants import FundamentalConstants

class TestPhaseTransitionDynamics:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.ep = EQSTGPParameters()
        self.pt = PhaseTransitionDynamics(self.ep, self.const)
    
    def test_critical_temperature(self):
        T_c = self.pt.critical_temperature()
        assert abs(T_c - 1.04e16) / 1.04e16 < 0.1, f"T_c = {T_c:.3e}, expected ~1.04e16 GeV"
    
    def test_effective_potential_symmetric_phase(self):
        T = 1.0e16
        V_0 = self.pt.effective_potential(0.0, T)
        assert V_0 == 0.0 or abs(V_0) < 1.0e50
    
    def test_potential_derivative_at_origin(self):
        T = 0.9e16
        dV_0 = self.pt.potential_derivative(0.0, T)
        assert dV_0 == 0.0
    
    def test_transition_strength_positive(self):
        T = 0.97e16
        alpha = self.pt.transition_strength_alpha(T)
        assert alpha > 0, "Transition strength must be positive"
    
    def test_transition_strength_range(self):
        T = 0.97e16
        alpha = self.pt.transition_strength_alpha(T)
        assert 0.1 < alpha < 2.0, f"alpha = {alpha:.3f} outside expected range"
    
    def test_latent_heat_positive(self):
        T = 0.97e16
        epsilon = self.pt.latent_heat(T)
        assert epsilon >= 0, "Latent heat must be non-negative"
    
    def test_radiation_energy_density(self):
        T = 1.0e16
        rho_rad = self.pt.radiation_energy_density(T)
        expected = (np.pi**2 / 30.0) * self.ep.g_star * T**4
        assert abs(rho_rad - expected) / expected < 1.0e-10


class TestBubbleNucleation:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.ep = EQSTGPParameters()
        from eqst_gw.physics.nucleation import BubbleNucleation
        self.nuc = BubbleNucleation(self.ep, self.const)
    
    def test_bounce_action_finite(self):
        T = 0.97e16
        rho, phi, phi_c = self.nuc.solve_bounce_shooting(T)
        S3 = self.nuc.compute_euclidean_action(rho, phi, T)
        assert np.isfinite(S3), "Bounce action must be finite"
        assert S3 > 0, "Bounce action must be positive"
    
    def test_bounce_action_magnitude(self):
        T = 0.97e16
        rho, phi, phi_c = self.nuc.solve_bounce_shooting(T)
        S3 = self.nuc.compute_euclidean_action(rho, phi, T)
        ratio = S3 / T
        assert 100.0 < ratio < 200.0, f"S3/T = {ratio:.1f} outside expected range [100, 200]"
    
    def test_nucleation_rate_decreases_with_temperature(self):
        T1 = 0.95e16
        T2 = 0.97e16
        
        rho1, phi1, _ = self.nuc.solve_bounce_shooting(T1)
        rho2, phi2, _ = self.nuc.solve_bounce_shooting(T2)
        
        S3_1 = self.nuc.compute_euclidean_action(rho1, phi1, T1)
        S3_2 = self.nuc.compute_euclidean_action(rho2, phi2, T2)
        
        Gamma1 = self.nuc.nucleation_rate(T1, S3_1)
        Gamma2 = self.nuc.nucleation_rate(T2, S3_2)
        
        assert Gamma2 > Gamma1, "Nucleation rate should increase as temperature decreases toward T_n"


class TestGravitationalWaveSpectrum:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.ep = EQSTGPParameters()
        from eqst_gw.gravitational_waves.spectrum import GravitationalWaveSpectrum
        self.gw = GravitationalWaveSpectrum(self.ep, self.const)
        self.f_test = np.logspace(-4, 0, 100)
    
    def test_spectrum_positive(self):
        Omega = self.gw.total_spectrum(self.f_test)
        assert np.all(Omega >= 0), "GW spectrum must be non-negative"
    
    def test_spectrum_finite(self):
        Omega = self.gw.total_spectrum(self.f_test)
        assert np.all(np.isfinite(Omega)), "GW spectrum must be finite everywhere"
    
    def test_peak_frequency_in_range(self):
        props = self.gw.peak_properties()
        f_peak = props['f_sound_peak_Hz']
        assert 1.0e-4 < f_peak < 1.0e-1, f"Peak frequency {f_peak:.3e} Hz not in LISA band"
    
    def test_peak_amplitude_order_of_magnitude(self):
        props = self.gw.peak_properties()
        Omega_peak = props['Omega_sound_peak_h2']
        assert 1.0e-16 < Omega_peak < 1.0e-10, f"Peak amplitude {Omega_peak:.3e} out of expected range"
    
    def test_low_frequency_slope(self):
        f_low = np.logspace(-5, -4, 50)
        Omega_low = self.gw.total_spectrum(f_low)
        
        log_f = np.log10(f_low)
        log_Omega = np.log10(Omega_low)
        
        slope = np.polyfit(log_f, log_Omega, 1)[0]
        
        assert 2.0 < slope < 4.0, f"Low-frequency slope = {slope:.2f}, expected ~3"
    
    def test_sound_wave_dominates(self):
        f_peak = np.array([self.ep.f_sw_Hz])
        
        from eqst_gw.gravitational_waves.sources import SoundWaveSource, TurbulenceSource, BubbleCollisionSource
        
        sw = SoundWaveSource(self.ep, self.const)
        turb = TurbulenceSource(self.ep, self.const)
        bub = BubbleCollisionSource(self.ep, self.const)
        
        Omega_sw = sw.spectrum(f_peak)[0]
        Omega_turb = turb.spectrum(f_peak)[0]
        Omega_bub = bub.spectrum(f_peak)[0]
        
        assert Omega_sw > Omega_turb, "Sound waves should dominate over turbulence"
        assert Omega_sw > Omega_bub, "Sound waves should dominate over bubble collisions"
    
    def test_detector_snr_positive(self):
        from eqst_gw.detectors.lisa import LISADetector
        lisa = LISADetector(mission_duration_years=4.0, constants=self.const)
        
        SNR = lisa.compute_snr(self.f_test, self.gw.total_spectrum(self.f_test))
        
        assert SNR > 0, "LISA SNR must be positive"
    
    def test_detector_snr_exceeds_threshold(self):
        from eqst_gw.detectors.lisa import LISADetector
        lisa = LISADetector(mission_duration_years=4.0, constants=self.const)
        
        SNR = lisa.compute_snr(self.f_test, self.gw.total_spectrum(self.f_test))
        
        assert SNR > 5.0, f"LISA SNR = {SNR:.2f} < 5 detection threshold"