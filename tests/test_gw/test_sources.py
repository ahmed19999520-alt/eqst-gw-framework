import numpy as np
import pytest
from eqst_gw.gravitational_waves.sources import BubbleCollisionSource, SoundWaveSource, TurbulenceSource
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.constants import FundamentalConstants


class TestGravitationalWaveSources:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.ep = EQSTGPParameters()
        self.bubble = BubbleCollisionSource(self.ep, self.const)
        self.sound = SoundWaveSource(self.ep, self.const)
        self.turb = TurbulenceSource(self.ep, self.const)
        self.f_test = np.logspace(-5, 0, 50)

    def test_bubble_efficiency_positive(self):
        kappa = self.bubble.efficiency_factor()
        assert kappa > 0

    def test_sound_efficiency_range(self):
        kappa = self.sound.efficiency_factor()
        assert 0.0 < kappa < 1.0

    def test_turb_efficiency_less_than_sound(self):
        kappa_sw = self.sound.efficiency_factor()
        kappa_turb = self.turb.efficiency_factor()
        assert kappa_turb < kappa_sw

    def test_bubble_spectrum_positive(self):
        Omega = self.bubble.spectrum(self.f_test)
        assert np.all(Omega >= 0)

    def test_sound_spectrum_positive(self):
        Omega = self.sound.spectrum(self.f_test)
        assert np.all(Omega >= 0)

    def test_turb_spectrum_positive(self):
        Omega = self.turb.spectrum(self.f_test)
        assert np.all(Omega >= 0)

    def test_sound_dominates_over_bubble(self):
        f_peak = np.array([self.ep.f_sw_Hz])
        assert self.sound.spectrum(f_peak)[0] > self.bubble.spectrum(f_peak)[0]

    def test_sound_dominates_over_turbulence(self):
        f_peak = np.array([self.ep.f_sw_Hz])
        assert self.sound.spectrum(f_peak)[0] > self.turb.spectrum(f_peak)[0]

    def test_bubble_peak_frequency_lisa_band(self):
        f_peak = self.bubble.peak_frequency_today(self.ep.beta_over_H, self.ep.v_w)
        assert 1.0e-5 < f_peak < 1.0e-1

    def test_sound_peak_frequency_lisa_band(self):
        f_peak = self.sound.peak_frequency_today()
        assert 1.0e-4 < f_peak < 1.0e-1

    def test_spectral_shape_normalised(self):
        f_peak = np.array([self.ep.f_sw_Hz])
        S_at_peak = self.sound.spectral_shape(f_peak)[0]
        assert abs(S_at_peak - 1.0) < 0.1