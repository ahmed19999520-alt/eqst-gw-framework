import numpy as np
import pytest
from eqst_gw.core.parameters import EQSTGPParameters


class TestEQSTGPParameters:
    def setup_method(self):
        self.ep = EQSTGPParameters()

    def test_chi_cy_value(self):
        assert self.ep.chi_CY == -960.0

    def test_chi_cy_error_positive(self):
        assert self.ep.chi_CY_err > 0

    def test_hodge_numbers_positive(self):
        assert self.ep.h11_CY > 0
        assert self.ep.h21_CY > 0

    def test_euler_characteristic_from_hodge(self):
        chi_computed = 2 * (self.ep.h11_CY - self.ep.h21_CY)
        assert abs(chi_computed - self.ep.chi_CY) < 100

    def test_critical_temperature_positive(self):
        assert self.ep.T_c_GeV > 0

    def test_nucleation_temperature_less_than_critical(self):
        assert self.ep.T_n_GeV < self.ep.T_c_GeV

    def test_alpha_PT_range(self):
        assert 0.0 < self.ep.alpha_PT < 5.0

    def test_beta_over_H_positive(self):
        assert self.ep.beta_over_H > 0

    def test_v_w_subluminal(self):
        assert 0.0 < self.ep.v_w < 1.0

    def test_g_star_positive(self):
        assert self.ep.g_star > 0

    def test_m_DM_GUT_scale(self):
        assert 1.0e14 < self.ep.m_DM_GeV < 1.0e18

    def test_sigma_DM_SM_tiny(self):
        assert self.ep.sigma_DM_SM_cm2 < 1.0e-60

    def test_peak_frequency_mhz_band(self):
        assert 1.0e-4 < self.ep.f_sw_Hz < 1.0e-1

    def test_peak_amplitude_positive(self):
        assert self.ep.Omega_sw_peak_h2 > 0

    def test_kappa_phi_small(self):
        assert self.ep.kappa_phi < 0.1

    def test_kappa_v_range(self):
        assert 0.0 < self.ep.kappa_v < 1.0

    def test_lambda_quartic_positive(self):
        assert self.ep.lambda_quartic > 0

    def test_to_dict_returns_dict(self):
        d = self.ep.to_dict()
        assert isinstance(d, dict)
        assert 'alpha_PT' in d

    def test_sample_parameters_shape(self):
        samples = self.ep.sample_parameters(n_samples=100)
        assert 'alpha_PT' in samples
        assert len(samples['alpha_PT']) == 100

    def test_sample_parameters_within_range(self):
        samples = self.ep.sample_parameters(n_samples=1000)
        assert np.all(samples['v_w'] > 0)
        assert np.all(samples['v_w'] < 1.0)

    def test_metadata_present(self):
        assert 'version' in self.ep.metadata
        assert 'reference' in self.ep.metadata