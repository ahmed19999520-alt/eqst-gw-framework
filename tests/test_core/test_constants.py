import numpy as np
import pytest
from eqst_gw.core.constants import FundamentalConstants


class TestFundamentalConstants:
    def setup_method(self):
        self.c = FundamentalConstants()

    def test_speed_of_light(self):
        assert abs(self.c.c - 2.99792458e8) < 1.0e-3

    def test_planck_mass_positive(self):
        assert self.c.M_pl_GeV > 0

    def test_planck_mass_value(self):
        assert abs(self.c.M_pl_GeV - 1.221e19) / 1.221e19 < 0.01

    def test_hubble_constant(self):
        assert 60.0 < self.c.H0_Planck2018 < 80.0

    def test_omega_m_plus_omega_lambda_approx_one(self):
        total = self.c.Omega_m_Planck2018 + self.c.Omega_Lambda_Planck2018
        assert abs(total - 1.0) < 0.01

    def test_mpc_to_m_conversion(self):
        assert abs(self.c.Mpc_to_m - 3.085677581e22) / 3.085677581e22 < 1.0e-5

    def test_year_to_seconds(self):
        assert abs(self.c.year_to_s - 31557600.0) / 31557600.0 < 1.0e-5

    def test_planck_length_positive(self):
        assert self.c.l_P_m > 0

    def test_gev_to_kg_conversion(self):
        assert abs(self.c.GeV_to_kg - 1.782661907e-27) / 1.782661907e-27 < 1.0e-6

    def test_derived_planck_mass_SI(self):
        assert self.c.M_pl_SI > 0
        assert abs(self.c.M_pl_SI - 2.176434e-8) / 2.176434e-8 < 0.01

    def test_t_CMB_positive(self):
        assert self.c.T_CMB_K > 0
        assert abs(self.c.T_CMB_K - 2.7255) < 0.01

    def test_sigma_8(self):
        assert 0.7 < self.c.sigma_8_Planck2018 < 0.95

    def test_n_s_spectral_index(self):
        assert 0.9 < self.c.n_s_Planck2018 < 1.0