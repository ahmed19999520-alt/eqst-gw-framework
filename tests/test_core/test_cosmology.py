import numpy as np
import pytest
from eqst_gw.core.cosmology import LambdaEffectiveCosmology
from eqst_gw.core.constants import FundamentalConstants


class TestLambdaEffectiveCosmology:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.cosmo = LambdaEffectiveCosmology(self.const)

    def test_lambda_eff_at_z0_positive(self):
        L = self.cosmo.Lambda_eff(0.0)
        assert L > 0

    def test_lambda_eff_at_z0_approx_lambda_cdm(self):
        L = self.cosmo.Lambda_eff(0.0)
        assert abs(L - self.const.Omega_Lambda_Planck2018) / self.const.Omega_Lambda_Planck2018 < 0.5

    def test_H_eff_at_z0_approx_H0(self):
        H = self.cosmo.H_eff(0.0)
        assert abs(H - self.const.H0_Planck2018) / self.const.H0_Planck2018 < 0.5

    def test_H_eff_increases_with_z(self):
        H_0 = self.cosmo.H_eff(0.0)
        H_1 = self.cosmo.H_eff(1.0)
        H_5 = self.cosmo.H_eff(5.0)
        assert H_1 > H_0
        assert H_5 > H_1

    def test_comoving_distance_positive(self):
        d = self.cosmo.comoving_distance(1.0)
        assert d > 0

    def test_comoving_distance_increases_with_z(self):
        d1 = self.cosmo.comoving_distance(0.5)
        d2 = self.cosmo.comoving_distance(1.0)
        assert d2 > d1

    def test_luminosity_distance_greater_than_comoving(self):
        z = 1.0
        d_L = self.cosmo.luminosity_distance(z)
        d_C = self.cosmo.comoving_distance(z)
        assert d_L > d_C

    def test_age_of_universe_positive(self):
        age = self.cosmo.age_of_universe()
        assert age > 0

    def test_age_of_universe_approx_range(self):
        age = self.cosmo.age_of_universe()
        assert 10.0 < age < 20.0

    def test_temperature_from_redshift(self):
        T_z0 = self.cosmo.temperature_from_redshift(0.0)
        T_z1 = self.cosmo.temperature_from_redshift(1.0)
        assert T_z1 > T_z0

    def test_R_factor_at_z_pivot(self):
        R = self.cosmo.R_factor(self.cosmo.z_pivot - 1.0)
        assert np.isfinite(R)

    def test_F_QCD_factor_range(self):
        F = self.cosmo.F_QCD_factor(np.array([0.0, 1.0, 5.0, 10.0]))
        assert np.all(F >= 0)
        assert np.all(F <= 1.0)

    def test_derived_parameters_complete(self):
        params = self.cosmo.compute_derived_parameters()
        assert 'rho_crit_kg_m3' in params
        assert 'age_Gyr' in params
        assert 'z_eq' in params
        assert params['rho_crit_kg_m3'] > 0