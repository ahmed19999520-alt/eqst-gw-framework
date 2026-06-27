import numpy as np
import pytest
from eqst_gw.simulations.dark_matter_halos import DMHaloProfile
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.constants import FundamentalConstants


class TestDMHaloProfile:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.ep = EQSTGPParameters()
        self.halo = DMHaloProfile(M_vir_solar=1.0e12, z=0.0, eqst_params=self.ep, constants=self.const)

    def test_virial_radius_positive(self):
        assert self.halo.r_vir > 0

    def test_scale_radius_less_than_virial(self):
        assert self.halo.r_s < self.halo.r_vir

    def test_concentration_positive(self):
        assert self.halo.concentration > 0

    def test_nfw_density_positive(self):
        r = np.logspace(18, 22, 50)
        rho = self.halo.nfw_density(r)
        assert np.all(rho > 0)

    def test_nfw_density_decreases_with_r(self):
        r = np.array([1.0e19, 1.0e20, 1.0e21])
        rho = self.halo.nfw_density(r)
        assert rho[0] > rho[1] > rho[2]

    def test_nfw_mass_enclosed_increases_with_r(self):
        r = np.array([self.halo.r_s, 2.0 * self.halo.r_s, self.halo.r_vir])
        M = self.halo.nfw_mass_enclosed(r)
        assert M[0] < M[1] < M[2]

    def test_mass_enclosed_at_virial_approx_virial_mass(self):
        M_virial = self.halo.nfw_mass_enclosed(np.array([self.halo.r_vir]))[0]
        assert abs(M_virial - self.halo.M_vir_kg) / self.halo.M_vir_kg < 0.01

    def test_eqst_density_positive(self):
        r = np.logspace(18, 22, 30)
        rho_eqst = self.halo.eqst_gp_density(r)
        assert np.all(rho_eqst >= 0)

    def test_eqst_density_less_equal_nfw_at_center(self):
        r_center = np.array([0.01 * self.halo.r_s])
        rho_nfw = self.halo.nfw_density(r_center)[0]
        rho_eqst = self.halo.eqst_gp_density(r_center)[0]
        assert rho_eqst <= rho_nfw

    def test_circular_velocity_positive(self):
        r = np.logspace(18, 22, 30)
        v_circ = self.halo.circular_velocity_eqst_gp(r)
        assert np.all(v_circ >= 0)

    def test_density_profile_summary_complete(self):
        summary = self.halo.density_profile_summary()
        assert 'r_vir_kpc' in summary
        assert 'v_max_km_s' in summary
        assert 'concentration' in summary
        assert summary['v_max_km_s'] > 0

    def test_annihilation_rate_finite(self):
        r = np.logspace(19, 22, 20)
        rate = self.halo.eqst_gp_dm_annihilation_rate(r)
        assert np.all(np.isfinite(rate))
        assert np.all(rate >= 0)

    def test_gravitational_potential_negative(self):
        r = np.logspace(18, 22, 30)
        Phi = self.halo.gravitational_potential(r)
        assert np.all(Phi < 0)