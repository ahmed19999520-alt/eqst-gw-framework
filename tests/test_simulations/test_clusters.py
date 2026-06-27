import numpy as np
import pytest
from eqst_gw.simulations.galaxy_clusters import GalaxyClusterSimulation
from eqst_gw.core.constants import FundamentalConstants


class TestGalaxyClusterSimulation:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.sim = GalaxyClusterSimulation(
            M_cluster1=1.0e15,
            M_cluster2=5.0e14,
            impact_parameter_kpc=500.0,
            relative_velocity_km_s=2000.0,
            N_particles_cluster1=500,
            N_particles_cluster2=250,
            constants=self.const
        )

    def test_positions_shape(self):
        assert self.sim.positions.shape[0] == 750
        assert self.sim.positions.shape[1] == 3

    def test_velocities_shape(self):
        assert self.sim.velocities.shape == self.sim.positions.shape

    def test_masses_positive(self):
        assert np.all(self.sim.masses > 0)

    def test_cluster_labels_valid(self):
        assert set(np.unique(self.sim.cluster_labels)) == {1, 2}

    def test_total_particles_correct(self):
        assert self.sim.N_total == 750

    def test_initial_positions_finite(self):
        assert np.all(np.isfinite(self.sim.positions))

    def test_initial_velocities_finite(self):
        assert np.all(np.isfinite(self.sim.velocities))

    def test_virial_radius_positive(self):
        r_vir = self.sim.virial_radius(self.sim.M1_kg, 0.0)
        assert r_vir > 0

    def test_nfw_sample_radii_positive(self):
        r_s = 1.0e21
        c = 5.0
        radii = self.sim.sample_nfw_radii(r_s, c, 100)
        assert np.all(radii > 0)

    def test_nfw_sample_radii_within_virial(self):
        r_s = 1.0e21
        c = 5.0
        radii = self.sim.sample_nfw_radii(r_s, c, 200)
        r_vir = c * r_s
        assert np.all(radii < r_vir)