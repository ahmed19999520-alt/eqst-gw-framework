import numpy as np
import pytest
from eqst_gw.physics.bubble_dynamics import BubbleDynamics
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.constants import FundamentalConstants


class TestBubbleDynamics:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.ep = EQSTGPParameters()
        self.bd = BubbleDynamics(self.ep, self.const)

    def test_wall_velocity_from_friction_positive(self):
        v_w = self.bd.wall_velocity_from_friction(self.ep.alpha_PT, self.ep.T_n)
        assert v_w > 0

    def test_wall_velocity_subluminal(self):
        v_w = self.bd.wall_velocity_from_friction(self.ep.alpha_PT, self.ep.T_n)
        assert v_w < 1.0

    def test_wall_velocity_increases_with_alpha(self):
        v_w_low = self.bd.wall_velocity_from_friction(0.1, self.ep.T_n)
        v_w_high = self.bd.wall_velocity_from_friction(1.0, self.ep.T_n)
        assert v_w_high >= v_w_low

    def test_wall_thickness_positive(self):
        L_w = self.bd.wall_thickness(self.ep.T_n)
        assert L_w > 0

    def test_bubble_wall_profile_boundary_conditions(self):
        r = np.linspace(0, 1.0e-14, 100)
        R_bubble = 5.0e-15
        L_wall = 5.0e-16
        phi_false = 0.0
        phi_true = 1.0
        phi = self.bd.bubble_wall_profile(r, R_bubble, L_wall, phi_false, phi_true)
        assert abs(phi[0] - phi_true) < 0.1
        assert abs(phi[-1] - phi_false) < 0.1

    def test_surface_tension_positive(self):
        phi_false = 0.0
        phi_true = 3.0 * self.ep.gamma_thermal * self.ep.T_n / self.ep.lambda_quartic
        sigma = self.bd.surface_tension(self.ep.T_n, phi_false, phi_true)
        assert sigma >= 0

    def test_bubble_expansion_simulation_runs(self):
        t, R, R_dot = self.bd.simulate_bubble_expansion(
            self.ep.alpha_PT, self.ep.T_n, t_max=1.0e-26, n_points=50
        )
        assert len(t) == 50
        assert len(R) == 50
        assert np.all(np.isfinite(R))