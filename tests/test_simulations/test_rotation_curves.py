import numpy as np
import pytest
from eqst_gw.simulations.rotation_curves import GalaxyRotationCurve
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.constants import FundamentalConstants


class TestGalaxyRotationCurve:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.ep = EQSTGPParameters()
        self.rc = GalaxyRotationCurve(M_vir=1.0e11, z=0.0, constants=self.const, eqst_params=self.ep)
        self.r_test = np.logspace(18, 22, 50)

    def test_virial_radius_positive(self):
        assert self.rc.r_vir > 0

    def test_concentration_parameter_positive(self):
        assert self.rc.concentration > 0

    def test_scale_radius_less_than_virial(self):
        assert self.rc.r_s < self.rc.r_vir

    def test_nfw_rotation_curve_positive(self):
        v_circ = self.rc.compute_rotation_curve_nfw(self.r_test)
        assert np.all(v_circ >= 0)

    def test_eqst_rotation_curve_positive(self):
        v_circ = self.rc.compute_rotation_curve_eqst_gp(self.r_test)
        assert np.all(v_circ >= 0)

    def test_eqst_rotation_curve_finite(self):
        v_circ = self.rc.compute_rotation_curve_eqst_gp(self.r_test)
        assert np.all(np.isfinite(v_circ))

    def test_baryonic_component_positive(self):
        M_disk = 1.0e10
        R_d = 3.0e19
        v_bar = self.rc.baryonic_component(self.r_test, M_disk, R_d)
        assert np.all(v_bar >= 0)

    def test_total_curve_greater_than_components(self):
        M_disk = 1.0e10
        R_d = 3.0e19
        v_total = self.rc.total_rotation_curve(self.r_test, M_disk, R_d, model='eqst_gp')
        v_bar = self.rc.baryonic_component(self.r_test, M_disk, R_d)
        assert np.all(v_total >= v_bar)

    def test_rotation_curve_has_maximum(self):
        v_circ = self.rc.compute_rotation_curve_eqst_gp(self.r_test)
        assert np.max(v_circ) > 0

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            self.rc.total_rotation_curve(self.r_test, 1.0e10, 3.0e19, model='invalid_model')