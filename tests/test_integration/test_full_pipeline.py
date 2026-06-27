import numpy as np
import pytest
import os
import tempfile
from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.cosmology import LambdaEffectiveCosmology
from eqst_gw.gravitational_waves.spectrum import GravitationalWaveSpectrum
from eqst_gw.detectors.lisa import LISADetector
from eqst_gw.data.loaders import load_desi_bao, load_pantheon_sn, load_planck_data
from eqst_gw.analysis.parameter_estimation import ParameterEstimator
from eqst_gw.simulations.dark_matter_halos import DMHaloProfile
from eqst_gw.multimessenger.cross_checks import MultiMessengerCrossChecks
from eqst_gw.io.hdf5_handler import HDF5Handler
from eqst_gw.utils.validators import validate_eqst_parameters


class TestFullPipeline:
    def setup_method(self):
        self.const = FundamentalConstants()
        self.ep = EQSTGPParameters()
        self.cosmo = LambdaEffectiveCosmology(self.const)
        self.tmpdir = tempfile.mkdtemp()

    def test_parameter_validation_passes(self):
        result = validate_eqst_parameters(self.ep)
        assert result is True

    def test_gw_spectrum_computes(self):
        gw = GravitationalWaveSpectrum(self.ep, self.const)
        f = np.logspace(-4, -1, 100)
        Omega = gw.total_spectrum(f)
        assert len(Omega) == 100
        assert np.all(np.isfinite(Omega))
        assert np.all(Omega >= 0)

    def test_lisa_snr_computable(self):
        gw = GravitationalWaveSpectrum(self.ep, self.const)
        lisa = LISADetector(mission_duration_years=4.0, constants=self.const)
        f = np.logspace(-4, -1, 200)
        Omega = gw.total_spectrum(f)
        snr = lisa.compute_snr(f, Omega)
        assert np.isfinite(snr)
        assert snr > 0

    def test_mock_data_loads(self):
        ell, D_ell, sigma = load_planck_data(use_mock=True)
        z, DM, DH, cov = load_desi_bao(use_mock=True)
        z_sn, mu, cov_sn = load_pantheon_sn(use_mock=True)
        assert len(ell) > 0
        assert len(z) > 0
        assert len(z_sn) > 0

    def test_bao_fitting_runs(self):
        pe = ParameterEstimator(self.ep, self.const, self.cosmo)
        z, DM, DH, cov = load_desi_bao(use_mock=True)
        results = pe.fit_bao(z, DM, DH, cov)
        assert 'Omega_m' in results
        assert 'h' in results
        assert results['chi2_min'] >= 0

    def test_sn_fitting_runs(self):
        pe = ParameterEstimator(self.ep, self.const, self.cosmo)
        z_sn, mu, cov_sn = load_pantheon_sn(use_mock=True)
        results = pe.fit_sn(z_sn[:50], mu[:50], cov_sn[:50, :50])
        assert 'Omega_m' in results
        assert results['chi2_min'] >= 0

    def test_halo_profile_computes(self):
        halo = DMHaloProfile(M_vir_solar=1.0e12, z=0.0, eqst_params=self.ep, constants=self.const)
        summary = halo.density_profile_summary()
        assert summary['r_vir_kpc'] > 0
        assert summary['v_max_km_s'] > 0

    def test_multimessenger_report_generates(self):
        mm = MultiMessengerCrossChecks(self.ep, self.const, self.cosmo)
        report = mm.generate_full_consistency_report()
        assert 'dark_matter' in report
        assert 'fundamental_constants' in report
        assert 'overall_consistency' in report

    def test_hdf5_export_and_reload(self):
        gw = GravitationalWaveSpectrum(self.ep, self.const)
        f = np.logspace(-4, -1, 100)
        components = gw.spectrum_components(f)
        handler = HDF5Handler(output_dir=self.tmpdir)
        filepath = handler.save_gw_spectrum(f, components, self.ep, filename='test_spectrum.h5')
        assert os.path.exists(filepath)
        f_loaded, comp_loaded, params_loaded = handler.load_gw_spectrum(filepath)
        assert len(f_loaded) == len(f)
        assert np.allclose(f_loaded, f)

    def test_spectrum_components_sum_to_total(self):
        gw = GravitationalWaveSpectrum(self.ep, self.const)
        f = np.logspace(-4, -1, 100)
        components = gw.spectrum_components(f)
        total_recomputed = (components['bubble_collisions'] + components['sound_waves'] + components['turbulence'])
        assert np.allclose(total_recomputed, components['total'], rtol=1.0e-10)

    def test_end_to_end_snr_above_threshold(self):
        gw = GravitationalWaveSpectrum(self.ep, self.const)
        lisa = LISADetector(mission_duration_years=4.0, constants=self.const)
        f = np.logspace(-4, -1, 500)
        Omega = gw.total_spectrum(f)
        snr = lisa.compute_snr(f, Omega)
        assert snr > 3.0