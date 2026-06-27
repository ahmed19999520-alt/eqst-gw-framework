import pytest
import numpy as np
import os
import tempfile

from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.cosmology import LambdaEffectiveCosmology
from eqst_gw.gravitational_waves.spectrum import GravitationalWaveSpectrum
from eqst_gw.detectors.lisa import LISADetector
from eqst_gw.detectors.sensitivity_curves import MultiDetectorNetwork
from eqst_gw.data.loaders import (
    load_planck_data,
    load_desi_bao,
    load_pantheon_sn,
    load_jwst_galaxies,
    load_nanograv_pta_data,
)
from eqst_gw.analysis.parameter_estimation import ParameterEstimator
from eqst_gw.analysis.mcmc import MCMCSampler
from eqst_gw.analysis.model_comparison import BayesianModelComparison
from eqst_gw.simulations.dark_matter_halos import DMHaloProfile
from eqst_gw.multimessenger.cross_checks import MultiMessengerCrossChecks
from eqst_gw.io.hdf5_handler import HDF5Handler
from eqst_gw.io.json_yaml import JSONYAMLHandler

_const = FundamentalConstants()
_params = EQSTGPParameters()
_cosmo = LambdaEffectiveCosmology(_const)

_tmpdir = tempfile.mkdtemp()

INT_GW_SPECTRUM = GravitationalWaveSpectrum(_params, _const)
INT_LISA = LISADetector(mission_duration_years=4.0, constants=_const)
INT_NETWORK = MultiDetectorNetwork(_const)
INT_NETWORK.initialize_standard_network()
INT_ESTIMATOR = ParameterEstimator(_params, _const, _cosmo)
INT_MCMC = MCMCSampler(_params, _const, _cosmo)
INT_MODEL_COMP = BayesianModelComparison()
INT_MM_CHECKS = MultiMessengerCrossChecks(_params, _const, _cosmo)
INT_HDF5 = HDF5Handler(output_dir=_tmpdir)
INT_JSON = JSONYAMLHandler(output_dir=_tmpdir)

INT_F_ARRAY = np.logspace(-5, 2, 1000)

_ell_cmb, _D_ell_cmb, _sigma_cmb = load_planck_data(use_mock=True)
_z_bao, _DM_bao, _DH_bao, _cov_bao = load_desi_bao(use_mock=True)
_z_sn, _mu_sn, _cov_sn = load_pantheon_sn(use_mock=True)
_jwst_data = load_jwst_galaxies(use_mock=True)
_pta_data = load_nanograv_pta_data(use_mock=True)

INT_MOCK_CMB = (_ell_cmb, _D_ell_cmb, _sigma_cmb)
INT_MOCK_BAO = (_z_bao, _DM_bao, _DH_bao, _cov_bao)
INT_MOCK_SN = (_z_sn, _mu_sn, _cov_sn)
INT_MOCK_JWST = _jwst_data
INT_MOCK_PTA = _pta_data

INTEGRATION_TMPDIR = _tmpdir

__all__ = [
    'INT_GW_SPECTRUM',
    'INT_LISA',
    'INT_NETWORK',
    'INT_ESTIMATOR',
    'INT_MCMC',
    'INT_MODEL_COMP',
    'INT_MM_CHECKS',
    'INT_HDF5',
    'INT_JSON',
    'INT_F_ARRAY',
    'INT_MOCK_CMB',
    'INT_MOCK_BAO',
    'INT_MOCK_SN',
    'INT_MOCK_JWST',
    'INT_MOCK_PTA',
    'INTEGRATION_TMPDIR',
]