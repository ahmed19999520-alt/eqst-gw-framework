__version__ = "1.0.0"
__author__ = "Ahmed Ali"
__email__ = "ahmed19999520@gmail.com"

from .core.constants import FundamentalConstants
from .core.parameters import EQSTGPParameters
from .core.cosmology import LambdaEffectiveCosmology

from .gravitational_waves.spectrum import GravitationalWaveSpectrum
from .gravitational_waves.sources import (
    BubbleCollisionSource,
    SoundWaveSource,
    TurbulenceSource
)

from .detectors.lisa import LISADetector
from .detectors.ligo import LIGODetector
from .detectors.virgo import VirgoDetector
from .detectors.einstein_telescope import EinsteinTelescopeDetector

from .simulations.dark_matter_halos import DMHaloProfile
from .simulations.galaxy_clusters import GalaxyClusterSimulation
from .simulations.rotation_curves import GalaxyRotationCurve
from .simulations.bubble_nucleation_sim import BubbleNucleationSimulation

from .analysis.parameter_estimation import ParameterEstimator
from .analysis.mcmc import MCMCSampler
from .analysis.model_comparison import BayesianModelComparison

from .data.loaders import (
    load_planck_data,
    load_desi_bao,
    load_pantheon_sn,
    load_jwst_galaxies
)

__all__ = [
    'FundamentalConstants',
    'EQSTGPParameters',
    'LambdaEffectiveCosmology',
    'GravitationalWaveSpectrum',
    'BubbleCollisionSource',
    'SoundWaveSource',
    'TurbulenceSource',
    'LISADetector',
    'LIGODetector',
    'VirgoDetector',
    'EinsteinTelescopeDetector',
    'DMHaloProfile',
    'GalaxyClusterSimulation',
    'GalaxyRotationCurve',
    'BubbleNucleationSimulation',
    'ParameterEstimator',
    'MCMCSampler',
    'BayesianModelComparison',
    'load_planck_data',
    'load_desi_bao',
    'load_pantheon_sn',
    'load_jwst_galaxies',
]