import pytest
import numpy as np

from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.gravitational_waves.spectrum import GravitationalWaveSpectrum
from eqst_gw.gravitational_waves.sources import (
    BubbleCollisionSource,
    SoundWaveSource,
    TurbulenceSource,
)
from eqst_gw.gravitational_waves.sound_waves import SoundWaveEvolution
from eqst_gw.gravitational_waves.turbulence import MHDTurbulence
from eqst_gw.gravitational_waves.bubble_collisions import BubbleCollisionDynamics
from eqst_gw.detectors.lisa import LISADetector
from eqst_gw.detectors.ligo import LIGODetector
from eqst_gw.detectors.virgo import VirgoDetector
from eqst_gw.detectors.kagra import KAGRADetector
from eqst_gw.detectors.einstein_telescope import EinsteinTelescopeDetector
from eqst_gw.detectors.sensitivity_curves import MultiDetectorNetwork

_const = FundamentalConstants()
_params = EQSTGPParameters()

GW_SPECTRUM_FIXTURE = GravitationalWaveSpectrum(_params, _const)
BUBBLE_SOURCE_FIXTURE = BubbleCollisionSource(_params, _const)
SOUND_SOURCE_FIXTURE = SoundWaveSource(_params, _const)
TURB_SOURCE_FIXTURE = TurbulenceSource(_params, _const)
SOUND_WAVE_EVO_FIXTURE = SoundWaveEvolution(_params, _const)
MHD_TURB_FIXTURE = MHDTurbulence(_params, _const)
BUBBLE_COLL_DYN_FIXTURE = BubbleCollisionDynamics(_params, _const)

LISA_FIXTURE = LISADetector(mission_duration_years=4.0, constants=_const)
LIGO_FIXTURE = LIGODetector(design='O4', constants=_const)
VIRGO_FIXTURE = VirgoDetector(design='O4', constants=_const)
KAGRA_FIXTURE = KAGRADetector(design='O4', constants=_const)
ET_FIXTURE = EinsteinTelescopeDetector(design='ET-D', constants=_const)

NETWORK_FIXTURE = MultiDetectorNetwork(_const)
NETWORK_FIXTURE.initialize_standard_network()

F_TEST_ARRAY = np.logspace(-5, 2, 500)
F_LISA_BAND = np.logspace(-4, -1, 200)
F_ET_BAND = np.logspace(0, 4, 200)

__all__ = [
    'GW_SPECTRUM_FIXTURE',
    'BUBBLE_SOURCE_FIXTURE',
    'SOUND_SOURCE_FIXTURE',
    'TURB_SOURCE_FIXTURE',
    'SOUND_WAVE_EVO_FIXTURE',
    'MHD_TURB_FIXTURE',
    'BUBBLE_COLL_DYN_FIXTURE',
    'LISA_FIXTURE',
    'LIGO_FIXTURE',
    'VIRGO_FIXTURE',
    'KAGRA_FIXTURE',
    'ET_FIXTURE',
    'NETWORK_FIXTURE',
    'F_TEST_ARRAY',
    'F_LISA_BAND',
    'F_ET_BAND',
]