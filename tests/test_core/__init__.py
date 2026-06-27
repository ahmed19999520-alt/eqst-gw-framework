import pytest
import numpy as np

from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.cosmology import LambdaEffectiveCosmology

CONST_FIXTURE = FundamentalConstants()
PARAMS_FIXTURE = EQSTGPParameters()
COSMO_FIXTURE = LambdaEffectiveCosmology(CONST_FIXTURE)

__all__ = [
    'CONST_FIXTURE',
    'PARAMS_FIXTURE',
    'COSMO_FIXTURE',
]