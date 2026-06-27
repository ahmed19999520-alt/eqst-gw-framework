import pytest
import numpy as np

from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.physics.phase_transition import PhaseTransitionDynamics
from eqst_gw.physics.bubble_dynamics import BubbleDynamics
from eqst_gw.physics.nucleation import BubbleNucleation
from eqst_gw.physics.effective_potential import EffectivePotential

_const = FundamentalConstants()
_params = EQSTGPParameters()

PHASE_TRANSITION_FIXTURE = PhaseTransitionDynamics(_params, _const)
BUBBLE_DYNAMICS_FIXTURE = BubbleDynamics(_params, _const)
NUCLEATION_FIXTURE = BubbleNucleation(_params, _const)
EFFECTIVE_POTENTIAL_FIXTURE = EffectivePotential(_params, _const)

T_NUCLEATION_TEST = _params.T_n
T_CRITICAL_TEST = _params.T_c_GeV
PHI_TRUE_APPROX = 3.0 * _params.gamma_thermal * _params.T_n / _params.lambda_quartic
PHI_FALSE = 0.0

__all__ = [
    'PHASE_TRANSITION_FIXTURE',
    'BUBBLE_DYNAMICS_FIXTURE',
    'NUCLEATION_FIXTURE',
    'EFFECTIVE_POTENTIAL_FIXTURE',
    'T_NUCLEATION_TEST',
    'T_CRITICAL_TEST',
    'PHI_TRUE_APPROX',
    'PHI_FALSE',
]