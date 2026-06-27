import pytest
import numpy as np

from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.cosmology import LambdaEffectiveCosmology
from eqst_gw.simulations.dark_matter_halos import DMHaloProfile
from eqst_gw.simulations.galaxy_clusters import GalaxyClusterSimulation
from eqst_gw.simulations.rotation_curves import GalaxyRotationCurve
from eqst_gw.simulations.hydrodynamics import RelativisticHydrodynamics
from eqst_gw.simulations.structure_formation import StructureFormationAnalysis

_const = FundamentalConstants()
_params = EQSTGPParameters()
_cosmo = LambdaEffectiveCosmology(_const)

HALO_MW_FIXTURE = DMHaloProfile(
    M_vir_solar=1.0e12,
    z=0.0,
    eqst_params=_params,
    constants=_const,
)

HALO_CLUSTER_FIXTURE = DMHaloProfile(
    M_vir_solar=1.0e15,
    z=0.3,
    eqst_params=_params,
    constants=_const,
)

CLUSTER_SIM_SMALL_FIXTURE = GalaxyClusterSimulation(
    M_cluster1=1.0e14,
    M_cluster2=5.0e13,
    impact_parameter_kpc=300.0,
    relative_velocity_km_s=1500.0,
    N_particles_cluster1=200,
    N_particles_cluster2=100,
    constants=_const,
)

ROTATION_CURVE_FIXTURE = GalaxyRotationCurve(
    M_vir=1.0e11,
    z=0.0,
    constants=_const,
    eqst_params=_params,
)

HYDRO_FIXTURE = RelativisticHydrodynamics(_params, _const)

STRUCTURE_FIXTURE = StructureFormationAnalysis(_const, _cosmo)

R_TEST_KPC = np.linspace(0.5, 50.0, 100)
R_TEST_M = R_TEST_KPC * 3.086e19
Z_TEST_ARRAY = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
K_TEST_ARRAY = np.logspace(-3, 1, 100)

__all__ = [
    'HALO_MW_FIXTURE',
    'HALO_CLUSTER_FIXTURE',
    'CLUSTER_SIM_SMALL_FIXTURE',
    'ROTATION_CURVE_FIXTURE',
    'HYDRO_FIXTURE',
    'STRUCTURE_FIXTURE',
    'R_TEST_KPC',
    'R_TEST_M',
    'Z_TEST_ARRAY',
    'K_TEST_ARRAY',
]