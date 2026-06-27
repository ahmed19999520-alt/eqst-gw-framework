from .dark_matter_halos import DMHaloProfile
from .galaxy_clusters import GalaxyClusterSimulation
from .bubble_nucleation_sim import BubbleNucleationSimulation
from .hydrodynamics import RelativisticHydrodynamics
from .rotation_curves import GalaxyRotationCurve
from .structure_formation import StructureFormationAnalysis

__all__ = [
    'DMHaloProfile',
    'GalaxyClusterSimulation',
    'BubbleNucleationSimulation',
    'RelativisticHydrodynamics',
    'GalaxyRotationCurve',
    'StructureFormationAnalysis',
]

_SIMULATION_REGISTRY = {
    'dark_matter_halos': {
        'class': DMHaloProfile,
        'description': 'EQST-GP modified NFW dark matter halo density profiles and rotation curves',
        'inputs': ['M_vir_solar', 'z'],
        'outputs': ['density_profile', 'rotation_curve', 'velocity_dispersion', 'annihilation_rate'],
        'eqst_gp_feature': 'Core suppression from topological DM stability and geometric interaction suppression',
    },
    'galaxy_clusters': {
        'class': GalaxyClusterSimulation,
        'description': 'N-body simulation of galaxy cluster mergers with EQST-GP DM profiles',
        'inputs': ['M_cluster1', 'M_cluster2', 'impact_parameter_kpc', 'relative_velocity_km_s'],
        'outputs': ['particle_positions', 'particle_velocities', 'density_profiles', 'animations'],
        'eqst_gp_feature': 'DM particles initialized from EQST-GP modified halo profiles',
    },
    'bubble_nucleation': {
        'class': BubbleNucleationSimulation,
        'description': '3D lattice field theory simulation of EQST-GP SU(4) -> SU(3)_C x U(1)_DM phase transition',
        'inputs': ['T_nucleation_GeV', 'alpha', 'beta_over_H', 'v_w', 'lattice_size'],
        'outputs': ['field_evolution', 'gw_power_spectrum', 'bubble_statistics', 'animations'],
        'eqst_gp_feature': 'Effective potential coefficients from M5-brane moduli stabilization',
    },
    'hydrodynamics': {
        'class': RelativisticHydrodynamics,
        'description': 'Relativistic hydrodynamics for GW-producing plasma around bubble walls',
        'inputs': ['alpha_PT', 'v_w'],
        'outputs': ['fluid_velocity_profile', 'kinetic_energy_efficiency', 'mhd_turbulence_spectrum'],
        'eqst_gp_feature': 'Jouguet velocity from (2,0) strongly-coupled plasma friction',
    },
    'rotation_curves': {
        'class': GalaxyRotationCurve,
        'description': 'Galaxy rotation curve predictions from EQST-GP DM density profiles',
        'inputs': ['M_vir', 'z'],
        'outputs': ['v_circular_eqst_gp', 'v_circular_nfw', 'baryonic_component', 'fit_results'],
        'eqst_gp_feature': 'Core density suppression creates observable differences from NFW at small radii',
    },
    'structure_formation': {
        'class': StructureFormationAnalysis,
        'description': 'Linear structure formation with EQST-GP Lambda_eff(z) cosmology',
        'inputs': ['z_range'],
        'outputs': ['growth_factor', 'growth_rate', 'matter_power_spectrum', 'sigma8', 'halo_mass_function'],
        'eqst_gp_feature': 'Modified expansion history from Lambda_eff(z) changes clustering at z < 3',
    },
}


def list_simulations() -> None:
    print("\nAvailable EQST-GP Simulation Modules")
    print("=" * 65)
    for name, info in _SIMULATION_REGISTRY.items():
        print(f"\n  [{name}]")
        print(f"    Class:    {info['class'].__name__}")
        print(f"    Purpose:  {info['description']}")
        print(f"    Inputs:   {', '.join(info['inputs'])}")
        print(f"    Outputs:  {', '.join(info['outputs'])}")
        print(f"    EQST-GP:  {info['eqst_gp_feature']}")
    print()


def get_simulation_class(name: str):
    if name not in _SIMULATION_REGISTRY:
        available = list(_SIMULATION_REGISTRY.keys())
        raise ValueError(f"Unknown simulation '{name}'. Available: {available}")
    return _SIMULATION_REGISTRY[name]['class']