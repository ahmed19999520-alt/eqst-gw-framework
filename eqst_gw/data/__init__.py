from .loaders import (
    load_planck_data,
    load_desi_bao,
    load_pantheon_sn,
    load_jwst_galaxies,
    load_ligo_gwosc_catalog,
    load_nanograv_pta_data,
)
from .preprocessors import (
    CMBPreprocessor,
    BAOPreprocessor,
    GWDataPreprocessor,
)
from .planck import PlanckDataManager
from .desi_bao import DESIBAODataManager
from .pantheon_sn import PantheonSNDataManager
from .jwst import JWSTDataManager

__all__ = [
    'load_planck_data',
    'load_desi_bao',
    'load_pantheon_sn',
    'load_jwst_galaxies',
    'load_ligo_gwosc_catalog',
    'load_nanograv_pta_data',
    'CMBPreprocessor',
    'BAOPreprocessor',
    'GWDataPreprocessor',
    'PlanckDataManager',
    'DESIBAODataManager',
    'PantheonSNDataManager',
    'JWSTDataManager',
]

_DATA_REGISTRY = {
    'planck_cmb': {
        'manager': PlanckDataManager,
        'loader': load_planck_data,
        'description': 'Planck 2018 CMB temperature and polarization power spectra',
        'reference': 'Planck Collaboration (2020), A&A 641, A6',
        'doi': '10.1051/0004-6361/201833910',
        'url': 'https://pla.esac.esa.int/',
        'data_dir': './data/observational/planck/',
        'files': [
            'COM_PowerSpect_CMB-TT-full_R3.01.txt',
            'COM_PowerSpect_CMB-TE-full_R3.01.txt',
            'planck_2018_cosmo_params.json',
        ],
        'eqst_gp_use': 'Constrains H0 and Omega_m via CMB acoustic scale; tests Lambda_eff(z) at recombination',
    },
    'desi_bao': {
        'manager': DESIBAODataManager,
        'loader': load_desi_bao,
        'description': 'DESI Baryon Acoustic Oscillation DR1 measurements',
        'reference': 'DESI Collaboration (2024), arXiv:2404.03002',
        'doi': '10.48550/arXiv.2404.03002',
        'url': 'https://data.desi.lbl.gov/public/dr1/',
        'data_dir': './data/observational/desi/',
        'files': [
            'DESI_BAO_2024_DR1.csv',
            'covariance_matrix.npy',
        ],
        'eqst_gp_use': 'Primary constraint on Lambda_eff(z) at 0.3 < z < 2.3; resolves Hubble tension',
    },
    'pantheon_sn': {
        'manager': PantheonSNDataManager,
        'loader': load_pantheon_sn,
        'description': 'Pantheon+ Type Ia supernovae distance moduli (1701 SNe)',
        'reference': 'Brout et al. (2022), ApJ 938, 110',
        'doi': '10.3847/1538-4357/ac8e04',
        'url': 'https://github.com/PantheonPlusSH0ES/DataRelease',
        'data_dir': './data/observational/pantheon_plus/',
        'files': [
            'Pantheon+SH0ES.dat',
            'systematics_covariance.fits',
        ],
        'eqst_gp_use': 'Constrains w0 and wa; tests late-time expansion with Lambda_eff(z)',
    },
    'jwst_galaxies': {
        'manager': JWSTDataManager,
        'loader': load_jwst_galaxies,
        'description': 'JWST high-redshift galaxy photometric and spectroscopic catalog',
        'reference': 'Various JWST papers (2022-2024)',
        'doi': 'Multiple',
        'url': 'https://mast.stsci.edu/',
        'data_dir': './data/observational/jwst/',
        'files': [
            'high_z_galaxies_catalog.csv',
            'stellar_mass_function.npy',
        ],
        'eqst_gp_use': 'Tests structure formation predictions; SFR density modified by Lambda_eff(z) at z > 8',
    },
    'ligo_gwosc': {
        'manager': None,
        'loader': load_ligo_gwosc_catalog,
        'description': 'LIGO-Virgo-KAGRA gravitational wave transient catalog GWTC-3',
        'reference': 'LIGO-Virgo-KAGRA (2021), Phys. Rev. X 13, 041039',
        'doi': '10.1103/PhysRevX.13.041039',
        'url': 'https://www.gw-openscience.org/',
        'data_dir': './data/observational/ligo_gwosc/',
        'files': [
            'gwtc3_confident.json',
            'event_parameters.h5',
        ],
        'eqst_gp_use': 'Provides astrophysical foreground characterization for SGWB searches',
    },
    'nanograv_pta': {
        'manager': None,
        'loader': load_nanograv_pta_data,
        'description': 'NANOGrav 15-year PTA stochastic GW background measurement',
        'reference': 'NANOGrav Collaboration (2023), ApJ Lett. 951, L8',
        'doi': '10.3847/2041-8213/acdac6',
        'url': 'https://nanograv.org/',
        'data_dir': './data/observational/',
        'files': [
            'nanograv_15yr_timing_residuals.h5',
        ],
        'eqst_gp_use': 'Cross-check: EQST-GP GW spectrum extrapolated to nHz band; spectral tilt comparison',
    },
}


def list_datasets() -> None:
    print("\nAvailable Observational Datasets")
    print("=" * 65)
    for name, info in _DATA_REGISTRY.items():
        print(f"\n  [{name}]")
        print(f"    Description: {info['description']}")
        print(f"    Reference:   {info['reference']}")
        print(f"    DOI:         {info['doi']}")
        print(f"    URL:         {info['url']}")
        print(f"    Files:       {', '.join(info['files'])}")
        print(f"    EQST-GP use: {info['eqst_gp_use']}")
    print()


def get_dataset_info(name: str) -> dict:
    if name not in _DATA_REGISTRY:
        available = list(_DATA_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return _DATA_REGISTRY[name]


def load_dataset(name: str, **kwargs):
    info = get_dataset_info(name)
    loader_func = info['loader']
    return loader_func(**kwargs)


def check_data_availability(data_dir_root: str = './data/observational/') -> dict:
    import os
    availability = {}
    for dataset_name, info in _DATA_REGISTRY.items():
        data_dir = info['data_dir']
        files_found = []
        files_missing = []
        for fname in info['files']:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                files_found.append(fname)
            else:
                files_missing.append(fname)
        availability[dataset_name] = {
            'all_present': len(files_missing) == 0,
            'files_found': files_found,
            'files_missing': files_missing,
            'will_use_mock': len(files_missing) > 0,
        }
    return availability


def print_data_status(data_dir_root: str = './data/observational/') -> None:
    availability = check_data_availability(data_dir_root)
    print("\nData Availability Status")
    print("=" * 65)
    for name, status in availability.items():
        symbol = "✓" if status['all_present'] else "~"
        mode = "REAL DATA" if status['all_present'] else "MOCK DATA"
        print(f"  [{symbol}] {name:<20s} -> {mode}")
        if status['files_missing']:
            for f in status['files_missing']:
                print(f"         Missing: {f}")
    print()