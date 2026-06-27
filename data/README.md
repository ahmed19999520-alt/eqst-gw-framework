# EQST-GP Framework: Observational Data Directory

This directory contains observational data files and simulation outputs used by the EQST-GP Gravitational Wave Analysis Framework.

## Directory Structure data/
data/
├── observational/
│ ├── planck/ Planck 2018 CMB power spectra and cosmological parameters
│ ├── desi/ DESI BAO DR1 measurements and covariance matrices
│ ├── pantheon_plus/ Pantheon+ Type Ia supernovae distance moduli
│ ├── jwst/ JWST high-z galaxy catalogs and stellar mass functions
│ └── ligo_gwosc/ LIGO-Virgo-KAGRA event catalogs (GWTC-3)
├── simulations/
│ ├── bubble_nucleation_grid.h5 Pre-computed GW spectrum parameter grid
│ ├── cluster_merger_snapshots/ N-body simulation snapshots
│ ├── rotation_curves_samples/ Sample galaxy rotation curve data
│ └── stellar_structure_models/ Stellar structure model inputs
└── templates/
├── gw_waveforms/ GW spectral templates for matched filtering
├── nfw_profiles/ Pre-tabulated NFW density profiles
└── phase_transition_templates/ Phase transition parameter templates

## Data Sources and Citations

### Planck 2018 CMB Data
- **Source**: Planck Legacy Archive (PLA) https://pla.esac.esa.int/
- **Reference**: Planck Collaboration (2020), A&A 641, A6
- **DOI**: 10.1051/0004-6361/201833910
- **License**: CC BY 4.0

### DESI BAO DR1
- **Source**: DESI Data Release 1 https://data.desi.lbl.gov/public/dr1/
- **Reference**: DESI Collaboration (2024), arXiv:2404.03002
- **License**: CC BY 4.0

### Pantheon+ Supernovae
- **Source**: https://github.com/PantheonPlusSH0ES/DataRelease
- **Reference**: Brout et al. (2022), ApJ 938, 110; DOI: 10.3847/1538-4357/ac8e04
- **License**: CC BY 4.0

### LIGO GWOSC Events
- **Source**: Gravitational Wave Open Science Center https://www.gw-openscience.org/
- **Reference**: LIGO-Virgo-KAGRA (2021), GWTC-3
- **License**: CC BY 4.0

### JWST Galaxy Catalogs
- **Source**: MAST Archive https://mast.stsci.edu/
- **Reference**: Various JWST papers (see individual catalog headers)
- **License**: CC BY 4.0

## Downloading Data

To automatically download all available public datasets, run:
```bash
bash scripts/download_observational_data.sh
```

To generate realistic mock data for testing (when real data is unavailable):
```bash
python scripts/generate_mock_catalogs.py
```

## Mock Data Notice

If real observational files are not present, the framework automatically generates scientifically realistic mock data based on published best-fit parameters. Mock data is clearly labeled in all outputs and analysis results. All mock data generators use published best-fit parameter values and realistic noise models.

## Data Format Reference

| File | Format | Description |
|------|--------|-------------|
| COM_PowerSpect_CMB-TT-full_R3.01.txt | ASCII | Planck TT power spectrum |
| DESI_BAO_2024_DR1.csv | CSV | DESI BAO measurements |
| covariance_matrix.npy | NumPy binary | BAO covariance matrix |
| Pantheon+SH0ES.dat | ASCII | SN distance moduli |
| high_z_galaxies_catalog.csv | CSV | JWST galaxy catalog |
| gwtc3_confident.json | JSON | GW event catalog |
| event_parameters.h5 | HDF5 | GW event parameters |
| bubble_nucleation_grid.h5 | HDF5 | GW spectrum grid |