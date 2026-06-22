# EQST-GP Gravitational Wave Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://doi.org/10.20944/preprints202602.0758.v2)
[![arXiv](https://img.shields.io/badge/arXiv-2501.00592-b31b1b.svg)](https://arxiv.org/abs/2501.00592)

Python code's that developed for numerical evaluation of integrals for computing gravitational wave spectra, dark matter halo profiles, galaxy rotation curves, galaxy cluster dynamics, and multi-messenger observables from the **Expanded Quantum String Theory with Gluonic Plasma (EQST-GP)** framework.

## Features

- **First-principles GW spectrum computation** from topological phase transitions
- **Multi-detector integration**: LISA, LIGO, Virgo, KAGRA, Einstein Telescope
- **Observational data fitting**: Planck CMB, DESI BAO, Pantheon+ SNe, JWST galaxies
- **N-body simulations**: Dark matter halos, galaxy clusters, bubble nucleation
- **Rotation curve predictions**: EQST-GP vs ΛCDM comparison
- **Pulsar timing analysis**: Gravitational wave background constraints
- **Bayesian inference**: MCMC, nested sampling, model comparison
- **Publication-ready visualizations**

## Installation

### From PyPI (when published)
```bash
pip install eqst-gw-framework
```

### From source
```bash
git clone https://github.com/ahmed19999520-alt/eqst-gw-framework.git
cd eqst-gw-framework
pip install -e .
```

### With conda
```bash
conda env create -f environment.yml
conda activate eqst-gw
```

## Quick Start

```python
from eqst_gw import EQSTGPParameters, GravitationalWaveSpectrum
from eqst_gw.detectors import LISADetector
import numpy as np

# Initialize EQST-GP parameters
params = EQSTGPParameters()

# Compute GW spectrum
gw = GravitationalWaveSpectrum(params)
frequencies = np.logspace(-5, 0, 1000)
spectrum = gw.total_spectrum(frequencies)

# Check LISA detectability
lisa = LISADetector(mission_duration_years=4.0)
snr = lisa.compute_snr(frequencies, spectrum)
print(f"LISA SNR: {snr:.2f}")
```

## Documentation

Full documentation available at: 

https://doi.org/10.20944/preprints202601.0003.v1

https://doi.org/10.20944/preprints202602.0758.v2

- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [API Reference](docs/api_reference.md)
- [Theory Background](docs/theory/eqst_gp_framework.md)

## Example Analyses

### 1. Galaxy Rotation Curves
```python
from eqst_gw.simulations import GalaxyRotationCurve
from eqst_gw.data import load_observed_rotation_curves

# Load observed data
observed_data = load_observed_rotation_curves('NGC1560')

# Predict EQST-GP rotation curve
rc = GalaxyRotationCurve(M_vir=1e11, z=0.0)
r, v_circ = rc.compute_eqst_gp_profile()

# Compare with ΛCDM
v_circ_lcdm = rc.compute_lcdm_profile(r)

# Fit and plot
rc.fit_to_data(observed_data)
rc.plot_comparison()
```

### 2. Galaxy Cluster Merger Simulation
```python
from eqst_gw.simulations import ClusterMergerSimulation

# Initialize two clusters
sim = ClusterMergerSimulation(
    M_cluster1=1e15,  # Solar masses
    M_cluster2=5e14,
    impact_parameter=500.0,  # kpc
    relative_velocity=2000.0  # km/s
)

# Run N-body simulation
sim.run(t_max=3.0, dt=0.01)  # Gyr

# Analyze DM distribution evolution
sim.compute_density_profiles()
sim.create_animation('cluster_merger.mp4')
```

### 3. Bubble Nucleation and Collision
```python
from eqst_gw.simulations import BubbleNucleationSimulation

# Initialize phase transition parameters
bubble_sim = BubbleNucleationSimulation(
    T_nucleation=9.71e15,  # GeV
    alpha=0.42,
    beta_over_H=94.7,
    lattice_size=256
)

# Run 3D lattice simulation
bubble_sim.nucleate_bubbles(n_bubbles=50)
bubble_sim.evolve_field(n_steps=1000)

# Compute GW power from collisions
P_gw = bubble_sim.compute_gw_power_spectrum()
bubble_sim.visualize_collision_snapshots()
```

### 4. Pulsar Timing Array Analysis
```python
from eqst_gw.multimessenger import PulsarTimingAnalysis

# Load NANOGrav 15-year data
pta = PulsarTimingAnalysis()
pta.load_nanograv_data('15yr_dataset')

# Compute EQST-GP predicted residuals
residuals_eqst = pta.compute_timing_residuals_eqst_gp()

# Compare with observed
chi2 = pta.fit_residuals(residuals_eqst)
pta.plot_residual_comparison()
```

## Observational Data Integration

The framework includes automated downloaders for:

- **Planck 2018 CMB**: Temperature and polarization power spectra
- **DESI BAO DR1**: Baryon acoustic oscillation measurements
- **Pantheon+ SNe**: Type Ia supernova distance moduli
- **JWST High-z Galaxies**: Stellar mass functions at z > 8
- **GWOSC Events**: LIGO/Virgo gravitational wave detections

Download all datasets:
```bash
bash scripts/download_observational_data.sh
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{Ali2025EQSTGP,
  title=(Swampland Conjectures Compatibility and Technical Refinements  in the Expanded Quantum String Theory with Gluonic Plasma (EQST-GP) Model),
  author={Ali, Ahmed},
  journal={Annals of Mathematics and Physics},
  volume={8},
  number={6},
  pages={273--283},
  year={2025},
  doi={10.17352/amp.000126}
https://dx.doi.org/10.17352/amp
}
```

```bibtex
@article{Ali2026EQSTGW,
  title={New Prediction for Gravitational Wave Background from Topological Phase Transitions in the Early Universe},
  author={Ali, Ahmed},
  journal={Preprints.org, Prepared for submission to JHEP},
  volume={2},
  number={758},
  pages={33},
  year={2026},
  doi={10.20944/preprints202602.0758.v2}
}
```


## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

- **Author**: Ahmed Ali
- **Email**: ahmed19999520@gmail.com
- **Institution**: Theoretical Physics Research Group at Max Plank institute 

## Acknowledgments

This framework builds upon:
- LISA Consortium data analysis tools
- LIGO Open Science Center (GWOSC)
- Planck Legacy Archive
- DESI Collaboration public data releases
- NANOGrav pulsar timing array data