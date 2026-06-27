# Contributing to the EQST-GP Gravitational Wave Framework

Thank you for your interest in contributing to this project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub:
- Describe the problem clearly with a minimal reproducible example
- Include your Python version, OS, and package versions
- For physics bugs, include the relevant equations and expected vs actual outputs

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following the code style guidelines
4. Add tests for new functionality in the `tests/` directory
5. Run the test suite: `pytest tests/ -v --cov=eqst_gw`
6. Ensure all tests pass and coverage remains above 80%
7. Submit a pull request with a clear description of your changes

### Code Style

- Follow PEP 8 with a maximum line length of 120 characters
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Use descriptive variable names

### Adding New Physics Modules

When adding new physics to the framework:
- Ensure all physical quantities carry proper units (document in docstrings)
- Validate against known analytic limits where possible
- Add unit tests in `tests/test_physics/`
- Reference the relevant papers in comments and docstrings

### Adding New Detectors

To add a new gravitational wave detector:
- Subclass `BaseDetector` in `eqst_gw/detectors/`
- Implement the `noise_psd(self, f)` method
- Add sensitivity data with proper citations
- Write tests in `tests/test_gw/test_detectors.py`

### Data Contributions

If you have real observational data to contribute:
- Ensure proper licensing and attribution
- Add a loader function in `eqst_gw/data/loaders.py`
- Document the data source, format, and any preprocessing applied
- Place data in the appropriate `data/observational/` subdirectory

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Contact

Ahmed Ali — ahmed19999520@gmail.com