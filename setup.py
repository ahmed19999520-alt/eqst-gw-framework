from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="eqst-gw-framework",
    version="1.0.0",
    author="Ahmed Ali",
    author_email="ahmed19999520@gmail.com",
    description="Complete gravitational wave and multi-messenger analysis framework for EQST-GP predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmed19999520-alt/eqst-gw-framework",
    project_urls={
        "Bug Tracker": "https://github.com/ahmed19999520-alt/eqst-gw-framework/issues",
        "Documentation": "https://eqst-gw-framework.readthedocs.io/",
        "Source Code": "https://github.com/ahmed19999520-alt/eqst-gw-framework",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "sphinx>=4.5",
            "sphinx-rtd-theme>=1.0",
        ],
        "gpu": [
            "cupy>=10.0",
            "pycuda>=2021.1",
        ],
        "mpi": [
            "mpi4py>=3.1",
        ],
    },
    include_package_data=True,
    package_data={
        "eqst_gw": [
            "data/templates/*.npy",
            "data/observational/*/*.txt",
            "configs/*.yaml",
        ],
    },
    entry_points={
        "console_scripts": [
            "eqst-gw-pipeline=eqst_gw.scripts.run_pipeline:main",
            "eqst-gw-download-data=eqst_gw.scripts.download_data:main",
            "eqst-gw-simulate-cluster=eqst_gw.scripts.simulate_cluster:main",
        ],
    },
)