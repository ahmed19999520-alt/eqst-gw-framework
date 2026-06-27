import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u_astropy
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class FITSHandler:
    def __init__(self, output_dir: str = './outputs/data/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_gw_spectrum_fits(self,
                               f: np.ndarray,
                               Omega_gw: np.ndarray,
                               components: Dict[str, np.ndarray],
                               eqst_params,
                               filename: str = 'gw_spectrum.fits') -> str:

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['TELESCOP'] = 'LISA'
        primary_hdu.header['INSTRUME'] = 'EQST-GP Framework'
        primary_hdu.header['DATE'] = datetime.now().isoformat()
        primary_hdu.header['AUTHOR'] = 'Ahmed Ali'
        primary_hdu.header['VERSION'] = '1.0.0'
        primary_hdu.header['T_N_GEV'] = (eqst_params.T_n, 'Nucleation temperature [GeV]')
        primary_hdu.header['ALPHA_PT'] = (eqst_params.alpha_PT, 'Phase transition strength')
        primary_hdu.header['BETA_H'] = (eqst_params.beta_over_H, 'Inverse duration / Hubble rate')
        primary_hdu.header['V_W'] = (eqst_params.v_w, 'Bubble wall velocity [c]')
        primary_hdu.header['G_STAR'] = (eqst_params.g_star, 'Relativistic DOF at transition')
        primary_hdu.header['M_DM_GEV'] = (eqst_params.m_DM_GeV, 'DM mass [GeV]')
        primary_hdu.header['F_SW_HZ'] = (eqst_params.f_sw_Hz, 'Sound wave peak frequency [Hz]')
        primary_hdu.header['OM_SW'] = (eqst_params.Omega_sw_peak_h2, 'Sound wave peak Omega h^2')

        cols = [
            fits.Column(name='FREQUENCY', format='D', unit='Hz', array=f),
            fits.Column(name='OMEGA_GW_TOTAL', format='D', unit='dimensionless', array=Omega_gw),
        ]

        for comp_name, comp_data in components.items():
            safe_name = comp_name.upper().replace(' ', '_')[:16]
            cols.append(fits.Column(name=f'OM_{safe_name}', format='D',
                                    unit='dimensionless', array=comp_data))

        spectrum_hdu = fits.BinTableHDU.from_columns(cols)
        spectrum_hdu.name = 'GW_SPECTRUM'
        spectrum_hdu.header['EXTNAME'] = 'GW_SPECTRUM'
        spectrum_hdu.header['TTYPE1'] = 'FREQUENCY'
        spectrum_hdu.header['TUNIT1'] = 'Hz'

        hdulist = fits.HDUList([primary_hdu, spectrum_hdu])

        filepath = os.path.join(self.output_dir, filename)
        hdulist.writeto(filepath, overwrite=True)
        print(f"GW spectrum FITS file saved to {filepath}")
        return filepath

    def load_gw_spectrum_fits(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        with fits.open(filepath) as hdulist:
            header = hdulist[0].header
            params = {
                'T_n_GeV': header.get('T_N_GEV', 9.71e15),
                'alpha_PT': header.get('ALPHA_PT', 0.42),
                'beta_over_H': header.get('BETA_H', 94.7),
                'v_w': header.get('V_W', 0.27),
                'g_star': header.get('G_STAR', 187.0),
                'm_DM_GeV': header.get('M_DM_GEV', 1.03e16)
            }

            table = Table.read(hdulist['GW_SPECTRUM'])
            f = np.array(table['FREQUENCY'])
            Omega_gw = np.array(table['OMEGA_GW_TOTAL'])

            components = {}
            for col_name in table.colnames:
                if col_name.startswith('OM_') and col_name != 'OM_TOTAL':
                    comp_key = col_name[3:].lower()
                    components[comp_key] = np.array(table[col_name])

        return f, Omega_gw, components, params

    def save_galaxy_catalog_fits(self,
                                  RA: np.ndarray,
                                  Dec: np.ndarray,
                                  z_phot: np.ndarray,
                                  M_stellar: np.ndarray,
                                  SFR: np.ndarray,
                                  additional_cols: Optional[Dict[str, np.ndarray]] = None,
                                  filename: str = 'galaxy_catalog.fits') -> str:

        cols = [
            fits.Column(name='RA', format='D', unit='deg', array=RA),
            fits.Column(name='DEC', format='D', unit='deg', array=Dec),
            fits.Column(name='Z_PHOT', format='D', unit='', array=z_phot),
            fits.Column(name='M_STELLAR', format='D', unit='M_sun', array=M_stellar),
            fits.Column(name='SFR', format='D', unit='M_sun/yr', array=SFR),
        ]

        if additional_cols is not None:
            for col_name, col_data in additional_cols.items():
                safe_name = col_name.upper().replace(' ', '_')[:16]
                cols.append(fits.Column(name=safe_name, format='D',
                                        unit='', array=col_data))

        catalog_hdu = fits.BinTableHDU.from_columns(cols)
        catalog_hdu.name = 'GALAXY_CATALOG'
        catalog_hdu.header['EXTNAME'] = 'GALAXY_CATALOG'
        catalog_hdu.header['DATE'] = datetime.now().isoformat()
        catalog_hdu.header['N_GAL'] = len(RA)
        catalog_hdu.header['Z_MIN'] = float(np.min(z_phot))
        catalog_hdu.header['Z_MAX'] = float(np.max(z_phot))
        catalog_hdu.header['SURVEY'] = 'EQST-GP Mock / JWST'
        catalog_hdu.header['FRAMEREF'] = 'ICRS'

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['DATE'] = datetime.now().isoformat()

        hdulist = fits.HDUList([primary_hdu, catalog_hdu])

        filepath = os.path.join(self.output_dir, filename)
        hdulist.writeto(filepath, overwrite=True)
        print(f"Galaxy catalog FITS file saved to {filepath}")
        return filepath

    def save_mcmc_posterior_fits(self,
                                  flat_chain: np.ndarray,
                                  param_names: List[str],
                                  log_prob: np.ndarray,
                                  filename: str = 'mcmc_posterior.fits') -> str:

        cols = []
        for i, name in enumerate(param_names):
            safe_name = name.upper().replace(' ', '_')[:16]
            cols.append(fits.Column(name=safe_name, format='D',
                                    unit='', array=flat_chain[:, i]))

        cols.append(fits.Column(name='LOG_PROB', format='D',
                                unit='', array=log_prob.flatten()[:len(flat_chain)]))

        posterior_hdu = fits.BinTableHDU.from_columns(cols)
        posterior_hdu.name = 'MCMC_POSTERIOR'
        posterior_hdu.header['EXTNAME'] = 'MCMC_POSTERIOR'
        posterior_hdu.header['DATE'] = datetime.now().isoformat()
        posterior_hdu.header['N_SAMP'] = len(flat_chain)
        posterior_hdu.header['N_PAR'] = len(param_names)
        posterior_hdu.header['SAMPLER'] = 'Metropolis-Hastings / emcee'

        for i, name in enumerate(param_names):
            samples = flat_chain[:, i]
            p16, p50, p84 = np.percentile(samples, [16, 50, 84])
            safe = name.upper().replace(' ', '_')[:8]
            posterior_hdu.header[f'MED_{safe}'] = (p50, f'Median {name}')
            posterior_hdu.header[f'S16_{safe}'] = (p16, f'16th percentile {name}')
            posterior_hdu.header[f'S84_{safe}'] = (p84, f'84th percentile {name}')

        primary_hdu = fits.PrimaryHDU()
        hdulist = fits.HDUList([primary_hdu, posterior_hdu])

        filepath = os.path.join(self.output_dir, filename)
        hdulist.writeto(filepath, overwrite=True)
        print(f"MCMC posterior FITS file saved to {filepath}")
        return filepath

    def save_simulation_snapshot_fits(self,
                                       positions: np.ndarray,
                                       velocities: np.ndarray,
                                       masses: np.ndarray,
                                       cluster_labels: np.ndarray,
                                       time_Gyr: float,
                                       filename: str = 'simulation_snapshot.fits') -> str:

        cols = [
            fits.Column(name='X', format='D', unit='m', array=positions[:, 0]),
            fits.Column(name='Y', format='D', unit='m', array=positions[:, 1]),
            fits.Column(name='Z', format='D', unit='m', array=positions[:, 2]),
            fits.Column(name='VX', format='D', unit='m/s', array=velocities[:, 0]),
            fits.Column(name='VY', format='D', unit='m/s', array=velocities[:, 1]),
            fits.Column(name='VZ', format='D', unit='m/s', array=velocities[:, 2]),
            fits.Column(name='MASS', format='D', unit='kg', array=masses),
            fits.Column(name='CLUSTER_ID', format='J', unit='', array=cluster_labels.astype(int)),
        ]

        snapshot_hdu = fits.BinTableHDU.from_columns(cols)
        snapshot_hdu.name = 'SNAPSHOT'
        snapshot_hdu.header['EXTNAME'] = 'SNAPSHOT'
        snapshot_hdu.header['TIME_GYR'] = (time_Gyr, 'Simulation time [Gyr]')
        snapshot_hdu.header['N_PART'] = len(masses)
        snapshot_hdu.header['DATE'] = datetime.now().isoformat()

        primary_hdu = fits.PrimaryHDU()
        hdulist = fits.HDUList([primary_hdu, snapshot_hdu])

        filepath = os.path.join(self.output_dir, filename)
        hdulist.writeto(filepath, overwrite=True)
        print(f"Simulation snapshot FITS file saved to {filepath}")
        return filepath

    def read_ligo_frame_file(self, filepath: str, channel: str = 'H1:LDAS-STRAIN') -> Tuple[np.ndarray, float, float]:
        try:
            from gwpy.timeseries import TimeSeries
            ts = TimeSeries.read(filepath, channel)
            strain = ts.value
            t_start = ts.t0.value
            dt = ts.dt.value
            return strain, t_start, dt
        except Exception as e:
            print(f"Could not read LIGO frame file: {e}")
            print("Generating mock strain data instead.")
            n_samples = 4096 * 32
            dt = 1.0 / 4096.0
            t_start = 1234567890.0
            strain = 1.0e-21 * np.random.randn(n_samples)
            return strain, t_start, dt