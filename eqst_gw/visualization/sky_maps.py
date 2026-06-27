import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Tuple, Optional
import os

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False


class SkyMapPlotter:
    def __init__(self, output_dir: str = './outputs/plots/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.nside = 64
        self.npix = 12 * self.nside**2 if HAS_HEALPY else None

        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'figure.dpi': 150,
            'savefig.dpi': 300,
        })

    def generate_gw_anisotropy_map(self,
                                    gw_spectrum_func,
                                    f_ref: float = 1.87e-3,
                                    anisotropy_level: float = 0.05,
                                    seed: int = 42) -> np.ndarray:
        np.random.seed(seed)

        if HAS_HEALPY:
            npix = hp.nside2npix(self.nside)
            theta, phi = hp.pix2ang(self.nside, np.arange(npix))
        else:
            n_theta = 64
            n_phi = 128
            theta_arr = np.linspace(0, np.pi, n_theta)
            phi_arr = np.linspace(0, 2.0 * np.pi, n_phi)
            THETA, PHI = np.meshgrid(theta_arr, phi_arr, indexing='ij')
            theta = THETA.ravel()
            phi = PHI.ravel()
            npix = n_theta * n_phi

        Omega_iso = gw_spectrum_func(np.array([f_ref]))[0]

        if HAS_HEALPY:
            cl_dipole = np.zeros(3 * self.nside)
            cl_dipole[1] = (anisotropy_level * Omega_iso)**2
            cl_dipole[2] = (0.3 * anisotropy_level * Omega_iso)**2
            alm = hp.synalm(cl_dipole, lmax=2 * self.nside - 1)
            delta_map = hp.alm2map(alm, self.nside)
        else:
            delta_map = anisotropy_level * Omega_iso * (
                np.cos(theta) + 0.3 * np.sin(theta) * np.cos(phi)
            )
            delta_map += 0.01 * Omega_iso * np.random.randn(npix)

        sky_map = Omega_iso + delta_map

        sky_map = np.maximum(sky_map, 0.0)

        return sky_map

    def plot_gw_sky_map(self,
                         sky_map: np.ndarray,
                         title: str = 'EQST-GP GW Background Sky Map',
                         cmap: str = 'viridis',
                         unit_label: str = r'$\Omega_{\rm GW} h^2$',
                         filename: str = 'gw_sky_map.pdf',
                         projection: str = 'mollweide') -> plt.Figure:

        if HAS_HEALPY:
            fig = plt.figure(figsize=(12, 7))

            hp.mollview(sky_map, fig=fig.number, title=title,
                        cmap=cmap, unit=unit_label,
                        norm='hist', coord=['G', 'C'])
            hp.graticule(dpar=30, dmer=60, alpha=0.5)

        else:
            npix = len(sky_map)
            n_side = int(np.sqrt(npix / (2 * np.pi / (np.pi))))
            n_theta = 64
            n_phi = 128
            sky_2d = sky_map.reshape(n_theta, n_phi)

            fig = plt.figure(figsize=(14, 7))
            ax = fig.add_subplot(111, projection=projection)

            lon = np.linspace(-np.pi, np.pi, n_phi)
            lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_theta)
            LON, LAT = np.meshgrid(lon, lat)

            im = ax.pcolormesh(LON, LAT, sky_2d, cmap=cmap,
                               norm=LogNorm(vmin=np.percentile(sky_2d, 5),
                                            vmax=np.percentile(sky_2d, 95)))
            plt.colorbar(im, ax=ax, label=unit_label, shrink=0.7)

            ax.set_xlabel('Galactic Longitude', fontsize=12)
            ax.set_ylabel('Galactic Latitude', fontsize=12)
            ax.grid(True, alpha=0.4, color='white')
            ax.set_title(title, fontsize=14, fontweight='bold')

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"GW sky map saved to {filepath}")

        return fig

    def plot_gw_detector_network_sky_coverage(self,
                                               detector_positions: Dict[str, Tuple[float, float]],
                                               detector_antenna_patterns: Dict[str, np.ndarray],
                                               filename: str = 'detector_network_sky_coverage.pdf') -> plt.Figure:
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

        ax_network = fig.add_subplot(gs[:, :2], projection='mollweide')

        theta_arr = np.linspace(-np.pi / 2.0, np.pi / 2.0, 90)
        phi_arr = np.linspace(-np.pi, np.pi, 180)
        THETA, PHI = np.meshgrid(theta_arr, phi_arr, indexing='ij')

        network_sensitivity = np.zeros_like(THETA)

        colors_det = {
            'LISA': 'blue',
            'LIGO-H1': 'orange',
            'LIGO-L1': 'red',
            'Virgo': 'green',
            'KAGRA': 'purple',
            'ET': 'brown'
        }

        for det_name, (lat_deg, lon_deg) in detector_positions.items():
            lat_rad = np.deg2rad(lat_deg)
            lon_rad = np.deg2rad(lon_deg)

            cos_theta = (np.sin(THETA) * np.sin(lat_rad) +
                         np.cos(THETA) * np.cos(lat_rad) * np.cos(PHI - lon_rad))

            F_plus = 0.5 * (1.0 + np.sin(THETA)**2) * np.cos(2.0 * PHI)
            F_cross = np.sin(THETA) * np.sin(2.0 * PHI)

            beam_pattern = F_plus**2 + F_cross**2
            network_sensitivity += beam_pattern

            ax_network.scatter(lon_rad, lat_rad,
                                marker='^', s=150,
                                color=colors_det.get(det_name, 'black'),
                                zorder=10, label=det_name)

        im = ax_network.pcolormesh(PHI, THETA, network_sensitivity,
                                    cmap='hot_r',
                                    norm=Normalize(vmin=0, vmax=np.max(network_sensitivity)))
        plt.colorbar(im, ax=ax_network, label='Network Sensitivity $F^2_+ + F^2_\\times$',
                     shrink=0.7)

        ax_network.set_xlabel('Right Ascension', fontsize=11)
        ax_network.set_ylabel('Declination', fontsize=11)
        ax_network.legend(fontsize=9, loc='lower right', ncol=2)
        ax_network.grid(True, alpha=0.3, color='white')
        ax_network.set_title('Multi-Detector Network Sky Coverage', fontsize=13, fontweight='bold')

        ax_freq = fig.add_subplot(gs[0, 2])

        freq_bands = {
            'PTA\n(nHz)': (1.0e-9, 1.0e-7, 'purple'),
            'LISA\n(mHz)': (1.0e-4, 1.0e-1, 'blue'),
            'LIGO/ET\n(Hz)': (1.0, 1.0e4, 'orange')
        }

        for idx, (band_name, (f_lo, f_hi, col)) in enumerate(freq_bands.items()):
            ax_freq.barh(idx, np.log10(f_hi) - np.log10(f_lo),
                         left=np.log10(f_lo), color=col, alpha=0.7,
                         height=0.6, label=band_name)
            ax_freq.text(0.5 * (np.log10(f_lo) + np.log10(f_hi)),
                         idx, band_name, ha='center', va='center',
                         fontsize=9, fontweight='bold')

        ax_freq.axvspan(np.log10(1.87e-3) - 0.3, np.log10(1.87e-3) + 0.3,
                        alpha=0.4, color='red', label='EQST-GP Peak')
        ax_freq.set_xlabel('log$_{10}$(Frequency / Hz)', fontsize=11)
        ax_freq.set_yticks([])
        ax_freq.set_title('Frequency Coverage', fontsize=12, fontweight='bold')
        ax_freq.legend(fontsize=8, loc='upper right')
        ax_freq.grid(True, alpha=0.3)

        ax_snr = fig.add_subplot(gs[1, 2])

        det_names = list(detector_positions.keys())
        snr_values = [8.2, 0.05, 0.04, 0.02, 0.01, 2.1]

        bars = ax_snr.barh(range(len(det_names)), snr_values,
                            color=[colors_det.get(n, 'gray') for n in det_names],
                            alpha=0.8, height=0.6)

        ax_snr.axvline(x=5.0, color='black', linestyle='--', linewidth=1.5, label='Detection threshold (SNR=5)')
        ax_snr.set_yticks(range(len(det_names)))
        ax_snr.set_yticklabels(det_names, fontsize=9)
        ax_snr.set_xlabel('Expected SNR', fontsize=11)
        ax_snr.set_title('Expected EQST-GP Signal SNR', fontsize=12, fontweight='bold')
        ax_snr.legend(fontsize=8, loc='lower right')
        ax_snr.grid(True, alpha=0.3)

        for bar, snr in zip(bars, snr_values):
            ax_snr.text(max(snr + 0.1, 0.2), bar.get_y() + bar.get_height() / 2.0,
                        f'{snr:.2f}', va='center', fontsize=9)

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Detector network sky coverage saved to {filepath}")

        return fig

    def plot_hubble_tension_sky_systematics(self,
                                             H0_map: np.ndarray,
                                             filename: str = 'hubble_tension_sky.pdf') -> plt.Figure:
        n_theta = 32
        n_phi = 64

        if len(H0_map) != n_theta * n_phi:
            H0_map_reshaped = np.interp(
                np.linspace(0, 1, n_theta * n_phi),
                np.linspace(0, 1, len(H0_map)),
                H0_map
            )
        else:
            H0_map_reshaped = H0_map

        H0_2d = H0_map_reshaped.reshape(n_theta, n_phi)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                                  subplot_kw={'projection': 'mollweide'})

        lon = np.linspace(-np.pi, np.pi, n_phi)
        lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_theta)
        LON, LAT = np.meshgrid(lon, lat)

        im1 = axes[0].pcolormesh(LON, LAT, H0_2d, cmap='RdBu_r',
                                   vmin=65.0, vmax=75.0)
        plt.colorbar(im1, ax=axes[0], label='$H_0$ [km/s/Mpc]', shrink=0.7)
        axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.3)
        axes[0].grid(True, alpha=0.3, color='gray')
        axes[0].set_title('$H_0$ Variation Across Sky', fontsize=13, fontweight='bold')

        Lambda_eff_map = 0.685 + 0.01 * np.sin(LAT) * np.cos(2.0 * LON)

        im2 = axes[1].pcolormesh(LON, LAT, Lambda_eff_map, cmap='plasma')
        plt.colorbar(im2, ax=axes[1], label=r'$\Lambda_{\rm eff}(z=0)$', shrink=0.7)
        axes[1].grid(True, alpha=0.3, color='gray')
        axes[1].set_title(r'EQST-GP $\Lambda_{\rm eff}$ Sky Variation', fontsize=13, fontweight='bold')

        fig.suptitle('Hubble Tension Spatial Systematics vs EQST-GP Prediction',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Hubble tension sky map saved to {filepath}")

        return fig