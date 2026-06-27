import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from typing import Tuple, Dict, Optional, List
import os

class SpectraPlotter:
    def __init__(self, output_dir: str = './outputs/plots/', style: str = 'publication'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if style == 'publication':
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'axes.labelsize': 14,
                'axes.titlesize': 15,
                'legend.fontsize': 11,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'figure.dpi': 150,
                'savefig.dpi': 300,
                'text.usetex': False
            })
    
    def plot_gw_spectrum_complete(self, f: np.ndarray, components: Dict[str, np.ndarray], sensitivity_curves: Dict[str, np.ndarray], filename: str = 'gw_spectrum_complete.pdf'):
        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
        
        ax1 = fig.add_subplot(gs[0, :])
        
        ax1.loglog(f, components['total'], 'k-', linewidth=2.5, label='EQST-GP Total', zorder=5)
        ax1.loglog(f, components['sound_waves'], 'b-', linewidth=2.0, alpha=0.8, label='Sound Waves', zorder=4)
        ax1.loglog(f, components['turbulence'], 'r-', linewidth=2.0, alpha=0.8, label='MHD Turbulence', zorder=3)
        ax1.loglog(f, components['bubble_collisions'], 'g--', linewidth=1.5, alpha=0.7, label='Bubble Collisions', zorder=2)
        
        colors_det = {'LISA': 'blue', 'LIGO': 'orange', 'ET': 'purple', 'Virgo': 'green'}
        for det_name, sens_curve in sensitivity_curves.items():
            color = colors_det.get(det_name, 'gray')
            ax1.loglog(f, sens_curve, '--', linewidth=1.5, color=color, alpha=0.6, label=f'{det_name} Sensitivity')
        
        ax1.fill_between(f, components['total'], sensitivity_curves.get('LISA', np.ones_like(f) * 1.0e-20), where=(components['total'] > sensitivity_curves.get('LISA', np.zeros_like(f))), alpha=0.15, color='blue', label='LISA Detectable Region')
        
        ax1.set_xlabel('Frequency [Hz]', fontsize=14)
        ax1.set_ylabel(r'$\Omega_{\rm GW} h^2$', fontsize=14)
        ax1.set_xlim(1.0e-5, 1.0e2)
        ax1.set_ylim(1.0e-18, 1.0e-10)
        ax1.legend(fontsize=10, loc='upper right', ncol=2)
        ax1.grid(True, which='both', alpha=0.3)
        ax1.set_title('EQST-GP Gravitational Wave Spectrum vs Detector Sensitivities', fontsize=15, fontweight='bold')
        
        ax2 = fig.add_subplot(gs[1, 0])
        
        f_LISA_band = f[(f >= 1.0e-4) & (f <= 1.0e-1)]
        Omega_LISA_band = components['total'][(f >= 1.0e-4) & (f <= 1.0e-1)]
        Omega_sound_band = components['sound_waves'][(f >= 1.0e-4) & (f <= 1.0e-1)]
        Omega_turb_band = components['turbulence'][(f >= 1.0e-4) & (f <= 1.0e-1)]
        
        ax2.loglog(f_LISA_band, Omega_LISA_band, 'k-', linewidth=2.5, label='Total')
        ax2.loglog(f_LISA_band, Omega_sound_band, 'b-', linewidth=2.0, alpha=0.8, label='Sound Waves')
        ax2.loglog(f_LISA_band, Omega_turb_band, 'r-', linewidth=2.0, alpha=0.8, label='Turbulence')
        
        if 'LISA' in sensitivity_curves:
            LISA_band_sens = sensitivity_curves['LISA'][(f >= 1.0e-4) & (f <= 1.0e-1)]
            ax2.loglog(f_LISA_band, LISA_band_sens, 'b--', linewidth=1.5, alpha=0.6, label='LISA Sensitivity')
        
        ax2.axvline(x=1.87e-3, color='black', linestyle=':', linewidth=1.5, alpha=0.8, label=r'$f_{\rm sw,peak}$')
        ax2.axvline(x=3.2e-3, color='red', linestyle=':', linewidth=1.5, alpha=0.8, label=r'$f_{\rm turb,peak}$')
        
        ax2.set_xlabel('Frequency [Hz]', fontsize=13)
        ax2.set_ylabel(r'$\Omega_{\rm GW} h^2$', fontsize=13)
        ax2.set_xlim(1.0e-4, 1.0e-1)
        ax2.set_ylim(1.0e-16, 1.0e-11)
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_title('LISA Band Detail', fontsize=13, fontweight='bold')
        
        ax3 = fig.add_subplot(gs[1, 1])
        
        f_ET_band = f[(f >= 1.0) & (f <= 1.0e4)]
        Omega_ET_band = components['total'][(f >= 1.0) & (f <= 1.0e4)]
        
        ax3.loglog(f_ET_band, Omega_ET_band, 'k-', linewidth=2.5, label='Total')
        
        if 'ET' in sensitivity_curves:
            ET_band_sens = sensitivity_curves['ET'][(f >= 1.0) & (f <= 1.0e4)]
            ax3.loglog(f_ET_band, ET_band_sens, 'purple', linestyle='--', linewidth=1.5, alpha=0.6, label='ET Sensitivity')
        
        ax3.set_xlabel('Frequency [Hz]', fontsize=13)
        ax3.set_ylabel(r'$\Omega_{\rm GW} h^2$', fontsize=13)
        ax3.legend(fontsize=10, loc='upper right')
        ax3.grid(True, which='both', alpha=0.3)
        ax3.set_title('Einstein Telescope Band', fontsize=13, fontweight='bold')
        
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Complete GW spectrum plot saved to {os.path.join(self.output_dir, filename)}")
    
    def plot_parameter_constraints(self, chain: np.ndarray, param_names: List[str], truths: Optional[List[float]] = None, filename: str = 'parameter_constraints.pdf'):
        try:
            import corner as corner_module
            
            n_walkers, n_steps, n_dim = chain.shape
            
            flat_chain = chain.reshape(-1, n_dim)
            
            labels = [f'${name}$' for name in param_names]
            
            fig = corner_module.corner(flat_chain, labels=labels, truths=truths, truth_color='red', quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={'fontsize': 11}, label_kwargs={'fontsize': 12}, hist_kwargs={'density': True, 'alpha': 0.7}, plot_contours=True, fill_contours=True, levels=(0.68, 0.95), smooth=1.0)
            
            fig.suptitle('EQST-GP Parameter Posterior Distributions', fontsize=16, fontweight='bold', y=1.02)
            
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Corner plot saved to {os.path.join(self.output_dir, filename)}")
            
        except ImportError:
            print("corner package not available, using basic scatter plot")
            
            n_walkers, n_steps, n_dim = chain.shape
            flat_chain = chain.reshape(-1, n_dim)
            
            fig, axes = plt.subplots(n_dim, n_dim, figsize=(14, 14))
            
            for i in range(n_dim):
                for j in range(n_dim):
                    ax = axes[i, j]
                    if i == j:
                        ax.hist(flat_chain[:, i], bins=50, density=True, color='steelblue', alpha=0.7)
                        if truths is not None:
                            ax.axvline(truths[i], color='red', linewidth=2)
                    elif i > j:
                        ax.scatter(flat_chain[::10, j], flat_chain[::10, i], s=1, alpha=0.3, color='steelblue')
                        if truths is not None:
                            ax.axvline(truths[j], color='red', linewidth=2)
                            ax.axhline(truths[i], color='red', linewidth=2)
                    else:
                        ax.axis('off')
                    
                    if i == n_dim - 1:
                        ax.set_xlabel(param_names[j], fontsize=11)
                    if j == 0 and i > 0:
                        ax.set_ylabel(param_names[i], fontsize=11)
            
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()