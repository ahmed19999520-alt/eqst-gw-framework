import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict, Optional
import os


class CornerPlotter:
    def __init__(self, output_dir: str = './outputs/plots/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.labelsize': 13,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 150,
            'savefig.dpi': 300,
        })

    def plot_corner(self,
                    flat_chain: np.ndarray,
                    param_names: List[str],
                    param_labels: Optional[List[str]] = None,
                    truths: Optional[List[float]] = None,
                    truth_color: str = 'red',
                    contour_levels: Tuple[float, ...] = (0.68, 0.95),
                    color: str = 'steelblue',
                    filename: str = 'corner_plot.pdf',
                    title: Optional[str] = None,
                    show_titles: bool = True,
                    smooth: float = 1.0,
                    range_percentile: float = 99.0) -> plt.Figure:

        n_params = flat_chain.shape[1]

        if param_labels is None:
            param_labels = [f'${name}$' for name in param_names]

        fig = plt.figure(figsize=(3.5 * n_params, 3.5 * n_params))
        gs = gridspec.GridSpec(n_params, n_params, figure=fig,
                               hspace=0.05, wspace=0.05)

        axes = np.empty((n_params, n_params), dtype=object)
        for i in range(n_params):
            for j in range(n_params):
                if i >= j:
                    axes[i, j] = fig.add_subplot(gs[i, j])
                else:
                    axes[i, j] = None

        param_ranges = []
        for i in range(n_params):
            lo = np.percentile(flat_chain[:, i], (100.0 - range_percentile) / 2.0)
            hi = np.percentile(flat_chain[:, i], 100.0 - (100.0 - range_percentile) / 2.0)
            padding = 0.1 * (hi - lo)
            param_ranges.append((lo - padding, hi + padding))

        for i in range(n_params):
            ax = axes[i, i]

            samples_i = flat_chain[:, i]
            lo_i, hi_i = param_ranges[i]

            x_hist = np.linspace(lo_i, hi_i, 200)
            try:
                kde = gaussian_kde(samples_i, bw_method='scott')
                y_hist = kde(x_hist)
                if smooth > 0:
                    y_hist = gaussian_filter(y_hist, sigma=smooth)
            except Exception:
                y_hist, x_edges = np.histogram(samples_i, bins=50,
                                                 range=(lo_i, hi_i), density=True)
                x_hist = 0.5 * (x_edges[:-1] + x_edges[1:])

            ax.plot(x_hist, y_hist, color=color, linewidth=1.5)
            ax.fill_between(x_hist, y_hist, alpha=0.3, color=color)

            p16 = np.percentile(samples_i, 16)
            p50 = np.percentile(samples_i, 50)
            p84 = np.percentile(samples_i, 84)

            ax.axvline(p16, color=color, linestyle='--', linewidth=1.0, alpha=0.8)
            ax.axvline(p50, color=color, linestyle='-', linewidth=1.5)
            ax.axvline(p84, color=color, linestyle='--', linewidth=1.0, alpha=0.8)

            if truths is not None and truths[i] is not None:
                ax.axvline(truths[i], color=truth_color, linewidth=2.0, linestyle='-')

            if show_titles:
                err_minus = p50 - p16
                err_plus = p84 - p50
                title_str = f'${p50:.3g}^{{+{err_plus:.2g}}}_{{-{err_minus:.2g}}}$'
                ax.set_title(title_str, fontsize=10, pad=4)

            ax.set_xlim(lo_i, hi_i)
            ax.set_yticks([])

            if i == n_params - 1:
                ax.set_xlabel(param_labels[i], fontsize=12)
            else:
                ax.set_xticklabels([])

        for i in range(n_params):
            for j in range(i):
                ax = axes[i, j]

                samples_j = flat_chain[:, j]
                samples_i = flat_chain[:, i]

                lo_j, hi_j = param_ranges[j]
                lo_i, hi_i = param_ranges[i]

                x_grid = np.linspace(lo_j, hi_j, 80)
                y_grid = np.linspace(lo_i, hi_i, 80)
                X, Y = np.meshgrid(x_grid, y_grid)

                try:
                    positions = np.vstack([samples_j, samples_i])
                    kde_2d = gaussian_kde(positions, bw_method='scott')
                    xy_grid = np.vstack([X.ravel(), Y.ravel()])
                    Z = kde_2d(xy_grid).reshape(X.shape)

                    if smooth > 0:
                        Z = gaussian_filter(Z, sigma=smooth)

                    Z_sorted = np.sort(Z.ravel())[::-1]
                    Z_cumsum = np.cumsum(Z_sorted)
                    Z_cumsum /= Z_cumsum[-1]

                    contour_values = []
                    for level in contour_levels:
                        idx = np.searchsorted(Z_cumsum, level)
                        idx = min(idx, len(Z_sorted) - 1)
                        contour_values.append(Z_sorted[idx])

                    contour_values = sorted(contour_values)

                    ax.contourf(X, Y, Z, levels=contour_values + [Z.max() * 1.01],
                                colors=[color], alphas=[0.15, 0.35])
                    ax.contour(X, Y, Z, levels=contour_values,
                               colors=[color], linewidths=[0.8, 1.2])

                except Exception:
                    ax.hexbin(samples_j, samples_i, gridsize=25,
                              cmap='Blues', mincnt=1, bins='log')

                if truths is not None:
                    if truths[j] is not None:
                        ax.axvline(truths[j], color=truth_color, linewidth=1.5, linestyle='-')
                    if truths[i] is not None:
                        ax.axhline(truths[i], color=truth_color, linewidth=1.5, linestyle='-')
                    if truths[j] is not None and truths[i] is not None:
                        ax.plot(truths[j], truths[i], 'o',
                                color=truth_color, markersize=4, zorder=10)

                ax.set_xlim(lo_j, hi_j)
                ax.set_ylim(lo_i, hi_i)

                if j == 0:
                    ax.set_ylabel(param_labels[i], fontsize=12)
                else:
                    ax.set_yticklabels([])

                if i == n_params - 1:
                    ax.set_xlabel(param_labels[j], fontsize=12)
                else:
                    ax.set_xticklabels([])

        for i in range(n_params):
            for j in range(n_params):
                if axes[i, j] is None:
                    ax_dummy = fig.add_subplot(gs[i, j])
                    ax_dummy.set_visible(False)

        if title is not None:
            fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Corner plot saved to {filepath}")

        return fig

    def plot_parameter_evolution(self,
                                  chain: np.ndarray,
                                  param_names: List[str],
                                  param_labels: Optional[List[str]] = None,
                                  truths: Optional[List[float]] = None,
                                  filename: str = 'chain_evolution.pdf') -> plt.Figure:

        n_walkers, n_steps, n_params = chain.shape

        if param_labels is None:
            param_labels = [f'${name}$' for name in param_names]

        fig, axes = plt.subplots(n_params + 1, 1,
                                  figsize=(12, 2.5 * (n_params + 1)),
                                  sharex=True)

        steps = np.arange(n_steps)

        for i in range(n_params):
            ax = axes[i]

            for w in range(min(n_walkers, 20)):
                ax.plot(steps, chain[w, :, i],
                        color='steelblue', linewidth=0.4, alpha=0.4)

            median_chain = np.median(chain[:, :, i], axis=0)
            p16_chain = np.percentile(chain[:, :, i], 16, axis=0)
            p84_chain = np.percentile(chain[:, :, i], 84, axis=0)

            ax.plot(steps, median_chain, color='navy', linewidth=1.5)
            ax.fill_between(steps, p16_chain, p84_chain, color='steelblue', alpha=0.25)

            if truths is not None and truths[i] is not None:
                ax.axhline(truths[i], color='red', linestyle='--', linewidth=1.5)

            ax.set_ylabel(param_labels[i], fontsize=12)
            ax.grid(True, alpha=0.3)

        ax_log = axes[-1]
        if hasattr(self, '_log_prob_cache') and self._log_prob_cache is not None:
            for w in range(min(n_walkers, 20)):
                ax_log.plot(steps, self._log_prob_cache[w, :],
                             color='darkorange', linewidth=0.4, alpha=0.4)
            ax_log.plot(steps, np.median(self._log_prob_cache, axis=0),
                         color='darkorange', linewidth=1.5)
        else:
            ax_log.plot(steps, np.zeros(n_steps), color='gray', linewidth=1.0)
            ax_log.text(0.5, 0.5, 'log_prob not available',
                        transform=ax_log.transAxes, ha='center', va='center')

        ax_log.set_ylabel('log posterior', fontsize=12)
        ax_log.set_xlabel('Step', fontsize=12)
        ax_log.grid(True, alpha=0.3)

        fig.suptitle('MCMC Chain Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Chain evolution plot saved to {filepath}")

        return fig

    def plot_1d_posteriors_comparison(self,
                                       chains_dict: Dict[str, np.ndarray],
                                       param_names: List[str],
                                       param_labels: Optional[List[str]] = None,
                                       truths: Optional[List[float]] = None,
                                       colors: Optional[List[str]] = None,
                                       filename: str = 'posterior_comparison.pdf') -> plt.Figure:

        n_params = len(param_names)
        model_names = list(chains_dict.keys())
        n_models = len(model_names)

        if param_labels is None:
            param_labels = [f'${name}$' for name in param_names]

        if colors is None:
            colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']
            colors = colors[:n_models]

        ncols = min(4, n_params)
        nrows = (n_params + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(4.5 * ncols, 3.5 * nrows))
        axes = np.atleast_2d(axes).reshape(nrows, ncols)

        for param_idx in range(n_params):
            row = param_idx // ncols
            col = param_idx % ncols
            ax = axes[row, col]

            x_all = []
            for model_name, flat_chain in chains_dict.items():
                x_all.extend(flat_chain[:, param_idx].tolist())

            x_lo = np.percentile(x_all, 0.5)
            x_hi = np.percentile(x_all, 99.5)
            x_plot = np.linspace(x_lo, x_hi, 300)

            for model_idx, (model_name, flat_chain) in enumerate(chains_dict.items()):
                samples = flat_chain[:, param_idx]
                color = colors[model_idx]

                try:
                    kde = gaussian_kde(samples, bw_method='scott')
                    y_kde = kde(x_plot)
                    y_kde = gaussian_filter(y_kde, sigma=1.0)
                except Exception:
                    y_kde, x_edges = np.histogram(samples, bins=50,
                                                    range=(x_lo, x_hi), density=True)
                    x_plot = 0.5 * (x_edges[:-1] + x_edges[1:])
                    y_kde = y_kde.astype(float)

                ax.plot(x_plot, y_kde, color=color, linewidth=2.0,
                        label=model_name)
                ax.fill_between(x_plot, y_kde, alpha=0.15, color=color)

                p50 = np.median(samples)
                ax.axvline(p50, color=color, linestyle='--', linewidth=1.0, alpha=0.8)

            if truths is not None and truths[param_idx] is not None:
                ax.axvline(truths[param_idx], color='black',
                           linestyle='-', linewidth=2.0, label='Truth', zorder=10)

            ax.set_xlabel(param_labels[param_idx], fontsize=12)
            ax.set_ylabel('Posterior density', fontsize=10)
            ax.set_xlim(x_lo, x_hi)
            ax.grid(True, alpha=0.3)

            if param_idx == 0:
                ax.legend(fontsize=9, loc='best')

        for idx in range(n_params, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].set_visible(False)

        fig.suptitle('Parameter Posterior Comparison: EQST-GP vs Alternative Models',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Posterior comparison plot saved to {filepath}")

        return fig