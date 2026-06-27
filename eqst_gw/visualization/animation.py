import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LogNorm, Normalize
from typing import Dict, List, Tuple, Optional, Callable
import os


class AnimationGenerator:
    def __init__(self, output_dir: str = './outputs/plots/animations/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'figure.dpi': 100,
        })

    def animate_bubble_nucleation_2d(self,
                                      field_history: List[np.ndarray],
                                      time_history: List[float],
                                      dx: float,
                                      z_slice: int = None,
                                      filename: str = 'bubble_nucleation.mp4',
                                      fps: int = 10,
                                      cmap: str = 'seismic') -> str:

        if len(field_history) == 0:
            print("No field history to animate.")
            return ""

        L = field_history[0].shape[0]
        if z_slice is None:
            z_slice = L // 2

        all_values = np.concatenate([f[:, :, z_slice].ravel() for f in field_history])
        v_abs = max(abs(np.percentile(all_values, 2)), abs(np.percentile(all_values, 98)))
        v_min, v_max = -v_abs, v_abs

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax_field = axes[0]
        ax_stats = axes[1]

        extent = [0, L * dx / 1.0e-15, 0, L * dx / 1.0e-15]

        field_slice_0 = field_history[0][:, :, z_slice]
        im = ax_field.imshow(field_slice_0.T, origin='lower', cmap=cmap,
                              vmin=v_min, vmax=v_max, interpolation='bilinear',
                              extent=extent, aspect='equal')
        cbar = plt.colorbar(im, ax=ax_field)
        cbar.set_label('Field $\\phi$', fontsize=11)
        ax_field.set_xlabel('X [fm]', fontsize=11)
        ax_field.set_ylabel('Y [fm]', fontsize=11)
        title_field = ax_field.set_title(f't = {time_history[0]:.2e} s', fontsize=12, fontweight='bold')

        n_frames = len(field_history)
        phi_mean_history = [np.mean(f[:, :, z_slice]) for f in field_history]
        phi_std_history = [np.std(f[:, :, z_slice]) for f in field_history]
        phi_max_history = [np.max(np.abs(f[:, :, z_slice])) for f in field_history]

        ax_stats.set_xlim(time_history[0], time_history[-1])
        ax_stats.set_xlabel('Time [s]', fontsize=11)
        ax_stats.set_ylabel('Field Statistics', fontsize=11)
        ax_stats.set_title('Phase Transition Dynamics', fontsize=12, fontweight='bold')
        ax_stats.grid(True, alpha=0.3)

        line_mean, = ax_stats.plot([], [], 'b-', linewidth=2, label=r'$\langle\phi\rangle$')
        line_std, = ax_stats.plot([], [], 'r-', linewidth=2, label=r'$\sigma_\phi$')
        line_max, = ax_stats.plot([], [], 'g--', linewidth=2, label=r'$|\phi|_{\rm max}$')
        ax_stats.legend(fontsize=9, loc='upper left')

        y_max_stats = max(max(phi_std_history), max(phi_max_history)) * 1.1
        y_min_stats = min(min(phi_mean_history) - max(phi_std_history), 0) * 1.1
        ax_stats.set_ylim(y_min_stats, y_max_stats)

        time_plot = []
        mean_plot = []
        std_plot = []
        max_plot = []

        def update(frame):
            field_slice = field_history[frame][:, :, z_slice]
            im.set_data(field_slice.T)
            title_field.set_text(f'Bubble Nucleation: t = {time_history[frame]:.2e} s')

            time_plot.append(time_history[frame])
            mean_plot.append(phi_mean_history[frame])
            std_plot.append(phi_std_history[frame])
            max_plot.append(phi_max_history[frame])

            line_mean.set_data(time_plot, mean_plot)
            line_std.set_data(time_plot, std_plot)
            line_max.set_data(time_plot, max_plot)

            return [im, title_field, line_mean, line_std, line_max]

        plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=n_frames,
                              interval=1000.0 / fps, blit=True)

        filepath = os.path.join(self.output_dir, filename)

        try:
            writer = FFMpegWriter(fps=fps, metadata={'artist': 'EQST-GP Framework'}, bitrate=2000)
            anim.save(filepath, writer=writer, dpi=120)
            print(f"Bubble nucleation animation (MP4) saved to {filepath}")
        except Exception:
            gif_path = filepath.replace('.mp4', '.gif')
            writer_gif = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer_gif, dpi=100)
            filepath = gif_path
            print(f"Bubble nucleation animation (GIF) saved to {filepath}")

        plt.close()
        return filepath

    def animate_cluster_merger_3d_projection(self,
                                              trajectory_history: List[Dict],
                                              cluster_labels: np.ndarray,
                                              filename: str = 'cluster_merger.mp4',
                                              fps: int = 10) -> str:

        if len(trajectory_history) == 0:
            print("No trajectory history to animate.")
            return ""

        all_pos = np.vstack([snap['positions'] for snap in trajectory_history])
        kpc = 3.086e19
        x_range = (np.min(all_pos[:, 0]) / kpc, np.max(all_pos[:, 0]) / kpc)
        y_range = (np.min(all_pos[:, 1]) / kpc, np.max(all_pos[:, 1]) / kpc)
        z_range = (np.min(all_pos[:, 2]) / kpc, np.max(all_pos[:, 2]) / kpc)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax_xy, ax_xz, ax_yz = axes

        mask1 = cluster_labels == 1
        mask2 = cluster_labels == 2

        pos0 = trajectory_history[0]['positions']

        sc1_xy = ax_xy.scatter(pos0[mask1, 0] / kpc, pos0[mask1, 1] / kpc,
                                s=1, c='blue', alpha=0.5, label='Cluster 1')
        sc2_xy = ax_xy.scatter(pos0[mask2, 0] / kpc, pos0[mask2, 1] / kpc,
                                s=1, c='red', alpha=0.5, label='Cluster 2')
        ax_xy.set_xlim(*x_range)
        ax_xy.set_ylim(*y_range)
        ax_xy.set_xlabel('X [kpc]', fontsize=11)
        ax_xy.set_ylabel('Y [kpc]', fontsize=11)
        ax_xy.legend(fontsize=9, loc='upper right', markerscale=5)
        ax_xy.grid(True, alpha=0.3)
        title_xy = ax_xy.set_title(f't = {trajectory_history[0]["time_Gyr"]:.2f} Gyr',
                                    fontsize=12, fontweight='bold')

        sc1_xz = ax_xz.scatter(pos0[mask1, 0] / kpc, pos0[mask1, 2] / kpc,
                                s=1, c='blue', alpha=0.5)
        sc2_xz = ax_xz.scatter(pos0[mask2, 0] / kpc, pos0[mask2, 2] / kpc,
                                s=1, c='red', alpha=0.5)
        ax_xz.set_xlim(*x_range)
        ax_xz.set_ylim(*z_range)
        ax_xz.set_xlabel('X [kpc]', fontsize=11)
        ax_xz.set_ylabel('Z [kpc]', fontsize=11)
        ax_xz.grid(True, alpha=0.3)
        title_xz = ax_xz.set_title('X-Z Projection', fontsize=12, fontweight='bold')

        sc1_yz = ax_yz.scatter(pos0[mask1, 1] / kpc, pos0[mask1, 2] / kpc,
                                s=1, c='blue', alpha=0.5)
        sc2_yz = ax_yz.scatter(pos0[mask2, 1] / kpc, pos0[mask2, 2] / kpc,
                                s=1, c='red', alpha=0.5)
        ax_yz.set_xlim(*y_range)
        ax_yz.set_ylim(*z_range)
        ax_yz.set_xlabel('Y [kpc]', fontsize=11)
        ax_yz.set_ylabel('Z [kpc]', fontsize=11)
        ax_yz.grid(True, alpha=0.3)
        title_yz = ax_yz.set_title('Y-Z Projection', fontsize=12, fontweight='bold')

        def update(frame):
            snapshot = trajectory_history[frame]
            pos = snapshot['positions']

            sc1_xy.set_offsets(pos[mask1, :2] / kpc)
            sc2_xy.set_offsets(pos[mask2, :2] / kpc)
            title_xy.set_text(f'Galaxy Cluster Merger: t = {snapshot["time_Gyr"]:.2f} Gyr')

            sc1_xz.set_offsets(np.column_stack([pos[mask1, 0], pos[mask1, 2]]) / kpc)
            sc2_xz.set_offsets(np.column_stack([pos[mask2, 0], pos[mask2, 2]]) / kpc)

            sc1_yz.set_offsets(np.column_stack([pos[mask1, 1], pos[mask1, 2]]) / kpc)
            sc2_yz.set_offsets(np.column_stack([pos[mask2, 1], pos[mask2, 2]]) / kpc)

            return [sc1_xy, sc2_xy, sc1_xz, sc2_xz, sc1_yz, sc2_yz,
                    title_xy, title_xz, title_yz]

        plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=len(trajectory_history),
                              interval=1000.0 / fps, blit=True)

        filepath = os.path.join(self.output_dir, filename)

        try:
            writer = FFMpegWriter(fps=fps, metadata={'artist': 'EQST-GP Framework'}, bitrate=3000)
            anim.save(filepath, writer=writer, dpi=120)
            print(f"Cluster merger animation (MP4) saved to {filepath}")
        except Exception:
            gif_path = filepath.replace('.mp4', '.gif')
            writer_gif = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer_gif, dpi=100)
            filepath = gif_path
            print(f"Cluster merger animation (GIF) saved to {filepath}")

        plt.close()
        return filepath

    def animate_gw_spectrum_evolution(self,
                                       f_array: np.ndarray,
                                       spectrum_history: List[np.ndarray],
                                       time_labels: List[str],
                                       sensitivity_curve: Optional[np.ndarray] = None,
                                       filename: str = 'gw_spectrum_evolution.mp4',
                                       fps: int = 5) -> str:

        if len(spectrum_history) == 0:
            print("No spectrum history to animate.")
            return ""

        fig, ax = plt.subplots(figsize=(12, 7))

        y_min = np.min([np.min(s[s > 0]) for s in spectrum_history if np.any(s > 0)]) * 0.1
        y_max = np.max([np.max(s) for s in spectrum_history]) * 10.0

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(f_array[0], f_array[-1])
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Frequency [Hz]', fontsize=13)
        ax.set_ylabel(r'$\Omega_{\rm GW} h^2$', fontsize=13)
        ax.grid(True, which='both', alpha=0.3)

        if sensitivity_curve is not None:
            ax.loglog(f_array, sensitivity_curve, 'b--', linewidth=2.0,
                      alpha=0.6, label='LISA Sensitivity')

        line_spectrum, = ax.plot([], [], 'k-', linewidth=2.5,
                                  label='EQST-GP Spectrum')

        ax.axvline(x=1.87e-3, color='red', linestyle=':', linewidth=1.5,
                   alpha=0.8, label=r'$f_{\rm sw,peak}$')
        ax.axvline(x=3.2e-3, color='orange', linestyle=':', linewidth=1.5,
                   alpha=0.8, label=r'$f_{\rm turb,peak}$')

        ax.legend(fontsize=11, loc='lower left')

        title = ax.set_title(f'GW Spectrum Evolution: {time_labels[0]}',
                              fontsize=13, fontweight='bold')

        def update(frame):
            spectrum = spectrum_history[frame]
            mask = spectrum > 0
            if np.any(mask):
                line_spectrum.set_data(f_array[mask], spectrum[mask])
            title.set_text(f'GW Spectrum Evolution: {time_labels[frame]}')
            return [line_spectrum, title]

        anim = FuncAnimation(fig, update, frames=len(spectrum_history),
                              interval=1000.0 / fps, blit=True)

        filepath = os.path.join(self.output_dir, filename)

        try:
            writer = FFMpegWriter(fps=fps, metadata={'artist': 'EQST-GP Framework'}, bitrate=1500)
            anim.save(filepath, writer=writer, dpi=120)
            print(f"GW spectrum evolution animation (MP4) saved to {filepath}")
        except Exception:
            gif_path = filepath.replace('.mp4', '.gif')
            writer_gif = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer_gif, dpi=100)
            filepath = gif_path
            print(f"GW spectrum evolution animation (GIF) saved to {filepath}")

        plt.close()
        return filepath

    def animate_rotation_curve_halo_growth(self,
                                            r_array: np.ndarray,
                                            v_curves_history: List[Dict[str, np.ndarray]],
                                            z_labels: List[float],
                                            filename: str = 'rotation_curve_evolution.mp4',
                                            fps: int = 5) -> str:

        if len(v_curves_history) == 0:
            print("No rotation curve history to animate.")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_rc, ax_profile = axes

        kpc = 3.086e19

        y_max_rc = max([max(np.max(vc.get('total', np.zeros(1))), 1.0)
                         for vc in v_curves_history]) / 1000.0 * 1.2

        ax_rc.set_xlim(r_array[0] / kpc, r_array[-1] / kpc)
        ax_rc.set_ylim(0, y_max_rc)
        ax_rc.set_xlabel('Radius [kpc]', fontsize=12)
        ax_rc.set_ylabel('Circular Velocity [km/s]', fontsize=12)
        ax_rc.grid(True, alpha=0.3)
        title_rc = ax_rc.set_title(f'Rotation Curve at z = {z_labels[0]:.2f}',
                                    fontsize=13, fontweight='bold')

        line_total, = ax_rc.plot([], [], 'k-', linewidth=2.5, label='EQST-GP Total')
        line_dm, = ax_rc.plot([], [], 'b--', linewidth=2.0, label='DM Component')
        line_baryon, = ax_rc.plot([], [], 'g:', linewidth=2.0, label='Baryonic')
        line_nfw, = ax_rc.plot([], [], 'r--', linewidth=2.0, alpha=0.7, label='NFW (ΛCDM)')
        ax_rc.legend(fontsize=10, loc='lower right')

        y_max_profile = max([max(np.max(vc.get('density', np.ones(1))), 1.0)
                               for vc in v_curves_history]) * 10.0
        y_min_profile = min([min(np.min(vc.get('density', np.ones(1))[v_curves_history[0].get('density', np.ones(1)) > 0]), 1.0)
                               for vc in v_curves_history]) * 0.1

        ax_profile.set_xscale('log')
        ax_profile.set_yscale('log')
        ax_profile.set_xlim(r_array[0] / kpc, r_array[-1] / kpc)
        ax_profile.set_ylim(max(y_min_profile, 1.0e-30), y_max_profile)
        ax_profile.set_xlabel('Radius [kpc]', fontsize=12)
        ax_profile.set_ylabel(r'$\rho_{\rm DM}$ [kg/m$^3$]', fontsize=12)
        ax_profile.grid(True, which='both', alpha=0.3)
        title_profile = ax_profile.set_title(f'DM Density Profile at z = {z_labels[0]:.2f}',
                                              fontsize=13, fontweight='bold')

        line_density_eqst, = ax_profile.plot([], [], 'b-', linewidth=2.5, label='EQST-GP')
        line_density_nfw, = ax_profile.plot([], [], 'r--', linewidth=2.0, label='NFW')
        ax_profile.legend(fontsize=10, loc='upper right')

        r_kpc = r_array / kpc

        def update(frame):
            vc = v_curves_history[frame]
            z_label = z_labels[frame]

            if 'total' in vc and len(vc['total']) == len(r_kpc):
                line_total.set_data(r_kpc, vc['total'] / 1000.0)
            if 'dm' in vc and len(vc['dm']) == len(r_kpc):
                line_dm.set_data(r_kpc, vc['dm'] / 1000.0)
            if 'baryon' in vc and len(vc['baryon']) == len(r_kpc):
                line_baryon.set_data(r_kpc, vc['baryon'] / 1000.0)
            if 'nfw' in vc and len(vc['nfw']) == len(r_kpc):
                line_nfw.set_data(r_kpc, vc['nfw'] / 1000.0)

            title_rc.set_text(f'Rotation Curve at z = {z_label:.2f}')

            if 'density' in vc:
                rho = vc['density']
                mask = rho > 0
                if np.any(mask):
                    line_density_eqst.set_data(r_kpc[mask], rho[mask])
            if 'density_nfw' in vc:
                rho_nfw = vc['density_nfw']
                mask_nfw = rho_nfw > 0
                if np.any(mask_nfw):
                    line_density_nfw.set_data(r_kpc[mask_nfw], rho_nfw[mask_nfw])

            title_profile.set_text(f'DM Density Profile at z = {z_label:.2f}')

            return [line_total, line_dm, line_baryon, line_nfw,
                    line_density_eqst, line_density_nfw,
                    title_rc, title_profile]

        plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=len(v_curves_history),
                              interval=1000.0 / fps, blit=True)

        filepath = os.path.join(self.output_dir, filename)

        try:
            writer = FFMpegWriter(fps=fps, metadata={'artist': 'EQST-GP Framework'}, bitrate=2000)
            anim.save(filepath, writer=writer, dpi=120)
            print(f"Rotation curve evolution animation (MP4) saved to {filepath}")
        except Exception:
            gif_path = filepath.replace('.mp4', '.gif')
            writer_gif = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer_gif, dpi=100)
            filepath = gif_path
            print(f"Rotation curve evolution animation (GIF) saved to {filepath}")

        plt.close()
        return filepath