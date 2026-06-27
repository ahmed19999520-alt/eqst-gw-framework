import numpy as np
from scipy.ndimage import convolve, gaussian_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Dict, Optional

class BubbleNucleationSimulation:
    def __init__(self,
                 T_nucleation_GeV: float = 9.71e15,
                 alpha: float = 0.42,
                 beta_over_H: float = 94.7,
                 v_w: float = 0.27,
                 lattice_size: int = 128,
                 lattice_spacing_m: float = 1.0e-15,
                 constants=None,
                 eqst_params=None):
        
        self.T_n = T_nucleation_GeV
        self.alpha = alpha
        self.beta_H = beta_over_H
        self.v_w = v_w
        self.L = lattice_size
        self.dx = lattice_spacing_m
        
        if constants is None:
            from ..core.constants import FundamentalConstants
            self.c = FundamentalConstants()
        else:
            self.c = constants
        
        if eqst_params is None:
            from ..core.parameters import EQSTGPParameters
            self.ep = EQSTGPParameters()
        else:
            self.ep = eqst_params
        
        self.field = np.zeros((self.L, self.L, self.L))
        self.field_dot = np.zeros((self.L, self.L, self.L))
        
        self.bubble_centers = []
        self.bubble_nucleation_times = []
        
        self.field_history = []
        self.time_history = []
        
        self.initialize_field()
    
    def initialize_field(self):
        self.field = np.random.normal(0.0, 0.01, (self.L, self.L, self.L))
        self.field_dot = np.zeros_like(self.field)
    
    def effective_potential(self, phi: np.ndarray) -> np.ndarray:
        T = self.T_n * 1.0e9
        mu_squared = (self.ep.mu_GeV * 1.0e9)**2
        kappa = self.ep.kappa_thermal
        gamma = self.ep.gamma_thermal
        lambda_q = self.ep.lambda_quartic
        
        V = 0.5 * (-mu_squared + kappa * T**2) * phi**2 - gamma * T * phi**3 + 0.25 * lambda_q * phi**4
        
        return V / (T**4)
    
    def potential_derivative(self, phi: np.ndarray) -> np.ndarray:
        T = self.T_n * 1.0e9
        mu_squared = (self.ep.mu_GeV * 1.0e9)**2
        kappa = self.ep.kappa_thermal
        gamma = self.ep.gamma_thermal
        lambda_q = self.ep.lambda_quartic
        
        dV = (-mu_squared + kappa * T**2) * phi - 3.0 * gamma * T * phi**2 + lambda_q * phi**3
        
        return dV / (T**4)
    
    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        kernel = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        
        laplacian = convolve(field, kernel, mode='wrap') / self.dx**2
        
        return laplacian
    
    def nucleate_bubble(self, center: Tuple[int, int, int], radius: float):
        i0, j0, k0 = center
        
        for i in range(self.L):
            for j in range(self.L):
                for k in range(self.L):
                    di = min(abs(i - i0), self.L - abs(i - i0))
                    dj = min(abs(j - j0), self.L - abs(j - j0))
                    dk = min(abs(k - k0), self.L - abs(k - k0))
                    
                    r = np.sqrt((di * self.dx)**2 + (dj * self.dx)**2 + (dk * self.dx)**2)
                    
                    if r < radius:
                        phi_true = 3.0 * self.ep.gamma_thermal * self.T_n * 1.0e9 / self.ep.lambda_quartic
                        self.field[i, j, k] = phi_true * (1.0 - r / radius)
    
    def evolve_field_step(self, dt: float):
        laplacian = self.compute_laplacian(self.field)
        
        dV_dphi = self.potential_derivative(self.field)
        
        friction = 3.0 * self.compute_hubble_parameter()
        
        field_ddot = laplacian - dV_dphi - friction * self.field_dot
        
        self.field_dot += dt * field_ddot
        
        self.field += dt * self.field_dot
    
    def compute_hubble_parameter(self) -> float:
        H0_SI = self.c.H0_Planck2018 * 1000.0 / self.c.Mpc_to_m
        E_z = np.sqrt(self.c.Omega_m_Planck2018 * (1.0 + 1.0e12)**3 + self.c.Omega_Lambda_Planck2018)
        H_z = H0_SI * E_z
        return H_z
    
    def run_simulation(self, n_bubbles: int = 20, n_steps: int = 500, dt: float = 1.0e-25):
        bubble_radius = 5.0 * self.dx
        
        for n in range(n_bubbles):
            center = (np.random.randint(0, self.L), np.random.randint(0, self.L), np.random.randint(0, self.L))
            nucleation_time = np.random.exponential(0.1 * n_steps)
            
            self.bubble_centers.append(center)
            self.bubble_nucleation_times.append(nucleation_time)
        
        for step in range(n_steps):
            for n, t_nuc in enumerate(self.bubble_nucleation_times):
                if abs(step - t_nuc) < 1.0:
                    self.nucleate_bubble(self.bubble_centers[n], bubble_radius)
            
            self.evolve_field_step(dt)
            
            if step % 50 == 0:
                self.field_history.append(self.field.copy())
                self.time_history.append(step * dt)
                print(f"Step {step}/{n_steps} ({100.0 * step / n_steps:.1f}%)")
        
        print("Bubble nucleation simulation complete!")
    
    def compute_gw_power_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.field_history) < 2:
            print("Not enough snapshots to compute GW power spectrum")
            return np.array([]), np.array([])
        
        field_final = self.field_history[-1]
        
        field_fft = np.fft.fftn(field_final)
        power_spectrum_3d = np.abs(field_fft)**2
        
        k_max = self.L // 2
        k_bins = np.arange(1, k_max)
        P_k = np.zeros(len(k_bins))
        
        kx = np.fft.fftfreq(self.L, d=self.dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(self.L, d=self.dx) * 2.0 * np.pi
        kz = np.fft.fftfreq(self.L, d=self.dx) * 2.0 * np.pi
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        for i, k_val in enumerate(k_bins):
            mask = (K >= k_val - 0.5) & (K < k_val + 0.5)
            P_k[i] = np.mean(power_spectrum_3d[mask])
        
        f_gw = k_bins / (2.0 * np.pi * self.dx) * self.c.c
        
        Omega_gw = (2.0 * np.pi**2 / 3.0) * f_gw**2 * P_k / ((self.c.H0_Planck2018 * 1000.0 / self.c.Mpc_to_m)**2)
        
        return f_gw, Omega_gw
    
    def visualize_2d_slice(self, snapshot_idx: int = -1, z_slice: int = None, filename: str = 'bubble_field_2d.pdf'):
        if z_slice is None:
            z_slice = self.L // 2
        
        field_snapshot = self.field_history[snapshot_idx]
        field_slice = field_snapshot[:, :, z_slice]
        
        fig, ax = plt.subplots(figsize=(10, 9))
        
        im = ax.imshow(field_slice.T, origin='lower', cmap='seismic', interpolation='bilinear', extent=[0, self.L * self.dx / 3.086e19 * 1.0e15, 0, self.L * self.dx / 3.086e19 * 1.0e15])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field Value $\phi$', fontsize=12)
        
        ax.set_xlabel('X [fm]', fontsize=13)
        ax.set_ylabel('Y [fm]', fontsize=13)
        ax.set_title(f'Bubble Nucleation Field: 2D Slice at z={z_slice}, t={self.time_history[snapshot_idx]:.2e} s', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"2D field slice saved to {filename}")
    
    def create_animation_2d(self, z_slice: int = None, filename: str = 'bubble_evolution.mp4', fps: int = 10):
        if z_slice is None:
            z_slice = self.L // 2
        
        fig, ax = plt.subplots(figsize=(10, 9))
        
        field_slice_init = self.field_history[0][:, :, z_slice]
        im = ax.imshow(field_slice_init.T, origin='lower', cmap='seismic', interpolation='bilinear', extent=[0, self.L * self.dx / 3.086e19 * 1.0e15, 0, self.L * self.dx / 3.086e19 * 1.0e15], vmin=-0.5, vmax=0.5)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field Value $\phi$', fontsize=12)
        
        ax.set_xlabel('X [fm]', fontsize=13)
        ax.set_ylabel('Y [fm]', fontsize=13)
        title = ax.set_title(f'Bubble Nucleation: t=0.00e+00 s', fontsize=14, fontweight='bold')
        
        def update(frame):
            field_slice = self.field_history[frame][:, :, z_slice]
            im.set_data(field_slice.T)
            title.set_text(f'Bubble Nucleation: t={self.time_history[frame]:.2e} s')
            return [im, title]
        
        anim = FuncAnimation(fig, update, frames=len(self.field_history), interval=1000/fps, blit=True)
        anim.save(filename, writer='ffmpeg', fps=fps, dpi=150)
        plt.close()
        
        print(f"2D animation saved to {filename}")