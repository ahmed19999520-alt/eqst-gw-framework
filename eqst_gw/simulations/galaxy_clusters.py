import numpy as np
from scipy.spatial import cKDTree
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Dict, Optional

class GalaxyClusterSimulation:
    def __init__(self,
                 M_cluster1: float,
                 M_cluster2: float,
                 impact_parameter_kpc: float = 500.0,
                 relative_velocity_km_s: float = 2000.0,
                 N_particles_cluster1: int = 10000,
                 N_particles_cluster2: int = 5000,
                 constants=None):
        
        self.M1_solar = M_cluster1
        self.M2_solar = M_cluster2
        self.b_kpc = impact_parameter_kpc
        self.v_rel_km_s = relative_velocity_km_s
        self.N1 = N_particles_cluster1
        self.N2 = N_particles_cluster2
        
        if constants is None:
            from ..core.constants import FundamentalConstants
            self.c = FundamentalConstants()
        else:
            self.c = constants
        
        self.M1_kg = M_cluster1 * 1.989e30
        self.M2_kg = M_cluster2 * 1.989e30
        self.b_m = impact_parameter_kpc * 3.086e19
        self.v_rel_m_s = relative_velocity_km_s * 1000.0
        
        self.positions = None
        self.velocities = None
        self.masses = None
        self.trajectory_history = []
        
        self.initialize_clusters()
    
    def initialize_clusters(self):
        pos1, vel1, mass1 = self.generate_nfw_cluster(self.M1_kg, self.N1, center=np.array([-self.b_m / 2.0, 0.0, 0.0]), velocity=np.array([self.v_rel_m_s / 2.0, 0.0, 0.0]))
        pos2, vel2, mass2 = self.generate_nfw_cluster(self.M2_kg, self.N2, center=np.array([self.b_m / 2.0, 0.0, 0.0]), velocity=np.array([-self.v_rel_m_s / 2.0, 0.0, 0.0]))
        
        self.positions = np.vstack([pos1, pos2])
        self.velocities = np.vstack([vel1, vel2])
        self.masses = np.concatenate([mass1, mass2])
        
        self.N_total = len(self.masses)
        self.cluster_labels = np.array([1] * self.N1 + [2] * self.N2)
    
    def generate_nfw_cluster(self, M_vir: float, N_particles: int, center: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H0_SI = self.c.H0_Planck2018 * 1000.0 / self.c.Mpc_to_m
        E_z = 1.0
        H_z = H0_SI * E_z
        rho_crit = 3.0 * H_z**2 / (8.0 * np.pi * self.c.G)
        
        Delta_vir = 200.0
        r_vir = (3.0 * M_vir / (4.0 * np.pi * Delta_vir * rho_crit))**(1.0/3.0)
        
        c = 5.0
        r_s = r_vir / c
        
        radii = self.sample_nfw_radii(r_s, c, N_particles)
        
        theta = np.arccos(2.0 * np.random.rand(N_particles) - 1.0)
        phi = 2.0 * np.pi * np.random.rand(N_particles)
        
        x = radii * np.sin(theta) * np.cos(phi)
        y = radii * np.sin(theta) * np.sin(phi)
        z = radii * np.cos(theta)
        
        positions = np.column_stack([x, y, z]) + center
        
        v_circ = np.sqrt(self.c.G * M_vir * (np.log(1.0 + radii / r_s) - (radii / r_s) / (1.0 + radii / r_s)) / radii)
        
        v_r = np.random.normal(0, v_circ / 3.0, N_particles)
        v_theta = np.random.normal(0, v_circ / 3.0, N_particles)
        v_phi = np.random.normal(v_circ, v_circ / 5.0, N_particles)
        
        vx = v_r * np.sin(theta) * np.cos(phi) + v_theta * np.cos(theta) * np.cos(phi) - v_phi * np.sin(phi)
        vy = v_r * np.sin(theta) * np.sin(phi) + v_theta * np.cos(theta) * np.sin(phi) + v_phi * np.cos(phi)
        vz = v_r * np.cos(theta) - v_theta * np.sin(theta)
        
        velocities = np.column_stack([vx, vy, vz]) + velocity
        
        masses = np.ones(N_particles) * (M_vir / N_particles)
        
        return positions, velocities, masses
    
    def sample_nfw_radii(self, r_s: float, c: float, N: int) -> np.ndarray:
        u = np.random.uniform(0, 1, N)
        f_c = np.log(1.0 + c) - c / (1.0 + c)
        
        x = np.zeros(N)
        for i in range(N):
            from scipy.optimize import brentq
            func = lambda x_val: np.log(1.0 + x_val) - x_val / (1.0 + x_val) - u[i] * f_c
            x[i] = brentq(func, 1e-6, c)
        
        return x * r_s
    
    def compute_accelerations(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        N = len(masses)
        accelerations = np.zeros_like(positions)
        
        softening = 1.0e20
        
        tree = cKDTree(positions)
        
        for i in range(N):
            neighbors = tree.query_ball_point(positions[i], r=1.0e23)
            
            for j in neighbors:
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    r_soft = np.sqrt(r_mag**2 + softening**2)
                    accelerations[i] += self.c.G * masses[j] * r_vec / r_soft**3
        
        return accelerations
    
    def leapfrog_step(self, dt: float):
        half_step_vel = self.velocities + 0.5 * dt * self.compute_accelerations(self.positions, self.masses)
        
        self.positions += dt * half_step_vel
        
        new_accel = self.compute_accelerations(self.positions, self.masses)
        
        self.velocities = half_step_vel + 0.5 * dt * new_accel
    
    def run_simulation(self, t_max_Gyr: float, dt_Myr: float = 10.0, save_interval: int = 10):
        t_max_s = t_max_Gyr * 1.0e9 * self.c.year_to_s
        dt_s = dt_Myr * 1.0e6 * self.c.year_to_s
        
        n_steps = int(t_max_s / dt_s)
        
        self.trajectory_history = []
        
        for step in range(n_steps):
            self.leapfrog_step(dt_s)
            
            if step % save_interval == 0:
                self.trajectory_history.append({
                    'time_Gyr': step * dt_s / (1.0e9 * self.c.year_to_s),
                    'positions': self.positions.copy(),
                    'velocities': self.velocities.copy()
                })
                
                if step % 100 == 0:
                    print(f"Step {step}/{n_steps} ({100.0 * step / n_steps:.1f}%)")
        
        print("Simulation complete!")
    
    def compute_density_profile(self, snapshot_idx: int = -1, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        snapshot = self.trajectory_history[snapshot_idx]
        positions = snapshot['positions']
        
        center_of_mass = np.average(positions, weights=self.masses, axis=0)
        
        r = np.linalg.norm(positions - center_of_mass, axis=1)
        
        r_max = np.max(r)
        r_bins = np.logspace(np.log10(r_max / 100.0), np.log10(r_max), n_bins + 1)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        
        rho = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
            volume = (4.0 / 3.0) * np.pi * (r_bins[i + 1]**3 - r_bins[i]**3)
            rho[i] = np.sum(self.masses[mask]) / volume
        
        return r_centers, rho
    
    def visualize_3d_snapshot(self, snapshot_idx: int = -1, filename: str = 'cluster_snapshot_3d.pdf'):
        snapshot = self.trajectory_history[snapshot_idx]
        positions = snapshot['positions']
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        mask1 = self.cluster_labels == 1
        mask2 = self.cluster_labels == 2
        
        ax.scatter(positions[mask1, 0] / 3.086e19, positions[mask1, 1] / 3.086e19, positions[mask1, 2] / 3.086e19, s=1, c='blue', alpha=0.6, label='Cluster 1')
        ax.scatter(positions[mask2, 0] / 3.086e19, positions[mask2, 1] / 3.086e19, positions[mask2, 2] / 3.086e19, s=1, c='red', alpha=0.6, label='Cluster 2')
        
        ax.set_xlabel('X [kpc]', fontsize=12)
        ax.set_ylabel('Y [kpc]', fontsize=12)
        ax.set_zlabel('Z [kpc]', fontsize=12)
        ax.legend(fontsize=11)
        ax.set_title(f"Galaxy Cluster Merger: t = {snapshot['time_Gyr']:.2f} Gyr", fontsize=14, fontweight='bold')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"3D snapshot saved to {filename}")
    
    def create_animation(self, filename: str = 'cluster_merger.mp4', fps: int = 10):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            snapshot = self.trajectory_history[frame]
            positions = snapshot['positions']
            
            mask1 = self.cluster_labels == 1
            mask2 = self.cluster_labels == 2
            
            ax.scatter(positions[mask1, 0] / 3.086e19, positions[mask1, 1] / 3.086e19, positions[mask1, 2] / 3.086e19, s=1, c='blue', alpha=0.6)
            ax.scatter(positions[mask2, 0] / 3.086e19, positions[mask2, 1] / 3.086e19, positions[mask2, 2] / 3.086e19, s=1, c='red', alpha=0.6)
            
            ax.set_xlabel('X [kpc]', fontsize=12)
            ax.set_ylabel('Y [kpc]', fontsize=12)
            ax.set_zlabel('Z [kpc]', fontsize=12)
            ax.set_title(f"Galaxy Cluster Merger: t = {snapshot['time_Gyr']:.2f} Gyr", fontsize=14, fontweight='bold')
            
            ax.set_xlim(-2000, 2000)
            ax.set_ylim(-2000, 2000)
            ax.set_zlim(-2000, 2000)
        
        anim = FuncAnimation(fig, update, frames=len(self.trajectory_history), interval=1000/fps)
        anim.save(filename, writer='ffmpeg', fps=fps, dpi=150)
        plt.close()
        
        print(f"Animation saved to {filename}")