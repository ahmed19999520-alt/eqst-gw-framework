import numpy as np
from eqst_gw.simulations import GalaxyClusterSimulation

def main():
    print("="*80)
    print("GALAXY CLUSTER MERGER SIMULATION")
    print("N-body simulation with EQST-GP dark matter")
    print("="*80)
    print()
    
    print("[1] Initializing cluster merger scenario...")
    M_cluster1 = 1.0e15
    M_cluster2 = 5.0e14
    impact_parameter_kpc = 500.0
    relative_velocity_km_s = 2000.0
    
    sim = GalaxyClusterSimulation(
        M_cluster1=M_cluster1,
        M_cluster2=M_cluster2,
        impact_parameter_kpc=impact_parameter_kpc,
        relative_velocity_km_s=relative_velocity_km_s,
        N_particles_cluster1=5000,
        N_particles_cluster2=2500
    )
    
    print(f"    Cluster 1: M = {M_cluster1:.2e} M_sun, N_particles = 5000")
    print(f"    Cluster 2: M = {M_cluster2:.2e} M_sun, N_particles = 2500")
    print(f"    Impact parameter: {impact_parameter_kpc} kpc")
    print(f"    Relative velocity: {relative_velocity_km_s} km/s")
    print()
    
    print("[2] Running N-body simulation...")
    print("    Duration: 3 Gyr")
    print("    Timestep: 10 Myr")
    sim.run_simulation(t_max_Gyr=3.0, dt_Myr=10.0, save_interval=10)
    print()
    
    print("[3] Computing density profiles...")
    r_bins, rho = sim.compute_density_profile(snapshot_idx=-1)
    print(f"    Computed density profile with {len(r_bins)} radial bins")
    print()
    
    print("[4] Visualizing final state...")
    sim.visualize_3d_snapshot(snapshot_idx=-1, filename='cluster_merger_final_state.pdf')
    print("    3D snapshot saved: cluster_merger_final_state.pdf")
    print()
    
    print("[5] Creating animation...")
    sim.create_animation(filename='cluster_merger_evolution.mp4', fps=10)
    print("    Animation saved: cluster_merger_evolution.mp4")
    print()
    
    print("="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print()
    print("Outputs:")
    print("  - cluster_merger_final_state.pdf")
    print("  - cluster_merger_evolution.mp4")
    print()

if __name__ == "__main__":
    main()