import numpy as np
from eqst_gw.simulations import BubbleNucleationSimulation

def main():
    print("="*80)
    print("BUBBLE NUCLEATION AND COLLISION SIMULATION")
    print("3D lattice field theory simulation of EQST-GP phase transition")
    print("="*80)
    print()
    
    print("[1] Initializing phase transition simulation...")
    bubble_sim = BubbleNucleationSimulation(
        T_nucleation_GeV=9.71e15,
        alpha=0.42,
        beta_over_H=94.7,
        v_w=0.27,
        lattice_size=64,
        lattice_spacing_m=5.0e-16
    )
    
    print(f"    Nucleation temperature: {bubble_sim.T_n:.2e} GeV")
    print(f"    Transition strength α: {bubble_sim.alpha}")
    print(f"    Lattice size: {bubble_sim.L}³")
    print(f"    Lattice spacing: {bubble_sim.dx:.2e} m")
    print()
    
    print("[2] Running bubble nucleation simulation...")
    bubble_sim.run_simulation(n_bubbles=15, n_steps=300, dt=1.0e-26)
    print()
    
    print("[3] Computing gravitational wave power spectrum from collisions...")
    f_gw, Omega_gw = bubble_sim.compute_gw_power_spectrum()
    print(f"    Computed power spectrum with {len(f_gw)} frequency bins")
    if len(f_gw) > 0:
        print(f"    Peak frequency: {f_gw[np.argmax(Omega_gw)]:.3e} Hz")
        print(f"    Peak amplitude: {np.max(Omega_gw):.3e}")
    print()
    
    print("[4] Visualizing field evolution...")
    bubble_sim.visualize_2d_slice(snapshot_idx=-1, filename='bubble_field_final.pdf')
    print("    2D field slice saved: bubble_field_final.pdf")
    print()
    
    print("[5] Creating animation of bubble evolution...")
    bubble_sim.create_animation_2d(filename='bubble_nucleation_evolution.mp4', fps=10)
    print("    Animation saved: bubble_nucleation_evolution.mp4")
    print()
    
    print("="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print()
    print("Outputs:")
    print("  - bubble_field_final.pdf")
    print("  - bubble_nucleation_evolution.mp4")
    print()

if __name__ == "__main__":
    main()