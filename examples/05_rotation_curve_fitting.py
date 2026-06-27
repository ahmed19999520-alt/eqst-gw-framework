import numpy as np
import matplotlib.pyplot as plt
from eqst_gw.simulations import GalaxyRotationCurve
from eqst_gw.core import FundamentalConstants, EQSTGPParameters

def load_observed_rotation_curve_ngc1560():
    r_obs_kpc = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
    v_obs_km_s = np.array([45, 65, 80, 95, 105, 112, 118, 120, 122, 123, 124, 124, 125, 125])
    v_err_km_s = np.array([5, 5, 4, 4, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6])
    
    r_obs_m = r_obs_kpc * 3.086e19
    v_obs_m_s = v_obs_km_s * 1000.0
    v_err_m_s = v_err_km_s * 1000.0
    
    return r_obs_m, v_obs_m_s, v_err_m_s

def main():
    print("="*80)
    print("GALAXY ROTATION CURVE FITTING: EQST-GP vs ΛCDM")
    print("Example: NGC 1560")
    print("="*80)
    print()
    
    print("[1] Loading observed rotation curve data...")
    r_obs, v_obs, v_err = load_observed_rotation_curve_ngc1560()
    print(f"    Loaded {len(r_obs)} data points")
    print()
    
    print("[2] Initializing EQST-GP galaxy model...")
    M_vir = 1.0e11
    rc = GalaxyRotationCurve(M_vir=M_vir, z=0.0)
    print(f"    Virial mass: {M_vir:.2e} M_sun")
    print(f"    Virial radius: {rc.r_vir / 3.086e19:.2f} kpc")
    print(f"    Scale radius: {rc.r_s / 3.086e19:.2f} kpc")
    print()
    
    print("[3] Fitting EQST-GP model to data...")
    results_eqst = rc.fit_to_observed_data(r_obs, v_obs, v_err, model='eqst_gp')
    print(f"    M_disk = {results_eqst['M_disk_solar']:.3e} ± {results_eqst['M_disk_err_solar']:.3e} M_sun")
    print(f"    R_disk = {results_eqst['R_d_kpc']:.2f} ± {results_eqst['R_d_err_kpc']:.2f} kpc")
    print(f"    χ² / dof = {results_eqst['chi2']:.2f} / {results_eqst['dof']} = {results_eqst['chi2_reduced']:.3f}")
    print()
    
    print("[4] Fitting NFW (ΛCDM) model to data...")
    results_nfw = rc.fit_to_observed_data(r_obs, v_obs, v_err, model='nfw')
    print(f"    M_disk = {results_nfw['M_disk_solar']:.3e} ± {results_nfw['M_disk_err_solar']:.3e} M_sun")
    print(f"    R_disk = {results_nfw['R_d_kpc']:.2f} ± {results_nfw['R_d_err_kpc']:.2f} kpc")
    print(f"    χ² / dof = {results_nfw['chi2']:.2f} / {results_nfw['dof']} = {results_nfw['chi2_reduced']:.3f}")
    print()
    
    print("[5] Model comparison...")
    delta_chi2 = results_nfw['chi2'] - results_eqst['chi2']
    if delta_chi2 > 0:
        print(f"    EQST-GP provides better fit by Δχ² = {delta_chi2:.2f}")
        print(f"    Statistical preference: {delta_chi2:.1f}σ")
    else:
        print(f"    NFW provides better fit by Δχ² = {-delta_chi2:.2f}")
    print()
    
    print("[6] Generating comparison plots...")
    rc.plot_rotation_curve_comparison(r_obs, v_obs, v_err, results_eqst['M_disk_solar'], results_eqst['R_d_kpc'] * 3.086e19, filename='rotation_curve_NGC1560_comparison.pdf')
    print("    Plot saved: rotation_curve_NGC1560_comparison.pdf")
    print()
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("Key findings:")
    print(f"  - EQST-GP χ²_red = {results_eqst['chi2_reduced']:.3f}")
    print(f"  - ΛCDM χ²_red = {results_nfw['chi2_reduced']:.3f}")
    print(f"  - Model preference: {'EQST-GP' if delta_chi2 > 0 else 'ΛCDM'}")
    print()

if __name__ == "__main__":
    main()