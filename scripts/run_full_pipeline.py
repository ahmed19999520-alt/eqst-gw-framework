import numpy as np
import os
import sys
import argparse
import json
import yaml
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.cosmology import LambdaEffectiveCosmology
from eqst_gw.gravitational_waves.spectrum import GravitationalWaveSpectrum
from eqst_gw.detectors.lisa import LISADetector
from eqst_gw.detectors.einstein_telescope import EinsteinTelescopeDetector
from eqst_gw.detectors.ligo import LIGODetector
from eqst_gw.detectors.virgo import VirgoDetector
from eqst_gw.data.loaders import load_planck_data, load_desi_bao, load_pantheon_sn, load_jwst_galaxies, load_nanograv_pta_data
from eqst_gw.analysis.parameter_estimation import ParameterEstimator
from eqst_gw.analysis.mcmc import MCMCSampler
from eqst_gw.analysis.model_comparison import BayesianModelComparison
from eqst_gw.analysis.fisher_matrix import FisherMatrixAnalysis
from eqst_gw.simulations.dark_matter_halos import DMHaloProfile
from eqst_gw.simulations.rotation_curves import GalaxyRotationCurve
from eqst_gw.simulations.galaxy_clusters import GalaxyClusterSimulation
from eqst_gw.simulations.bubble_nucleation_sim import BubbleNucleationSimulation
from eqst_gw.simulations.structure_formation import StructureFormationAnalysis
from eqst_gw.multimessenger.dark_matter import MajoranaGluonDarkMatter
from eqst_gw.multimessenger.hubble_tension import HubbleTensionResolution
from eqst_gw.multimessenger.cross_checks import MultiMessengerCrossChecks
from eqst_gw.visualization.spectra_plots import SpectraPlotter
from eqst_gw.io.hdf5_handler import HDF5Handler
from eqst_gw.utils.validators import validate_eqst_parameters

def parse_arguments():
    parser = argparse.ArgumentParser(description='EQST-GP Complete Analysis Pipeline')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='./outputs/', help='Output directory')
    parser.add_argument('--use-mock-data', action='store_true', help='Use mock observational data')
    parser.add_argument('--run-simulations', action='store_true', help='Run N-body and lattice simulations')
    parser.add_argument('--run-mcmc', action='store_true', help='Run MCMC parameter estimation')
    parser.add_argument('--n-mcmc-steps', type=int, default=2000, help='Number of MCMC steps')
    parser.add_argument('--n-walkers', type=int, default=32, help='Number of MCMC walkers')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        return {
            'eqst_gp': {'alpha_PT': 0.42, 'beta_over_H': 94.7, 'v_w': 0.27, 'T_n_GeV': 9.71e15},
            'lisa': {'mission_duration_years': 4.0},
            'analysis': {'f_min': 1e-5, 'f_max': 1e-1, 'n_freq_points': 2000},
            'simulations': {'n_bubble': 20, 'lattice_size': 64}
        }


def main():
    args = parse_arguments()
    config = load_config(args.config)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'reports'), exist_ok=True)
    
    print("="*80)
    print("EQST-GP COMPLETE ANALYSIS PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    const = FundamentalConstants()
    ep = EQSTGPParameters()
    
    if 'eqst_gp' in config:
        for key, val in config['eqst_gp'].items():
            if hasattr(ep, key):
                setattr(ep, key, val)
    
    validate_eqst_parameters(ep)
    
    cosmo = LambdaEffectiveCosmology(const)
    
    print("\n[STEP 1] Computing EQST-GP Gravitational Wave Spectrum...")
    
    f_min = config.get('analysis', {}).get('f_min', 1.0e-5)
    f_max = config.get('analysis', {}).get('f_max', 1.0e2)
    n_freq = config.get('analysis', {}).get('n_freq_points', 2000)
    
    f_array = np.logspace(np.log10(f_min), np.log10(f_max), n_freq)
    
    gw = GravitationalWaveSpectrum(ep, const)
    
    components = gw.spectrum_components(f_array)
    
    props = gw.peak_properties()
    
    print(f"    Sound wave peak: f = {props['f_sound_peak_Hz']:.4e} Hz, Omega h^2 = {props['Omega_sound_peak_h2']:.4e}")
    print(f"    Turbulence peak: f = {props['f_turb_peak_Hz']:.4e} Hz, Omega h^2 = {props['Omega_turb_peak_h2']:.4e}")
    
    print("\n[STEP 2] Computing Multi-Detector Sensitivities and SNRs...")
    
    lisa = LISADetector(mission_duration_years=config.get('lisa', {}).get('mission_duration_years', 4.0), constants=const)
    et = EinsteinTelescopeDetector(design='ET-D', constants=const)
    ligo = LIGODetector(design='O4', constants=const)
    virgo = VirgoDetector(design='O4', constants=const)
    
    sensitivity_curves = {
        'LISA': lisa.omega_sensitivity(f_array),
        'ET': et.omega_sensitivity(f_array),
        'LIGO': ligo.omega_sensitivity(f_array),
        'Virgo': virgo.omega_sensitivity(f_array)
    }
    
    SNR_LISA = lisa.compute_snr(f_array, components['total'])
    SNR_ET = et.compute_snr(f_array, components['total'])
    
    print(f"    LISA SNR (4 yr): {SNR_LISA:.2f}")
    print(f"    Einstein Telescope SNR: {SNR_ET:.4f}")
    
    print("\n[STEP 3] Loading Observational Data...")
    
    ell_cmb, D_ell_cmb, sigma_cmb = load_planck_data(use_mock=args.use_mock_data)
    print(f"    CMB data: {len(ell_cmb)} multipoles")
    
    z_bao, DM_bao, DH_bao, cov_bao = load_desi_bao(use_mock=args.use_mock_data)
    print(f"    DESI BAO data: {len(z_bao)} redshift bins")
    
    z_sn, mu_sn, cov_sn = load_pantheon_sn(use_mock=args.use_mock_data)
    print(f"    Pantheon+ SNe: {len(z_sn)} supernovae")
    
    jwst_data = load_jwst_galaxies(use_mock=args.use_mock_data)
    print(f"    JWST galaxies: {len(jwst_data['z_phot'])} high-z galaxies")
    
    pta_data = load_nanograv_pta_data(use_mock=args.use_mock_data)
    print(f"    NANOGrav PTA: {len(pta_data['f_gw_Hz'])} frequency bins")
    
    print("\n[STEP 4] Fitting BAO and Cosmological Parameters...")
    
    pe = ParameterEstimator(ep, const, cosmo)
    
    bao_results = pe.fit_bao(z_bao, DM_bao, DH_bao, cov_bao)
    
    print(f"    BAO fit: Omega_m = {bao_results['Omega_m']:.4f} +/- {bao_results['Omega_m_err']:.4f}")
    print(f"    BAO fit: h = {bao_results['h']:.4f} +/- {bao_results['h_err']:.4f}")
    print(f"    BAO chi2/dof = {bao_results['chi2_min']:.2f} / {bao_results['dof']}")
    
    sn_results = pe.fit_sn(z_sn, mu_sn, cov_sn)
    
    print(f"    SN fit: Omega_m = {sn_results['Omega_m']:.4f}")
    print(f"    SN fit: w0 = {sn_results['w0']:.3f}, wa = {sn_results['wa']:.3f}")
    
    print("\n[STEP 5] Fisher Matrix Forecast...")
    
    fisher = FisherMatrixAnalysis(ep, const)
    
    LISA_noise = lisa.omega_sensitivity(f_array)
    
    forecast = fisher.forecast_constraints(f_array, LISA_noise, T_obs_years=4.0)
    
    print(f"    Forecasted constraints:")
    for param_name, err in forecast['parameter_errors'].items():
        rel_err = forecast['relative_errors'][param_name]
        print(f"      {param_name}: +/- {err:.4f} ({rel_err*100:.2f}%)")
    
    if args.run_mcmc:
        print("\n[STEP 6] Running MCMC Parameter Estimation...")
        
        mcmc = MCMCSampler(ep, const, cosmo)
        
        data_dict = {'BAO': (z_bao, DM_bao, DH_bao, cov_bao)}
        
        def log_posterior_combined(params):
            Omega_m, h, w0, wa = params
            lp = mcmc.log_prior_cosmology(params)
            if not np.isfinite(lp):
                return -np.inf
            ll = mcmc.log_likelihood_bao(params, z_bao, DM_bao, DH_bao, cov_bao)
            return lp + ll
        
        initial_params = np.array([0.315, 0.674, -1.0, 0.0])
        
        chain, log_prob, acceptance = mcmc.run_metropolis_hastings(
            log_posterior_combined,
            initial_params,
            n_walkers=args.n_walkers,
            n_steps=args.n_mcmc_steps
        )
        
        param_stats = mcmc.get_parameter_statistics(['Omega_m', 'h', 'w0', 'wa'], burn_in=args.n_mcmc_steps // 4)
        
        print(f"    MCMC results (median +/- 1sigma):")
        for pname, pstats in param_stats.items():
            print(f"      {pname} = {pstats['median']:.4f} +{pstats['sigma_plus']:.4f} -{pstats['sigma_minus']:.4f}")
        
        gelman_rubin = mcmc.gelman_rubin_statistic(burn_in=args.n_mcmc_steps // 4)
        print(f"    Gelman-Rubin R-hat: {gelman_rubin}")
    
    print("\n[STEP 7] Hubble Tension Analysis...")
    
    ht = HubbleTensionResolution(const, cosmo)
    
    tensions = ht.compute_all_tensions()
    
    print(f"    Planck vs SH0ES tension: {tensions['Planck_vs_SH0ES']:.2f} sigma")
    print(f"    EQST-GP H0 = {tensions['H0_EQST_GP']:.2f} km/s/Mpc")
    print(f"    EQST-GP vs Planck: {tensions['EQST_vs_Planck']:.2f} sigma")
    print(f"    EQST-GP vs SH0ES: {tensions['EQST_vs_SH0ES']:.2f} sigma")
    
    S8_analysis = ht.S8_tension_analysis()
    print(f"    S8 (EQST-GP) = {S8_analysis['S8_EQST_GP']:.4f}")
    print(f"    S8 Planck-KiDS tension: {S8_analysis['tension_Planck_KiDS']:.2f} sigma")
    
    print("\n[STEP 8] Multi-Messenger Cross-Checks...")
    
    mm_checks = MultiMessengerCrossChecks(ep, const, cosmo)
    
    full_report = mm_checks.generate_full_consistency_report()
    
    print(f"    Consistency checks passed: {full_report['overall_consistency']['n_checks_passed']} / {full_report['overall_consistency']['n_checks_total']}")
    print(f"    Framework status: {full_report['overall_consistency']['framework_status']}")
    print(f"    DM abundance tension: {full_report['dark_matter']['tension_sigma']:.2f} sigma")
    print(f"    alpha_EM prediction deviation: {full_report['fundamental_constants']['alpha_EM_deviation_ppm']:.2f} ppm")
    print(f"    m_proton prediction deviation: {full_report['fundamental_constants']['m_proton_deviation_ppm']:.2f} ppm")
    
    pta_consistency = mm_checks.pta_consistency_check(pta_data['f_gw_Hz'], pta_data['Omega_gw_h2'], pta_data['sigma_Omega_gw_h2'])
    print(f"    PTA chi2_red (EQST-GP): {pta_consistency['chi2_reduced_EQST']:.3f}")
    print(f"    PTA preferred model: {pta_consistency['preferred_model']}")
    
    if args.run_simulations:
        print("\n[STEP 9] Running Simulations...")
        
        print("    [9a] Dark Matter Halo Profile...")
        halo = DMHaloProfile(M_vir_solar=1.0e12, z=0.0, eqst_params=ep, constants=const)
        halo_summary = halo.density_profile_summary()
        print(f"         Virial radius: {halo_summary['r_vir_kpc']:.2f} kpc")
        print(f"         v_max: {halo_summary['v_max_km_s']:.2f} km/s")
        
        print("    [9b] Galaxy Rotation Curve...")
        rc = GalaxyRotationCurve(M_vir=1.0e11, z=0.0, constants=const, eqst_params=ep)
        r_test = np.linspace(1.0e19, 1.0e22, 100)
        v_eqst = rc.compute_rotation_curve_eqst_gp(r_test)
        v_nfw = rc.compute_rotation_curve_nfw(r_test)
        print(f"         Max EQST-GP velocity: {np.max(v_eqst)/1000.:.2f} km/s")
        print(f"         Max NFW velocity: {np.max(v_nfw)/1000.:.2f} km/s")
        
        print("    [9c] Bubble Nucleation Simulation...")
        bubble_sim = BubbleNucleationSimulation(T_nucleation_GeV=ep.T_n, alpha=ep.alpha_PT, beta_over_H=ep.beta_over_H, v_w=ep.v_w, lattice_size=config.get('simulations', {}).get('lattice_size', 32), constants=const, eqst_params=ep)
        bubble_sim.run_simulation(n_bubbles=config.get('simulations', {}).get('n_bubble', 10), n_steps=100)
        f_gw_sim, Omega_gw_sim = bubble_sim.compute_gw_power_spectrum()
        if len(f_gw_sim) > 0:
            print(f"         Bubble simulation GW peak: {f_gw_sim[np.argmax(Omega_gw_sim)]:.3e} Hz")
        
        print("    [9d] Structure Formation...")
        sfa = StructureFormationAnalysis(const, cosmo)
        sigma8_eqst = sfa.sigma8_eqst_gp()
        D_z1 = sfa.linear_growth_factor(np.array([1.0]))[0]
        print(f"         sigma_8 (EQST-GP): {sigma8_eqst:.4f}")
        print(f"         D(z=1) / D(z=0): {D_z1:.4f}")
    
    print("\n[STEP 10] Generating Visualizations...")
    
    plotter = SpectraPlotter(output_dir=os.path.join(args.output_dir, 'plots'))
    
    plotter.plot_gw_spectrum_complete(
        f_array,
        components,
        sensitivity_curves,
        filename='eqst_gp_gw_spectrum_complete.pdf'
    )
    
    if args.run_mcmc:
        plotter.plot_parameter_constraints(
            chain[:, args.n_mcmc_steps // 4:, :],
            ['Omega_m', 'h', 'w_0', 'w_a'],
            truths=[0.315, 0.674, -1.0, 0.0],
            filename='mcmc_corner_plot.pdf'
        )
    
    print("\n[STEP 11] Exporting Data Products...")
    
    io_handler = HDF5Handler(output_dir=os.path.join(args.output_dir, 'data'))
    
    io_handler.save_gw_spectrum(f_array, components, ep, filename='eqst_gp_gw_spectrum.h5')
    
    full_results = {
        'timestamp': datetime.now().isoformat(),
        'eqst_gp_parameters': ep.to_dict(),
        'gw_spectrum': {
            'peak_frequency_sound_Hz': float(props['f_sound_peak_Hz']),
            'peak_amplitude_sound_h2': float(props['Omega_sound_peak_h2']),
            'peak_frequency_turb_Hz': float(props['f_turb_peak_Hz']),
            'peak_amplitude_turb_h2': float(props['Omega_turb_peak_h2'])
        },
        'detector_snr': {
            'LISA_4yr': float(SNR_LISA),
            'ET': float(SNR_ET)
        },
        'bao_fit': bao_results,
        'sn_fit': sn_results,
        'fisher_forecast': {k: float(v) if np.isscalar(v) else str(v) for k, v in forecast['parameter_errors'].items()},
        'hubble_tension': {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in tensions.items()},
        'multi_messenger': {
            'dark_matter_tension_sigma': float(full_report['dark_matter']['tension_sigma']),
            'framework_status': full_report['overall_consistency']['framework_status'],
            'alpha_EM_deviation_ppm': float(full_report['fundamental_constants']['alpha_EM_deviation_ppm']),
            'm_proton_deviation_ppm': float(full_report['fundamental_constants']['m_proton_deviation_ppm'])
        }
    }
    
    report_path = os.path.join(args.output_dir, 'reports', 'full_analysis_report.json')
    with open(report_path, 'w') as jf:
        json.dump(full_results, jf, indent=4, default=str)
    
    print(f"    Full results saved to {report_path}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    print("Summary:")
    print(f"  LISA SNR:                 {SNR_LISA:.2f}  ({'DETECTABLE' if SNR_LISA > 5 else 'BELOW THRESHOLD'})")
    print(f"  Hubble tension:           {tensions['Planck_vs_SH0ES']:.2f} sigma (standard)")
    print(f"  EQST-GP H0:              {tensions['H0_EQST_GP']:.2f} km/s/Mpc")
    print(f"  DM abundance tension:     {full_report['dark_matter']['tension_sigma']:.2f} sigma")
    print(f"  Framework status:         {full_report['overall_consistency']['framework_status']}")
    print()
    print("Output files:")
    print(f"  Plots:    {os.path.join(args.output_dir, 'plots')}/")
    print(f"  Data:     {os.path.join(args.output_dir, 'data')}/")
    print(f"  Reports:  {os.path.join(args.output_dir, 'reports')}/")


if __name__ == "__main__":
    main()