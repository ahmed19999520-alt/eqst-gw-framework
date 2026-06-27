import numpy as np
import matplotlib.pyplot as plt
import os
from eqst_gw.core.constants import FundamentalConstants
from eqst_gw.core.parameters import EQSTGPParameters
from eqst_gw.core.cosmology import LambdaEffectiveCosmology
from eqst_gw.data.loaders import load_desi_bao
from eqst_gw.analysis.mcmc import MCMCSampler
from eqst_gw.visualization.corner_plots import CornerPlotter
from eqst_gw.io.hdf5_handler import HDF5Handler

os.makedirs('./outputs/plots', exist_ok=True)
os.makedirs('./outputs/data', exist_ok=True)

const = FundamentalConstants()
ep = EQSTGPParameters()
cosmo = LambdaEffectiveCosmology(const)

z_bao, DM_obs, DH_obs, cov_bao = load_desi_bao(use_mock=False)

mcmc = MCMCSampler(ep, const, cosmo)

def log_posterior(params):
    Omega_m, h, w0, wa = params
    lp = mcmc.log_prior_cosmology(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = mcmc.log_likelihood_bao(params, z_bao, DM_obs, DH_obs, cov_bao)
    return lp + ll

initial_params = np.array([0.315, 0.674, -1.0, 0.0])

print("="*65)
print("Running MCMC parameter estimation...")
print(f"Initial params: Omega_m={initial_params[0]}, h={initial_params[1]}, w0={initial_params[2]}, wa={initial_params[3]}")
print("="*65)

N_STEPS = 2000
N_WALKERS = 16

chain, log_prob, acceptance = mcmc.run_metropolis_hastings(log_posterior, initial_params, n_walkers=N_WALKERS, n_steps=N_STEPS)

burn_in = N_STEPS // 4
param_stats = mcmc.get_parameter_statistics(['Omega_m', 'h', 'w0', 'wa'], burn_in=burn_in)

print("\nPosterior Summary (median +/- 1sigma):")
for pname, pstats in param_stats.items():
    print(f"  {pname:12s} = {pstats['median']:.4f} +{pstats['sigma_plus']:.4f} -{pstats['sigma_minus']:.4f}")

gelman_rubin = mcmc.gelman_rubin_statistic(burn_in=burn_in)
auto_times = mcmc.autocorrelation_time(burn_in=burn_in)

print(f"\nConvergence Diagnostics:")
for i, pname in enumerate(['Omega_m', 'h', 'w0', 'wa']):
    print(f"  {pname:12s}: R-hat = {gelman_rubin[i]:.4f},  tau = {auto_times[i]:.1f} steps")

print(f"\nMean acceptance rate: {np.mean(acceptance):.3f}")

plotter = CornerPlotter(output_dir='./outputs/plots/')
flat_chain = chain[:, burn_in:, :].reshape(-1, 4)
plotter.plot_corner(flat_chain, param_names=['Omega_m', 'h', 'w_0', 'w_a'],
                     param_labels=['$\\Omega_m$', '$h$', '$w_0$', '$w_a$'],
                     truths=[0.315, 0.674, -1.0, 0.0],
                     filename='03_mcmc_corner_plot.pdf',
                     title='EQST-GP Cosmological Parameter Posterior')

plotter.plot_parameter_evolution(chain, param_names=['Omega_m', 'h', 'w_0', 'w_a'],
                                  param_labels=['$\\Omega_m$', '$h$', '$w_0$', '$w_a$'],
                                  truths=[0.315, 0.674, -1.0, 0.0],
                                  filename='03_mcmc_chain_evolution.pdf')

io = HDF5Handler(output_dir='./outputs/data/')
io.save_mcmc_chains(chain, log_prob, ['Omega_m', 'h', 'w0', 'wa'], filename='03_mcmc_chains.h5')

print("\nOutputs saved:")
print("  ./outputs/plots/03_mcmc_corner_plot.pdf")
print("  ./outputs/plots/03_mcmc_chain_evolution.pdf")
print("  ./outputs/data/03_mcmc_chains.h5")