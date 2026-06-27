import numpy as np
from typing import Tuple, Dict, Optional, Callable, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants
from ..core.cosmology import LambdaEffectiveCosmology

class MCMCSampler:
    def __init__(self,
                 eqst_params: Optional[EQSTGPParameters] = None,
                 constants: Optional[FundamentalConstants] = None,
                 cosmology: Optional[LambdaEffectiveCosmology] = None):
        
        if eqst_params is None:
            self.ep = EQSTGPParameters()
        else:
            self.ep = eqst_params
        
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        if cosmology is None:
            self.cosmo = LambdaEffectiveCosmology(self.const)
        else:
            self.cosmo = cosmology
        
        self.chain = None
        self.log_prob_chain = None
        self.acceptance_rate = None
        self.sampler_type = None
    
    def log_prior_gw(self, params: np.ndarray) -> float:
        alpha, beta_H, v_w, T_n = params
        
        if alpha <= 0.0 or alpha >= 3.0:
            return -np.inf
        if beta_H <= 1.0 or beta_H >= 1000.0:
            return -np.inf
        if v_w <= 0.05 or v_w >= 0.99:
            return -np.inf
        if T_n <= 1.0e13 or T_n >= 1.0e18:
            return -np.inf
        
        log_p = 0.0
        
        log_p += -0.5 * ((alpha - self.ep.alpha_PT) / self.ep.alpha_PT_err)**2
        log_p += -0.5 * ((beta_H - self.ep.beta_over_H) / self.ep.beta_over_H_err)**2
        log_p += -0.5 * ((v_w - self.ep.v_w) / self.ep.v_w_err)**2
        log_p += -0.5 * ((T_n - self.ep.T_n) / self.ep.T_n_err)**2
        
        return log_p
    
    def log_likelihood_gw(self, params: np.ndarray, f_data: np.ndarray, Omega_data: np.ndarray, sigma_data: np.ndarray) -> float:
        alpha, beta_H, v_w, T_n = params
        
        ep_temp = EQSTGPParameters()
        ep_temp.alpha_PT = alpha
        ep_temp.beta_over_H = beta_H
        ep_temp.v_w = v_w
        ep_temp.T_n = T_n
        
        from ..gravitational_waves.spectrum import GravitationalWaveSpectrum
        gw = GravitationalWaveSpectrum(ep_temp, self.const)
        
        Omega_model = gw.total_spectrum(f_data)
        
        residuals = (Omega_data - Omega_model) / sigma_data
        
        log_L = -0.5 * np.sum(residuals**2 + np.log(2.0 * np.pi * sigma_data**2))
        
        return log_L
    
    def log_posterior_gw(self, params: np.ndarray, f_data: np.ndarray, Omega_data: np.ndarray, sigma_data: np.ndarray) -> float:
        lp = self.log_prior_gw(params)
        
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood_gw(params, f_data, Omega_data, sigma_data)
        
        return lp + ll
    
    def log_prior_cosmology(self, params: np.ndarray) -> float:
        Omega_m, h, w0, wa = params
        
        if Omega_m <= 0.1 or Omega_m >= 0.6:
            return -np.inf
        if h <= 0.5 or h >= 0.9:
            return -np.inf
        if w0 <= -3.0 or w0 >= 0.0:
            return -np.inf
        if wa <= -3.0 or wa >= 3.0:
            return -np.inf
        
        log_p = 0.0
        
        log_p += -0.5 * ((Omega_m - 0.315) / 0.007)**2
        log_p += -0.5 * ((h - 0.674) / 0.005)**2
        
        return log_p
    
    def log_likelihood_bao(self, params: np.ndarray, z_eff: np.ndarray, DM_obs: np.ndarray, DH_obs: np.ndarray, cov_matrix: np.ndarray) -> float:
        Omega_m, h = params[:2]
        
        self.cosmo.Omega_m = Omega_m
        self.cosmo.H0 = h * 100.0
        self.cosmo.h = h
        
        from ..analysis.parameter_estimation import ParameterEstimator
        pe = ParameterEstimator(self.ep, self.const, self.cosmo)
        r_d = pe.sound_horizon_rs(Omega_m, h)
        
        c_km_s = self.const.c / 1000.0
        
        DM_theory = np.zeros_like(z_eff)
        DH_theory = np.zeros_like(z_eff)
        
        for i, z in enumerate(z_eff):
            from scipy.integrate import quad
            d_C, _ = quad(lambda zp: c_km_s / self.cosmo.H_eff(zp, Omega_m), 0, z)
            DM_theory[i] = d_C / r_d
            DH_theory[i] = (c_km_s / self.cosmo.H_eff(z, Omega_m)) / r_d
        
        data_vec = np.concatenate([DM_obs, DH_obs])
        theory_vec = np.concatenate([DM_theory, DH_theory])
        
        residual = data_vec - theory_vec
        
        log_L = -0.5 * residual @ np.linalg.inv(cov_matrix) @ residual
        log_L -= 0.5 * np.log(np.linalg.det(2.0 * np.pi * cov_matrix))
        
        return log_L
    
    def run_emcee_sampler(self,
                          log_prob_func: Callable,
                          initial_params: np.ndarray,
                          n_walkers: int = 32,
                          n_steps: int = 5000,
                          n_burnin: int = 1000,
                          moves=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            import emcee
            
            n_dim = len(initial_params)
            
            pos = initial_params + 1.0e-4 * np.random.randn(n_walkers, n_dim)
            pos = pos * (1.0 + 1.0e-5 * np.random.randn(n_walkers, n_dim))
            
            if moves is None:
                moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
            
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_func, moves=moves)
            
            sampler.run_mcmc(pos, n_steps, progress=True)
            
            chain = sampler.get_chain(discard=n_burnin, flat=False)
            flat_chain = sampler.get_chain(discard=n_burnin, flat=True)
            log_prob = sampler.get_log_prob(discard=n_burnin, flat=False)
            acceptance = sampler.acceptance_fraction
            
            self.chain = chain
            self.log_prob_chain = log_prob
            self.acceptance_rate = acceptance
            self.sampler_type = 'emcee'
            
            return chain, log_prob, acceptance
            
        except ImportError:
            return self.run_metropolis_hastings(log_prob_func, initial_params, n_walkers, n_steps)
    
    def run_metropolis_hastings(self,
                                log_prob_func: Callable,
                                initial_params: np.ndarray,
                                n_walkers: int = 32,
                                n_steps: int = 5000,
                                proposal_scale: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_dim = len(initial_params)
        
        chain = np.zeros((n_walkers, n_steps, n_dim))
        log_prob_chain = np.zeros((n_walkers, n_steps))
        
        current_pos = initial_params + proposal_scale * np.random.randn(n_walkers, n_dim)
        current_log_prob = np.array([log_prob_func(current_pos[i]) for i in range(n_walkers)])
        
        acceptance_count = np.zeros(n_walkers)
        
        for step in range(n_steps):
            proposal = current_pos + proposal_scale * np.random.randn(n_walkers, n_dim)
            
            proposal_log_prob = np.array([log_prob_func(proposal[i]) for i in range(n_walkers)])
            
            log_acceptance = proposal_log_prob - current_log_prob
            
            u = np.log(np.random.uniform(0, 1, n_walkers))
            
            accept = u < log_acceptance
            
            current_pos[accept] = proposal[accept]
            current_log_prob[accept] = proposal_log_prob[accept]
            acceptance_count += accept.astype(float)
            
            chain[:, step, :] = current_pos
            log_prob_chain[:, step] = current_log_prob
        
        acceptance_rate = acceptance_count / n_steps
        
        self.chain = chain
        self.log_prob_chain = log_prob_chain
        self.acceptance_rate = acceptance_rate
        self.sampler_type = 'metropolis_hastings'
        
        return chain, log_prob_chain, acceptance_rate
    
    def run_nested_sampling(self,
                            log_likelihood_func: Callable,
                            log_prior_transform: Callable,
                            n_dim: int,
                            n_live: int = 400,
                            dlogz: float = 0.5) -> Dict:
        try:
            import dynesty
            
            sampler = dynesty.NestedSampler(
                log_likelihood_func,
                log_prior_transform,
                n_dim,
                nlive=n_live,
                sample='rwalk',
                bound='multi'
            )
            
            sampler.run_nested(dlogz=dlogz, print_progress=True)
            
            results = sampler.results
            
            log_Z = results.logz[-1]
            log_Z_err = results.logzerr[-1]
            
            weights = np.exp(results.logwt - results.logz[-1])
            samples = results.samples
            
            return {
                'log_evidence': log_Z,
                'log_evidence_err': log_Z_err,
                'samples': samples,
                'weights': weights,
                'results': results
            }
        
        except ImportError:
            print("dynesty not available, returning None")
            return None
    
    def get_parameter_statistics(self, param_names: List[str], burn_in: int = 0) -> Dict:
        if self.chain is None:
            raise ValueError("No chain available. Run sampler first.")
        
        n_walkers, n_steps, n_dim = self.chain.shape
        
        flat_chain = self.chain[:, burn_in:, :].reshape(-1, n_dim)
        
        stats = {}
        
        for i, name in enumerate(param_names):
            samples = flat_chain[:, i]
            
            mean = np.mean(samples)
            median = np.median(samples)
            std = np.std(samples)
            
            percentile_16 = np.percentile(samples, 16)
            percentile_84 = np.percentile(samples, 84)
            percentile_5 = np.percentile(samples, 5)
            percentile_95 = np.percentile(samples, 95)
            
            stats[name] = {
                'mean': mean,
                'median': median,
                'std': std,
                'sigma_minus': median - percentile_16,
                'sigma_plus': percentile_84 - median,
                '90_lower': percentile_5,
                '90_upper': percentile_95
            }
        
        return stats
    
    def autocorrelation_time(self, burn_in: int = 0) -> np.ndarray:
        if self.chain is None:
            raise ValueError("No chain available. Run sampler first.")
        
        n_walkers, n_steps, n_dim = self.chain.shape
        
        chain_after_burnin = self.chain[:, burn_in:, :]
        
        tau = np.zeros(n_dim)
        
        for i in range(n_dim):
            param_chain = chain_after_burnin[:, :, i].flatten()
            
            N = len(param_chain)
            mean = np.mean(param_chain)
            var = np.var(param_chain)
            
            if var == 0:
                tau[i] = np.inf
                continue
            
            acf = np.correlate(param_chain - mean, param_chain - mean, mode='full')
            acf = acf[N-1:]
            acf /= acf[0]
            
            tau[i] = 1.0 + 2.0 * np.sum(acf[1:min(50, N//4)])
        
        return tau
    
    def gelman_rubin_statistic(self, burn_in: int = 0) -> np.ndarray:
        if self.chain is None:
            raise ValueError("No chain available. Run sampler first.")
        
        n_walkers, n_steps, n_dim = self.chain.shape
        
        n_samples = n_steps - burn_in
        chain_after_burnin = self.chain[:, burn_in:, :]
        
        R_hat = np.zeros(n_dim)
        
        for i in range(n_dim):
            walker_means = np.mean(chain_after_burnin[:, :, i], axis=1)
            
            grand_mean = np.mean(walker_means)
            
            B = n_samples / (n_walkers - 1) * np.sum((walker_means - grand_mean)**2)
            
            W = np.mean([np.var(chain_after_burnin[j, :, i], ddof=1) for j in range(n_walkers)])
            
            Var_plus = (n_samples - 1.0) / n_samples * W + B / n_samples
            
            R_hat[i] = np.sqrt(Var_plus / W)
        
        return R_hat