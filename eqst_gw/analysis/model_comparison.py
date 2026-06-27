import numpy as np
from typing import Tuple, Dict, Optional, Callable, List
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants

class BayesianModelComparison:
    def __init__(self,
                 constants: Optional[FundamentalConstants] = None):
        
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        self.models = {}
    
    def register_model(self, name: str, log_likelihood_func: Callable, prior_ranges: List[Tuple[float, float]], description: str = ""):
        self.models[name] = {
            'log_L': log_likelihood_func,
            'prior_ranges': prior_ranges,
            'description': description,
            'log_evidence': None
        }
    
    def compute_evidence_thermodynamic_integration(self, model_name: str, data: Dict, n_temps: int = 20, n_steps: int = 2000, n_walkers: int = 20) -> Tuple[float, float]:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered.")
        
        model = self.models[model_name]
        prior_ranges = model['prior_ranges']
        n_params = len(prior_ranges)
        
        betas = np.linspace(0, 1, n_temps)**5
        
        log_L_means = np.zeros(n_temps)
        log_L_vars = np.zeros(n_temps)
        
        for t_idx, beta in enumerate(betas):
            def log_posterior_beta(params):
                in_prior = all(prior_ranges[i][0] < params[i] < prior_ranges[i][1] for i in range(n_params))
                if not in_prior:
                    return -np.inf
                
                try:
                    log_L = model['log_L'](params, data)
                except:
                    return -np.inf
                
                return beta * log_L
            
            samples = np.zeros((n_walkers * n_steps, n_params))
            log_L_samples = np.zeros(n_walkers * n_steps)
            
            current = np.array([np.random.uniform(p_min, p_max) for p_min, p_max in prior_ranges])
            current_log_p = log_posterior_beta(current)
            
            for step in range(n_walkers * n_steps):
                proposal = current + 0.05 * np.random.randn(n_params)
                proposal_log_p = log_posterior_beta(proposal)
                
                if np.log(np.random.rand()) < proposal_log_p - current_log_p:
                    current = proposal
                    current_log_p = proposal_log_p
                
                samples[step] = current
                log_L_samples[step] = model['log_L'](current, data) if np.isfinite(current_log_p) else -1.0e10
            
            log_L_means[t_idx] = np.mean(log_L_samples[n_walkers * n_steps // 2:])
            log_L_vars[t_idx] = np.var(log_L_samples[n_walkers * n_steps // 2:])
        
        log_Z = np.trapz(log_L_means, betas)
        log_Z_err = np.sqrt(np.trapz(log_L_vars, betas) / (n_walkers * n_steps // 2))
        
        self.models[model_name]['log_evidence'] = log_Z
        
        return log_Z, log_Z_err
    
    def compute_evidence_importance_sampling(self, model_name: str, data: Dict, n_samples: int = 100000) -> Tuple[float, float]:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered.")
        
        model = self.models[model_name]
        prior_ranges = model['prior_ranges']
        n_params = len(prior_ranges)
        
        prior_samples = np.zeros((n_samples, n_params))
        for i, (p_min, p_max) in enumerate(prior_ranges):
            prior_samples[:, i] = np.random.uniform(p_min, p_max, n_samples)
        
        log_L_values = np.zeros(n_samples)
        for s in range(n_samples):
            try:
                log_L_values[s] = model['log_L'](prior_samples[s], data)
            except:
                log_L_values[s] = -1.0e10
        
        log_L_max = np.max(log_L_values[np.isfinite(log_L_values)])
        
        L_normalized = np.exp(log_L_values - log_L_max)
        
        prior_volume = np.prod([p_max - p_min for p_min, p_max in prior_ranges])
        
        Z = np.mean(L_normalized) * prior_volume
        log_Z = np.log(Z) + log_L_max
        
        Z_err = np.std(L_normalized) / np.sqrt(n_samples) * prior_volume
        log_Z_err = Z_err / Z
        
        self.models[model_name]['log_evidence'] = log_Z
        
        return log_Z, log_Z_err
    
    def bayes_factor(self, model_1: str, model_2: str) -> Tuple[float, str]:
        if self.models[model_1]['log_evidence'] is None:
            raise ValueError(f"Evidence for {model_1} not computed yet.")
        if self.models[model_2]['log_evidence'] is None:
            raise ValueError(f"Evidence for {model_2} not computed yet.")
        
        log_BF = self.models[model_1]['log_evidence'] - self.models[model_2]['log_evidence']
        BF = np.exp(log_BF)
        
        interpretation = self.jeffreys_interpretation(np.abs(log_BF), model_1 if log_BF > 0 else model_2)
        
        return BF, log_BF, interpretation
    
    def jeffreys_interpretation(self, log_BF_abs: float, favored_model: str) -> str:
        if log_BF_abs < np.log(1.0):
            return f"Inconclusive (no preference)"
        elif log_BF_abs < np.log(3.0):
            return f"Weak evidence for {favored_model}"
        elif log_BF_abs < np.log(10.0):
            return f"Moderate evidence for {favored_model}"
        elif log_BF_abs < np.log(30.0):
            return f"Strong evidence for {favored_model}"
        elif log_BF_abs < np.log(100.0):
            return f"Very strong evidence for {favored_model}"
        else:
            return f"Decisive evidence for {favored_model}"
    
    def aic_bic_comparison(self, model_results: Dict[str, Dict]) -> Dict:
        comparison = {}
        
        for model_name, results in model_results.items():
            chi2_min = results.get('chi2_min', 0)
            n_params = results.get('n_params', 4)
            n_data = results.get('n_data', 100)
            
            log_L_max = -0.5 * chi2_min
            
            AIC = 2.0 * n_params - 2.0 * log_L_max
            
            BIC = n_params * np.log(n_data) - 2.0 * log_L_max
            
            AICc = AIC + 2.0 * n_params * (n_params + 1) / (n_data - n_params - 1)
            
            comparison[model_name] = {
                'log_L_max': log_L_max,
                'AIC': AIC,
                'BIC': BIC,
                'AICc': AICc,
                'n_params': n_params,
                'n_data': n_data
            }
        
        min_AIC = min([v['AIC'] for v in comparison.values()])
        min_BIC = min([v['BIC'] for v in comparison.values()])
        
        for model_name in comparison:
            comparison[model_name]['delta_AIC'] = comparison[model_name]['AIC'] - min_AIC
            comparison[model_name]['delta_BIC'] = comparison[model_name]['BIC'] - min_BIC
            comparison[model_name]['Akaike_weight'] = np.exp(-0.5 * comparison[model_name]['delta_AIC'])
        
        total_weight = sum([v['Akaike_weight'] for v in comparison.values()])
        for model_name in comparison:
            comparison[model_name]['Akaike_weight'] /= total_weight
        
        return comparison

