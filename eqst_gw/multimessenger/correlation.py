import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.signal import correlate, coherence
from typing import Tuple, Dict, Optional, List, Callable
from ..core.parameters import EQSTGPParameters
from ..core.constants import FundamentalConstants
from ..core.cosmology import LambdaEffectiveCosmology


class GWMultiMessengerCorrelator:
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

    def gw_cmb_cross_correlation(self,
                                  f_gw: np.ndarray,
                                  Omega_gw: np.ndarray,
                                  ell_cmb: np.ndarray,
                                  C_ell_cmb: np.ndarray) -> Dict:
        f_gw_arr = np.atleast_1d(f_gw)
        Omega_arr = np.atleast_1d(Omega_gw)
        ell_arr = np.atleast_1d(ell_cmb)
        C_arr = np.atleast_1d(C_ell_cmb)

        k_gw = 2.0 * np.pi * f_gw_arr / self.const.c
        a_today = 1.0
        chi_rec = self.cosmo.comoving_distance(1090.0) * self.const.Mpc_to_m / 1000.0
        ell_from_k = k_gw * chi_rec

        Omega_interp = interp1d(np.log10(f_gw_arr), np.log10(np.maximum(Omega_arr, 1.0e-40)),
                                 kind='cubic', fill_value='extrapolate')

        C_interp = interp1d(ell_arr, C_arr, kind='cubic',
                             bounds_error=False, fill_value=0.0)

        ell_overlap = np.logspace(np.log10(max(ell_arr[0], ell_from_k[0])),
                                   np.log10(min(ell_arr[-1], ell_from_k[-1])), 200)

        f_at_ell = ell_overlap / chi_rec / (2.0 * np.pi) * self.const.c
        Omega_at_ell = 10.0**Omega_interp(np.log10(f_at_ell))
        C_at_ell = C_interp(ell_overlap)

        Omega_norm = Omega_at_ell / np.max(Omega_at_ell)
        C_norm = C_at_ell / np.max(np.abs(C_at_ell))

        pearson_r, pearson_p = pearsonr(Omega_norm, C_norm)
        spearman_r, spearman_p = spearmanr(Omega_norm, C_norm)

        cross_power = np.trapz(Omega_norm * C_norm, np.log(ell_overlap))
        auto_gw = np.trapz(Omega_norm**2, np.log(ell_overlap))
        auto_cmb = np.trapz(C_norm**2, np.log(ell_overlap))

        coherence_value = cross_power / np.sqrt(auto_gw * auto_cmb)

        sachs_wolfe_correlation = self._sachs_wolfe_gw_overlap(ell_overlap, Omega_at_ell, C_at_ell)

        return {
            'pearson_r': pearson_r,
            'pearson_p_value': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p_value': spearman_p,
            'coherence': coherence_value,
            'cross_power_integral': cross_power,
            'sachs_wolfe_overlap': sachs_wolfe_correlation,
            'ell_overlap_range': (ell_overlap[0], ell_overlap[-1])
        }

    def _sachs_wolfe_gw_overlap(self,
                                  ell: np.ndarray,
                                  Omega_gw: np.ndarray,
                                  C_ell: np.ndarray) -> float:
        SW_kernel = 1.0 / (ell * (ell + 1.0))
        SW_kernel /= np.max(SW_kernel)

        integrand = Omega_gw * C_ell * SW_kernel

        overlap = np.trapz(integrand, np.log(ell))

        normalisation = np.sqrt(np.trapz(Omega_gw**2, np.log(ell)) *
                                 np.trapz(C_ell**2, np.log(ell)))

        if normalisation > 0:
            return overlap / normalisation
        return 0.0

    def gw_bao_correlation(self,
                            f_gw: np.ndarray,
                            Omega_gw: np.ndarray,
                            z_bao: np.ndarray,
                            DM_over_rd: np.ndarray,
                            DH_over_rd: np.ndarray) -> Dict:
        f_gw_arr = np.atleast_1d(f_gw)
        Omega_arr = np.atleast_1d(Omega_gw)

        f_peak_idx = np.argmax(Omega_arr)
        f_peak = f_gw_arr[f_peak_idx]
        Omega_peak = Omega_arr[f_peak_idx]

        T_n_eV = self.ep.T_n * 1.0e9
        T_CMB_eV = self.const.T_CMB_K * 8.617333262e-5
        z_phase_transition = T_n_eV / T_CMB_eV - 1.0

        z_test = np.linspace(0.1, 10.0, 500)
        Lambda_eff_z = self.cosmo.Lambda_eff(z_test)
        H_z = self.cosmo.H_eff(z_test, self.const.Omega_m_Planck2018)

        H_interp = interp1d(z_test, H_z, kind='cubic', fill_value='extrapolate')

        H_at_bao = H_interp(z_bao)
        DH_predicted_ratio = (self.const.c / 1000.0) / H_at_bao / 147.09

        residuals_DH = DH_over_rd - DH_predicted_ratio

        gw_amplitude_effect = Omega_peak * 1.0e13
        bao_gw_coupling = gw_amplitude_effect * np.sin(np.pi * z_bao / z_phase_transition)

        coupling_correlation, p_val = pearsonr(residuals_DH, bao_gw_coupling)

        return {
            'peak_frequency_Hz': f_peak,
            'peak_amplitude_h2': Omega_peak,
            'z_phase_transition': z_phase_transition,
            'DH_residuals': residuals_DH,
            'gw_bao_coupling_correlation': coupling_correlation,
            'coupling_p_value': p_val,
            'H_at_bao_z': H_at_bao,
            'DH_predicted_ratio': DH_predicted_ratio
        }

    def gw_dm_spatial_correlation(self,
                                   f_gw: np.ndarray,
                                   Omega_gw: np.ndarray,
                                   r_dm: np.ndarray,
                                   rho_dm: np.ndarray,
                                   z_halo: float = 0.0) -> Dict:
        f_gw_arr = np.atleast_1d(f_gw)
        Omega_arr = np.atleast_1d(Omega_gw)
        r_arr = np.atleast_1d(r_dm)
        rho_arr = np.atleast_1d(rho_dm)

        Omega_total = np.trapz(Omega_arr, np.log(f_gw_arr))
        M_DM_total = np.trapz(4.0 * np.pi * r_arr**2 * rho_arr, r_arr)

        m_DM_kg = self.ep.m_DM_GeV * self.const.GeV_to_kg
        n_DM_profile = rho_arr / m_DM_kg

        gw_production_rate = Omega_arr / np.trapz(Omega_arr, np.log(f_gw_arr))

        dm_formation_kernel = np.exp(-f_gw_arr / self.ep.f_sw_Hz)

        spatial_gw_dm_coupling = np.outer(dm_formation_kernel, n_DM_profile)

        mean_coupling = np.mean(spatial_gw_dm_coupling)
        std_coupling = np.std(spatial_gw_dm_coupling)

        correlation_length = r_arr[np.argmax(rho_arr)]

        return {
            'Omega_gw_total': Omega_total,
            'M_DM_total_kg': M_DM_total,
            'n_DM_peak_m3': np.max(n_DM_profile),
            'mean_gw_dm_coupling': mean_coupling,
            'std_gw_dm_coupling': std_coupling,
            'correlation_length_m': correlation_length,
            'spatial_coupling_matrix': spatial_gw_dm_coupling
        }

    def gw_sn_correlation(self,
                           f_gw: np.ndarray,
                           Omega_gw: np.ndarray,
                           z_sn: np.ndarray,
                           mu_obs: np.ndarray,
                           mu_lcdm: np.ndarray) -> Dict:
        f_gw_arr = np.atleast_1d(f_gw)
        Omega_arr = np.atleast_1d(Omega_gw)

        mu_residuals = mu_obs - mu_lcdm

        c_km_s = self.const.c / 1000.0
        H_z_sn = self.cosmo.H_eff(z_sn, self.const.Omega_m_Planck2018)

        delta_mu_from_gw = np.zeros_like(z_sn)
        for i, z in enumerate(z_sn):
            Lambda_eqst = self.cosmo.Lambda_eff(z)
            Lambda_lcdm = self.const.Omega_Lambda_Planck2018
            delta_Lambda = Lambda_eqst - Lambda_lcdm
            if abs(delta_Lambda) > 1.0e-10:
                d_C_delta, _ = quad(lambda zp: (c_km_s / (self.const.H0_Planck2018 *
                                     np.sqrt(self.const.Omega_m_Planck2018 * (1 + zp)**3 +
                                     Lambda_lcdm + delta_Lambda))) -
                                     (c_km_s / (self.const.H0_Planck2018 *
                                     np.sqrt(self.const.Omega_m_Planck2018 * (1 + zp)**3 +
                                     Lambda_lcdm))), 0, z)
                delta_mu_from_gw[i] = 5.0 * np.log10(1.0 + d_C_delta / self.cosmo.comoving_distance(z))
            else:
                delta_mu_from_gw[i] = 0.0

        if np.std(mu_residuals) > 0 and np.std(delta_mu_from_gw) > 0:
            corr_r, corr_p = pearsonr(mu_residuals, delta_mu_from_gw)
        else:
            corr_r, corr_p = 0.0, 1.0

        chi2_improvement = np.sum(mu_residuals**2) - np.sum((mu_residuals - delta_mu_from_gw)**2)

        return {
            'mu_residuals_mean': np.mean(mu_residuals),
            'mu_residuals_std': np.std(mu_residuals),
            'delta_mu_from_Lambda_eff': delta_mu_from_gw,
            'correlation_r': corr_r,
            'correlation_p_value': corr_p,
            'chi2_improvement': chi2_improvement,
            'n_SN': len(z_sn)
        }

    def full_multimessenger_correlation_matrix(self,
                                                datasets: Dict[str, np.ndarray]) -> np.ndarray:
        observable_names = list(datasets.keys())
        N = len(observable_names)

        corr_matrix = np.eye(N)

        for i in range(N):
            for j in range(i + 1, N):
                obs_i = np.atleast_1d(datasets[observable_names[i]])
                obs_j = np.atleast_1d(datasets[observable_names[j]])

                min_len = min(len(obs_i), len(obs_j))
                if min_len > 2:
                    obs_i_interp = obs_i[:min_len]
                    obs_j_interp = obs_j[:min_len]

                    if np.std(obs_i_interp) > 0 and np.std(obs_j_interp) > 0:
                        r, _ = pearsonr(obs_i_interp, obs_j_interp)
                        corr_matrix[i, j] = r
                        corr_matrix[j, i] = r

        return corr_matrix

    def time_delay_correlation(self,
                                signal_1: np.ndarray,
                                signal_2: np.ndarray,
                                dt: float,
                                max_lag_s: float = 1.0e-2) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(signal_1)
        max_lag_samples = int(max_lag_s / dt)
        max_lag_samples = min(max_lag_samples, n_samples // 2)

        ccf = np.zeros(2 * max_lag_samples + 1)
        lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=float) * dt

        std_1 = np.std(signal_1)
        std_2 = np.std(signal_2)

        if std_1 == 0 or std_2 == 0:
            return lags, ccf

        sig1_norm = (signal_1 - np.mean(signal_1)) / std_1
        sig2_norm = (signal_2 - np.mean(signal_2)) / std_2

        for idx, lag in enumerate(range(-max_lag_samples, max_lag_samples + 1)):
            if lag >= 0:
                n_overlap = n_samples - lag
                if n_overlap > 1:
                    ccf[idx] = np.mean(sig1_norm[lag:] * sig2_norm[:n_overlap])
            else:
                n_overlap = n_samples + lag
                if n_overlap > 1:
                    ccf[idx] = np.mean(sig1_norm[:n_overlap] * sig2_norm[-lag:])

        return lags, ccf

    def gw_21cm_cross_spectrum(self,
                                f_gw: np.ndarray,
                                Omega_gw: np.ndarray,
                                z_21cm: np.ndarray,
                                T_21cm_mK: np.ndarray) -> Dict:
        f_gw_arr = np.atleast_1d(f_gw)
        Omega_arr = np.atleast_1d(Omega_gw)
        z_arr = np.atleast_1d(z_21cm)
        T_arr = np.atleast_1d(T_21cm_mK)

        f_21cm_rest = 1420.4057e6

        f_21cm_obs = f_21cm_rest / (1.0 + z_arr)

        H_z = self.cosmo.H_eff(z_arr, self.const.Omega_m_Planck2018)

        Lambda_eff_z = self.cosmo.Lambda_eff(z_arr)

        dT_dLambda = -50.0 * np.ones_like(z_arr)

        T_21cm_eqst_correction = dT_dLambda * (Lambda_eff_z - self.const.Omega_Lambda_Planck2018)

        gw_at_21cm_freqs = np.interp(f_21cm_obs / 1.0e9,
                                       f_gw_arr,
                                       Omega_arr,
                                       left=0.0, right=0.0)

        if np.std(T_arr) > 0 and np.std(gw_at_21cm_freqs) > 0:
            cross_corr, p_val = pearsonr(T_arr, gw_at_21cm_freqs)
        else:
            cross_corr, p_val = 0.0, 1.0

        return {
            'f_21cm_obs_Hz': f_21cm_obs,
            'T_21cm_eqst_correction_mK': T_21cm_eqst_correction,
            'Omega_gw_at_21cm_freq': gw_at_21cm_freqs,
            'cross_correlation': cross_corr,
            'p_value': p_val,
            'H_z_at_21cm': H_z,
            'Lambda_eff_at_21cm': Lambda_eff_z
        }

    def pta_gw_background_correlation(self,
                                       f_pta: np.ndarray,
                                       Omega_pta: np.ndarray,
                                       sigma_pta: np.ndarray,
                                       f_lisa: np.ndarray,
                                       Omega_lisa: np.ndarray) -> Dict:
        f_pta_arr = np.atleast_1d(f_pta)
        Omega_pta_arr = np.atleast_1d(Omega_pta)
        sigma_pta_arr = np.atleast_1d(sigma_pta)
        f_lisa_arr = np.atleast_1d(f_lisa)
        Omega_lisa_arr = np.atleast_1d(Omega_lisa)

        f_all = np.concatenate([f_pta_arr, f_lisa_arr])
        idx_sort = np.argsort(f_all)
        f_all_sorted = f_all[idx_sort]

        Omega_eqst_at_pta = np.interp(f_pta_arr, f_lisa_arr, Omega_lisa_arr)

        chi2_eqst_pta = np.sum(((Omega_pta_arr - Omega_eqst_at_pta) / sigma_pta_arr)**2)
        dof_pta = len(f_pta_arr)

        power_law_amp = Omega_pta_arr[len(Omega_pta_arr)//2]
        power_law_index = 2.0
        Omega_power_law = power_law_amp * (f_pta_arr / f_pta_arr[len(f_pta_arr)//2])**power_law_index

        chi2_pl_pta = np.sum(((Omega_pta_arr - Omega_power_law) / sigma_pta_arr)**2)

        spectral_tilt_pta = np.polyfit(np.log10(f_pta_arr),
                                        np.log10(np.maximum(Omega_pta_arr, 1.0e-30)), 1)[0]

        spectral_tilt_eqst_at_pta = np.polyfit(np.log10(f_pta_arr),
                                                  np.log10(np.maximum(Omega_eqst_at_pta, 1.0e-30)), 1)[0]

        spectral_tilt_match = abs(spectral_tilt_pta - spectral_tilt_eqst_at_pta) < 0.5

        frequency_extrapolation_consistency = (
            Omega_pta_arr[-1] / Omega_eqst_at_pta[-1]
            if Omega_eqst_at_pta[-1] > 0 else np.inf
        )

        return {
            'chi2_EQST_vs_PTA': chi2_eqst_pta,
            'chi2_power_law_vs_PTA': chi2_pl_pta,
            'chi2_reduced_EQST': chi2_eqst_pta / dof_pta,
            'chi2_reduced_power_law': chi2_pl_pta / dof_pta,
            'spectral_tilt_PTA': spectral_tilt_pta,
            'spectral_tilt_EQST_at_PTA': spectral_tilt_eqst_at_pta,
            'spectral_tilt_consistent': spectral_tilt_match,
            'frequency_extrapolation_ratio': frequency_extrapolation_consistency,
            'Omega_eqst_predicted_at_pta': Omega_eqst_at_pta,
            'preferred_model': 'EQST-GP' if chi2_eqst_pta < chi2_pl_pta else 'Power Law'
        }

    def angular_power_spectrum_gw_lss(self,
                                       f_gw: np.ndarray,
                                       Omega_gw: np.ndarray,
                                       ell: np.ndarray,
                                       P_matter: np.ndarray,
                                       k_matter: np.ndarray,
                                       z_lss: float = 1.0) -> Dict:
        f_arr = np.atleast_1d(f_gw)
        Omega_arr = np.atleast_1d(Omega_gw)
        ell_arr = np.atleast_1d(ell)
        P_arr = np.atleast_1d(P_matter)
        k_arr = np.atleast_1d(k_matter)

        chi_lss = self.cosmo.comoving_distance(z_lss)

        k_from_ell = (ell_arr + 0.5) / chi_lss

        P_at_ell = np.interp(k_from_ell, k_arr, P_arr, left=0.0, right=0.0)

        f_at_ell = (ell_arr + 0.5) / (chi_lss * self.const.Mpc_to_m / 1000.0) * self.const.c / (2.0 * np.pi)

        Omega_at_ell = np.interp(f_at_ell, f_arr, Omega_arr, left=0.0, right=0.0)

        C_ell_cross = Omega_at_ell * P_at_ell / chi_lss**2

        if np.std(P_at_ell) > 0 and np.std(Omega_at_ell) > 0:
            cross_corr, p_val = pearsonr(P_at_ell, Omega_at_ell)
        else:
            cross_corr, p_val = 0.0, 1.0

        return {
            'C_ell_cross': C_ell_cross,
            'ell': ell_arr,
            'P_matter_at_ell': P_at_ell,
            'Omega_gw_at_ell': Omega_at_ell,
            'k_from_ell': k_from_ell,
            'cross_correlation': cross_corr,
            'p_value': p_val,
            'chi_lss_Mpc': chi_lss
        }