import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

class CMBPreprocessor:
    def __init__(self):
        pass
    
    def apply_beam_deconvolution(self, ell: np.ndarray, D_ell: np.ndarray, theta_beam_arcmin: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        theta_rad = theta_beam_arcmin * np.pi / (60.0 * 180.0)
        sigma_beam = theta_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        B_ell = np.exp(-0.5 * ell * (ell + 1.0) * sigma_beam**2)
        
        D_ell_deconv = D_ell / B_ell**2
        
        D_ell_deconv[B_ell < 0.01] = D_ell[B_ell < 0.01]
        
        return ell, D_ell_deconv
    
    def subtract_noise_bias(self, ell: np.ndarray, D_ell: np.ndarray, N_ell: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        D_ell_signal = D_ell - ell * (ell + 1.0) * N_ell / (2.0 * np.pi)
        return ell, D_ell_signal
    
    def binning(self, ell: np.ndarray, D_ell: np.ndarray, sigma: np.ndarray, delta_ell: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ell_min = ell[0]
        ell_max = ell[-1]
        
        bins = np.arange(ell_min, ell_max + delta_ell, delta_ell)
        
        ell_bins = np.zeros(len(bins) - 1)
        D_ell_bins = np.zeros(len(bins) - 1)
        sigma_bins = np.zeros(len(bins) - 1)
        
        for i in range(len(bins) - 1):
            mask = (ell >= bins[i]) & (ell < bins[i + 1])
            if np.sum(mask) > 0:
                weights = 1.0 / sigma[mask]**2
                ell_bins[i] = np.average(ell[mask], weights=weights)
                D_ell_bins[i] = np.average(D_ell[mask], weights=weights)
                sigma_bins[i] = 1.0 / np.sqrt(np.sum(weights))
        
        return ell_bins, D_ell_bins, sigma_bins
    
    def smooth_spectrum(self, ell: np.ndarray, D_ell: np.ndarray, window_length: int = 21, polyorder: int = 3) -> np.ndarray:
        return savgol_filter(D_ell, window_length, polyorder)


class BAOPreprocessor:
    def __init__(self):
        pass
    
    def fiducial_sound_horizon(self, Omega_b_h2: float = 0.02237, Omega_m_h2: float = 0.1432) -> float:
        z_drag = 1291.0 * Omega_m_h2**0.251 / (1.0 + 0.659 * Omega_m_h2**0.828) * (1.0 + 0.395 * Omega_m_h2**(-0.569))
        
        a_drag = 1.0 / (1.0 + z_drag)
        
        R_eq = 31500.0 * Omega_b_h2 * (2.7255 / 2.7)**(-4) / 3400.0
        
        r_d = 1.0 / (0.0628 * np.sqrt(Omega_m_h2)) * np.log(np.sqrt(1.0 + R_eq) + np.sqrt(R_eq + a_drag * R_eq / a_drag)) / np.sqrt(1.0 + R_eq)
        
        return r_d * 3000.0 / 0.674
    
    def reconstruct_power_spectrum(self, positions: np.ndarray, velocities: np.ndarray, f_growth: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        smoothing_length = 15.0
        
        n_grid = 256
        density_field = np.zeros((n_grid, n_grid, n_grid))
        
        L_box = np.max(positions) - np.min(positions)
        
        grid_indices = ((positions - np.min(positions)) / L_box * n_grid).astype(int) % n_grid
        
        for i in range(len(positions)):
            idx = tuple(grid_indices[i])
            density_field[idx] += 1.0
        
        density_field /= np.mean(density_field)
        density_field -= 1.0
        
        delta_k = np.fft.fftn(density_field)
        
        kx = np.fft.fftfreq(n_grid, d=L_box / n_grid) * 2.0 * np.pi
        ky = np.fft.fftfreq(n_grid, d=L_box / n_grid) * 2.0 * np.pi
        kz = np.fft.fftfreq(n_grid, d=L_box / n_grid) * 2.0 * np.pi
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        K[0, 0, 0] = 1.0
        
        k_bins = np.linspace(0.01, 0.5, 50)
        P_k = np.zeros(len(k_bins) - 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
        
        for i in range(len(k_bins) - 1):
            mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
            P_k[i] = np.mean(np.abs(delta_k[mask])**2) * (L_box / n_grid)**3
        
        return k_centers, P_k


class GWDataPreprocessor:
    def __init__(self):
        pass
    
    def whiten_strain_data(self, strain: np.ndarray, psd: np.ndarray, dt: float) -> np.ndarray:
        strain_fft = np.fft.rfft(strain)
        freqs = np.fft.rfftfreq(len(strain), d=dt)
        
        psd_interp = interp1d(np.linspace(freqs[0], freqs[-1], len(psd)), psd, bounds_error=False, fill_value='extrapolate')
        
        psd_at_freqs = psd_interp(freqs)
        psd_at_freqs[psd_at_freqs <= 0] = 1.0e-40
        
        whitened_fft = strain_fft / np.sqrt(psd_at_freqs)
        
        whitened_strain = np.fft.irfft(whitened_fft, n=len(strain))
        
        return whitened_strain
    
    def compute_power_spectral_density(self, strain: np.ndarray, dt: float, window: str = 'hann', nperseg: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.signal import welch
        
        freqs, psd = welch(strain, fs=1.0/dt, window=window, nperseg=nperseg, detrend='constant', scaling='density')
        
        return freqs, psd
    
    def bandpass_filter(self, strain: np.ndarray, dt: float, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
        from scipy.signal import butter, filtfilt
        
        fs = 1.0 / dt
        Wn = [f_low / (fs / 2.0), f_high / (fs / 2.0)]
        
        b, a = butter(order, Wn, btype='band')
        
        filtered = filtfilt(b, a, strain)
        
        return filtered
    
    def remove_spectral_lines(self, strain: np.ndarray, dt: float, line_frequencies: List[float], notch_width: float = 1.0) -> np.ndarray:
        strain_fft = np.fft.rfft(strain)
        freqs = np.fft.rfftfreq(len(strain), d=dt)
        
        for f_line in line_frequencies:
            mask = np.abs(freqs - f_line) < notch_width
            strain_fft[mask] = 0.0
        
        strain_cleaned = np.fft.irfft(strain_fft, n=len(strain))
        
        return strain_cleaned
    
    def identify_glitches(self, strain: np.ndarray, dt: float, threshold_sigma: float = 5.0) -> np.ndarray:
        rolling_std = np.std(strain)
        
        snr = np.abs(strain) / rolling_std
        
        glitch_mask = snr > threshold_sigma
        
        return glitch_mask
    
    def gap_filling_inpainting(self, strain: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
        strain_filled = strain.copy()
        
        valid_indices = np.where(~gap_mask)[0]
        gap_indices = np.where(gap_mask)[0]
        
        if len(gap_indices) > 0 and len(valid_indices) > 0:
            strain_interp = interp1d(valid_indices, strain[valid_indices], kind='cubic', fill_value='extrapolate')
            strain_filled[gap_indices] = strain_interp(gap_indices)
        
        return strain_filled
