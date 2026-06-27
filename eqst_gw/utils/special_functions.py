import numpy as np
from scipy.special import erf, erfc, gamma as scipy_gamma, zeta as scipy_zeta
from scipy.integrate import quad
from typing import Union, Tuple, Optional


def thermal_function_boson_full(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.zeros_like(x_arr)

    for i, x_val in enumerate(x_arr):
        if x_val < 0:
            result[i] = 0.0
        elif x_val < 1.0e-10:
            result[i] = -np.pi**4 / 45.0
        elif x_val < 1.0:
            result[i] = (-np.pi**4 / 45.0 +
                         (np.pi**2 / 12.0) * x_val -
                         (np.pi / 6.0) * x_val**(3.0/2.0) +
                         x_val**2 * np.log(x_val) / 32.0)
        elif x_val < 100.0:
            integrand = lambda t: t**2 * np.log(1.0 - np.exp(-np.sqrt(t**2 + x_val)))
            integral, _ = quad(integrand, 0, 50.0, limit=100)
            result[i] = integral
        else:
            result[i] = -(np.pi**2 / 12.0) * x_val * np.exp(-np.sqrt(x_val))

    return result if x_arr.size > 1 else result[0]


def thermal_function_fermion_full(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.zeros_like(x_arr)

    for i, x_val in enumerate(x_arr):
        if x_val < 0:
            result[i] = 0.0
        elif x_val < 1.0e-10:
            result[i] = 7.0 * np.pi**4 / 360.0
        elif x_val < 1.0:
            result[i] = (7.0 * np.pi**4 / 360.0 +
                         (np.pi**2 / 24.0) * x_val +
                         x_val**2 * np.log(x_val) / 32.0)
        elif x_val < 100.0:
            integrand = lambda t: t**2 * np.log(1.0 + np.exp(-np.sqrt(t**2 + x_val)))
            integral, _ = quad(integrand, 0, 50.0, limit=100)
            result[i] = integral
        else:
            result[i] = (np.pi**2 / 24.0) * x_val * np.exp(-np.sqrt(x_val))

    return result if x_arr.size > 1 else result[0]


def daisy_resummation_correction(phi: np.ndarray,
                                   T: float,
                                   Pi_T: float) -> np.ndarray:
    phi_arr = np.atleast_1d(phi)

    m_phi_sq = phi_arr**2

    correction = -(T / (12.0 * np.pi)) * (
        (m_phi_sq + Pi_T)**(3.0/2.0) - m_phi_sq**(3.0/2.0)
    )

    correction[m_phi_sq < 0] = 0.0

    return correction if phi_arr.size > 1 else correction[0]


def euclidean_bounce_action_thin_wall(delta_V: float,
                                       sigma_wall: float,
                                       epsilon: float) -> float:
    if abs(epsilon) < 1.0e-30 or sigma_wall <= 0:
        return np.inf

    R_bubble = 3.0 * sigma_wall / epsilon

    S3_thin_wall = 4.0 * np.pi * R_bubble**2 * sigma_wall / 3.0

    return S3_thin_wall


def nucleation_rate_prefactor(T: float, S3: float, g_star: float) -> float:
    A_prefactor = T**4 * (S3 / (2.0 * np.pi * T))**(3.0/2.0)
    return A_prefactor


def kolmogorov_spectrum(k: np.ndarray,
                         epsilon_dissipation: float,
                         k_integral: float,
                         k_dissipation: float,
                         C_K: float = 1.6) -> np.ndarray:
    k_arr = np.atleast_1d(k)
    E_k = np.zeros_like(k_arr)

    mask_inertial = (k_arr >= k_integral) & (k_arr < k_dissipation)
    mask_injection = k_arr < k_integral
    mask_dissipation = k_arr >= k_dissipation

    E_k[mask_inertial] = C_K * epsilon_dissipation**(2.0/3.0) * k_arr[mask_inertial]**(-5.0/3.0)

    E_k[mask_injection] = C_K * epsilon_dissipation**(2.0/3.0) * k_integral**(-5.0/3.0) * (k_arr[mask_injection] / k_integral)**4

    E_k[mask_dissipation] = C_K * epsilon_dissipation**(2.0/3.0) * k_arr[mask_dissipation]**(-5.0/3.0) * np.exp(-(k_arr[mask_dissipation] / k_dissipation - 1.0)**2)

    return E_k if k_arr.size > 1 else E_k[0]


def transfer_function_radiation_domination(k: np.ndarray,
                                            k_eq: float,
                                            Omega_m: float = 0.315,
                                            h: float = 0.674) -> np.ndarray:
    k_arr = np.atleast_1d(k)

    Gamma = Omega_m * h * np.exp(-Omega_m - np.sqrt(2.0 * h) / Omega_m)
    q = k_arr / (Gamma * h)

    T_k = (np.log(1.0 + 2.34 * q) / (2.34 * q) *
           (1.0 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**(-0.25))

    return T_k if k_arr.size > 1 else T_k[0]


def jeans_length(T_gas: float, rho_gas: float, mu: float = 1.22) -> float:
    k_B = 1.380649e-23
    m_H = 1.6735575e-27
    G = 6.67430e-11

    c_s = np.sqrt(5.0 / 3.0 * k_B * T_gas / (mu * m_H))

    lambda_J = c_s * np.sqrt(np.pi / (G * rho_gas))

    return lambda_J


def scale_factor_from_temperature(T: np.ndarray, T_0: float = 2.7255) -> np.ndarray:
    T_arr = np.atleast_1d(T)

    T_eV = T_arr / 8.617333262e-5
    T_0_eV = T_0 / 8.617333262e-5

    g_star_S_T = np.where(T_arr > 1.0e9, 106.75, np.where(T_arr > 1.0e2, 10.75, 3.91))
    g_star_S_0 = 3.91

    a = (g_star_S_0 / g_star_S_T)**(1.0/3.0) * T_0_eV / T_eV

    return a if T_arr.size > 1 else a[0]


def gravitational_wave_energy_from_stress(Pi_ij: np.ndarray,
                                           k: np.ndarray,
                                           t_duration: float,
                                           G_newton: float = 6.67430e-11) -> np.ndarray:
    k_arr = np.atleast_1d(k)
    Pi_arr = np.atleast_1d(Pi_ij)

    c = 2.99792458e8

    rho_gw_k = (G_newton / c**3) * k_arr**2 * Pi_arr**2 * t_duration

    return rho_gw_k if k_arr.size > 1 else rho_gw_k[0]


def overlap_reduction_function_isotropic(f: np.ndarray,
                                          detector_separation: float,
                                          c: float = 2.99792458e8) -> np.ndarray:
    f_arr = np.atleast_1d(f)

    alpha = 2.0 * np.pi * f_arr * detector_separation / c

    gamma_f = (3.0 / (2.0 * alpha)) * (np.sin(alpha) * (1.0 - 1.0 / alpha**2) +
                                         np.cos(alpha) / alpha)

    return gamma_f if f_arr.size > 1 else gamma_f[0]


def sound_horizon_approximation(Omega_b_h2: float = 0.02237,
                                 Omega_m_h2: float = 0.1432) -> float:
    z_drag = (1291.0 * Omega_m_h2**0.251 /
               (1.0 + 0.659 * Omega_m_h2**0.828) *
               (1.0 + 0.395 * Omega_m_h2**(-0.569)))

    R_drag = 31500.0 * Omega_b_h2 * (2.7255 / 2.7)**(-4) / z_drag

    r_s = (147.09 * (Omega_m_h2 / 0.1432)**(-0.255) *
            (Omega_b_h2 / 0.02237)**(-0.128))

    return r_s


def lorentz_boost_velocity(v1: float, v2: float, c: float = 1.0) -> float:
    return (v1 + v2) / (1.0 + v1 * v2 / c**2)


def relativistic_enthalpy(rho: np.ndarray, p: np.ndarray) -> np.ndarray:
    return rho + p


def jouguet_detonation_velocity(alpha: float, c_s: float = 1.0 / np.sqrt(3.0)) -> float:
    numerator = c_s**2 * (1.0 + np.sqrt(3.0 * alpha * (2.0 + 3.0 * alpha)))
    denominator = 1.0 + alpha

    v_J = numerator / denominator

    return min(v_J, 1.0)