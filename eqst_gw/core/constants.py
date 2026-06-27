import numpy as np
from dataclasses import dataclass

@dataclass
class FundamentalConstants:
    c: float = 2.99792458e8
    hbar: float = 1.054571817e-34
    G: float = 6.67430e-11
    k_B: float = 1.380649e-23
    
    M_pl_GeV: float = 1.221e19
    l_P_m: float = 1.616255e-35
    
    eV_to_J: float = 1.602176634e-19
    GeV_to_kg: float = 1.782661907e-27
    Mpc_to_m: float = 3.085677581e22
    year_to_s: float = 31557600.0
    
    H0_Planck2018: float = 67.4
    h_Planck2018: float = 0.674
    Omega_m_Planck2018: float = 0.315
    Omega_Lambda_Planck2018: float = 0.685
    Omega_b_h2_Planck2018: float = 0.02237
    Omega_cdm_h2_Planck2018: float = 0.1200
    
    T_CMB_K: float = 2.7255
    n_s_Planck2018: float = 0.9649
    sigma_8_Planck2018: float = 0.811
    tau_reio_Planck2018: float = 0.054
    
    def __post_init__(self):
        self.M_pl_SI = np.sqrt(self.hbar * self.c / self.G)
        self.t_Pl_s = self.l_P_m / self.c
        self.rho_Pl_SI = self.c**5 / (self.hbar * self.G**2)