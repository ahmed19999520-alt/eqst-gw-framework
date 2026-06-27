import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize
from typing import Tuple, Dict, Optional
from ..core.constants import FundamentalConstants
from ..core.cosmology import LambdaEffectiveCosmology

class HubbleTensionResolution:
    def __init__(self,
                 constants: Optional[FundamentalConstants] = None,
                 cosmology: Optional[LambdaEffectiveCosmology] = None):
        
        if constants is None:
            self.const = FundamentalConstants()
        else:
            self.const = constants
        
        if cosmology is None:
            self.cosmo = LambdaEffectiveCosmology(self.const)
        else:
            self.cosmo = cosmology
        
        self.H0_Planck = 67.4
        self.H0_Planck_err = 0.5
        self.H0_SH0ES = 73.04
        self.H0_SH0ES_err = 1.04
        self.H0_DESI = 68.52
        self.H0_DESI_err = 0.62
        self.H0_CCHP = 69.96
        self.H0_CCHP_err = 1.05
        self.H0_TDCOSMO = 74.2
        self.H0_TDCOSMO_err = 1.6
    
    def effective_H0_from_lambda_eff(self, Omega_m: float = 0.315) -> float:
        theta_star_target = 0.0104109
        
        def chi2_theta(H0):
            self.cosmo.H0 = H0
            self.cosmo.h = H0 / 100.0
            
            z_star = 1090.0
            z_drag = 1060.0
            
            c_km_s = self.const.c / 1000.0
            
            r_s, _ = quad(lambda z: c_km_s / (self.cosmo.H_eff(z, Omega_m) * np.sqrt(3.0 * (1.0 + 31500.0 * self.const.Omega_b_h2_Planck2018 * (2.7255 / 2.7)**(-4) / (1.0 + z)))), z_drag, 1200.0)
            
            d_A, _ = quad(lambda z: c_km_s / self.cosmo.H_eff(z, Omega_m), 0, z_star)
            
            theta_pred = r_s / d_A
            
            return (theta_pred - theta_star_target)**2
        
        result = minimize_scalar(chi2_theta, bounds=(60.0, 80.0), method='bounded')
        
        H0_eff = result.x
        
        return H0_eff
    
    def tension_sigma(self, H0_1: float, sigma_1: float, H0_2: float, sigma_2: float) -> float:
        delta_H0 = abs(H0_1 - H0_2)
        sigma_total = np.sqrt(sigma_1**2 + sigma_2**2)
        return delta_H0 / sigma_total
    
    def compute_all_tensions(self) -> Dict:
        H0_EQST = self.effective_H0_from_lambda_eff()
        
        tensions = {
            'Planck_vs_SH0ES': self.tension_sigma(self.H0_Planck, self.H0_Planck_err, self.H0_SH0ES, self.H0_SH0ES_err),
            'Planck_vs_TDCOSMO': self.tension_sigma(self.H0_Planck, self.H0_Planck_err, self.H0_TDCOSMO, self.H0_TDCOSMO_err),
            'EQST_vs_Planck': self.tension_sigma(H0_EQST, 1.3, self.H0_Planck, self.H0_Planck_err),
            'EQST_vs_SH0ES': self.tension_sigma(H0_EQST, 1.3, self.H0_SH0ES, self.H0_SH0ES_err),
            'EQST_vs_DESI': self.tension_sigma(H0_EQST, 1.3, self.H0_DESI, self.H0_DESI_err),
            'H0_EQST_GP': H0_EQST
        }
        
        return tensions
    
    def S8_tension_analysis(self) -> Dict:
        sigma_8_Planck = 0.811
        Omega_m_Planck = 0.315
        S8_Planck = sigma_8_Planck * np.sqrt(Omega_m_Planck / 0.3)
        S8_Planck_err = 0.006
        
        S8_KiDS = 0.766
        S8_KiDS_err = 0.020
        
        S8_DES = 0.759
        S8_DES_err = 0.023
        
        from ..simulations.structure_formation import StructureFormationAnalysis
        sfa = StructureFormationAnalysis(self.const, self.cosmo)
        sigma_8_EQST = sfa.sigma8_eqst_gp()
        
        Omega_m_EQST = 0.315
        S8_EQST = sigma_8_EQST * np.sqrt(Omega_m_EQST / 0.3)
        S8_EQST_err = 0.015
        
        return {
            'S8_Planck': S8_Planck,
            'S8_Planck_err': S8_Planck_err,
            'S8_KiDS': S8_KiDS,
            'S8_KiDS_err': S8_KiDS_err,
            'S8_DES': S8_DES,
            'S8_DES_err': S8_DES_err,
            'S8_EQST_GP': S8_EQST,
            'S8_EQST_GP_err': S8_EQST_err,
            'tension_Planck_KiDS': self.tension_sigma(S8_Planck, S8_Planck_err, S8_KiDS, S8_KiDS_err),
            'tension_EQST_KiDS': self.tension_sigma(S8_EQST, S8_EQST_err, S8_KiDS, S8_KiDS_err)
        }
