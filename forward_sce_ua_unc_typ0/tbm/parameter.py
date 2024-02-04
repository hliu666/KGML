import numpy as np
from tbm.constants import O, P, T2K
from tbm.resistance_funcs import calc_d_0, calc_z_0M

"""
Parameters for TBM model
"""


class TBM_Pars:
    def __init__(self, params):
        """
        Carbon pool parameters and constants
        """
        # I.C. for carbon pools gCm-2     range
        self.clab = params['parclab']  # (10,1e3)
        self.cf = params['parcf']  # (10,1e3)
        self.cr = params['parcr']  # (10,1e3)
        self.cw = params['parcw']  # (3e3,3e4)
        self.cl = params['parcl']  # (10,1e3)
        self.cs = params['parcs']  # (1e3, 1e5)

        self.clspan = params['parclspan']  # clspan, leaf lifespan               (1.0001 - 5)
        self.lma = params['parlma']  # clma, leaf mass per area          (81 - 120) g C m-2
        self.f_auto = params['parf_auto']  # f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.f_fol = params['parf_fol']  # f_fol, frac GPP to foliage        (0.01 - 0.5)
        self.f_lab = params['parf_lab']  # f_lab, frac GPP to clab           (0.01 - 0.5)
        self.f_roo = 1 - self.f_fol - self.f_lab  # f_roo, frac GPP to fine roots    (0.01 - 0.5)
        self.Theta = params['parTheta']  # Theta, temp dependence exp fact (0.018 - 0.08)
        self.theta_min = params['partheta_min']  # theta_min, cl to cs decomp      (1e-5 - 1e-2) day-1
        self.theta_woo = params['partheta_woo']  # theta_woo, wood C turnover      (2.5e-5 - 1e-3) day-1
        self.theta_roo = params['partheta_roo']  # theta_roo, root C turnover rate(1e-4 - 1e-2) day-1
        self.theta_lit = params['partheta_lit']  # theta_lit, litter C turnover     (1e-4 - 1e-2) day-1
        self.theta_som = params['partheta_som']  # theta_som, SOM C turnover       (1e-7 - 1e-3) day-1

        self.d_onset = params['pard_onset']  # d_onset, clab release date       (1 - 365) (60,150)
        self.cronset = params['parcronset']  # cronset, clab release period      (10 - 100)
        self.d_fall = params['pard_fall']  # d_fall, date of leaf fall        (1 - 365) (242,332)
        self.crfall = params['parcrfall']  # crfall, leaf fall period          (10 - 100)

        """
        Canopy structure parameters
        Clumping Index (CI_flag):
            0: CI varied with the zenith angle
            1: CI as a constant 
            2: Without considering CI effect            
        """
        self.CI_thres = params['parCI']  # 0.72  # clumping index
        self.CI_flag = 0  # clumping index
        self.lidfa = params['parlidf']  # 55  # 63.6
        self.lidfb = 0.0

        """
        Photosynthesis parameters and constants
        """
        # 1. Initial values of parameters
        self.RUB = params['parRUB']  # [umol sites m-2] Rubisco density
        self.Rdsc = params['parRdsc']  # [] Scalar for mitochondrial (dark) respiration
        self.CB6F = params['parCB6F']  # [umol sites m-2] Cyt b6f density
        self.gm = params['pargm']  # [] mesophyll conductance to CO2
        self.e = 3.0  # 0.92

        # 2. Initial values of constants
        self.BallBerrySlope = params['parBallBerrySlope']
        self.BallBerry0 = params['parBallBerry0']  # intercept of Ball-Berry stomatal conductance model
        self.Rd25 = self.Rdsc * self.RUB

        # Cytochrome b6f complex
        self.kc0 = 1.0  # [umol CO2 umol sites-1 s-1] Rubisco kcat for CO2
        self.Eac = 58000  # [J umol-1] Activation energy

        self.kq0 = 1.0  # 35  # [umol e-1 umol sites-1 s-1] Cyt b6f kcat for PQH2
        self.Eaq = 37000  # [J mol-1] Activation energy

        # Cytochrome b6f-limited rates
        self.Kp1 = 14.5E9  # [s-1] Rate constant for photochemistry at PSI
        self.Kf = 0.05E9  # [s-1] Rate constant for fluoresence at PSII and PSI
        self.Kd = 0.55E9  # [s-1] Rate constant for constitutive heat loss at PSII and PSI

        self.a2 = self.e / (1 + self.e)  # [] PSII, mol PPFD abs PS2 mol-1 PPFD incident
        self.a1 = 1 - self.a2  # [] PSI, mol PPFD abs PS1 mol-1 PPFD incident

        self.nl = 0.75  # [ATP/e-] ATP per e- in linear flow
        self.nc = 1.00  # [ATP/e-] ATP per e- in cyclic flow

        self.spfy25 = 2444  # specificity (Computed from Bernacchhi et al. 2001 paper)
        self.ppm2bar = 1E-6 * (P * 1E-3)  # convert all to bar: CO2 was supplied in ppm, O2 in permil, and pressure in mBar
        self.O_c3 = (O * 1E-3) * (P * 1E-3)
        self.Gamma_star25 = 0.5 * self.O_c3 / self.spfy25  # [ppm] compensation point in absence of Rd

        # temperature correction for Gamma_star
        self.Eag = 37830  # Unit is [J K^-1]

        # temperature correction for Rd
        self.Ear = 46390  # Unit is [J K^-1]

        self.Tyear = 7.4
        self.Tref = 25 + T2K  # [K] absolute temperature at 25 degrees

        # temperature correction of Vcmax
        self.Eav = 55729  # Unit is [J K^-1]
        self.deltaSv = (-1.07 * self.Tyear + 668)  # Unit is [J mol^-1 K^-1]
        self.Hdv = 200000  # Unit is [J mol^-1]

        self.Kc25 = 404.9 * 1E-6  # [mol mol-1]
        self.Ko25 = 278.4 * 1E-3  # [mol mol-1]

        # temperature correction for Kc
        self.Ec = 79430  # Unit is  [J K^-1]

        # temperature correction for Ko
        self.Eo = 36380  # Unit is  [J K^-1]

        self.minCi = 0.3

        """
        Fluorescence (Jen) parameters and constants 
        """
        # 1. Initial values of parameters
        self.eta = 5E-5

        # 2. Initial values of photochemical constants
        self.Kn1 = 14.5E9  # [s-1] Rate constant for regulated heat loss at PSI
        self.Kp2 = 4.5E9  # [s-1] Rate constant for photochemistry at PSII
        self.Ku2 = 0E9  # [s-1] Rate constant for exciton sharing at PSII

        self.eps1 = 0.5  # [mol PSI F to detector mol-1 PSI F emitted] PS I transfer function
        self.eps2 = 0.5  # [mol PSII F to detector mol-1 PSII F emitted] PS II transfer function

        """
        Resistence parameters and constants 
        """
        # 1. Initial values of parameters
        self.leaf_width = 0.1  # efective leaf width size (m)
        self.h_C = 10.0  # vegetation height
        self.zm = 10.0  # Measurement height of meteorological data
        self.z_u = 10.0  # Height of measurement of windspeed (m).
        self.CM_a = 0.01  # Choudhury and Monteith 1988 leaf drag coefficient

        # 2. Initial values of constants
        self.d_0 = calc_d_0(self.h_C)  # displacement height
        self.z_0M = calc_z_0M(self.h_C)  # roughness length for momentum of the canopy

        """
        Leaf traits parameters and constants 
        """
        self.Cab = params['parcab']
        self.Car = 5.563160774
        self.Cm = self.lma / 10000.0
        self.Cbrown = 0.185385  # brown pigments concentration (unitless).
        self.Cw = 0.00597  # equivalent water thickness (g cm-2 or cm).
        self.Ant = 1.96672  # Anthocianins concentration (mug cm-2).
        self.Alpha = 600  # constant for the optimal size of the leaf scattering element
        self.fLMA_k = 2519.65
        self.gLMA_k = -631.54
        self.gLMA_b = 0.0086

        self.rho = 0.01  # [1]               Leaf/needle reflection
        self.tau = 0.01  # [1]               Leaf/needle transmission
        self.rs = 0.06  # [1]               Soil reflectance
        """
        Soil parameters and constants 
        """
        self.rsoil = 0.5  # brightness
        self.Soil = {
            "soc_top": [43, 39, 18],  # Soil composition, 1 by 3, percentages of sand, silt and clay
            "Zr_top": 0.5,
            "sti_top": 2,  # [] soil texture ID
        }
        self.sm0 = 0.8
        self.w0 = 0.0

        """
        Phenology
        """
        self.radconv = 365.25 / np.pi
