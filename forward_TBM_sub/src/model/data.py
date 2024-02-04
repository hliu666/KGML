import numpy as np

from RTM_initial import sip_leaf, soil_spectra, atmoE
from RTM_initial import cal_lidf, weighted_sum_over_lidf_vec, CIxy
from RTM_initial import hemi_initial, dif_initial, hemi_dif_initial
from RTM_initial import calc_sun_angles
from Ebal_initial import calc_extinc_coeff_pars
from SIF import creat_sif_matrix
from hydraulics_funcs import cal_thetas, hygroscopic_point, field_capacity, saturated_matrix_potential, calc_b

xrange = range


class TBM_Data:
    """
    Data for TBM model
    """

    def __init__(self, p, forcing, data):
        """ Extracts data from netcdf file
        :param start_yr: year for model runs to begin as an integer (year)
        :param end_yr: year for model runs to end as an integer (year)
        :return:
        """
        [rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr] = data

        self.pft = forcing[0]

        # 'Driving Data'
        self.sw = forcing[1]
        self.par = forcing[2]
        self.t_mean = forcing[3]
        self.vpd = forcing[4] * 100
        self.wds = forcing[5]
        self.lai = max(forcing[6], 1e-16)

        self.Cab = p.Cab
        self.Car = p.Car
        self.Cm = p.Cm
        self.Cbrown = p.Cbrown  # brown pigments concentration (unitless).
        self.Cw = p.Cw  # equivalent water thickness (g cm-2 or cm).
        self.Ant = p.Ant  # Anthocianins concentration (mug cm-2).
        self.Alpha = p.Alpha  # constant for the optimal size of the leaf scattering element
        self.fLMA_k = p.fLMA_k
        self.gLMA_k = p.gLMA_k
        self.gLMA_b = p.gLMA_b

        """ 
        Initialization of Leaf-level SIF  
        """
        self.Kab, self.nr, self.Kall, self.leaf = sip_leaf(prospectpro, self.Cab, self.Car, self.Cbrown, self.Cw,
                                                           self.Cm, self.Ant, self.Alpha, self.fLMA_k, self.gLMA_k,
                                                           self.gLMA_b, p.tau, p.rho)

        """ 
        Initialization of soil model
        """
        self.soil = soil_spectra(soil, p.rsoil, p.rs)

        """
        The spectral response curve 
        """
        self.rsr_red = rsr_red
        self.rsr_nir = rsr_nir
        self.rsr_sw1 = rsr_sw1
        self.rsr_sw2 = rsr_sw2

        """ 
        Initialization of sun's spectral curve
        """
        self.wl, self.atmoMs = atmoE(TOCirr)

        """
        Sun-sensor geometry
        """

        # non-leap/leap year
        self.tts = np.array([forcing[7]])
        self.tto = np.array([forcing[8]])
        self.psi = np.array([forcing[9]])

        """
        Initialization of leaf angle distribution
        """
        self.lidf = cal_lidf(p.lidfa, p.lidfb)

        """
        Clumping Index (CI_flag)      
        """
        self.CIs = CIxy(p.CI_flag, self.tts, p.CI_thres)
        self.CIo = CIxy(p.CI_flag, self.tto, p.CI_thres)

        """ 
        Initialization of canopy-level Radiative Transfer Model 
        """
        self.ks, self.ko, _, self.sob, self.sof = weighted_sum_over_lidf_vec(self.lidf, self.tts, self.tto, self.psi)

        self.hemi_pars = hemi_initial(p.CI_flag, self.tts, self.lidf, p.CI_thres)
        self.dif_pars = dif_initial(p.CI_flag, self.tto, self.lidf, p.CI_thres)
        self.hemi_dif_pars = hemi_dif_initial(p.CI_flag, self.lidf, p.CI_thres)

        """
        Initialization of extinction coefficient
        """
        self.extinc_k, self.extinc_sum0 = calc_extinc_coeff_pars(p.CI_flag, p.CI_thres, self.lidf)

        """
        Initialization of hydraulics model
        """
        self.sm_top = p.sm0
        self.w_can = p.w0

        p.Soil["theta_sat"] = cal_thetas(p.Soil['soc_top'])
        p.Soil["fc_top"] = field_capacity(p.Soil['soc_top'])
        p.Soil["sh_top"] = hygroscopic_point(p.Soil['soc_top'])

        p.Soil["phis_sat"] = saturated_matrix_potential(p.Soil["soc_top"][0])
        p.Soil["b1"] = calc_b(p.Soil["soc_top"][2])
