import numpy as np

from tbm.RTM_initial import sip_leaf, soil_spectra, atmoE
from tbm.RTM_initial import cal_lidf, weighted_sum_over_lidf_vec, CIxy
from tbm.RTM_initial import hemi_initial, dif_initial, hemi_dif_initial
from tbm.RTM_initial import calc_sun_angles
from tbm.Ebal_initial import calc_extinc_coeff_pars
from tbm.hydraulics_funcs import cal_thetas, hygroscopic_point, field_capacity, saturated_matrix_potential, calc_b

xrange = range


class TBM_Data:
    """
    Data for TBM model
    """

    def __init__(self, p, lat, lon, data):
        """ Extracts data from netcdf file
        :param start_yr: year for model runs to begin as an integer (year)
        :param end_yr: year for model runs to end as an integer (year)
        :return:
        """
        [flux_arr, rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr, phiI, phiII] = data

        self.flux_data = flux_arr # [(flux_arr['year'] >= start_yr) & (flux_arr['year'] < end_yr+1)]

        # 'Driving Data'
        self.sw = self.flux_data['sw']
        self.par = self.flux_data['par']
        self.t_mean = self.flux_data['ta']
        self.vpd = self.flux_data['vpd'] * 100
        self.precip = self.flux_data['p']
        self.wds = self.flux_data['wds']

        self.year = self.flux_data['year']  # Year
        self.month = self.flux_data['month']  # Month
        self.day = self.flux_data['day']  # Date in month
        self.D = self.flux_data['doy']  # day of year
        self.hour = self.flux_data['hour']  # day of year

        self.len_run = self.flux_data[['year', 'month', 'day']].drop_duplicates().shape[0]

        self.Cab = np.full((1, self.len_run), p.Cab)
        self.Car = np.full((1, self.len_run), p.Car)
        self.Cm = np.full((1, self.len_run), p.Cm)
        self.Cbrown = np.full((1, self.len_run), p.Cbrown)  # brown pigments concentration (unitless).
        self.Cw = np.full((1, self.len_run), p.Cw)  # equivalent water thickness (g cm-2 or cm).
        self.Ant = np.full((1, self.len_run), p.Ant)  # Anthocianins concentration (mug cm-2).
        self.Alpha = np.full((1, self.len_run), p.Alpha)  # constant for the optimal size of the leaf scattering element
        self.fLMA_k = np.full((1, self.len_run), p.fLMA_k)
        self.gLMA_k = np.full((1, self.len_run), p.gLMA_k)
        self.gLMA_b = np.full((1, self.len_run), p.gLMA_b)

        """ 
        Initialization of Leaf-level SIF  
        """
        self.Kab, self.nr, self.Kall, self.leaf = sip_leaf(prospectpro, self.Cab, self.Car, self.Cbrown, self.Cw, \
                                                           self.Cm, self.Ant, self.Alpha, self.fLMA_k, self.gLMA_k, \
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
        stdlon = (int(lon / 15) + -1 * (1 if abs(lon) % 15 > 7.5 else 0)) * 15

        # non-leap/leap year
        self.tts = self.flux_data['sza']
        self.tto = self.flux_data['vza']
        self.psi = self.flux_data['raa']

        """
        Initialization of leaf angle distribution
        """
        # the proportion of leaves with different angles
        self.lidf = cal_lidf(p.lidfa, p.lidfb)

        """
        Clumping Index (CI_flag)      
        """
        self.CIs = CIxy(p.CI_flag, self.tts, p.CI_thres)
        self.CIo = CIxy(p.CI_flag, self.tto, p.CI_thres)

        """ 
        Initialization of canopy-level Radiative Transfer Model 
        """
        _, _, self.ks, self.ko, _, self.sob, self.sof = weighted_sum_over_lidf_vec(self.lidf, self.tts, self.tto, self.psi)
        self.ks[self.tts > 75] = 1
        self.ko[self.tto > 75] = 1

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
        self.sm_top = np.full(len(self.flux_data) + 1, p.sm0)
        self.w_can = np.full(len(self.flux_data) + 1, p.w0)

        p.Soil["theta_sat"] = cal_thetas(p.Soil['soc_top'])
        p.Soil["fc_top"] = field_capacity(p.Soil['soc_top'])
        p.Soil["sh_top"] = hygroscopic_point(p.Soil['soc_top'])

        p.Soil["phis_sat"] = saturated_matrix_potential(p.Soil["soc_top"][0])
        p.Soil["b1"] = calc_b(p.Soil["soc_top"][2])
