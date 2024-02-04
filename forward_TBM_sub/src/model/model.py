"""
TBM model class takes a data class and then uses functions to run the TBM model.
"""
import numpy as np
from constants import T2K

from RTM_Optical import rtm_o, BRF_hemi_dif_func
from Ebal import Ebal
from PhotoSynth_Jen import PhotoSynth_Jen, calc_resp
from hydraulics import calc_hy_f

xrange = range


class TBM_Model:

    def __init__(self, dataclass, pramclass):
        """ Model class for running DALEC2
        :param dataclass: TBM data class containing data to run model
        :return:
        """
        self.d = dataclass
        self.p = pramclass

    # ------------------------------------------------------------------------------
    # Model functions
    # ------------------------------------------------------------------------------
    def tbm(self):

        lai = self.d.lai

        hemi_dif_brf = BRF_hemi_dif_func(self.d.hemi_dif_pars, lai)
        rtm_o_dict = rtm_o(self.d, self.p, lai, hemi_dif_brf)
        ebal_dict, netrad_sw_dict, netrad_lw_dict = Ebal(self.d, self.p, lai, rtm_o_dict)
        if (self.d.tts < 75) and \
                (netrad_sw_dict['ERnuc'] > 1) and \
                (netrad_sw_dict['ERnhc'] > 1) and \
                (lai > 0.1):
            # ----------------------canopy intercepted wator and soil moisture factor---------------------
            # self.d.w_can, fwet, self.d.sm_top, sf = calc_hy_f(self.d, self.p, lai, Ebal_dict['Ev'], Ebal_dict['ET'])
            sf = 1.0

            # ----------------------two leaf model---------------------
            APARu_leaf = netrad_sw_dict['APARu'] / (lai * netrad_lw_dict['Fc'])
            APARh_leaf = netrad_sw_dict['APARh'] / (lai * (1 - netrad_lw_dict['Fc']))

            meteo_u = [APARu_leaf, ebal_dict['Ccu'], ebal_dict['Tcu'], ebal_dict['ecu'], sf]
            meteo_h = [APARh_leaf, ebal_dict['Cch'], ebal_dict['Tch'], ebal_dict['ech'], sf]

            rcw_u, _, Anu, fqe2u, fqe1u = PhotoSynth_Jen(meteo_u, self.p)
            rcw_h, _, Anh, fqe2h, fqe1h = PhotoSynth_Jen(meteo_h, self.p)

            An = (Anu * ebal_dict['Fc'] + Anh * (1 - ebal_dict['Fc'])) * lai

        else:
            # ----------------------canopy intercepted wator and soil moisture factor---------------------
            # self.d.w_can, self.d.sm_top = self.d.w_can, self.d.sm_top

            Rdu = -calc_resp(self.p.Rd25, self.p.Ear, ebal_dict['Tcu'] + T2K)
            Rdh = -calc_resp(self.p.Rd25, self.p.Ear, ebal_dict['Tch'] + T2K)

            An = (Rdu * ebal_dict['Fc'] + Rdh * (1 - ebal_dict['Fc'])) * lai

        sur_refl = rtm_o_dict['BRF'] * netrad_sw_dict['ratio'] + rtm_o_dict['BRF_dif'] * (1 - netrad_sw_dict['ratio'])
        sur_refl_red = float(np.nansum(sur_refl[220:271].flatten() * self.d.rsr_red.flatten()) / np.nansum(self.d.rsr_red.flatten()))
        sur_refl_nir = float(np.nansum(sur_refl[441:477].flatten() * self.d.rsr_nir.flatten()) / np.nansum(self.d.rsr_nir.flatten()))

        brf_refl = rtm_o_dict['BRF']
        brf_refl_red = float(np.nansum(brf_refl[220:271].flatten() * self.d.rsr_red.flatten()) / np.nansum(self.d.rsr_red.flatten()))
        brf_refl_nir = float(np.nansum(brf_refl[441:477].flatten() * self.d.rsr_nir.flatten()) / np.nansum(self.d.rsr_nir.flatten()))

        fpar = sum((rtm_o_dict['A'] * netrad_sw_dict['ratio'] + rtm_o_dict['A_dif'] * (1 - netrad_sw_dict['ratio']))[0:301]) / 301

        out = np.array([An[0], ebal_dict['LST'][0], fpar, sur_refl_red, sur_refl_nir, brf_refl_red, brf_refl_nir])

        return out
