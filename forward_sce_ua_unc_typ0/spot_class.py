import numpy as np
import pandas as pd
import spotpy
from spotpy.parameter import Uniform, Normal
from dataloader import Dataloader
from predict import predict


class spot_setup(object):
    clab = Uniform(low=1, high=2000)
    cf = Uniform(low=1, high=10)
    cr = Uniform(low=1, high=2000)
    cw = Uniform(low=1, high=1E5)
    cl = Uniform(low=1, high=2000)
    cs = Uniform(low=1, high=1E5)

    clspan = Uniform(low=1.0001, high=8)
    lma = Uniform(low=10.429688, high=119.570312)
    f_auto = Uniform(low=0.3, high=0.7)
    f_fol = Uniform(low=0.01, high=0.5)
    f_lab = Uniform(low=0.01, high=0.5)

    Theta = Uniform(low=0.018, high=0.08)
    theta_min = Uniform(low=1E-5, high=1E-2)
    theta_woo = Uniform(low=2.5E-5, high=1E-3)
    theta_roo = Uniform(low=1E-4, high=1E-2)
    theta_lit = Uniform(low=1E-4, high=1E-2)
    theta_som = Uniform(low=1E-7, high=1E-3)

    d_onset = Uniform(low=0, high=180)
    cronset = Uniform(low=10, high=100)
    d_fall = Uniform(low=180, high=365)
    crfall = Uniform(low=20, high=150)

    CI = Uniform(low=0.601562, high=0.998437)
    lidf = Uniform(low=0.234375, high=59.765625)
    cab = Uniform(low=10.273438, high=79.726562)

    RUB = Uniform(low=50.390625, high=149.609375)
    Rdsc = Uniform(low=0.000139, high=0.009961)
    CB6F = Uniform(low=50.390625, high=149.609375)
    gm = Uniform(low=0.029492, high=4.980508)
    BallBerrySlope = Uniform(low=0.058594, high=14.941406)
    BallBerry0 = Uniform(low=0.003906, high=0.996094)

    def __init__(self, root, vs, ms, hourly_obs, daily_obs):
        self.obj_func = spotpy.objectivefunctions.rmse
        self.root = root
        self.vs = vs
        self.ms = ms
        self.batch_size = 1

        self.hourly_obs = hourly_obs
        self.daily_obs = daily_obs

    def simulation(self, pars):
        hourly_df = self.hourly_obs.copy()
        daily_df = self.daily_obs.copy()

        daily_df['gpp'] = 0.001

        daily_df['clab'] = pars[0]
        daily_df['cf'] = pars[1]
        daily_df['cr'] = pars[2]
        daily_df['cw'] = pars[3]
        daily_df['cl'] = pars[4]
        daily_df['cs'] = pars[5]

        daily_df['clspan'] = pars[6]
        daily_df['lma'] = pars[7]
        daily_df['f_auto'] = pars[8]
        daily_df['f_fol'] = pars[9]
        daily_df['f_lab'] = pars[10]

        daily_df['Theta'] = pars[11]
        daily_df['theta_min'] = pars[12]
        daily_df['theta_woo'] = pars[13]
        daily_df['theta_roo'] = pars[14]
        daily_df['theta_lit'] = pars[15]
        daily_df['theta_som'] = pars[16]

        daily_df['d_onset'] = pars[17]
        daily_df['cronset'] = pars[18]
        daily_df['d_fall'] = pars[19]
        daily_df['crfall'] = pars[20]

        # daily_df['pft'] = self.pft

        hourly_df['fpar'] = 0.0
        hourly_df['an'] = 0.0

        hourly_df['CI'] = pars[21]
        hourly_df['lidf'] = pars[22]
        hourly_df['cab'] = pars[23]
        hourly_df['lma'] = pars[7]

        hourly_df['RUB'] = pars[24]
        hourly_df['Rdsc'] = pars[25]
        hourly_df['CB6F'] = pars[26]
        hourly_df['gm'] = pars[27]
        hourly_df['BallBerrySlope'] = pars[28]
        hourly_df['BallBerry0'] = pars[29]

        # hourly_df['pft'] = self.pft

        dL = Dataloader(self.root, daily_df, hourly_df, self.vs, self.batch_size)
        lai_sim, nee_sim, lst_sim, ref_red_sim, ref_nir_sim = predict(daily_df, hourly_df, self.ms, self.vs, dL)

        return lai_sim, nee_sim, lst_sim, ref_red_sim, ref_nir_sim

    def evaluation(self):
        return self.daily_obs, self.hourly_obs

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        lai_sim, nee_sim, lst_sim, ref_red_sim, ref_nir_sim = simulation
        daily_obs, hourly_obs = evaluation

        daily_lai = daily_obs[['year', 'lai', 'lai_std']].copy()
        hourly_nee = hourly_obs[['year', 'nee', 'nee_unc']].copy()
        hourly_lst = hourly_obs[['lst', 'lst_unc']].copy()
        hourly_ref_red = hourly_obs[['doy', 'ref_red', 'ref_red_unc']].copy()
        hourly_ref_nir = hourly_obs[['doy', 'ref_nir', 'ref_nir_unc']].copy()

        daily_lai['lai_sim'] = lai_sim
        hourly_nee['nee_sim'] = nee_sim
        hourly_lst['lst_sim'] = lst_sim
        hourly_ref_red['ref_red_sim'] = ref_red_sim
        hourly_ref_nir['ref_nir_sim'] = ref_nir_sim

        daily_lai['lai_sim'] = daily_lai['lai_sim'].replace([np.inf, -np.inf], np.nan)
        hourly_nee['nee_sim'] = hourly_nee['nee_sim'].replace([np.inf, -np.inf], np.nan)
        hourly_lst['lst_sim'] = hourly_lst['lst_sim'].replace([np.inf, -np.inf], np.nan)
        hourly_ref_red['ref_red_sim'] = hourly_ref_red['ref_red_sim'].replace([np.inf, -np.inf], np.nan)
        hourly_ref_nir['ref_nir_sim'] = hourly_ref_nir['ref_nir_sim'].replace([np.inf, -np.inf], np.nan)

        daily_lai_nan = daily_lai['lai_sim'].isna().any()
        hourly_nee_nan = hourly_nee['nee_sim'].isna().any()
        hourly_lst_nan = hourly_lst['lst_sim'].isna().any()
        hourly_ref_red_nan = hourly_ref_red['ref_red_sim'].isna().any()
        hourly_ref_nir_nan = hourly_ref_nir['ref_nir_sim'].isna().any()

        if daily_lai_nan or hourly_nee_nan or hourly_lst_nan or hourly_ref_red_nan or hourly_ref_nir_nan:
            like = np.Inf
            return like

        ###########################
        # EDC1: annual LAI
        ###########################
        annual_avg_lai = daily_lai.groupby('year')['lai_sim'].mean()
        # check if the ratio of consecutive years' LAI is between 5/6 and 6/5
        var_annual_lai = True
        for i in range(1, len(annual_avg_lai) - 1):
            ratio1 = annual_avg_lai.iloc[i] / annual_avg_lai.iloc[i - 1]
            ratio2 = annual_avg_lai.iloc[i] / annual_avg_lai.iloc[i + 1]
            if not (5 / 6 <= ratio1 <= 6 / 5) or not (5 / 6 <= ratio2 <= 6 / 5):
                var_annual_lai = False
        if not var_annual_lai:
            like = np.Inf
            return like

        ###########################
        # cost function
        ###########################
        else:
            like1 = sum(((daily_lai['lai'].values - daily_lai['lai_sim'].values) / daily_lai['lai_std'].values)**2) / len(daily_lai)
            like2 = sum(((hourly_nee['nee'].values - hourly_nee['nee_sim'].values) / hourly_nee['nee_unc'].values)**2) / len(hourly_nee)
            like3 = sum(((hourly_lst['lst'].values - hourly_lst['lst_sim'].values) / hourly_lst['lst_unc'].values)**2) / len(hourly_lst)
            like4 = sum(((hourly_ref_red['ref_red'].values - hourly_ref_red['ref_red_sim'].values) / hourly_ref_red['ref_red_unc'].values)**2) / len(hourly_ref_red)
            like5 = sum(((hourly_ref_nir['ref_nir'].values - hourly_ref_nir['ref_nir_sim'].values) / hourly_ref_nir['ref_nir_unc'].values)**2) / len(hourly_ref_nir)

            # print(round(like1, 2), round(like2, 2), round(like3, 2), round(like4, 2), round(like5, 2))
            like = (like1 + like2 + like3 + like4 + like5)

            return like
