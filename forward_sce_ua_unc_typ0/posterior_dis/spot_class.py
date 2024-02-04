import numpy as np
import pandas as pd
import spotpy
from spotpy.parameter import Uniform, Normal
from dataloader import Dataloader
from predict import predict
from scipy.stats import chi2_contingency


class spot_setup(object):

    def __init__(self, root, vs, ms, hourly_obs, daily_obs, pars, likes):
        self.step = 0.1

        clab_low, clab_high = 1, 2000
        cf_low, cf_high = 1, 10
        cr_low, cr_high = 1, 2000
        cw_low, cw_high = 1, 1E5
        cl_low, cl_high = 1, 2000
        cs_low, cs_high = 1, 1E5

        clspan_low, clspan_high = 1.0001, 8
        lma_low, lma_high = 10, 120
        f_auto_low, f_auto_high = 0.3, 0.7
        f_fol_low, f_fol_high = 0.01, 0.5
        f_lab_low, f_lab_high = 0.01, 0.5

        Theta_low, Theta_high = 0.018, 0.08
        theta_min_low, theta_min_high = 1E-5, 1E-2
        theta_woo_low, theta_woo_high = 2.5E-5, 1E-3
        theta_roo_low, theta_roo_high = 1E-4, 1E-2
        theta_lit_low, theta_lit_high = 1E-4, 1E-2
        theta_som_low, theta_som_high = 1E-7, 1E-3

        d_onset_low, d_onset_high = 0, 180
        cronset_low, cronset_high = 10, 100
        d_fall_low, d_fall_high = 180, 365
        crfall_low, crfall_high = 20, 150

        CI_low, CI_high = 0.601562, 0.998437
        lidf_low, lidf_high = 0.234375, 59.765625
        cab_low, cab_high = 10.273438, 79.726562

        RUB_low, RUB_high = 50.390625, 149.609375
        Rdsc_low, Rdsc_high = 0.000139, 0.009961
        CB6F_low, CB6F_high = 50.390625, 149.609375
        gm_low, gm_high = 0.029492, 4.980508
        BallBerrySlope_low, BallBerrySlope_high = 0.058594, 14.941406
        BallBerry0_low, BallBerry0_high = 0.003906, 0.996094

        self.params = [spotpy.parameter.Normal('clab', mean=pars['parclab'], stddev=(clab_high - clab_low)*self.step),
                       spotpy.parameter.Normal('cf', mean=pars['parcf'], stddev=(cf_high - cf_low)*self.step),
                       spotpy.parameter.Normal('cr', mean=pars['parcr'], stddev=(cr_high - cr_low)*self.step),
                       spotpy.parameter.Normal('cw', mean=pars['parcw'], stddev=(cw_high - cw_low)*self.step),
                       spotpy.parameter.Normal('cl', mean=pars['parcl'], stddev=(cl_high - cl_low)*self.step),
                       spotpy.parameter.Normal('cs', mean=pars['parcs'], stddev=(cs_high - cs_low)*self.step),

                       spotpy.parameter.Normal('clspan', mean=pars['parclspan'], stddev=(clspan_high - clspan_low)*self.step),
                       spotpy.parameter.Normal('lma', mean=pars['parlma'], stddev=(lma_high - lma_low)*self.step),
                       spotpy.parameter.Normal('f_auto', mean=pars['parf_auto'], stddev=(f_auto_high - f_auto_low)*self.step),
                       spotpy.parameter.Normal('f_fol', mean=pars['parf_fol'], stddev=(f_fol_high - f_fol_low)*self.step),
                       spotpy.parameter.Normal('f_lab', mean=pars['parf_lab'], stddev=(f_lab_high - f_lab_low)*self.step),

                       spotpy.parameter.Normal('Theta', mean=pars['parTheta'], stddev=(Theta_high - Theta_low)*self.step),
                       spotpy.parameter.Normal('theta_min', mean=pars['partheta_min'], stddev=(theta_min_high - theta_min_low)*self.step),
                       spotpy.parameter.Normal('theta_woo', mean=pars['partheta_woo'], stddev=(theta_woo_high - theta_woo_low)*self.step),
                       spotpy.parameter.Normal('theta_roo', mean=pars['partheta_roo'], stddev=(theta_roo_high - theta_roo_low)*self.step),
                       spotpy.parameter.Normal('theta_lit', mean=pars['partheta_lit'], stddev=(theta_lit_high - theta_lit_low)*self.step),
                       spotpy.parameter.Normal('theta_som', mean=pars['partheta_som'], stddev=(theta_som_high - theta_som_low)*self.step),

                       spotpy.parameter.Normal('d_onset', mean=pars['pard_onset'], stddev=(d_onset_high - d_onset_low)*self.step),
                       spotpy.parameter.Normal('cronset', mean=pars['parcronset'], stddev=(cronset_high - cronset_low)*self.step),
                       spotpy.parameter.Normal('d_fall', mean=pars['pard_fall'], stddev=(d_fall_high - d_fall_low)*self.step),
                       spotpy.parameter.Normal('crfall', mean=pars['parcrfall'], stddev=(crfall_high - crfall_low)*self.step),

                       spotpy.parameter.Normal('CI', mean=pars['parCI'], stddev=(CI_high - CI_low)*self.step),
                       spotpy.parameter.Normal('lidf', mean=pars['parlidf'], stddev=(lidf_high - lidf_low)*self.step),
                       spotpy.parameter.Normal('cab', mean=pars['parlidf'], stddev=(cab_high - cab_low)*self.step),

                       spotpy.parameter.Normal('RUB', mean=pars['parRUB'], stddev=(RUB_high - RUB_low)*self.step),
                       spotpy.parameter.Normal('Rdsc', mean=pars['parRdsc'], stddev=(Rdsc_high - Rdsc_low)*self.step),
                       spotpy.parameter.Normal('CB6F', mean=pars['parCB6F'], stddev=(CB6F_high - CB6F_low)*self.step),
                       spotpy.parameter.Normal('gm', mean=pars['pargm'], stddev=(gm_high - gm_low)*self.step),
                       spotpy.parameter.Normal('BallBerrySlope', mean=pars['parBallBerrySlope'], stddev=(BallBerrySlope_high - BallBerrySlope_low)*self.step),
                       spotpy.parameter.Normal('BallBerry0', mean=pars['parBallBerry0'], stddev=(BallBerry0_high - BallBerry0_low)*self.step)
                       ]

        self.obj_func = spotpy.objectivefunctions.rmse
        self.root = root
        self.vs = vs
        self.ms = ms
        self.batch_size = 1

        self.hourly_obs = hourly_obs
        self.daily_obs = daily_obs

        self.likes = likes

    def parameters(self):
        return spotpy.parameter.generate(self.params)

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
            daily_lai = daily_lai.dropna()
            hourly_nee = hourly_nee.dropna()
            hourly_lst = hourly_lst.dropna()
            hourly_ref_red = hourly_ref_red.dropna()
            hourly_ref_nir = hourly_ref_nir.dropna()

            like1 = sum(((daily_lai['lai'].values - daily_lai['lai_sim'].values) / daily_lai['lai_std'].values)**2) / len(daily_lai)
            like2 = sum(((hourly_nee['nee'].values - hourly_nee['nee_sim'].values) / hourly_nee['nee_unc'].values)**2) / len(hourly_nee)
            like3 = sum(((hourly_lst['lst'].values - hourly_lst['lst_sim'].values) / hourly_lst['lst_unc'].values)**2) / len(hourly_lst)
            like4 = sum(((hourly_ref_red['ref_red'].values - hourly_ref_red['ref_red_sim'].values) / hourly_ref_red['ref_red_unc'].values)**2) / len(hourly_ref_red)
            like5 = sum(((hourly_ref_nir['ref_nir'].values - hourly_ref_nir['ref_nir_sim'].values) / hourly_ref_nir['ref_nir_unc'].values)**2) / len(hourly_ref_nir)

            observed = np.array([like1, like2, like3, like4, like5])
            expected = np.array(self.likes)

            normalized_observed = observed / np.min(observed)
            normalized_expected = expected / np.min(expected)

            chi2, p_value, _, _ = chi2_contingency([normalized_observed, normalized_expected])

            # if p_value <= 0.1:
            #    like = 99999
            #else:
            #    like = 1 - p_value

            return 1 - p_value
