import pandas as pd
import numpy as np
from forward.pars import Par, Var_bicm, Var_rtmo, Var_carp
from forward.model import Model
from dataloader import Dataloader
from predict import predict


def initial_ml(ml_root):
    carp_frcngs_vars = ['gpp', 'doy', 'ta']
    carp_initls_vars = ['clab', 'cf', 'cr', 'cw', 'cl', 'cs']
    carp_params_vars = ['clspan', 'lma', 'f_auto', 'f_fol', 'f_lab',
                        'Theta', 'theta_min', 'theta_woo', 'theta_roo', 'theta_lit', 'theta_som',
                        'd_onset', 'cronset', 'd_fall', 'crfall']
    carp_obsrvs_vars = ['lai']

    carp_v = Var_carp(carp_frcngs_vars, carp_initls_vars, carp_params_vars, carp_obsrvs_vars)

    """
    2. rtmo model
    """
    rtmo_hidden_dim = 120
    rtmo_batch_size = 24
    rtmo_epochs = 1000
    rtmo_learn_rate = 0.001
    rtmo_lr_decay = 0.98

    rtmo_frcngs_vars = ['lai', 'sza', 'vza', 'raa', 'pft']
    rtmo_params_vars = ['CI', 'lidf', "cab", "lma"]
    rtmo_obsrvs_var1 = ['fpar']
    rtmo_obsrvs_var2 = ['ref_red', 'ref_nir']

    rtmo_p = Par(rtmo_hidden_dim, rtmo_batch_size, rtmo_epochs, rtmo_learn_rate, rtmo_lr_decay)
    rtmo_v = Var_rtmo(rtmo_frcngs_vars, rtmo_params_vars, rtmo_obsrvs_var1, rtmo_obsrvs_var2)
    rtmo_m = Model(rtmo_v, rtmo_p)

    rtmo_m.load("rtmo", ml_root)

    """
    3. bicm model
    """
    bicm_hidden_dim = 128
    bicm_batch_size = 24
    bicm_epochs = 1000
    bicm_learn_rate = 0.001
    bicm_lr_decay = 0.94

    bicm_frcngs_vars = ['lai', 'sw', 'ta', 'wds', 'sza', 'fpar', 'par', 'vpd', 'pft']
    bicm_params_vars = ['RUB', 'Rdsc', 'CB6F', 'gm', 'BallBerrySlope', 'BallBerry0']
    bicm_obsrvs_var1 = ['an']
    bicm_obsrvs_var2 = ['lst']

    bicm_p = Par(bicm_hidden_dim, bicm_batch_size, bicm_epochs, bicm_learn_rate, bicm_lr_decay)
    bicm_v = Var_bicm(bicm_frcngs_vars, bicm_params_vars, bicm_obsrvs_var1, bicm_obsrvs_var2)
    bicm_m = Model(bicm_v, bicm_p)

    bicm_m.load("bicm", ml_root)

    vs = [carp_v, rtmo_v, bicm_v]
    ms = [rtmo_m.model, bicm_m.model]

    return vs, ms


def calc_sims(site_ID, site_LC, root, pars):
    vs, ms = initial_ml(root + 'forward/')

    pft_dict = {"ENF": 1, "DBF": 4, "MF": 3, "OSH": 7, "GRA": 10, "WET": 11, "CRO": 12}

    pft = pft_dict[site_LC]
    daily_obs, hourly_obs = load_par(root, site_ID, pft, pars)
    daily_df_ml, hourly_df_ml = daily_obs.copy(), hourly_obs.copy()

    batch_size = 1
    dL = Dataloader(root, daily_df_ml, hourly_df_ml, vs, batch_size)
    lai_sim_ml, nee_sim_ml, lst_sim_ml, ref_red_sim_ml, ref_nir_sim_ml = predict(daily_df_ml, hourly_df_ml, ms, vs, dL)

    daily_lai = daily_obs[['year', 'lai', 'lai_std']].copy()
    hourly_nee = hourly_obs[['year', 'nee', 'nee_unc']].copy()
    hourly_lst = hourly_obs[['lst', 'lst_unc']].copy()
    hourly_ref_red = hourly_obs[['doy', 'ref_red', 'ref_red_unc']].copy()
    hourly_ref_nir = hourly_obs[['doy', 'ref_nir', 'ref_nir_unc']].copy()

    daily_lai['lai_sim'] = lai_sim_ml
    hourly_nee['nee_sim'] = nee_sim_ml
    hourly_lst['lst_sim'] = lst_sim_ml
    hourly_ref_red['ref_red_sim'] = ref_red_sim_ml
    hourly_ref_nir['ref_nir_sim'] = ref_nir_sim_ml

    daily_lai['lai_sim'] = daily_lai['lai_sim'].replace([np.inf, -np.inf], np.nan)
    hourly_nee['nee_sim'] = hourly_nee['nee_sim'].replace([np.inf, -np.inf], np.nan)
    hourly_lst['lst_sim'] = hourly_lst['lst_sim'].replace([np.inf, -np.inf], np.nan)
    hourly_ref_red['ref_red_sim'] = hourly_ref_red['ref_red_sim'].replace([np.inf, -np.inf], np.nan)
    hourly_ref_nir['ref_nir_sim'] = hourly_ref_nir['ref_nir_sim'].replace([np.inf, -np.inf], np.nan)

    daily_lai = daily_lai.dropna()
    hourly_nee = hourly_nee.dropna()
    hourly_lst = hourly_lst.dropna()
    hourly_ref_red = hourly_ref_red.dropna()
    hourly_ref_nir = hourly_ref_nir.dropna()

    like1 = sum(((daily_lai['lai'].values - daily_lai['lai_sim'].values) / daily_lai['lai_std'].values) ** 2) / len(daily_lai)
    like2 = sum(((hourly_nee['nee'].values - hourly_nee['nee_sim'].values) / hourly_nee['nee_unc'].values) ** 2) / len(hourly_nee)
    like3 = sum(((hourly_lst['lst'].values - hourly_lst['lst_sim'].values) / hourly_lst['lst_unc'].values) ** 2) / len(hourly_lst)
    like4 = sum(((hourly_ref_red['ref_red'].values - hourly_ref_red['ref_red_sim'].values) / hourly_ref_red['ref_red_unc'].values) ** 2) / len(hourly_ref_red)
    like5 = sum(((hourly_ref_nir['ref_nir'].values - hourly_ref_nir['ref_nir_sim'].values) / hourly_ref_nir['ref_nir_unc'].values) ** 2) / len(hourly_ref_nir)

    # print(round(like1, 2), round(like2, 2), round(like3, 2), round(like4, 2), round(like5, 2))
    # like = (like1 + like2 + like3 + like4 + like5)

    return [like1, like2, like3, like4, like5]


def load_par(flux_root, site, pft, pars):
    hourly_df = pd.read_csv(flux_root + f"flux/{site}.csv")
    daily_df = pd.read_csv(flux_root + f"flux_d/{site}.csv")

    hourly_df['datetime'] = pd.to_datetime(hourly_df[['year', 'month', 'day', 'hour']])
    hourly_df.set_index('datetime', inplace=True)

    daily_df['gpp'] = 0.001

    daily_df['clab'] = pars['parclab']
    daily_df['cf'] = pars['parcf']
    daily_df['cr'] = pars['parcr']
    daily_df['cw'] = pars['parcw']
    daily_df['cl'] = pars['parcl']
    daily_df['cs'] = pars['parcs']

    daily_df['clspan'] = pars['parclspan']
    daily_df['lma'] = pars['parlma']
    daily_df['f_auto'] = pars['parf_auto']
    daily_df['f_fol'] = pars['parf_fol']
    daily_df['f_lab'] = pars['parf_lab']

    daily_df['Theta'] = pars['parTheta']
    daily_df['theta_min'] = pars['partheta_min']
    daily_df['theta_woo'] = pars['partheta_woo']
    daily_df['theta_roo'] = pars['partheta_roo']
    daily_df['theta_lit'] = pars['partheta_lit']
    daily_df['theta_som'] = pars['partheta_som']

    daily_df['d_onset'] = pars['pard_onset']
    daily_df['cronset'] = pars['parcronset']
    daily_df['d_fall'] = pars['pard_fall']
    daily_df['crfall'] = pars['parcrfall']

    hourly_df['fpar'] = 0.0
    hourly_df['an'] = 0.0

    hourly_df['CI'] = pars['parCI']
    hourly_df['lidf'] = pars['parlidf']
    hourly_df['cab'] = pars['parcab']
    hourly_df['lma'] = pars['parlma']

    hourly_df['RUB'] = pars['parRUB']
    hourly_df['Rdsc'] = pars['parRdsc']
    hourly_df['CB6F'] = pars['parCB6F']
    hourly_df['gm'] = pars['pargm']
    hourly_df['BallBerrySlope'] = pars['parBallBerrySlope']
    hourly_df['BallBerry0'] = pars['parBallBerry0']

    hourly_df['pft'] = pft

    return daily_df, hourly_df
