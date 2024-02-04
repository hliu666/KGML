root = "C:/Users/liuha/Desktop/TBM_DA/TBM_DAv3/"

import sys

sys.path.append(root + 'sce_ua/tbm')
sys.path.append(root + 'forward')

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import spotpy
from pars import Par, Var_bicm, Var_rtmo, Var_carp
from model import Model
from dataloader import Dataloader
from predict import predict
from tbm.main import run_tbm

pft_dict = {"ENF": 1, "DBF": 4, "MF": 3, "OSH": 7, "GRA": 10, "WET": 11, "CRO": 12}

site_file_path = root + 'siteInfo.csv'
site_pd = pd.read_csv(site_file_path, sep=',')


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
    rtmo_frcngs_vars = ['lai', 'sza', 'vza', 'raa', 'pft']
    rtmo_params_vars = ['CI', 'lidf', "cab", "lma"]
    rtmo_obsrvs_var1 = ['fpar']
    rtmo_obsrvs_var2 = ['ref_red', 'ref_nir']

    rtmo_p = Par(rtmo_hidden_dim, rtmo_batch_size, rtmo_epochs, rtmo_learn_rate)
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
    bicm_frcngs_vars = ['lai', 'sw', 'ta', 'wds', 'sza', 'fpar', 'par', 'vpd', 'p', 'pft']
    bicm_params_vars = ['RUB', 'Rdsc', 'CB6F', 'gm', 'BallBerrySlope', 'BallBerry0']
    bicm_obsrvs_var1 = ['an']
    bicm_obsrvs_var2 = ['lst']

    bicm_p = Par(bicm_hidden_dim, bicm_batch_size, bicm_epochs, bicm_learn_rate)
    bicm_v = Var_bicm(bicm_frcngs_vars, bicm_params_vars, bicm_obsrvs_var1, bicm_obsrvs_var2)
    bicm_m = Model(bicm_v, bicm_p)

    bicm_m.load("bicm", ml_root)

    vs = [carp_v, rtmo_v, bicm_v]
    ms = [rtmo_m.model, bicm_m.model]

    return vs, ms


def load_par(flux_root, site, pft, pars):
    hourly_df = pd.read_csv(flux_root + f"flux/{site}.csv")

    hourly_df['datetime'] = pd.to_datetime(hourly_df[['year', 'month', 'day', 'hour']])
    hourly_df.set_index('datetime', inplace=True)

    daily_df = hourly_df.resample('D').agg({
        'year': 'first',
        'doy': 'mean',
        'ta': 'mean',
        'nee': 'mean',
        'lai': 'mean'
    })

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


def plot_funcs(site_ID, typ, df, sim_ml, sim_tbm, pars):
    if typ == "lai":
        df['year_doy'] = df['year'].astype(int).astype(str) + '-' + df['doy'].astype(int).astype(str)
        df['datetime'] = pd.to_datetime(df['year_doy'], format='%Y-%j')
    else:
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

    df[f'{typ}_sim_ml'] = sim_ml[0:len(df)]
    df[f'{typ}_sim_tbm'] = sim_tbm[0:len(df)]

    df_nonan = df[['datetime', 'doy', f'{typ}', f'{typ}_sim_ml', f'{typ}_sim_tbm']].copy().dropna()
    """
    if typ == "lai":
        df_nonan = df_nonan[df_nonan['lai'] > 1]
    elif typ == "ref_red" or typ == "ref_nir":
        sos = pars[17]
        eos = pars[19] + pars[20]/2.0
        df_nonan = df_nonan[(df_nonan['doy'] > sos) & (df_nonan['doy'] < eos)]
    """

    r2_sim_ml = round(stats.pearsonr(df_nonan[f'{typ}'].values, df_nonan[f'{typ}_sim_ml'].values)[0] ** 2, 2)
    r2_sim_tbm = round(stats.pearsonr(df_nonan[f'{typ}'].values, df_nonan[f'{typ}_sim_tbm'].values)[0] ** 2, 2)

    rmse_sim_ml = round(np.nanmean((df_nonan[f'{typ}'].values - df_nonan[f'{typ}_sim_ml'].values) ** 2) ** 0.5, 2)
    rmse_sim_tbm = round(np.nanmean((df_nonan[f'{typ}'].values - df_nonan[f'{typ}_sim_tbm'].values) ** 2) ** 0.5, 2)

    if typ == "ref_red" or typ == "ref_nir":
        # Plotting
        fig = plt.figure(figsize=(16, 9))
        ax = plt.subplot(1, 1, 1)
        plt.plot(df_nonan['datetime'], df_nonan[f'{typ}'], 'r+', color='red')
        plt.plot(df_nonan['datetime'], df_nonan[f'{typ}_sim_ml'], marker='o', markersize=6,
                 markerfacecolor='none', markeredgecolor='black', linestyle='none',
                 label=f'{typ}_sim_ml: TBM R2:{r2_sim_tbm} RMSE:{rmse_sim_tbm}')
        plt.plot(df_nonan['datetime'], df_nonan[f'{typ}_sim_tbm'], marker='o', markersize=6,
                 markerfacecolor='none', markeredgecolor='blue', linestyle='none',
                 label=f'{typ}_sim_tbm: TBM R2:{r2_sim_tbm} RMSE:{rmse_sim_tbm}')

        plt.title(f'Time Series of {typ}')
        plt.xlabel('Date')
        plt.ylabel(f'{typ}')
        plt.legend(fontsize=18, loc='upper right')
        plt.tight_layout()
        plt.show()
        fig.savefig(f'SCEUA_{site_ID}_{typ}.png', dpi=300)

    else:
        # Plotting
        fig = plt.figure(figsize=(16, 9))
        ax = plt.subplot(1, 1, 1)
        plt.plot(df_nonan['datetime'], df_nonan[f'{typ}'], 'r+', color='red')
        plt.plot(df['datetime'], df[f'{typ}_sim_ml'], color='black',
                 label=f'{typ}_sim_ml: R2:{r2_sim_ml} RMSE:{rmse_sim_ml}')
        plt.plot(df['datetime'], df[f'{typ}_sim_tbm'], color='blue',
                 label=f'{typ}_sim_tbm: TBM R2:{r2_sim_tbm} RMSE:{rmse_sim_tbm}')

        plt.title(f'Time Series of {typ}')
        plt.xlabel('Date')
        plt.ylabel(f'{typ}')
        plt.legend(fontsize=18, loc='upper right')
        plt.tight_layout()
        plt.show()
        fig.savefig(f'SCEUA_{site_ID}_{typ}.png', dpi=300)


for index, row in site_pd.iterrows():
    if index != 1:
        continue
    site_ID = row['Site ID']
    latitude = row['Latitude']
    longitude = row['Longitude']
    site_LC = row['LC']

    for typ in ["_typ0"]: # ["_typ0", "_typ1", "_typ2", "_typ3"]:
        # df = pd.read_csv(root + f"hpc/{site_ID}_SCEUA{typ}.csv")
        df = pd.read_csv(f"{site_ID}_SCEUA{typ}.csv")

        pars_row = df.loc[df['like1'].idxmin()]

        vs, ms = initial_ml(root + 'forward/')

        pft = pft_dict[site_LC]
        daily_df, hourly_df = load_par(root, site_ID, pft, pars_row)
        daily_df_ml, hourly_df_ml = daily_df.copy(), hourly_df.copy()

        batch_size = 1
        dL = Dataloader(root, daily_df_ml, hourly_df_ml, vs, batch_size)
        lai_sim_ml, nee_sim_ml, lst_sim_ml, ref_red_sim_ml, ref_nir_sim_ml = predict(daily_df_ml, hourly_df_ml, ms, vs, dL)

        # step 2. tbm model simulations
        lai_sim_tbm, nee_sim_tbm, fpar_sim_tbm, ref_red_sim_tbm, ref_nir_sim_tbm, lst_sim_tbm = run_tbm(root, site_ID, latitude, longitude, pars_row)

        plot_funcs(site_ID, 'lai', daily_df, lai_sim_ml, lai_sim_tbm, pars_row)
        plot_funcs(site_ID, 'nee', hourly_df, nee_sim_ml, nee_sim_tbm, pars_row)
        plot_funcs(site_ID, 'lst', hourly_df, lst_sim_ml, lst_sim_tbm, pars_row)
        plot_funcs(site_ID, 'ref_red', hourly_df, ref_red_sim_ml, ref_red_sim_tbm, pars_row)
        plot_funcs(site_ID, 'ref_nir', hourly_df, ref_nir_sim_ml, ref_nir_sim_tbm, pars_row)
