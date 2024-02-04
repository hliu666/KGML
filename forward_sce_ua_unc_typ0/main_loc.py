# root = "/home/hliu666/TBM_DAv3/"
root = "C:/Users/liuha/Desktop/TBM_DA/TBM_DAv3/"
model_root = root + "forward/"

import sys
sys.path.append(model_root)

from pars import Par, Var_bicm, Var_rtmo, Var_carp
from model import Model
from spot_class import spot_setup
import spotpy
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

pft_dict = {"ENF": 1, "DBF": 4, "MF": 3, "OSH": 7, "GRA": 10, "WET": 11, "CRO": 12}

site_id = 1

# from joblib import Parallel, delayed


def calc_sceua(i, df, vs, ms):
    row = df.iloc[i]
    site_ID = row['Site ID']

    hourly_obs = pd.read_csv(root + f"flux/{site_ID}.csv")
    daily_obs = pd.read_csv(root + f"flux_d/{site_ID}.csv")
    hourly_obs['pft'] = pft_dict[row['LC']]
    daily_obs['pft'] = pft_dict[row['LC']]

    spot = spot_setup(root, vs, ms, hourly_obs, daily_obs)

    sampler = spotpy.algorithms.sceua(spot, dbname=f'{site_ID}_SCEUA_typ0', dbformat='csv', save_sim=False) # , parallel='mpi', save_sim=False)
    rep = 100000

    sampler.sample(rep) # , ngs=40)


def main():
    """
    1. carbon pool model

    Use DALEC based carbon pool model instead of machine learning model
    """
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
    rtmo_hidden_dim = 80
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

    rtmo_m.load("rtmo", model_root)

    """
    3. bicm model
    """
    bicm_hidden_dim = 80
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

    bicm_m.load("bicm", model_root)

    vs = [carp_v, rtmo_v, bicm_v]
    ms = [rtmo_m.model, bicm_m.model]

    site_file_path = root + 'siteInfo.csv'
    site_pd = pd.read_csv(site_file_path, sep=',')

    # calc_sceua(int(sys.argv[1]), site_pd, vs, ms)
    calc_sceua(site_id, site_pd, vs, ms)

    # Parallel(n_jobs=-1)(delayed(calc_sceua)(i, site_pd, vs, ms) for i in range(len(site_pd)))


if __name__ == '__main__':
    main()
