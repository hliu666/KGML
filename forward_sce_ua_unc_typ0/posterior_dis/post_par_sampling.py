import pandas as pd
from spot_class import spot_setup
import spotpy

from best_like_func import initial_ml


def calc_sceua(root, site_ID, site_LC, pars, likes, typ, rep=1000):
    vs, ms = initial_ml(root + 'forward/')

    pft_dict = {"ENF": 1, "DBF": 4, "MF": 3, "OSH": 7, "GRA": 10, "WET": 11, "CRO": 12}

    hourly_obs = pd.read_csv(root + f"flux/{site_ID}.csv")
    daily_obs = pd.read_csv(root + f"flux_d/{site_ID}.csv")

    hourly_obs['pft'] = pft_dict[site_LC]
    daily_obs['pft'] = pft_dict[site_LC]

    spot = spot_setup(root, vs, ms, hourly_obs, daily_obs, pars, likes)

    sampler = spotpy.algorithms.sceua(spot, dbname=f'{site_ID}_SCEUA{typ}_post', dbformat='csv', save_sim=False) # , parallel='mpi', save_sim=False)

    sampler.sample(rep) # , ngs=40)
