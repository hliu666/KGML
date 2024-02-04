import os
import time
import cProfile
import pandas as pd
import numpy as np
import scipy.io

from tbm.data import TBM_Data
from tbm.parameter import TBM_Pars
from tbm.model import TBM_Model


def main(pars, lat, lon, start_yr, end_yr, output_dim1, output_dim2, driving_data):
    print("-----------start-------------")

    p = TBM_Pars(pars)
    d = TBM_Data(p, lat, lon, start_yr, end_yr, driving_data)
    m = TBM_Model(d, p)

    model_output_daily, model_output_hourly, mod_list_spectral = m.mod_list(output_dim1, output_dim2)

    model_output_hourly = model_output_hourly.reshape(-1, model_output_hourly.shape[-1])

    lai_sim = model_output_daily[:, -1]

    nee_sim = model_output_hourly[:, 0]
    fpar_sim = model_output_hourly[:, 1]
    ref_red_sim = model_output_hourly[:, 2]
    ref_nir_sim = model_output_hourly[:, 3]
    lst_sim = model_output_hourly[:, 4]

    return lai_sim, nee_sim, fpar_sim, ref_red_sim, ref_nir_sim, lst_sim


def run_tbm(root, site, lat, lon, pars):

    os.chdir(root + 'forward_sce_ua_typ0/tbm')

    flux_arr = pd.read_csv(root + "flux/{0}.csv".format(site), na_values="nan")
    rsr_red = np.genfromtxt("support/rsr_red.txt")
    rsr_nir = np.genfromtxt("support/rsr_nir.txt")
    rsr_sw1 = np.genfromtxt("support/rsr_swir1.txt")
    rsr_sw2 = np.genfromtxt("support/rsr_swir2.txt")
    prospectpro = np.loadtxt("support/dataSpec_PDB.txt")
    soil = np.genfromtxt("support/soil_reflectance.txt")
    TOCirr = np.loadtxt("support/atmo.txt", skiprows=1)
    phiI = scipy.io.loadmat('support/phiI.mat')['phiI']
    phiII = scipy.io.loadmat('support/phiII.mat')['phiII']
    driving_data = [flux_arr, rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr, phiI, phiII]

    start_yr, end_yr = flux_arr['year'].unique()[0], flux_arr['year'].unique()[-1]
    output_dim1, output_dim2 = 8, 5

    lai_sim, nee_sim, fpar_sim, ref_red_sim, ref_nir_sim, lst_sim = main(pars, lat, lon, start_yr, end_yr, output_dim1, output_dim2, driving_data)

    return lai_sim, nee_sim, fpar_sim, ref_red_sim, ref_nir_sim, lst_sim
