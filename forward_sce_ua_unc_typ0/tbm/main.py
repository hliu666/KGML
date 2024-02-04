
import pandas as pd
import numpy as np
import scipy.io
import sys

from data import TBM_Data
from parameter import TBM_Pars
from model import TBM_Model


# Reference: https://www.kaggle.com/hijest/gaps-features-tf-lstm-resnet-like-ff
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def main():
    site_file_path = 'siteInfo.csv'
    site_pd = pd.read_csv(site_file_path, sep=',')

    typ_id, site_id, i, js = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    row = site_pd.loc[site_id]

    site_ID = row['Site ID']

    flux_arr = pd.read_csv(f"{site_ID}.csv", na_values="nan")  # "data/driving/{0}.csv".format(site)
    rsr_red = np.genfromtxt("rsr_red.txt")  # "src/model/support/rsr_red.txt"
    rsr_nir = np.genfromtxt("rsr_nir.txt")  # "src/model/support/rsr_nir.txt"
    rsr_sw1 = np.genfromtxt("rsr_swir1.txt")  # "src/model/support/rsr_swir1.txt"
    rsr_sw2 = np.genfromtxt("rsr_swir2.txt")  # "src/model/support/rsr_swir2.txt"
    prospectpro = np.loadtxt("dataSpec_PDB.txt")  # "src/model/support/dataSpec_PDB.txt"
    soil = np.genfromtxt("soil_reflectance.txt")  # "src/model/support/soil_reflectance.txt"
    TOCirr = np.loadtxt("atmo.txt", skiprows=1)  # "src/model/support/atmo.txt"
    phiI = scipy.io.loadmat("phiI.mat")['phiI']  # 'src/model/support/phiI.mat'
    phiII = scipy.io.loadmat("phiII.mat")['phiII']  # 'src/model/support/phiII.mat'

    driving_data = [flux_arr, rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr, phiI, phiII]

    lat, lon = row['Latitude'], row['Longitude']

    output_dim1, output_dim2 = 11, 11

    for j in range(0, js):
        k = i + j

        pars = pd.read_csv(f"{site_ID}_SCEUA_typ{typ_id}_pars.csv").iloc[-(1 + k)]

        p = TBM_Pars(pars)
        d = TBM_Data(p, lat, lon, driving_data)
        m = TBM_Model(d, p)

        model_output_daily, model_output_hourly, mod_list_spectral = m.mod_list(output_dim1, output_dim2)

        model_output_daily = model_output_daily[1:].reshape(-1, output_dim1)
        model_output_hourly = model_output_hourly[1:].reshape(-1, output_dim2)

        df_daily = pd.DataFrame(model_output_daily,
                                columns=['clab', 'cf', 'cr', 'cw', 'cl', 'cs', 'Year', 'Doy', 'hour', 'gpp', 'lai'])
        df_hourly = pd.DataFrame(model_output_hourly,
                                 columns=['Year', 'Doy', 'hour', 'nee', 'fPAR', 'ref_red', 'ref_nir', 'LST',
                                          'tts', 'tto', 'psi'])

        df_daily = reduce_mem_usage(df_daily)
        df_hourly = reduce_mem_usage(df_hourly)

        df_daily.to_csv('{0}_{1}_daily_typ{2}.csv'.format(k, site_ID, typ_id), index=False)
        df_hourly.to_csv('{0}_{1}_hourly_typ{2}.csv'.format(k, site_ID, typ_id), index=False)


if __name__ == '__main__':
    main()