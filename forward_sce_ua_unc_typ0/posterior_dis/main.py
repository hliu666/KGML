import pandas as pd

from best_like_func import calc_sims
from post_par_sampling import calc_sceua

import warnings
warnings.filterwarnings(action='ignore')

root = "C:/Users/liuha/Desktop/TBM_DA/TBM_DAv3/"


def main():
    site_file_path = root + 'siteInfo.csv'
    site_pd = pd.read_csv(site_file_path, sep=',')

    for index, row in site_pd.iterrows():
        if index != 1:
            continue
        site_ID = row['Site ID']
        latitude = row['Latitude']
        longitude = row['Longitude']
        site_LC = row['LC']

        for typ in ["_typ0"]:  # ["_typ0", "_typ1", "_typ2", "_typ3"]:
            df = pd.read_csv(root + f"hpc/{site_ID}_SCEUA{typ}.csv")
            best_pars = df.loc[df['like1'].idxmin()] # get best like array

            best_likes = calc_sims(site_ID, site_LC, root, best_pars)

            # calc_sceua(int(sys.argv[1]), site_pd, vs, ms)
            calc_sceua(root, site_ID, site_LC, best_pars, best_likes, typ, rep=2000)


if __name__ == '__main__':
    main()
