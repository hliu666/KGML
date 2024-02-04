import pandas as pd
import numpy as np

fields = ["parclab", "parcf", "parcr", "parcw", "parcl", "parcs", "parclspan", \
          "parlma", "parf_auto", "parf_fol", "parf_lab", "parTheta", "partheta_min", \
          "partheta_woo", "partheta_roo", "partheta_lit", "partheta_som", "pard_onset", \
          "parcronset", "pard_fall", "parcrfall", "parCI", "parlidf", "parcab", \
          "parRUB", "parRdsc", "parCB6F", "pargm", "parBallBerrySlope", "parBallBerry0"]

site_df = pd.read_csv(f"siteInfo.csv")
type_value = 0
for site in ["CA-Oas"]:
    df = pd.read_csv(f"../{site}_SCEUA_typ{type_value}_post.csv")
    sub_df = df[(df['like1'] < 0.9) & (df['like1'] != float('-inf'))]
    if len(sub_df) > 1000:
        sub_df = sub_df.iloc[-1000:]
    else:
        sub_df = sub_df
    sub_df = sub_df[fields]
    sub_df.to_csv(f'{site}_SCEUA_typ{type_value}_pars.csv', index=False)

    interval = 1
    sub_files = 1
    sub_lens = int(len(sub_df) / sub_files)

    row_value = site_df.index[site_df['Site ID'] == site].tolist()[0]
    row = np.full((sub_lens, 1), row_value)  # Create an array with the same value repeated
    typ_row = np.full((sub_lens, 1), type_value)  # Create an array with the same value repeated

    for i in range(0, sub_files):
        sid, eid = i * sub_lens, (i + 1) * sub_lens
        x = np.array(range(sid, eid, interval)).astype(int)
        y = np.repeat(interval, len(x))
        id_arr = np.hstack((typ_row.reshape(-1, 1), row.reshape(-1, 1), x.reshape(-1, 1), y.reshape(-1, 1)))
        np.savetxt(f'pars{i}_{site}_typ{type_value}_post.txt', id_arr, '%-d', delimiter=',')  # X is an array
