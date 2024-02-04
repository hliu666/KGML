import numpy as np
import pandas as pd

site_vars = pd.read_csv('site_vars.csv')
site_pars = pd.read_csv('site_pars.csv')

out_list = []
for index_pars, row_pars in site_pars.iterrows():
    print(index_pars)
    for index_vars, row_vars in site_vars.iterrows():
        concat_rows = np.concatenate((row_vars.values, row_pars.values), axis=0)
        out_list.append(concat_rows)
    # out_arr = np.vstack(out_list)
    # np.save('input_data_test.npy', out_arr)
    # print("ok")

out_arr = np.vstack(out_list)
np.save('input_data.npy', out_arr)

interval = len(site_vars)
sub_files = 1
sub_lens = int(len(site_pars) * len(site_vars) / sub_files)

for i in range(0, sub_files):
    sid, eid = i * sub_lens, (i + 1) * sub_lens
    x = np.array(range(sid, eid, interval)).astype(int)
    y = np.repeat(interval, len(x))
    id_arr = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    np.savetxt('pars{0}_f.txt'.format(i), id_arr, '%-d', delimiter=',')  # X is an array
