import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import joblib
import numpy as np


class Dataloader:
    def __init__(self, root, daily_df, hourly_df, vs, batch_size):
        self.carp_obsrvs_scaler = joblib.load(root + 'scaler/carp_obsrvs_scaler.pkl')

        self.rtmo_frcngs_scaler = joblib.load(root + 'scaler/rtmo_frcngs_scaler.pkl')
        self.rtmo_params_scaler = joblib.load(root + 'scaler/rtmo_params_scaler.pkl')
        self.rtmo_obsrvs_scaler = joblib.load(root + 'scaler/rtmo_obsrvs_scaler.pkl')

        self.bicm_frcngs_scaler = joblib.load(root + 'scaler/bicm_frcngs_scaler.pkl')
        self.bicm_params_scaler = joblib.load(root + 'scaler/bicm_params_scaler.pkl')
        self.bicm_obsrvs_scaler = joblib.load(root + 'scaler/bicm_obsrvs_scaler.pkl')

        carp_v, rtmo_v, bicm_v = vs

        carp_input, carp_label = self.create_carp_dataset(daily_df, carp_v)
        rtmo_input, rtmo_label = self.create_rtmo_dataset(hourly_df, rtmo_v)
        bicm_input, bicm_label = self.create_bicm_dataset(hourly_df, bicm_v)

        dataset = TensorDataset(torch.from_numpy(carp_input), torch.from_numpy(carp_label),
                                torch.from_numpy(rtmo_input), torch.from_numpy(rtmo_label),
                                torch.from_numpy(bicm_input), torch.from_numpy(bicm_label))

        self.predict_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True)

    def create_carp_dataset(self, df, v):
        carp_frcngs = df[v.x_vars].values
        carp_initls = df[v.c_vars].values
        carp_params = df[v.x_pars].values
        carp_obsrvs = df[v.y_vars].values

        carp_input = np.hstack((carp_frcngs, carp_initls, carp_params))
        carp_label = carp_obsrvs

        return carp_input, carp_label

    def create_rtmo_dataset(self, df, v):
        rtmo_frcngs = self.rtmo_frcngs_scaler.transform(df[v.x_vars].values)
        rtmo_params = self.rtmo_params_scaler.transform(df[v.x_pars].values)
        rtmo_obsrvs = self.rtmo_obsrvs_scaler.transform(df[v.y1_vars + v.y2_vars].values)

        input = np.hstack((rtmo_frcngs, rtmo_params))
        label = rtmo_obsrvs

        num_hours = df.shape[0]
        num_col_input = input.shape[1]
        num_col_label = label.shape[1]

        num_days = num_hours // 24
        frcngs_re = input.reshape((num_days, 24, num_col_input))
        params_re = label.reshape((num_days, 24, num_col_label))

        return frcngs_re, params_re

    def create_bicm_dataset(self, df, v):
        bicm_frcngs = self.bicm_frcngs_scaler.transform(df[v.x_vars].values)
        bicm_params = self.bicm_params_scaler.transform(df[v.x_pars].values)
        bicm_obsrvs = self.bicm_obsrvs_scaler.transform(df[v.y1_vars + v.y2_vars].values)

        input = np.hstack((bicm_frcngs, bicm_params))
        label = bicm_obsrvs

        num_hours = df.shape[0]
        num_col_input = input.shape[1]
        num_col_label = label.shape[1]

        num_days = num_hours // 24
        frcngs_re = input.reshape((num_days, 24, num_col_input))
        params_re = label.reshape((num_days, 24, num_col_label))

        return frcngs_re, params_re