import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import joblib


class Loader:
    def __init__(self, mdl, frcngs_arr, params_arr, obsrvs_arr, p):
        self.frcngs_scaler = joblib.load(f'../scaler/{mdl}_frcngs_scaler.pkl')
        self.params_scaler = joblib.load(f'../scaler/{mdl}_params_scaler.pkl')
        self.obsrvs_scaler = joblib.load(f'../scaler/{mdl}_obsrvs_scaler.pkl')

        self.train_loader, self.valid_loader, self.test_loader = self.dataLoader(frcngs_arr, params_arr, obsrvs_arr, p)

    def dataLoader(self, frcngs, params, obsrvs, p):
        dataset = TensorDataset(torch.from_numpy(np.hstack((frcngs, params))), torch.from_numpy(obsrvs))

        # Split data into train/test portions and combining all data from different files into a single array
        total_len = len(frcngs)
        train_len = int(total_len * (1 - 2 * p.test_portion))
        valid_len = int(total_len * p.test_portion)
        test_len = total_len - train_len - valid_len

        # Assuming data is your Dataset object
        lengths = [train_len, valid_len, test_len]
        generator = torch.Generator().manual_seed(0)
        train, valid, test = random_split(dataset, lengths, generator)

        train_loader = DataLoader(train, shuffle=True, batch_size=p.batch_size, drop_last=True, generator=generator)
        valid_loader = DataLoader(valid, shuffle=True, batch_size=p.batch_size, drop_last=True, generator=generator)
        test_loader = DataLoader(test, shuffle=False, batch_size=p.batch_size, drop_last=True, generator=generator)

        return train_loader, valid_loader, test_loader

