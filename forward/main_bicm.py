from pars import Par, Var_bicm
from loader import Loader
from model import Model
from plot import plot_bicm
import numpy as np
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)


"""
define model structure
"""
mdl = 'bicm'

hidden_dim = 128
batch_size = 1015
epochs = 1000
learn_rate = 0.001
lr_decay = 0.94

frcngs_vars = ['lai', 'sw', 'ta', 'wds', 'sza', 'fpar', 'par', 'vpd', 'pft']
params_vars = ['RUB', 'Rdsc', 'CB6F', 'gm', 'BallBerrySlope', 'BallBerry0']
obsrvs_var1 = ['an']
obsrvs_var2 = ['lst']


"""
initialize model
"""
p = Par(hidden_dim, batch_size, epochs, learn_rate, lr_decay)
v = Var_bicm(frcngs_vars, params_vars, obsrvs_var1, obsrvs_var2)

frcngs_arr = np.load(f'../forward_input/{mdl}_frcngs_arr.npy')
params_arr = np.load(f'../forward_input/{mdl}_params_arr.npy')
obsrvs_arr = np.load(f'../forward_input/{mdl}_obsrvs_arr.npy')

"""
load data
"""
dL = Loader(mdl, frcngs_arr, params_arr, obsrvs_arr, p)

"""
model training
"""
m = Model(v, p)

m.train(dL, p)
torch.save(m.model.state_dict(), 'out/bicm_model.pth')

# m.load(mdl)

plot_bicm(m.model, dL.test_loader, dL.obsrvs_scaler)
