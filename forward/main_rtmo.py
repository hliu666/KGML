from pars import Par, Var_rtmo
from loader import Loader
from model import Model
from plot import plot_rtmo
import numpy as np
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

"""
define model structure
"""
mdl = 'rtmo'

hidden_dim = 120
batch_size = 3700
epochs = 1000
learn_rate = 0.0001  # 0.001
lr_decay = 0.98

frcngs_vars = ['lai', 'sza', 'vza', 'raa', 'pft']
params_vars = ['CI', 'lidf', "cab", "lma"]
obsrvs_var1 = ['fpar']
obsrvs_var2 = ['ref_red', 'ref_nir']

"""
initialize model
"""
p = Par(hidden_dim, batch_size, epochs, learn_rate, lr_decay)
v = Var_rtmo(frcngs_vars, params_vars, obsrvs_var1, obsrvs_var2)

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
torch.save(m.model.state_dict(), 'out/rtmo_model.pth')

# m.load(mdl)

plot_rtmo(m.model, dL.test_loader, dL.obsrvs_scaler)
