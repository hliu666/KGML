from pars import Par, Var
from loader import Dataloader
from model import Model
from plot import plot_bicm
import pandas as pd
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

"""
define model structure
"""
mdl = 'bicm'

hidden_dim = 60
batch_size = 256 * 24
epochs = 1000
x_vars = ['LAI', 'SW', 'TA', 'wds', 'fPAR', 'PAR_up', 'VPD']
x_pars = ['RUB', 'Rdsc', 'CB6F', 'gm', 'BallBerrySlope', 'BallBerry0']
y_vars = ['An']
df = pd.read_csv("../input/debug.csv", na_values="nan")

"""
initialize model
"""
p = Par(hidden_dim, batch_size, epochs)
v = Var(x_vars, x_pars, y_vars)
m = Model(v, p)

"""
load data
"""
dL = Dataloader(mdl, df, v, p)

"""
model training
"""
# m.train(dL, p)
# torch.save(m.model.state_dict(), 'out/bicm_model.pth')

m.load(mdl)

plot_bicm(m.model, dL.test_loader, dL.label_scaler)

