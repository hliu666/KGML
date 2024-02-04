# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 21:49:29 2022

@author: hliu
"""

# this will fail if SALib isn't properly installed
from SALib.sample import saltelli
import pandas as pd

fields = ["vza", "raa", "CI", "lidf", "cab", "lma",
          "RUB", "Rdsc", "CB6F", "gm", "BallBerrySlope", "BallBerry0"]

# Specify the model inputs and their bounds. The default probability
# distribution is a uniform distribution between lower and upper bounds.
# %% Leaf traits parameter sensitivity analysis
# ref:
# https://gmd.copernicus.org/articles/15/1789/2022/gmd-15-1789-2022-supplement.pdf
# https://gmd.copernicus.org/articles/15/1789/2022/
problem = {
    "num_vars": len(fields),
    "names": fields,
    "bounds": [[0, 60], [0, 180], [0.6, 1.0], [0, 60], [10, 80], [10, 120], \
               [50, 150], [0.0001, 0.01], [50, 150], [0.01, 5], [0, 15], [0, 1]]
}

N = 128
# generate the input sample
sample = saltelli.sample(problem, N)
df = pd.DataFrame(sample, columns=fields)

df = df[fields]
df.to_csv('site_pars.csv', index=False)

