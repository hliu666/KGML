# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:07:40 2023

@author: hliu
"""

import pandas as pd

# 只读取第一行来获取所有的列名
col_names = pd.read_csv('CA-Oas_SCEUA.csv', nrows=0).columns.tolist()

# 选择前50列的列名
columns_to_read = col_names[:50]

# 使用前50列的列名来读取数据
df = pd.read_csv('CA-Oas_SCEUA.csv', usecols=columns_to_read)

df.to_csv("CA-Oas_pars.csv")
