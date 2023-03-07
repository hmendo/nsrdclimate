#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:31:30 2023

@author: hmendo
"""

import numpy as np
import glob
#import netCDF4 as nc
import os
import pandas as pd
import datetime

models = np.load('00_model_names.npy')
sites = pd.read_pickle('00_site_df.pkl')
mysites = sites.loc[:, 'site_name']
save_path = '04a_ts_precip'

# conversion formula:
# [kg/ m^2 s] * [(3*3600) s of time per datapoint] * [1/1000 m^3 / kg density of water] * [1000 mm / m] = [mm of rain during the 3hr period]
def convert_to_mm_in_3hour(x):
    x = x * (3*3600) * (1/1000) * 1000
    return x
    
df_files = glob.glob(os.path.join(save_path, '*'))

precip_daily = {}
plot_daily = {}


for mod in range(len(df_files)):
    mymodel = df_files[mod][30:-4]
    precip_full = pd.read_pickle(df_files[mod])
    for s in range(len(precip_full)):
        print(s, mysites[s])
        precip_df = precip_full.loc[mysites[s]]
        precip_df['mm'] = precip_df['precip'].apply(convert_to_mm_in_3hour)
        precip_df['day'] = precip_df['time'].dt.date
        
        precip_daily_df = precip_df.groupby(['day']).sum(numeric_only=True)
        precip_daily_df.reset_index(inplace=True)
        precip_daily_df.rename(columns={'day':'date', 'precip':'mm_precip'}, inplace=True)
        
        precip_daily[mysites[s]] = precip_daily_df
        
    precip_daily = pd.Series(precip_daily)
    precip_daily.to_pickle(os.path.join(save_path,'daily_precip_df_'+mymodel+'.pkl'))
    