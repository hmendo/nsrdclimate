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
save_path = '04a_ts_wind'

    
df_files = glob.glob(os.path.join(save_path, '*'))

wind_daily = {}
plot_daily = {}


for mod in range(len(df_files)):
    mymodel = df_files[mod][32:-4]
    wind_full = pd.read_pickle(df_files[mod])
    for s in range(len(wind_full)):
        print(s, mysites[s])
        wind_df = wind_full.loc[mysites[s]]
        wind_df['day'] = wind_df['time'].dt.date
        
        wind_daily_df = wind_df.groupby(['day']).sum(numeric_only=True)
        wind_daily_df.reset_index(inplace=True)
        wind_daily_df.rename(columns={'day':'date', 'wind':'avg_wind'}, inplace=True)
        
        wind_daily[mysites[s]] = wind_daily_df
        
    wind_daily = pd.Series(wind_daily)
    wind_daily.to_pickle(os.path.join(save_path,'daily_wind_df_'+mymodel+'.pkl'))
    