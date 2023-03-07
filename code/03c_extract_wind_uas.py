#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:25:25 2023

@author: hmendo
"""
import numpy as np
import glob
import netCDF4 as nc
import os
import pandas as pd
import re

myos = 'Linux' #Windows

array_info_ser = pd.read_pickle('02_array_info.pkl')
models = np.load('00_model_names.npy')
sites = pd.read_pickle('00_site_df.pkl')
#model_data_dfs = pd.read_pickle('01b_model_data_dfs.pkl')
homedir = os.path.expanduser('~')

if myos == 'Windows':
    wind_dir = os.path.join('//snl', 'Collaborative', 'nsrd_climate_impacts','data', 'raw', 'wind')
    wind_files_all = glob.glob(wind_dir)
elif myos == 'Mac':
#NOTE: the below needs to be previously mounted in terminal with: mount_smbfs //snl/Collaborative/nsrd_climate_impacts/data/raw/wind/ ~/tempmount
    wind_dir = os.path.join(homedir,'tempmount2') 
    wind_files_all = sorted(glob.glob(os.path.join(wind_dir, 'uas*nc')))
elif myos == 'Linux':
    wind_dir = os.path.join('.','wind')
    wind_files_all = sorted(glob.glob(os.path.join(wind_dir, 'uas*nc')))

#models = [models[0]]
for m in range(len(models)):
    print('m = ',m)
    mymodel = models[m]
    df = array_info_ser[mymodel]
    wind_files = [x for x in wind_files_all if re.search(mymodel,x)]
    wind_files = sorted(wind_files)
    #sites = [sites.iloc[[0]]]
    for s in range(len(sites)):
        print('s = ',s)
        mysite = df.iloc[s,:]['site_name']
        print(mysite)
        myyc = df.loc[df.loc[:,'site_name']==mysite, 'array_dim1'].iloc[0] #[0] is needed because the command returns a series
        myxc = df.loc[df.loc[:,'site_name']==mysite, 'array_dim2'].iloc[0] #[0] is needed because the command returns a series
        
        wind_data = {}
        #time_data = {}
        #wind_files = [wind_files[0]]
        for i in range(len(wind_files)):
            print('i = ',i)
            print(mymodel, "site", s, mysite, i, sep=' ')
            ncfilepath = wind_files[i]
            print(ncfilepath+'\n')
            ncfile = nc.Dataset(ncfilepath)

            nc_wind = ncfile.variables['uas'][:]
            wind_data[i] = nc_wind[:, myyc, myxc] #see nc files attributes => float32 pr(time, yc, xc)
            #time_data[i] = ncfile.variables['time_bnds'][:] #see nc files attributes => float32 time_bnds(time, bnds)
            del(ncfile, nc_wind)
            
        wind_data = pd.Series(wind_data)
        wind_data.to_pickle(os.path.join('03_data', mysite+'_'+mymodel+'_uas.pkl'))
        #time_data = pd.Series(time_data)
        #time_data.to_pickle(os.path.join('03_data', mysite+'_'+mymodel+'_time.pkl'))
        del(mysite, myxc, myyc, wind_data)
        
    del(mymodel, df, wind_files)
        
    