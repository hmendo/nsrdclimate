#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:33:50 2023

@author: hmendo
"""

import numpy as np
import glob
import netCDF4 as nc
import os
import pandas as pd

myos = 'Mac' #Windows

models = np.load('00_model_names.npy')
homedir = os.path.expanduser('~')

if myos == 'Windows':
    precip_dir = os.path.join('//snl', 'Collaborative', 'nsrd_climate_impacts','data', 'raw', 'precipitation')
    precip_files_all = glob.glob(precip_dir)
elif myos == 'Mac':
#NOTE: the below needs to be previously mounted in terminal with: mount_smbfs //snl/Collaborative/nsrd_climate_impacts/data/raw/precipitation/ ~/tempmount
    precip_dir = os.path.join(homedir,'tempmount') 
    precip_files_all = glob.glob(os.path.join(precip_dir, '*nc'))
    
    
def parse_data(mymodel):
    myfile = 'pr_'+mymodel+"_1968010103.nc"
    ncfilepath = os.path.join(precip_dir, myfile)
    ncfile = nc.Dataset(ncfilepath)
    
    nc_lat = ncfile.variables['lat'][:]
    nc_lon = ncfile.variables['lon'][:]
    if mymodel[:4] != 'HRM3':
        nc_lon = nc_lon - 360
    
    data_df = pd.DataFrame({'lat': nc_lat.flatten(), 'lon': nc_lon.flatten()})
    return data_df


data_df = parse_data("HRM3_gfdl")


model_dfs_list = list(map(parse_data, models))
model_dfs = {i:j for i,j in zip(models,model_dfs_list)}
model_dfs = pd.Series(model_dfs)
model_dfs.to_pickle('01b_model_data_dfs.pkl')

