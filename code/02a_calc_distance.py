#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:23:27 2023

@author: hmendo
"""

import numpy as np
import glob
import netCDF4 as nc
import os
import pandas as pd
from geopy.distance import great_circle

myos = 'Mac' #Windows

models = np.load('00_model_names.npy')
sites = pd.read_pickle('00_site_df.pkl')
model_data_dfs = pd.read_pickle('01b_model_data_dfs.pkl')
homedir = os.path.expanduser('~')

if myos == 'Windows':
    precip_dir = os.path.join('//snl', 'Collaborative', 'nsrd_climate_impacts','data', 'raw', 'precipitation')
    precip_files_all = glob.glob(precip_dir)
elif myos == 'Mac':
#NOTE: the below needs to be previously mounted in terminal with: mount_smbfs //snl/Collaborative/nsrd_climate_impacts/data/raw/precipitation/ ~/tempmount
    precip_dir = os.path.join(homedir,'tempmount') 
    precip_files_all = glob.glob(os.path.join(precip_dir, '*nc'))

def get_closest_point(site, mymodel):
    myfile = 'pr_'+mymodel+"_1968010103.nc"
    ncfilepath = os.path.join(precip_dir, myfile)
    ncfile = nc.Dataset(ncfilepath)
    
    nc_lat = ncfile.variables['lat'][:]
    nc_lon = ncfile.variables['lon'][:]
    if mymodel[:4] != 'HRM3':
        nc_lon = nc_lon - 360
        
    df = model_data_dfs[mymodel]
    df_sp = df
    df_sp['coords'] = list(zip(df['lat'], df['lon']))
    
    site_loc = (sites.iloc[site,:]["lat"], sites.iloc[site,:]["lon"])

    my_dist = df_sp['coords'].apply(lambda x: great_circle(x, site_loc).km) #distance in km, len(my_dist) = len(df_sp) = len(df)
    dist_min_idx = my_dist.idxmin()
    close_loc = df.iloc[dist_min_idx,:][['lat', 'lon']]
    
    arr_ind = np.where((nc_lon==close_loc[1]) & (nc_lat==close_loc[0]))
    
    out = {'site_lon': site_loc[1], 
             'site_lat': site_loc[0],
             'data_lon': close_loc[1],
             'data_lat': close_loc[0],
             'dist': my_dist[dist_min_idx],
             'array_dim1': arr_ind[0][0],
             'array_dim2': arr_ind[1][0],
#             'array_ind': arr_ind,
#             'nc_lon': nc_lon,
#             'nc_lat': nc_lat
             }
    return out
    
test_gcp = get_closest_point(site = 0, mymodel = models[4])

array_info = {}
for j in range(len(models)):
    site_info = pd.DataFrame(columns = ['site_lon', 
                                        'site_lat', 
                                        'data_lon', 
                                        'data_lat',
                                        'dist',
                                        'array_dim1',
                                        'array_dim2'
                                        ])
    for i in range(len(sites)):
        out = get_closest_point(i, models[j])
        site_info = pd.concat([site_info, pd.DataFrame(out, index=[0])], ignore_index=True)
        
    site_info.loc[:,'model'] = models[j]
    site_info.loc[:,'site_name'] = sites.loc[:,'site_name'].values
    
    array_info[models[j]] = site_info
    
array_info_ser = pd.Series(array_info)

array_info_ser.to_pickle('02_array_info.pkl')
    
    


