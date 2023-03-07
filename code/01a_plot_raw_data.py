# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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


crcm1 = precip_files_all[0:13]

def plot_data(mymodel, myfile):
    ncfilepath = os.path.join(precip_dir, myfile)
    ncfile = nc.Dataset(ncfilepath)
    
    nc_lat = ncfile.variables['lat'][:]
    nc_lon = ncfile.variables['lon'][:]
    
    plot_df = pd.DataFrame({'lat': nc_lat.flatten(), 'lon': nc_lon.flatten()})
    