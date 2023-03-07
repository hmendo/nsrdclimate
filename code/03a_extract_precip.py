import numpy as np
import glob
import netCDF4 as nc
import os
import pandas as pd
import re

#does geopandas open netcdf4 files??
#can't write out a layerd raster as a single TIF
#arcpy

myos = 'Linux' #Windows

array_info_ser = pd.read_pickle('02_array_info.pkl')
models = np.load('00_model_names.npy')
sites = pd.read_pickle('00_site_df.pkl')
#model_data_dfs = pd.read_pickle('01b_model_data_dfs.pkl')
homedir = os.path.expanduser('~')

if myos == 'Windows':
    precip_dir = os.path.join('\\snl', 'Collaborative', 'nsrd_climate_impacts','data', 'raw', 'precipitation')
    precip_files_all = sorted(glob.glob(os.path.join(precip_dir, '*nc')))
elif myos == 'Mac':
#NOTE: the below needs to be previously mounted in terminal with: mount_smbfs //snl/Collaborative/nsrd_climate_impacts/data/raw/precipitation/ ~/tempmount
    precip_dir = os.path.join(homedir,'tempmount') 
    precip_files_all = sorted(glob.glob(os.path.join(precip_dir, '*nc')))
elif myos == 'Linux':
    precip_dir = os.path.join('.','precipitation')
    precip_files_all = sorted(glob.glob(os.path.join(precip_dir, '*nc')))

#models = [models[0]]
for m in range(len(models)):
    print('m = ',m)
    mymodel = models[m]
    df = array_info_ser[mymodel]
    precip_files = [x for x in precip_files_all if re.search(mymodel,x)]
    precip_files = sorted(precip_files)
    
    for s in range(len(sites)):
        print('s = ',s)
        mysite = df.iloc[s,:]['site_name']
        print(mysite)
        myyc = df.loc[df.loc[:,'site_name']==mysite, 'array_dim1'].iloc[0] #[0] is needed because the command returns a series
        myxc = df.loc[df.loc[:,'site_name']==mysite, 'array_dim2'].iloc[0] #[0] is needed because the command returns a series
        
        precip_data = {}
        time_data = {}
        
        for i in range(len(precip_files)):
            print('i = ',i)
            print(mymodel, "site", s, mysite, i, sep=' ')
            ncfilepath = os.path.join(precip_files[i])
            print(ncfilepath+'\n')
            ncfile = nc.Dataset(ncfilepath)

            nc_precip = ncfile.variables['pr'][:]
            precip_data[i] = nc_precip[:, myyc, myxc] #see nc files attributes => float32 pr(time, yc, xc)
            time_data[i] = ncfile.variables['time_bnds'][:] #see nc files attributes => float32 time_bnds(time, bnds)
            #del(ncfile, nc_precip)
            
        precip_data = pd.Series(precip_data)
        precip_data.to_pickle(os.path.join('03_data', mysite+'_'+mymodel+'_precip.pkl'))
        time_data = pd.Series(time_data)
        time_data.to_pickle(os.path.join('03_data', mysite+'_'+mymodel+'_time.pkl'))
        #del(mysite, myxc, myyc, precip_data, time_data)
        
    #del(mymodel, df, precip_files)
        
    