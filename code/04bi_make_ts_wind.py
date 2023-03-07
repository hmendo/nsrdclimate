# time data is the same for all sites in the same model - only need to extract 6 times, one for each model
# start time for HRM3 data is 1968 for both current and future, for other models start time for current is 1968 and for future is 2038


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import glob
#import netCDF4 as nc
import os
import pandas as pd
import datetime

myos = 'Mac' #Windows

models = np.load('00_model_names.npy')
sites = pd.read_pickle('00_site_df.pkl')

homedir = os.path.expanduser('~')
save_path = '04a_ts_wind'
if not os.path.exists(save_path):
    os.makedirs(save_path)

if myos == 'Windows':
    precip_dir = os.path.join('//snl', 'Collaborative', 'nsrd_climate_impacts','data', 'raw', 'precipitation')
    precip_files_all = glob.glob(precip_dir)
elif myos == 'Mac':
#NOTE: the below needs to be previously mounted in terminal with: mount_smbfs //snl/Collaborative/nsrd_climate_impacts/data/raw/precipitation/ ~/tempmount
    precip_dir = os.path.join(homedir,'tempmount') 
    precip_files_all = glob.glob(os.path.join(precip_dir, '*nc'))
elif myos == 'Linux':
    precip_dir = os.path.join('.','precipitation')
    precip_files_all = sorted(glob.glob(os.path.join(precip_dir, '*nc')))

data_files_all = glob.glob(os.path.join('.','03_data', '*pkl'))
#time_data_files = glob.glob(os.path.join('.','03_data','*time*pkl'))
#data_files_mod = glob.glob(os.path.join('.','03_data','*precip*pkl'))

models = models[0]
for mod in range(len(models)):
    mymodel = models[mod]
    print('\n'+mymodel)
    
    #load time data
    time_data_files = [i for i in data_files_all if ('time' in i) and (mymodel in i)]
    #print(time_data_files)
    time_data = pd.read_pickle(time_data_files[0])
    
    
    all_times = pd.Series()
    for i in range(len(time_data)):
        start_time = "1968-01-01 00:00:00" if (mymodel[0:4]=='HRM3' or i<8) else "2038-01-01 00:00:00"       
        time_deltas = pd.to_timedelta(time_data[i][:,0], unit='days')
        times_parsed = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S") + pd.Series(time_deltas)
        all_times = pd.concat([all_times, times_parsed])
        
    data_files_uas = [i for i in data_files_all if ('uas' in i) and (mymodel in i)]
    data_files_vas = [i for i in data_files_all if ('vas' in i) and (mymodel in i)]

    
    wind_full = {}
    for s in range(len(sites)):
        mysite = sites.loc[s, 'site_name']
        print(s, mysite)
        
        mysite_string = [i for i in data_files_uas if mysite in i][0]
        uas_data = pd.read_pickle(mysite_string)
        
        mysite_string = [i for i in data_files_vas if mysite in i][0]
        vas_data = pd.read_pickle(mysite_string)
        
        all_uas = np.array([])
        for i in range(len(uas_data)):
            all_uas = np.concatenate([all_uas, uas_data[i].data])
            
        all_vas = np.array([])
        for i in range(len(vas_data)):
            all_vas = np.concatenate([all_vas, vas_data[i].data])
        
        wind_data = np.sqrt(all_uas**2 + all_vas**2)
        
        # make df
        wind_df = pd.DataFrame({'time': all_times, 'wind': wind_data})
        wind_full[mysite] = wind_df
    
    wind_full = pd.Series(wind_full)
    wind_full.to_pickle(os.path.join(save_path,'full_wind_df_'+mymodel+'.pkl'))
        
        
        
    

#datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
#use "pd.to_timdelta" with the datetime.datetime => put in a dictionary and
#Then transpose to maybe concatenate/flatten?!!? => 
#td11 = pd.to_timedelta(time_data[1][:2,0].data, unit='s')
#np.array([td11.values,td11.values]).transpose().flatten()
#maybe seek difference between "pd.to_timedelta" and "pd.Timedelta" => different!

#time_data is a series of shape (14,)
#each element of time_data has shape (large_number, 2) => she's just taking the first value of each row

#------Question fo Audrey
# What are the two vaules in time_data, and why only take 1?












# library(lubridate)
# library(ggplot2)

# data_files_all <- list.files("audrey_sandbox/03_data")
# load("audrey_sandbox/00_data_years.Rdata")
# load("audrey_sandbox/00_model_names.Rdata")
# load("audrey_sandbox/00_site_df.Rdata")