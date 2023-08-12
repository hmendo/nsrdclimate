import numpy as np
import glob
import netCDF4 as nc
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import great_circle
import pyextremes


pd.set_option('display.max_columns', None)


def parse_data_orig(ncfilepath, offset):
    """
    Extracts latitude and longitude from a netCDF (network Common Data Form)
    format and returns them in a data frame
    
    Parameters
    ----------
    ncfilepath : str
        A filepath in the form of a string that points to the location of the
        netCDF file of interest
        
    offset : list
        Contains the lattitude and longitude offest in the form of [lat, lon]
        
        
    Returns
    -------
    Dictionary
        Dictionary with latitude and longitude components corresponding to the 
        input file
    
    """
    
    ncfile = nc.Dataset(ncfilepath)
    
    nc_lat = ncfile.variables['lat'][:]
    nc_lat = np.ma.getdata(nc_lat) + offset[0]
    nc_lon = ncfile.variables['lon'][:]
    nc_lon = np.ma.getdata(nc_lon) + offset[1]
    
    data_dict = {'lat': nc_lat, 'lon': nc_lon}
    return data_dict

def get_closest_point(site_loc, ncfilepath, offset):
    """
    Finds the closest data point between location of interest (site_loc) and
    weather stations captured by .nc files (ncfilepath)
    
    Parameters
    ----------
    site_loc : tuple
        Latitude and longitude in the form of (lat, lon) for site of interest
        
    ncfilepath : str
        A filepath in the form of a string that points to the location of the
        netCDF file of interest
        
    offset : list
        Contains the lattitude and longitude offest in the form of [lat, lon]
        
        
    Returns
    -------
    Dictionary containing lat/lon, distance, and corresponding x/y indices for
    weather station that is closest to the input site_loc

    
    """

    data_dict = parse_data_orig(ncfilepath, offset)
    data_dict_flat = {i:j.flatten() for i,j in data_dict.items()}
    data_df = pd.DataFrame(data_dict_flat)
        
    data_df['coords'] = list(zip(data_df['lat'], data_df['lon']))
    
    my_dist = data_df['coords'].apply(lambda x: great_circle(x, site_loc).km) 
    dist_min_idx = my_dist.idxmin()
    close_loc_nc = data_df.iloc[dist_min_idx,:]['coords']
    
    nc_ind = np.where((data_dict['lat']==close_loc_nc[0]) & (data_dict['lon']==close_loc_nc[1]))
    
    out = {
             'data_lat': close_loc_nc[0],
             'data_lon': close_loc_nc[1],
             'dist': my_dist[dist_min_idx],
             'nc_xindex': nc_ind[1][0], 
             'nc_yindex': nc_ind[0][0], 

             }
    return out

def weather_data_calcs(nc_filepath, wstat_indices, ncwvar):
    """
    Extracts the weather and time data from a .nc file (nc_filepath) for 
    specified location(wstat_coords)
    
    Parameters
    ----------
    ncfilepath : str
        A filepath in the form of a string that points to the location of the
        netCDF file of interest
        
    wstat_indices : tuple
        xc and yc indices corresponding to specified location 
        (weather station of interest) in the form of (xc,yc)
        
    ncwvar : str
        Weather variable of interest to be extracted from you .nc file 
        (e.g., 'pr', 'uas', or 'vas')
        
        
    Returns
    -------
    time_data : Numpy array 
        Numpy array corresponding to time data
    
    weather_data: Numpy array
        Numpy array corresponding to weather data

    """

    nc_file = nc.Dataset(nc_filepath)
    nc_weather = nc_file.variables[ncwvar][:]
    myxc = wstat_indices[0]
    myyc = wstat_indices[1]
    weather_data = nc_weather[:, myyc, myxc] 
    weather_data = np.ma.getdata(weather_data) 
    time_data = nc_file.variables['time'][:] 
    time_data = nc.ma.getdata(time_data) 
    return time_data, weather_data

def make_ordered_time_weather_lists(nc_files, wstat_indices, ncwvar):
    """
    Takes a list of .nc files corresponding to different time frames for a
        specific weather model and produces an ordered list of time and weather
        numpy arrays for an identified weather station location (wstat_loc)
    
    Parameters
    ----------
    nc_files : list
        A list of strings, where each string is a filepath that points to the
        locations of the netCDF files corresponding to different years of a
        climate model
        
    wstat_indices : tuple
        xc and yc indices corresponding to specified location 
        (weather station of interest) in the form of (xc,yc)
        
    ncwvar : str
        Weather variable of interest to be extracted from you .nc file 
        (e.g., 'pr', 'uas', or 'vas')
        
        
    Returns
    -------
    time_list : list 
        List of numpy arrays corresponding to time data
    
    weather_list: list
        List of numpy arrays corresponding to weather data

    """
    nclen = len(nc_files)
    myxc = wstat_indices[0]
    myyc = wstat_indices[1]    
    mymap = map(weather_data_calcs,
                nc_files,
                np.tile([myxc, myyc], (nclen,1)), 
                np.repeat(ncwvar, nclen)
                )
    mymap_list = list(mymap)
    
    time_list = [j[0] for i,j in zip(range(nclen), mymap_list)]
    weather_list = [j[1] for i,j in zip(range(nclen), mymap_list)]
    
    return time_list, weather_list


def timedelta_to_continuous_w_weather(time_data_list, weather_data_list, wvar, start_time, start_time_format="%Y-%m-%d %H:%M:%S", td_unit='days'):
    """
    Takes a list of time and weather numpy arrays that correspond to a 
    specific location from a single climate model. Concatenates the time 
    and weather lists into a single pandas data frame with continuous time 
    and weather data as two separate columns of the data frame.
        
       
    Parameters
    ----------
    time_data_list : list 
        List of numpy arrays corresponding to time data for a specific climate
        model
    
    weather_data_list: list
        List of numpy arrays corresponding to weather data for a specific 
        climate model
    
    wvar : str
        Column name given to weather variable of interest in weather data frame
        (e.g., 'precip', 'uas', or 'vas')
        
    start_time : str
        Time stamp corresponding to the start time of the arrays in 
        time_data_list (e.g., '1968-01-01 00:00:00')
        
    start_time_format : str
        Format in which the start_time is input (Default is '%Y-%m-%d %H:%M:%S')
    
    td_unit : str
        Unit in which the arrays in time_data_list are provided (e.g., 'days')
        
        
    Returns
    -------
    Pandas DataFrame with continuous time and weather data as two separate
    columns of the data frame

    """        
    
    all_times = pd.Series(dtype='datetime64[ns]')
    all_weather = np.array([])
    for i in range(len(time_data_list)):
        time_deltas = pd.to_timedelta(time_data_list[i], unit=td_unit) 
        times_parsed = datetime.datetime.strptime(start_time, start_time_format) + pd.Series(time_deltas)
        all_times = pd.concat([all_times, times_parsed], ignore_index=True)
        
        all_weather = np.concatenate([all_weather, weather_data_list[i]]) 
    
    timeweather_DF = pd.DataFrame({'time': all_times, wvar: all_weather})
    return timeweather_DF

def convert_PrecipRate_to_mm(precipDF, tint, wvar):
   
    """
    Converts precipitation rate (in [kg/m^2 s]) to total depth (in mm) 
    when time interval in between timestamps is provided

        
       
    Parameters
    ----------
    precipDF : pandas.DataFrame 
        Index:
            RangeIndex
        Columns:
            Name: time, dtype: datetime64[ns], description: time stamps
            Name: precip, dtype: float64, description: precipitation rate in (kg/m^2-s)
    
    tint : float
        Time interval in between the timestamps in precipDF
    
    wvar : str
        Name given to preciptation column in precipitation data frame
        
    Returns
    -------
    Pandas DataFrame similar to input but with precipitation rate converted to
        total depth in mm.  Time stamps with NaNs in precipitation column are 
        dropped from the data frame.

    """
    
    
    precipDF_mm = precipDF.copy()
    precipDF_mm.loc[:,wvar] = precipDF_mm.loc[:,wvar]*(tint*3600)*(1/1000)*1000
    precipDF_mm = precipDF_mm.dropna(subset=wvar)
    return precipDF_mm
    
def convert_to_daily(wDF, tvar, aggfun):
    
    """
    Converts a weather data frame with hourly time stamps to daily time 
    stamps based on a user defined aggregation method

        
       
    Parameters
    ----------
    wDF : pandas.DataFrame 
        Index:
            RangeIndex
        Columns:
            Name: time, dtype: datetime64[ns], description: time stamps
            Name: any weather variable, dtype: float64, description: can be 
            either precipitation (in mm) or wind speed (in m/s)
    
    tvar : str
        Name given to time column in weather data frame
    
    aggfun : string
        Aggregation function that is read in by the 'aggregate' method for 
        pandas groupby objects (e.g., 'sum', 'mean', etc.)
        
    Returns
    -------
    Pandas DataFrame similar to input but with daily time intervals instead of
    more frequent time intervals (e.g., hourly)

    """
    
    wDF = wDF.copy()
    wDF.loc[:,'date'] = wDF.loc[:, tvar].dt.date
    
    wDF_daily = wDF.groupby(['date']).aggregate(aggfun, numeric_only=True)
    #DF_daily.drop(columns=tvar, inplace=True)
    wDF_daily.reset_index(inplace=True)
    return wDF_daily

def convert_to_mm_daily(precipDF, tint, tvar, wvar, aggfun):
    
    """
    Converts precipitation rate (in [kg/m^2 s]) to daily total depth (in mm) 
    when time interval in between timestamps is provided

        
       
    Parameters
    ----------
    precipDF : pandas.DataFrame 
        Index:
            RangeIndex
        Columns:
            Name: time, dtype: datetime64[ns], description: time stamps
            Name: precip, dtype: float64, description: precipitation rate in (kg/m^2-s)
    
    tint : float
        Time interval in between the timestamps in precipDF
    
    tvar : str
        Name given to time column in weather data frame
        
    wvar : str
        Name given to preciptation column in precipitation data frame
        
    aggfun : string
        Aggregation function that is read in by the 'aggregate' method for 
        pandas groupby objects (e.g., 'sum', 'mean', etc.)
        
    Returns
    -------
    Pandas DataFrame similar to input but with precipitation rate converted to
        daily total depth (in mm)

    """
        
    precipDF = convert_PrecipRate_to_mm(precipDF, tint, wvar)
    precipDF = convert_to_daily(precipDF, tvar, aggfun)
    return precipDF

def convert_uasvas_to_wind(uasDF, vasDF, ucol, vcol, tvar, wndvar):
    
    """
    Converts u and v (east/west and north/south) components of surface wind
    into wind speed (in m/s).  Any time stamps that result in NaNs for the
    output wind speed column are dropped from the output data frame.

        
       
    Parameters
    ----------
    uasDF : pandas.DataFrame 
        Index:
            RangeIndex
        Columns:
            Name: time, dtype: datetime64[ns], description: time stamps
            Name: uas, dtype: float64, description: u-component of surface wind (in m/s)
            
    vasDF : pandas.DataFrame 
        Index:
            RangeIndex
        Columns:
            Name: time, dtype: datetime64[ns], description: time stamps
            Name: uas, dtype: float64, description: v-component of surface wind (in m/s)
    
    ucol : str
        Name given to wind column in uasDF
        
    ucol : str
        Name given to wind column in vasDF
        
    tvar : str
        Name given to time column in uasDF
        
    wndvar : str
        Name to be given to wind speed column in output data frame
        
    Returns
    -------
    Pandas DataFrame with u and v components of wind converted to wind speed (in m/s)

    """
    
    time = uasDF.loc[:, tvar] 
    uas = uasDF.loc[:,ucol] 
    vas = vasDF.loc[:,vcol] 
    wind = np.sqrt(uas**2 + vas**2)
    windDF = pd.concat([time.rename(tvar), wind.rename(wndvar)], axis=1)
    windDF = windDF.dropna(subset=wndvar)
    return windDF
    
def summary_stat_fun(dailyDF_current, dailyDF_future, wmodel, site):
    
    """
    Takes in a pandas data frame for daily weather and outputs statstical
    summaries for that data set (min, max, mean, and std. dev.)

        
       
    Parameters
    ----------
    dailyDF_current : pandas.DataFrame 
        Index:
            RangeIndex
        Columns:
            Name: date, dtype: datetime.date object, description: historical dates
            Name: precip or wind, dtype: float64, description: any daily weather
                parameter (e.g., precipitation or wind)
            
    dailyDF_future : pandas.DataFrame 
        Index:
            RangeIndex
        Columns:
            Name: date, dtype: datetime.date object, description: future dates
            Name: precip or wind, dtype: float64, description: any daily weather
                parameter (e.g., precipitation or wind)
    wmodel : str
        Name for the climate model used for weather data sets
        
    site : str
        Name for the specific site whose weather being analyzed
        

        
    Returns
    -------
    pandas.DataFrames with statistical summaries (min, max, mean, and std. dev.)

    """
    
    
    dailyDF_current = dailyDF_current.copy()
    dailyDF_future = dailyDF_future.copy()
    dailyDF_current.set_index('date', inplace=True) 
    dailyDF_future.set_index('date', inplace=True)
    
    summary_stats_current = pd.DataFrame({'model':[wmodel], 
                                          'time_period':['current'],
                                          'site_name':[site],
                                          'min':dailyDF_current.min().values,
                                          'mean':dailyDF_current.mean().values,
                                          'max':dailyDF_current.max().values,
                                          'sd':dailyDF_current.std().values},
                                         index=[0])
    summary_stats_future = pd.DataFrame({'model':[wmodel],
                                         'time_period':['future'],
                                         'site_name':[site],
                                         'min':dailyDF_future.min().values,
                                         'mean':dailyDF_future.mean().values,
                                         'max':dailyDF_future.max().values,
                                         'sd':dailyDF_future.std().values},
                                        index=[0])
    return summary_stats_current, summary_stats_future
    
def get_annual_max(dailyDF, max_dict_rename):
    """
    Takes in a pandas data frame for daily weather and outputs annual maximum
    for the years covered in the input data.  Function also renames new data
    frame columns to user-defined preference.

        
       
    Parameters
    ----------
    dailyDF: pandas.DataFrame 
        Index:
            RangeIndex
        Columns:
            Name: date, dtype: datetime.date object, description: historical dates
            Name: precip or wind, dtype: float64, description: any daily weather
                parameter (e.g., precipitation or wind)
            
    max_dict_rename : dict
        Dictionary used to rename weather data column in output data frame so  
        as to distinguish from input data (e.g., from 'precip' in dailyDF to 
        'MaxPrecip' in output data frame).
                                                                
    Returns
    -------
    pandas.DataFrames with annual maximum weather for years covered in input data

    """
    dailyDF = dailyDF.copy()
    dailyDF['year'] = dailyDF['date'].apply(lambda x: x.year)
    dailyDF.drop(columns='date', inplace=True)
    dailyDF = dailyDF.groupby('year').max().reset_index()
    dailyDF.rename(columns=max_dict_rename, inplace=True)
    return dailyDF

def myextremes(annDF, rp_array, wvar, alpha=None):
    """
    Uses pyextremes to fit extreme (maximum annual) weather data to a generalized
    extreme value distribution (GEVD).  The GEVD fit is then used to calculate
    return values for user-defined return period and user-defined confidence
    intervals.


    Parameters
    ----------
    annDF : pandas.DataFrame 
        Index:
            RangeIndex
        Columns:
            Name: date, dtype: datetime.date object, description: historical dates
            Name: precip or wind, dtype: float64, description: any daily weather
                parameter (e.g., precipitation or wind)
                
    rp_array : numpy.ndarray
        Numpy array with return periods for which return values should be 
        calculated with pyextremes
        
    wvar : str
        Column name given to weather variable of interest in annDF
        (e.g., 'MaxPrecip', 'MaxWind', etc)

    alpha : float
        Width of confidence interval (0,1).  If None (default), return None for
        upper and lower confidence interval bounds.
                                                                
    Returns
    -------
    pandas.DataFrame providing return values for user-requested return periods.  
    Upper and lower confidence bounds are also output if alpha is non-zero.

    """
    
    
    annDF = annDF.copy()
    annDF.index = pd.to_datetime(annDF.year, format='%Y')
    Ser = annDF[wvar]
    
    
    extmod = pyextremes.EVA(Ser)
    extmod.get_extremes(method='BM', block_size='365.2425D')
    extmod.fit_model()
    
    out = extmod.get_summary(return_period=rp_array, alpha=alpha, n_samples=1000)
    

    return out

def find_RL_location(future_DF, RL):
    """
    Finds a return period corresponding to a specified return level in a data
    frame of return-periods/return-levels as output by pyextremes


    Parameters
    ----------
    future_DF : pandas.DataFrame 
        Index:
            return periods
        Columns:
            Name: return value, dtype: float64, description: return values of specified return period (index)
            Name: lower ci, dtype: float64, description: lower confidence interval for return values of specified return period (index)
            Name: upper ci, dtype: float64, description: lower confidence interval for return values of specified return period (index)
                 
    RL : str
        Return level of interest for which a corresponding return period should
        be identified.
                                                                
    Returns
    -------
    Return period in future_DF that most closely approximates the specified return level

    """
    
    DF = future_DF.copy()
    rp_diff = DF.loc[:, 'return value']-RL #return period
    futureRP_fromcurrentRL = (abs(rp_diff)).idxmin() #future location
    return futureRP_fromcurrentRL
