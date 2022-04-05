import pandas as pd
import datetime
import math
import dask.dataframe as dd
import gcsfs
import json
import numpy as np

S = 37.25411
N = 42.59896
E = -118.73987
W = -124.51868

square_side_miles = 1
miles_per_degree = 24901.461 / 360 # circumference of earth at equator divided by 360
square_side_degrees = square_side_miles / miles_per_degree

local = False

GCS_PREFIX = 'gs://generic-bucket-name'
if local == True:
    GCS_PREFIX = 'data'

folder = '{}/modeling-2'.format(GCS_PREFIX)

WEATHER_METRICS = ['precipIntensity', 'precipIntensityMax', 'temperatureHigh', 'temperatureLow',
                   'humidity', 'windSpeed', 'windGust', 'windBearing', 'cloudCover', 'windSpeedSquared']

def get_ref_dictionaries(reverse = False):

    step = square_side_degrees
    if 'gs://' in GCS_PREFIX:
        fs = gcsfs.GCSFileSystem()

        with fs.open("{}/refs/Lat_refs.json".format(GCS_PREFIX)) as file:
            lat_refs = json.load(file)

        with fs.open("{}/refs/Long_refs.json".format(GCS_PREFIX)) as file:
            long_refs = json.load(file)

        with fs.open("{}/refs/Date_refs.json".format(GCS_PREFIX)) as file:
            date_ids = json.load(file)

    else:

        with open("{}/refs/Lat_refs.json".format(GCS_PREFIX)) as file:
            lat_refs = json.load(file)

        with open("{}/refs/Long_refs.json".format(GCS_PREFIX)) as file:
            long_refs = json.load(file)

        with open("{}/refs/Date_refs.json".format(GCS_PREFIX)) as file:
            date_ids = json.load(file)


    for i in range(30):
        lat_refs[str(-i)] = lat_refs['1'] - (i + 1) * step
        lat_refs[str(370 + i)] = lat_refs['369'] + (i + 1) * step
        long_refs[str(-i)] = long_refs['1'] - (i + 1) * step
        long_refs[str(400 + i)] = long_refs['399'] + (i + 1) * step

    if reverse:

        lat_refs_reverse = {round(value, 3): key for (key, value) in lat_refs.items()}
        long_refs_reverse = {round(value, 3): key for (key, value) in long_refs.items()}
        date_ids_reverse = {value: key for (key, value) in date_ids.items()}

        return lat_refs_reverse, long_refs_reverse, date_ids_reverse

    else:

        return lat_refs, long_refs, date_ids


def get_weather_coords():

    coords = []

    for x in range(-2, 2 + 1):
        for y in range(-2, 2 + 1):

            if (math.sqrt(x**2 + y**2) <= 4) & (x % 2 == 0) & (y % 2 == 0):
                coords.append(str((x,y,0)))
                coords.append(str((x,y,1)))

    return coords

def get_neighbor_coords(fire_spread_thresh, dates_back = 0):

    coords = []

    for x in range(-fire_spread_thresh, fire_spread_thresh + 1):
        for y in range(-fire_spread_thresh, fire_spread_thresh + 1):

            if math.sqrt(x**2 + y**2) <= fire_spread_thresh:
                if not dates_back:
                    coords.append(str((x,y)))
                else:
                    for d in range(dates_back + 1):
                        coords.append(str((x, y, -d)))

    return coords


def get_historic_raw():
    historic_archive1 = pd.read_csv('{}/nasa_archive/fire_archive_SV-C2_233133.csv'.format(GCS_PREFIX))
    historic_archive2 = pd.read_csv('{}/nasa_archive/fire_archive_SV-C2_246698.csv'.format(GCS_PREFIX))

    historic_archive = historic_archive1.append(historic_archive2)

    del (historic_archive1)
    del (historic_archive2)

    historic_archive = historic_archive[['latitude', 'longitude', 'acq_date', 'acq_time', 'brightness']]
    historic_archive = historic_archive.drop_duplicates(keep='last')

    historic_daily = dd.read_csv('{}/nasa-firms/suomi-npp-viirs-c2/*.txt'.format(GCS_PREFIX))
    historic_daily = historic_daily.compute()
    historic_daily = historic_daily[['latitude', 'longitude', 'acq_date', 'acq_time']]
    historic_daily = historic_daily[historic_daily.acq_date >= '2022-01-01']

    historic = historic_archive.append(historic_daily)

    return historic


def get_weather_raw():

    df1 = pd.read_csv('{}/weather/darksky_points_thresh1_withValues.csv'.format(GCS_PREFIX))
    df2 = pd.read_csv('{}/weather/darksky_thresh2minus1.csv'.format(GCS_PREFIX))
    df3 = pd.read_csv('{}/weather/darksky_remaining_results.csv'.format(GCS_PREFIX))

    df = df1.append(df2).append(df3)

    del(df1)
    del(df2)
    del(df3)

    df = df[['Date', 'Lat', 'Long', 'precipIntensity', 'precipIntensityMax', 'temperatureHigh', 'temperatureLow',
             'humidity', 'windSpeed', 'windGust', 'windBearing', 'cloudCover']].dropna()

    df['windSpeedSquared'] = df['windSpeed'].apply(lambda x: x**2)

    df['Date'] = df['Date'].astype(str)

    return df



def convert_to_datetime(date, time):

    try:
        datetime = pd.to_datetime('{} {:0>2}:{}'.format(date, str(time)[:-2], str(time)[-2:]))
    except:
        datetime = pd.to_datetime('{} {}'.format(date, time))

    return datetime



def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if col=='Pred':
                    df[col] = df[col].astype(np.float64)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.float64)

        else:
            pass
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df