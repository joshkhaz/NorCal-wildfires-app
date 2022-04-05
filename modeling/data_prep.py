import pandas as pd
import numpy as np
import datetime
import math
import multiprocessing
import gcsfs
import json
import pytz

import common

folder = common.folder

lat_refs, long_refs, date_ids = common.get_ref_dictionaries()
lat_refs_reverse, long_refs_reverse, date_ids_reverse = common.get_ref_dictionaries(reverse = True)

step = common.square_side_degrees

def get_refs(df, step, coords, coord_refs, coord_string, coord_ref_string, coord_ref_id_string, buffer):
    for coord in coords:
        print(coord, min(coords), max(coords))
        refs_to_check = {k: coord_refs[k] for k in coord_refs if
                         (coord < coord_refs[k] + step) & (coord > coord_refs[k] - step)}
        if refs_to_check:
            for ID in refs_to_check:
                coord_ref = refs_to_check[ID]
                if abs(coord - coord_ref) < step/2:
                    df.loc[df[coord_string] == coord, coord_ref_string] = coord_ref
                    df.loc[df[coord_string] == coord, coord_ref_id_string] = ID
                    break
        else:
            dist_to_min_coord_ref = abs(coord - coord_refs[min(coord_refs)])
            dist_to_max_coord_ref = abs(coord - coord_refs[max(coord_refs)])
            if dist_to_min_coord_ref < dist_to_max_coord_ref:
                df.loc[df[coord_string] == coord, coord_ref_string] = coord_refs[min(coord_refs)]
                df.loc[df[coord_string] == coord, coord_ref_id_string] = min(coord_refs)
            else:
                df.loc[df[coord_string] == coord, coord_ref_string] = coord_refs[max(coord_refs)]
                df.loc[df[coord_string] == coord, coord_ref_id_string] = max(coord_refs)

    return df

def label():
    df = common.get_historic_raw()
    df = df[(df['latitude'] >= common.S) & (df['latitude'] <= common.N) & (df['longitude'] >= common.W) & (
            df['longitude'] <= common.E)]
    df['datetime'] = df.apply(lambda x: common.convert_to_datetime(x.acq_date, x.acq_time), axis=1)
    df = df[['latitude', 'longitude', 'datetime']]
    df = df.drop_duplicates(subset=['latitude', 'longitude', 'datetime'])
    df['date'] = df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))

    df['latitude_rounded'] = df.latitude.apply(lambda x: round(x, 4))
    df['longitude_rounded'] = df.longitude.apply(lambda x: round(x, 4))

    lat_refs = np.arange(common.S, common.N, step)[:-1] + step / 2
    long_refs = np.arange(common.W, common.E, step)[:-1] + step / 2

    lat_refs = dict(zip(range(1, len(lat_refs) + 1), lat_refs))
    long_refs = dict(zip(range(1, len(long_refs) + 1), long_refs))

    min_date = df.date.min()
    max_date = pd.to_datetime(df.date.max()) + datetime.timedelta(days=365 * 10)
    dates = [d.strftime('%Y-%m-%d') for d in pd.date_range(min_date, max_date)]
    date_ids = dict(zip(range(1, len(dates) + 1), dates))

    df['lat_ref'] = np.nan
    df['long_ref'] = np.nan
    df['lat_ref_id'] = np.nan
    df['long_ref_id'] = np.nan

    lats_rounded = list(df['latitude_rounded'].unique())
    longs_rounded = list(df['longitude_rounded'].unique())

    lats_rounded.sort()
    longs_rounded.sort()

    df = get_refs(df, step, lats_rounded, lat_refs, 'latitude_rounded', 'lat_ref', 'lat_ref_id', .00005)
    df = get_refs(df, step, longs_rounded, long_refs, 'longitude_rounded', 'long_ref', 'long_ref_id', .00005)

    lats_remaining = list(df[df['lat_ref'].isna()].latitude.unique())
    longs_remaining = list(df[df['long_ref'].isna()].longitude.unique())

    lats_remaining.sort()
    longs_remaining.sort()

    df = get_refs(df, step, lats_remaining, lat_refs, 'latitude', 'lat_ref', 'lat_ref_id', 0)
    df = get_refs(df, step, longs_remaining, long_refs, 'longitude', 'long_ref', 'long_ref_id', 0)

    df['date_id'] = df['date'].map({value: key for (key, value) in date_ids.items()})

    df.to_csv('{}/labelled_1Mile.csv'.format(folder))


def prepare_modeling_points(df, thresh):
    modeling_coords = common.get_neighbor_coords(thresh)

    start = datetime.datetime.now(pytz.timezone('US/Pacific'))
    for col in modeling_coords:
        print(col, modeling_coords.index(col) + 1, len(modeling_coords),
              start + (datetime.datetime.now(pytz.timezone('US/Pacific')) - start) * len(modeling_coords) / (
                          modeling_coords.index(col) + 1))
        lat_shift = eval(col)[0]
        long_shift = eval(col)[1]
        df[col] = df.apply(lambda x: (x['lat_ref_id'] + lat_shift, x['long_ref_id'] + long_shift), axis=1)

    melt = pd.melt(df, id_vars=['date_id', 'lat_ref_id', 'long_ref_id'], value_vars=modeling_coords)
    melt['new_lat_ref_id'] = melt['value'].apply(lambda x: x[0])
    melt['new_long_ref_id'] = melt['value'].apply(lambda x: x[1])
    df_new = melt[['new_lat_ref_id', 'new_long_ref_id', 'date_id']].drop_duplicates(
        subset=['new_lat_ref_id', 'new_long_ref_id', 'date_id'])
    df_new = df_new.rename(columns={'new_lat_ref_id': 'lat_ref_id',
                                    'new_long_ref_id': 'long_ref_id'})
    df_new['num_points'] = 0
    df_new['fire'] = 'W'

    df = df[['lat_ref_id', 'long_ref_id', 'date_id', 'num_points', 'fire']]
    df = df.append(df_new)
    df = df.drop_duplicates(subset=['lat_ref_id', 'long_ref_id', 'date_id'])

    return df


def prepare_weather_predictors(df):
    print("getting weather")
    weather = common.get_weather_raw()

    print("fillna")
    weather = weather.fillna({'precipAccumulation': 0})
    weather_metrics = common.WEATHER_METRICS

    print("getting lats")
    weather['lat_ref_id'] = weather.Lat.apply(lambda x: round(x, 3)).map(lat_refs_reverse).astype(int)
    print("getting longs")
    weather['long_ref_id'] = weather.Long.apply(lambda x: round(x, 3)).map(long_refs_reverse).astype(int)
    print("get dates")
    weather['date_id'] = weather.Date.map(date_ids_reverse).astype(int)

    weather_coords = common.get_weather_coords()
    start = datetime.datetime.now(pytz.timezone('US/Pacific'))
    for col in weather_coords:
        print(col, weather_coords.index(col) + 1, len(weather_coords),
              start + (datetime.datetime.now(pytz.timezone('US/Pacific')) - start) * len(weather_coords) / (
                          weather_coords.index(col) + 1))
        lat_shift = eval(col)[0]
        long_shift = eval(col)[1]
        date_shift = eval(col)[2]
        df[col] = df.apply(
            lambda x: (x['lat_ref_id'] + lat_shift, x['long_ref_id'] + long_shift, x['date_id'] + date_shift), axis=1)
        df['new_lat_ref_id{}'.format(col)] = df[col].apply(lambda x: x[0])
        df['new_long_ref_id{}'.format(col)] = df[col].apply(lambda x: x[1])
        df['new_date_id{}'.format(col)] = df[col].apply(lambda x: x[2])
        df = df.merge(weather[['date_id', 'lat_ref_id', 'long_ref_id'] + weather_metrics].rename(
            columns={'date_id': 'date_id_weather',
                     'lat_ref_id': 'lat_ref_id_weather',
                     'long_ref_id': 'long_ref_id_weather'}),
            left_on=['new_date_id{}'.format(col), 'new_lat_ref_id{}'.format(col),
                     'new_long_ref_id{}'.format(col)],
            right_on=['date_id_weather', 'lat_ref_id_weather', 'long_ref_id_weather'], how='left')
        df = df.rename(columns=lambda metric: '{}{}'.format(metric, col) if metric in weather_metrics else metric)
        df = df.drop(
            columns=[col, 'new_date_id{}'.format(col), 'new_lat_ref_id{}'.format(col), 'new_long_ref_id{}'.format(col),
                     'date_id_weather', 'lat_ref_id_weather', 'long_ref_id_weather'])

    return df


def prepare_nasa_predictors(df, df_fire):
    nasa_coords = common.get_neighbor_coords(3, 2)

    start = datetime.datetime.now(pytz.timezone('US/Pacific'))
    for col in nasa_coords:
        print(col, nasa_coords.index(col) + 1, len(nasa_coords),
              start + (datetime.datetime.now(pytz.timezone('US/Pacific')) - start) * len(nasa_coords) / (
                          nasa_coords.index(col) + 1))
        lat_shift = eval(col)[0]
        long_shift = eval(col)[1]
        date_shift = eval(col)[2]
        df[col] = df.apply(
            lambda x: (x['lat_ref_id'] + lat_shift, x['long_ref_id'] + long_shift, x['date_id'] + date_shift), axis=1)
        df['new_lat_ref_id{}'.format(col)] = df[col].apply(lambda x: x[0])
        df['new_long_ref_id{}'.format(col)] = df[col].apply(lambda x: x[1])
        df['new_date_id{}'.format(col)] = df[col].apply(lambda x: x[2])
        df = df.merge(
            df_fire[['date_id', 'lat_ref_id', 'long_ref_id', 'num_points']].rename(columns={'date_id': 'date_id_fire',
                                                                                            'lat_ref_id': 'lat_ref_id_fire',
                                                                                            'long_ref_id': 'long_ref_id_fire',
                                                                                            'num_points': 'num_points_fire'}),
            left_on=['new_date_id{}'.format(col), 'new_lat_ref_id{}'.format(col), 'new_long_ref_id{}'.format(col)],
            right_on=['date_id_fire', 'lat_ref_id_fire', 'long_ref_id_fire'],
            how='left')
        df = df.rename(columns={'num_points_fire': 'num_points{}'.format(col)})
        df = df.drop(
            columns=[col, 'new_date_id{}'.format(col), 'new_lat_ref_id{}'.format(col), 'new_long_ref_id{}'.format(col),
                     'date_id_fire', 'lat_ref_id_fire', 'long_ref_id_fire'])
        df = df.fillna({'num_points{}'.format(col): 0})

    return df


def prepare_target(df, df_fire):
    df['target_date'] = df['date_id'].apply(lambda x: x + 1)
    df = df.merge(
        df_fire[['date_id', 'lat_ref_id', 'long_ref_id', 'num_points']].rename(columns={'date_id': 'date_id_fire',
                                                                                        'lat_ref_id': 'lat_ref_id_fire',
                                                                                        'long_ref_id': 'long_ref_id_fire',
                                                                                        'num_points': 'Target'}),
        left_on=['target_date', 'lat_ref_id', 'long_ref_id'],
        right_on=['date_id_fire', 'lat_ref_id_fire', 'long_ref_id_fire'],
        how='left')
    df = df.drop(columns=['date_id_fire', 'lat_ref_id_fire', 'long_ref_id_fire'])
    df['Target'] = df['Target'].apply(lambda x: 0 if np.isnan(x) else 1)

    return df


def prepare_for_modeling():

    df = pd.read_csv('{}/labelled_1Mile.csv'.format(folder))
    df = common.reduce_mem_usage(df)

    df = df[['lat_ref_id', 'long_ref_id', 'date_id']]
    df['num_points'] = 1

    df = df.groupby(by=['lat_ref_id', 'long_ref_id', 'date_id']).count().reset_index()
    df['fire'] = 'R'
    df_fire = df.copy()

    df1 = prepare_modeling_points(df, 1)
    df0 = df1[df1.num_points > 0]
    df1 = df1[df1.num_points == 0]

    df2 = prepare_modeling_points(df, 2)

    df0['thresh'] = 0
    df1['thresh'] = 1
    df2['thresh'] = 2
    df = df0.append(df1).append(df2)

    del (df0)
    del (df1)
    del (df2)

    df = df.drop_duplicates(subset=[col for col in df.columns if col != 'thresh'])

    df = common.reduce_mem_usage(df)
    df = prepare_weather_predictors(df)

    df = prepare_nasa_predictors(df, df_fire)

    df['month'] = df['date_id'].apply(lambda x: str(int(x))).map(date_ids).apply(lambda x: pd.to_datetime(x).month)
    df = pd.get_dummies(df, columns=['month'], dtype='int64')

    df = prepare_target(df, df_fire)

    df.to_csv('{}/modeling_df.csv'.format(folder))


if __name__ == '__main__':
    label()
    prepare_for_modeling()