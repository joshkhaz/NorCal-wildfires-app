import math
import datetime
import pandas as pd
import pytz

import common

"""
INPUTS:
    df={o1,o2,...,on} Set of objects
    spatial_threshold = Maximum geographical coordinate (spatial) distance value
    temporal_threshold = Maximum non-spatial distance value
    min_neighbors = Minimun number of points within Eps1 and Eps2 distance
OUTPUT:
    C = {c1,c2,...,ck} Set of clusters
"""

def identify_wildfires(df,
                       spatial_threshold = 1.01 * common.square_side_degrees, #* 1.01 * math.sqrt(2) * common.square_side_degrees,
                       temporal_threshold = 1):

    cluster_label = 0
    UNMARKED = -999
    stack = []

    # initialize each point with unmarked
    df['wildfire_id'] = UNMARKED
    df['datetime'] = df['date'].apply(lambda x: pd.to_datetime(x))

    df = df.sort_values(by = 'datetime')
    df = df.reset_index(drop = True)

    start = datetime.datetime.now(pytz.timezone('US/Pacific'))
    # for each point in database
    for index, point in df.iterrows():
        if df.loc[index]['wildfire_id'] == UNMARKED:
            neighborhood = retrieve_neighbors(index, df, spatial_threshold, temporal_threshold)

            cluster_label = cluster_label + 1
            df.at[index, 'wildfire_id'] = cluster_label  # assign a label to core point
            print(df[df.wildfire_id!=UNMARKED].shape[0], df.shape[0],
                  start + (datetime.datetime.now(pytz.timezone('US/Pacific')) - start) * df.shape[0] / df[df.wildfire_id!=UNMARKED].shape[0])
            for neig_index in neighborhood:  # assign core's label to its neighborhood
                df.at[neig_index, 'wildfire_id'] = cluster_label
                print(df[df.wildfire_id != UNMARKED].shape[0], df.shape[0],
                      start + (datetime.datetime.now(pytz.timezone('US/Pacific')) - start) * df.shape[0] /
                      df[df.wildfire_id != UNMARKED].shape[0])
                stack.append(neig_index)  # append neighborhood to stack

            while len(stack) > 0:  # find new neighbors from core point neighborhood
                current_point_index = stack.pop()
                new_neighborhood = retrieve_neighbors(current_point_index, df, spatial_threshold,
                                                      temporal_threshold)

                for neig_index in new_neighborhood:
                    neig_cluster = df.loc[neig_index]['wildfire_id']
                    if (neig_cluster == UNMARKED):
                        df.at[neig_index, 'wildfire_id'] = cluster_label
                        print(df[df.wildfire_id != UNMARKED].shape[0], df.shape[0],
                              start + (datetime.datetime.now(pytz.timezone('US/Pacific')) - start) * df.shape[0] /
                              df[df.wildfire_id != UNMARKED].shape[0])
                        stack.append(neig_index)

    assert UNMARKED not in df.wildfire_id.unique()
    df = df.sort_values(by='datetime')
    df = df.drop(columns=['datetime'])

    return df


def retrieve_neighbors(index_center, df, spatial_threshold, temporal_threshold):

    neigborhood = []

    center_point = df.loc[index_center]

    # filter by time
    min_time = center_point['datetime'] - datetime.timedelta(days=temporal_threshold)
    max_time = center_point['datetime'] + datetime.timedelta(days=temporal_threshold)
    df = df[(df['datetime'] >= min_time) & (df['datetime'] <= max_time)]

    min_lat = center_point['latitude'] - spatial_threshold
    max_lat = center_point['latitude'] + spatial_threshold
    min_long = center_point['longitude'] - spatial_threshold
    max_long = center_point['longitude'] + spatial_threshold
    df = df[(df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) & (df['longitude'] >= min_long) & (df['longitude'] <= max_long)]

    # filter by distance
    for index, point in df.iterrows():
        if index != index_center:
            neigborhood.append(index)

    return neigborhood
