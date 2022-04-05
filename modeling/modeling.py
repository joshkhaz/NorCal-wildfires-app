import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
import pickle
import gcsfs
import shap
import datetime
import pytz
import json

import common

folder = common.folder

lat_refs, long_refs, date_ids = common.get_ref_dictionaries()

date_id_start = 2558


def feature_engineering(df, thresh):
    for i in range(0, 2):
        point_cols = [col for col in df.columns if '(0, 0, {})'.format(i) in col]
        for col in point_cols:
            if 'num_points' not in col:
                prefix = col.split('(')[0]
                df['{}_x_num_points(0, 0, {})'.format(prefix, i)] = df[col] * df['num_points(0, 0, 0)']

    cardinal_directions = {'N': '({}, 0, 0)',
                           'E': '(0, {}, 0)',
                           'S': '(-{}, 0, 0)',
                           'W': '(0, -{}, 0)'}

    for d in cardinal_directions:
        point_cols = [col for col in df.columns if (
                    ('num_points' not in col) & ('windBearing' not in col) & (cardinal_directions[d].format(2) in col))]
        for col in point_cols:
            prefix = col.split('(')[0]
            df['{}_x_num_points({})'.format(prefix, d)] = df[col] * df[
                'num_points{}'.format(cardinal_directions[d]).format(thresh)]

    cardinal_directions_forecast = {'N': '({}, 0, 1)',
                                    'E': '(0, {}, 1)',
                                    'S': '(-{}, 0, 1)',
                                    'W': '(0, -{}, 1)'}

    for d in cardinal_directions_forecast:
        point_cols = [col for col in df.columns if (('num_points' not in col) & ('windBearing' not in col) & (
                    cardinal_directions_forecast[d].format(2) in col))]
        for col in point_cols:
            prefix = col.split('(')[0]
            df['{}_x_num_points({})(forecast)'.format(prefix, d)] = df[col] * df[
                'num_points{}'.format(cardinal_directions[d]).format(thresh)]

    intercardinal_directions = {'NE': '({}, {}, 0)',
                                'NW': '({}, -{}, 0)',
                                'SE': '(-{}, {}, 0)',
                                'SW': '(-{}, -{}, 0)'}

    for d in intercardinal_directions:
        point_cols = [col for col in df.columns if (('num_points' not in col) & ('windBearing' not in col) & (
                    intercardinal_directions[d].format(2, 2) in col))]
        for col in point_cols:
            prefix = col.split('(')[0]
            df['{}_x_num_points({})'.format(prefix, d)] = df[col] * df[
                'num_points{}'.format(intercardinal_directions[d]).format(thresh, thresh)]

    intercardinal_directions_forecast = {'NE': '({}, {}, 1)',
                                         'NW': '({}, -{}, 1)',
                                         'SE': '(-{}, {}, 1)',
                                         'SW': '(-{}, -{}, 1)'}

    for d in intercardinal_directions_forecast:
        point_cols = [col for col in df.columns if (('num_points' not in col) & ('windBearing' not in col) & (
                    intercardinal_directions_forecast[d].format(2, 2) in col))]
        for col in point_cols:
            prefix = col.split('(')[0]
            df['{}_x_num_points({})(forecast)'.format(prefix, d)] = df[col] * df[
                'num_points{}'.format(intercardinal_directions[d]).format(thresh, thresh)]

    df.columns = [col.replace('num_points', 'FireStrength') for col in df.columns]

    predictors = [col for col in df.columns if col not in ['FireStrength', 'fire', 'target_date']]

    return df, predictors


def upsampling(df):
    df_fire = df[df.Target == 1]
    df = df[df.Target == 0]

    multiplier = math.ceil(df.shape[0] / df_fire.shape[0])

    for i in range(multiplier):
        df = df.append(df_fire)

    return df


def model(df, thresh):
    df, predictors = feature_engineering(df, thresh)

    train, test = df[df.date_id < date_id_start], df[df.date_id >= date_id_start]
    train = upsampling(train)
    del (df)

    X_train, y_train, X_test, y_test = train[predictors], train['Target'], test[predictors], test['Target']
    del (train)
    del (test)

    cols_to_exclude = ['date_id', 'Target', 'lat_ref_id', 'long_ref_id']

    model = RandomForestClassifier(n_estimators=100, max_features=10, random_state=0, verbose=2, n_jobs=64)
    model.fit(X_train[[col for col in X_train.columns if col not in cols_to_exclude]], y_train)
    del (y_train)

    filename = '{}/model_thresh{}.sav'.format(folder, thresh)
    if 'gs://' in common.GCS_PREFIX:
        fs = gcsfs.GCSFileSystem()
        with fs.open(filename, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    preds = list(model.predict_proba(X_test[[col for col in X_test.columns if col not in cols_to_exclude]])[::, 1])

    X_test['Pred'] = preds
    X_test['latitude'] = X_test.lat_ref_id.apply(lambda x: str(int(x))).map(lat_refs)
    X_test['longitude'] = X_test.long_ref_id.apply(lambda x: str(int(x))).map(long_refs)
    X_test['date'] = X_test.date_id.apply(lambda x: str(int(x))).map(date_ids)
    X_test['shap_dict_key'] = X_test.apply(
        lambda x: '{}_{}_{}'.format(x.date, round(x.latitude, 3), round(x.longitude, 3)), axis=1)

    X_test.to_csv('{}/X_test_thresh{}.csv'.format(folder, thresh))


if __name__ == '__main__':
    file = '{}/modeling_df.csv'.format(folder)
    df = pd.read_csv(file, index_col=0)
    df = df[df.date_id < 2955].dropna()

    df0or1 = df[df.thresh < 2].drop(columns=['thresh'])
    df2 = df[df.thresh == 2].drop(columns=['thresh'])
    del (df)

    model(df0or1, 1)
    model(df2, 2)