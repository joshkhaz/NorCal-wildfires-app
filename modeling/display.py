import pandas as pd
import itertools
import re
import numpy as np
import pickle
import gcsfs
import json
import multiprocessing
import shap

from identify_wildfires import identify_wildfires
import common

folder = common.folder

lat_refs, long_refs, date_ids = common.get_ref_dictionaries()
lats = list(lat_refs.values())
longs = list(long_refs.values())

county_df = pd.read_csv('{}/counties/ALL_county_coords.csv'.format(common.GCS_PREFIX), index_col = 0)

dates = pd.date_range('2021-01-01','2022-02-01')
dates = [d.strftime('%Y-%m-%d') for d in dates]

def create_shap_dict(X_test):

    for date in dates:
        print(date)

        manager = multiprocessing.Manager()
        shap_dict_date = manager.dict({})

        def get_shap_args(sample, index, explainer, shap_dict_key, pred, shape, date, first_index):
            print("START", index - first_index, shape, date)

            try:
                shap_values = explainer.shap_values(sample, check_additivity=True)
                p = shap.force_plot(explainer.expected_value[1], shap_values[1], features=sample.columns)
                base_value = explainer.expected_value[1]
                features = p.data['features']
                for i in features:
                    features[i]['value'] = sample.loc[index][i]
            except:
                shap_values = explainer.shap_values(sample, check_additivity=False)
                p = shap.force_plot(explainer.expected_value[1], shap_values[1], features=sample.columns)
                base_value = explainer.expected_value[1]
                features = p.data['features']
                try:
                    standardizer = sum([i['effect'] for i in list(features.values())]) / (
                                pred - base_value)  # maintain same relative effects when SHAP output slightly mismatches prediction
                except:
                    standardizer = sum([i['effect'] for i in list(features.values())]) / (pred - base_value + .0001)
                for i in features:
                    try:
                        features[i]['effect'] /= standardizer
                    except:
                        features[i]['effect'] /= (standardizer + .0001)
                    features[i]['value'] = sample.loc[index][i]

            shap_dict = {'baseValue': base_value,
                         'features': features}

            shap_dict_date[shap_dict_key] = shap_dict

            print("FINISH", index - first_index, shape, date)

        X_test = X_test.sort_values(by='date_id')
        X_test = X_test.reset_index(drop=True)

        cols_to_exclude = ['date_id', 'Target', 'lat_ref_id', 'long_ref_id', 'Pred', 'shap_dict_key', 'latitude',
                           'longitude', 'date', 'model']
        df = X_test[[col for col in X_test.columns if col not in cols_to_exclude]]

        filename1 = '{}/model_thresh1.sav'.format(folder)
        filename2 = '{}/model_thresh2.sav'.format(folder)

        if 'gs://' in common.GCS_PREFIX:
            fs = gcsfs.GCSFileSystem()
            model1 = pickle.load(fs.open(filename1, 'rb'))
            model2 = pickle.load(fs.open(filename2, 'rb'))
        else:
            model1 = pickle.load(open(filename1, 'rb'))
            model2 = pickle.load(open(filename2, 'rb'))

        explainer1 = shap.TreeExplainer(model1)
        explainer2 = shap.TreeExplainer(model2)

        process_list = []
        try:
            first_index = X_test[X_test.date==date].index[0]
        except:
            first_index = 0
        shape = X_test[X_test.date==date].shape[0]
        for index, row in X_test.iterrows():

            if (row.date == date):

                sample = df.loc[[index]]
                if row.model == 1:
                    process = multiprocessing.Process(target=get_shap_args,
                                                  args=[sample, index, explainer1, row.shap_dict_key, row.Pred, shape, date, first_index])
                elif row.model == 2:
                    process = multiprocessing.Process(target=get_shap_args,
                                                      args=[sample, index, explainer2, row.shap_dict_key, row.Pred, shape, date, first_index])
                process.start()
                process_list.append(process)

        for p in process_list:
            print(p)
            p.join()
        print('joined')

        if 'gs://' in common.GCS_PREFIX:
            fs = gcsfs.GCSFileSystem()
            with fs.open("{}/shap/shap_dict_{}.json".format(folder, date), "w") as outfile:
                json.dump(shap_dict_date.copy(), outfile)
        else:
            with open("{}/shap/shap_dict_{}.json".format(folder, date), "w") as outfile:
                json.dump(shap_dict_date.copy(), outfile)
        print('wrote')


def create_display_frame(X_test):

    df_pred = X_test[['latitude', 'longitude', 'date', 'Target', 'Pred']]

    df_pred['type'] = 'Prediction'

    df_historic = pd.read_csv('{}/labelled_1Mile.csv'.format(folder), index_col=0)
    df_historic = df_historic[['latitude', 'longitude', 'date']]
    df_historic['type'] = 'Historic'
    df_historic['Target'] = np.nan
    df_historic['Pred'] = np.nan

    df_display = df_historic.append(df_pred)
    del(df_historic)

    df_display = df_display[['latitude', 'longitude', 'date', 'type', 'Target', 'Pred']]
    df_display.to_csv('{}/historic_and_predictions.csv'.format(common.folder))

    df_pred = df_pred[['latitude', 'longitude', 'date']]

    return df_pred


def get_county(lat_min, lat_max, long_min, long_max):

    df = county_df[(county_df.latitude >= lat_min) & (county_df.latitude <= lat_max) & (county_df.longitude >= long_min) & (county_df.longitude <= long_max)]

    counties = df.county.unique()
    counties_to_display = []

    for c in counties:

        perc = df.county.value_counts()[c] / df.shape[0]
        if perc >= .01:
            counties_to_display.append(c.replace(" County", ""))

    return str(counties_to_display)


def create_id_frame(df):

    df = identify_wildfires(df)

    display_cols = ['wildfire_id', 'date', 'lat_min', 'lat_max', 'long_min', 'long_max', 'county']
    df_display = pd.DataFrame(columns=display_cols)

    for fire in df.wildfire_id.unique():

        df_fire = df[df.wildfire_id == fire]

        for date in df_fire.date.unique():

            df_date = df_fire[df_fire.date == date]

            lat_min = df_date.latitude.min() - common.square_side_degrees / 2
            lat_max = df_date.latitude.max() + common.square_side_degrees / 2
            long_min = df_date.longitude.min() - common.square_side_degrees / 2
            long_max = df_date.longitude.max() + common.square_side_degrees / 2
            county = get_county(lat_min, lat_max, long_min, long_max)

            df_new = pd.DataFrame([[fire, date, lat_min, lat_max, long_min, long_max, county]], columns=display_cols)

            df_display = df_display.append(df_new)

    df_display.to_csv('{}/fire_ids.csv'.format(common.folder))


if __name__=='__main__':

    X_test_0or1 = pd.read_csv('{}/X_test_thresh{}.csv'.format(folder, 1), index_col =0)
    X_test_0or1['model'] = 1

    X_test_2 = pd.read_csv('{}/X_test_thresh{}.csv'.format(folder, 2), index_col =0)
    X_test_2['model'] = 2

    X_test = X_test_0or1.append(X_test_2)

    print("creating shap")
    create_shap_dict(X_test)

    df = create_display_frame(X_test)
    create_id_frame(df)