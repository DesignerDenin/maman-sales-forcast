from utils.common import get_data
from utils.common import average
from utils.common import weighted_moving_average
from datetime import timedelta
import numpy as np

def sales(data, end_date, prefix):
    '''
    This function generates feature dictionary for train, cv, test
    Features generated are:
    moving average, weighted moving average, standard deviation observed, 
    moving average of DOW, weighted moving average of DOW, having total sales day,
    last sales day in n days, first sales day in n days
    '''
    days_list = [3, 7, 16, 30, 60, 120] # These are the list of days used for extracting above mentioned features 
    feature_dict = {'{}_average_{}_days'.format(prefix, days): average(get_data(data, end_date, days, days).values)  for days in days_list}
    feature_dict.update({'{}_WMA_{}_days'.format(prefix, days): weighted_moving_average(get_data(data, end_date, days, days)) for days in days_list})
    feature_dict.update({'{}_std_{}_days'.format(prefix, days) : get_data(data, end_date, days, days).std(axis = 1).values for days in days_list})
    feature_dict.update({'{}_6avgdow_{}_days'.format(prefix, day) : get_data(data, end_date, 42 - day, 6, freq = '7D').mean(axis =1).values for day in range(7)})
    feature_dict.update({'{}_20avgdow_{}_days'.format(prefix, day) : get_data(data, end_date, 140 - day, 20, freq = '7D').mean(axis =1).values for day in range(7)})
    feature_dict.update({'{}_6WMAdow_{}_days'.format(prefix, day) : weighted_moving_average(get_data(data, end_date, 42 - day, 6, freq = '7D')) for day in range(7)})
    feature_dict.update({'{}_20WMAdow_{}_days'.format(prefix, day) : weighted_moving_average(get_data(data, end_date, 140 - day, 20, freq = '7D')) for day in range(7)})
    feature_dict.update({'{}_has_sale_day_{}'.format(prefix, days) : (get_data(data, end_date, days, days) > 0).sum(axis = 1).values for days in days_list})
    feature_dict.update({'{}_last_has_sale_day_{}'.format(prefix, days) : days - ((get_data(data, end_date, days, days) > 0) * np.arange(days)).max(axis = 1).values for days in days_list})
    feature_dict.update({'{}_first_has_sale_day_{}'.format(prefix, days) : ((get_data(data, end_date, days, days) > 0) * np.arange(days, 0, -1)).max(axis = 1).values for days in days_list})

    return feature_dict

def promo(data, class_array, family_array, item_array, store_array, store_state_array, store_city_array, store_type_array, store_cluster_array, class_family_df, end_date, prefix):
    '''
    This function uses promo information and categorical array to create features
    features created are---
    promo: total_promo, future promo information, promo days in 15 days, last promo in 15 days, first promo in 15 days
    categorical: class, item, store, family, city, state, clsuter, type 
    '''
    try:
        days_list = [16, 30, 60, 120]
        feature_dict = {'{}_totalpromo_{}_days'.format(prefix, days) : get_data(data, end_date, days, days).sum(axis = 1).values for days in days_list}
        feature_dict.update({'{}_totalpromoafter_{}_days'.format(prefix, days) : get_data(data, end_date + timedelta(days = 16), 16, days).sum(axis = 1).values for days in [5, 10, 15]})
        feature_dict.update({'{}_promo_{}_day'.format(prefix, abs(day - 1)): get_data(data, end_date, day, 1).values.ravel() for day in range(-15, 1)})
        feature_dict.update({'promo_day_in_15_days' : (get_data(data, end_date + timedelta(days=16), 15, 15) > 0).sum(axis = 1).values})
        feature_dict.update({'last_promo_day_in_15_days' : 15 - ((get_data(data, end_date + timedelta(days=16), 15, 15) > 0) * np.arange(15)).max(axis = 1).values})
        feature_dict.update({'firt_promo_day_in_15_days' : ((get_data(data, end_date + timedelta(days=16), 15, 15) > 0) * np.arange(15, 0, -1)).max(axis = 1).values})
        feature_dict.update({'class_{}'.format(i+1) : class_array[:, i] for i in range(class_array.shape[1])})
        feature_dict.update({'item_{}'.format(i+1) : item_array[:, i] for i in range(item_array.shape[1])})
        feature_dict.update({'store_{}'.format(i+1) : store_array[:, i] for i in range(store_array.shape[1])})
        feature_dict.update({'family_{}'.format(i+1) : family_array[:, i] for i in range(family_array.shape[1])})
        feature_dict.update({'city_{}'.format(i+1) : store_city_array[:, i] for i in range(store_city_array.shape[1])})
        feature_dict.update({'state_{}'.format(i+1) : store_state_array[:, i] for i in range(store_state_array.shape[1])})
        feature_dict.update({'cluster_{}'.format(i+1) : store_cluster_array[:, i] for i in range(store_cluster_array.shape[1])})
        feature_dict.update({'type_{}'.format(i+1) : store_type_array[:, i] for i in range(store_type_array.shape[1])})
        feature_dict.update({'perishable' : class_family_df['perishable'].values})
        
        return feature_dict
    except: 
        return None
