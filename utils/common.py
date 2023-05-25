import pandas as pd
import datetime
import numpy as np

def get_data(data, dt_end, days, period, freq='D'):
    '''
    This function gives us the selected columns based on a range of dates passed.
    '''
    try:
        return data[[str(col)[0:10] for col in pd.date_range(dt_end - datetime.timedelta(days = days), periods = period, freq = freq)]]
    except:
        print(end='')

def average(data):
    '''
    Here we are calculating simple average
    '''
    return np.mean(data, axis = 1)

def weighted_moving_average(data):
    '''
    This function computes weighted moving average, 
    higher weights are given to recent observations.
    '''
    data = data.values
    weight_len = data.shape[1]
    denom = (weight_len *(weight_len + 1))/2
    weights = [i+1/denom for i in range(weight_len)]
    data = average(data * weights)
    return data

def merge(predictions):
    test_data = pd.read_csv('data/test.csv')
    items_data = pd.read_csv('data/items.csv')
    stores_data = pd.read_csv('data/stores.csv')
    final_file = pd.merge(test_data[['id', 'item_nbr', 'store_nbr']], predictions[['id', 'unit_sales']], on='id')
    final_file['unit_sales'] = final_file['unit_sales'].round()
    final_file = pd.merge(final_file[['id', 'item_nbr', 'store_nbr', 'unit_sales']], items_data[['item_nbr', 'family']], on='item_nbr')
    final_file = pd.merge(final_file[['id', 'item_nbr', 'store_nbr', 'unit_sales', 'family']], stores_data[['store_nbr', 'state']], on='store_nbr')
    return final_file

def top_categories(predictions):
    selected = predictions[['family', 'unit_sales']]
    grouped_data = selected.groupby('family')['unit_sales'].sum().reset_index()
    sorted = grouped_data.sort_values(by='unit_sales', ascending=False)
    return sorted.head(10)

def top_stores(predictions):
    selected = predictions[['state', 'unit_sales']]
    grouped_data = selected.groupby('state')['unit_sales'].sum().reset_index()
    sorted = grouped_data.sort_values(by='unit_sales', ascending=False)

    return sorted.head(8)

def top_items(predictions):
    selected = predictions[['item_nbr', 'unit_sales']]
    sorted = selected.sort_values(by='unit_sales', ascending=False)
    return sorted.head(10)