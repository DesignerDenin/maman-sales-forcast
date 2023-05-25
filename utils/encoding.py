
import pandas as pd
import category_encoders as ce

def cat_encoding(cat_data, category):
    '''
    This function takes a df and the category and generate
    binary encoded vectors for the same
    '''
    encoder = ce.BinaryEncoder()
    return encoder.fit_transform(cat_data[category]).values

def generate_cat_features(sales_data):
    '''
    This function uses cat_encoding function and does binary encoding for all the categorical variables
    '''
    items_df = pd.read_csv('data/items.csv')
    stores_df = pd.read_csv('data/stores.csv')

    class_family_df = pd.DataFrame(sales_data['item_nbr']).merge(items_df[['item_nbr', 'class', 'family', 'perishable']], on = 'item_nbr', how = 'left')
    class_family_df['class'] = class_family_df['class'].astype('str')
    class_family_df['item_nbr'] = class_family_df['item_nbr'].astype('str')

    store_detail_df = pd.DataFrame(sales_data['store_nbr']).merge(stores_df[['store_nbr', 'state', 'city', 'type', 'cluster']], on = 'store_nbr', how = 'left')
    store_detail_df['store_nbr'] = store_detail_df['store_nbr'].astype('str')
    store_detail_df['cluster'] = store_detail_df['cluster'].astype('str')

    class_array = cat_encoding(class_family_df, 'class')
    family_array = cat_encoding(class_family_df, 'family')
    item_array = cat_encoding(class_family_df, 'item_nbr')


    store_array = cat_encoding(store_detail_df, 'store_nbr')
    store_state_array = cat_encoding(store_detail_df, 'state')
    store_city_array = cat_encoding(store_detail_df, 'city')
    store_type_array = cat_encoding(store_detail_df, 'type')
    store_cluster_array = cat_encoding(store_detail_df, 'cluster')

    return class_array, family_array, item_array, store_array, store_state_array, store_city_array, store_type_array, store_cluster_array, class_family_df
