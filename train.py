import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from joblib import dump
from utils import features, encoding
import xgboost as xgb

def train(X):
    '''
    This function takes raw input, generate features using the raw input and trains the model.
    '''

    X.loc[(X.unit_sales<0),'unit_sales'] = 0
    X['unit_sales'] =  X['unit_sales'].apply(lambda x : np.log1p(x))
    X = X.replace(to_replace = [False, True], value = [0, 1])

    sales_data = X.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
    sales_data.columns = sales_data.columns.get_level_values(1)
    sales_data = sales_data.reset_index()

    train_promo = X.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(level=-1).fillna(0)
    train_promo.columns = train_promo.columns.get_level_values(1)

    promo_data = pd.concat([train_promo], axis=1)
    promo_data = promo_data.reset_index()
    del train_promo
    print('Data Collected!!!')
    print('Shape of sales and promo data is: {} and {}'.format(sales_data.shape, promo_data.shape))

    print('Generating categorical variables features')
    class_array, family_array, item_array, store_array, store_state_array, store_city_array, store_type_array, store_cluster_array, class_family_df = encoding.generate_cat_features(sales_data)
    print('Categorical variables features generated')

    print('Extracting features for training using sales information')
    x_lst, y_lst = [], []
    num_of_intervals = 8
    dates = [date(2017, 5, 31) + timedelta(days=7 * interval) for interval in range(num_of_intervals)]
    for train_date in dates:
        train_dict = features.sales(sales_data, train_date,'item_store')
        x_lst.append(pd.DataFrame(train_dict, index = [i for i in range(len(list(train_dict.values())[0]))]))
        y_lst.append(sales_data[[str(col)[0:10] for col in pd.date_range(train_date, periods = 16)]].values)

    train_item_store_x = pd.concat(x_lst, axis=0)
    train_y = np.concatenate(y_lst, axis=0)
    del x_lst, y_lst

    print('Extracting features for training using promo information')
    x_lst = []
    num_of_intervals = 8
    dates = [date(2017, 5, 31) + timedelta(days=7 * interval) for interval in range(num_of_intervals)]
    for train_date in dates:
        train_dict = features.promo(promo_data, class_array, family_array, item_array, store_array, store_state_array, store_city_array, store_type_array, store_cluster_array, class_family_df, train_date,'item_store')
        x_lst.append(pd.DataFrame(train_dict, index = [i for i in range(len(list(train_dict.values())[0]))]))

    train_item_store_x1 = pd.concat(x_lst, axis=0)
    del x_lst
    train_x = train_item_store_x.reset_index(drop = True).merge(train_item_store_x1.reset_index(drop = True), left_index=True, right_index=True)
    del train_item_store_x, train_item_store_x1
    [train_x[col].update((train_x[col] - train_x[col].min()) / (train_x[col].max() - train_x[col].min())) for col in train_x.columns]
    print('Shape of train_x and corresponding train_y is {} & {}'.format(train_x.shape, train_y.shape))

    trained_models = []
    for i in range(train_y.shape[1]):
        print('step{}'.format(i+1))
        xgb_model = xgb.XGBRegressor()
        xgb_model.fit(train_x, train_y[:, i])
        trained_models.append(xgb_model)
        xgb_model.save_model('models/model_{}.xgb'.format(i))
    
    # Save the trained models
    print("Models saved successfully.")