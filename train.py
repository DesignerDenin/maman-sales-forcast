import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from joblib import dump
from utils import features, encoding
import xgboost as xgb

def train(X):
    X.loc[(X.unit_sales<0),'unit_sales'] = 0
    X['unit_sales'] =  X['unit_sales'].apply(lambda x : np.log1p(x))
    X = X.replace(to_replace = [False, True], value = [0, 1])

    print('Generating sales data for feature engg')
    sales_data = X.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
    sales_data.columns = sales_data.columns.get_level_values(1)
    sales_data = sales_data.reset_index()
        
    print('Extracting features for training using sales information')
    x_lst, y_lst = [], []
    num_of_intervals = 8
    dates = [date(2016, 5, 31) + timedelta(days=7 * interval) for interval in range(num_of_intervals)]
    for train_date in dates:
        train_dict = features.sales(sales_data, train_date,'item_store')
        x_lst.append(pd.DataFrame(train_dict, index = [i for i in range(len(list(train_dict.values())[0]))]))
        y_lst.append(sales_data[[str(col)[0:10] for col in pd.date_range(train_date, periods = 1)]].values)

    train_item_store_x = pd.concat(x_lst, axis=0)
    train_y = np.concatenate(y_lst, axis=0)
    del x_lst, y_lst
    
    train_x = train_item_store_x.reset_index(drop = True)
    del train_item_store_x
    [train_x[col].update((train_x[col] - train_x[col].min()) / (train_x[col].max() - train_x[col].min())) for col in train_x.columns]
    print('Shape of train_x and corresponding train_y is {} & {}'.format(train_x.shape, train_y.shape))

    print('Creating and Saving Model')
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(train_x, train_y[:, 0])
    xgb_model.save_model('models/model.xgb')
       
    # Save the trained models
    print("Models saved successfully.")