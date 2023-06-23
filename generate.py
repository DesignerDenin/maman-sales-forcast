import pandas as pd
import numpy as np
from datetime import date
from joblib import load
import xgboost as xgb
from utils import features, encoding

def generate(X, year, month, day):
    print('Generating sales data for feature engg')
    X.loc[(X.unit_sales<0),'unit_sales'] = 0
    X['unit_sales'] =  X['unit_sales'].apply(lambda x : np.log1p(x))
    X = X.replace(to_replace = [False, True], value = [0, 1])

    sales_data = X.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
    sales_data.columns = sales_data.columns.get_level_values(1)
    sales_data = sales_data.reset_index()

    print('Extracting features for prediction on test data using sales information')
    test_date = date(year, month, day)
    test_dict = features.sales(sales_data, test_date, 'item_store')
    test_item_store_x = pd.DataFrame(test_dict, index = [i for i in range(len(list(test_dict.values())[0]))])
    
    test_x = test_item_store_x.reset_index(drop = True)
    [test_x[col].update((test_x[col] - test_x[col].min()) / (test_x[col].max() - test_x[col].min())) for col in test_x.columns]

    trained_models = []
    xgb_model = xgb.Booster()
    xgb_model.load_model('models/model.xgb')
    trained_models.append(xgb_model)
    
    test_pred = []
    test_dmatrix = xgb.DMatrix(test_x) 
    
    test_pred.append(trained_models[0].predict(test_dmatrix))

    print('Prediction done on test data, generating final output')
    y_test = np.array(test_pred).transpose()

    pred_df = pd.DataFrame(y_test, columns = pd.date_range(str(year) + '-' + str(month) + '-' + str(day), periods=1))
    pred_df = sales_data[['item_nbr', 'store_nbr']].merge(pred_df, left_index=True, right_index=True)
    pred_df = pred_df.melt(id_vars=['item_nbr', 'store_nbr'], var_name='date', value_name='unit_sales')
    pred_df['unit_sales'] = pred_df['unit_sales'].apply(lambda x : np.expm1(x))

    print('Prediction df generated, loading test file and merging results with test file')
    test_df = pd.read_csv('data/test.csv')
    test_df['date'] = pd.to_datetime(str(year) + '-' + str(month) + '-' + str(day))

    test_df = test_df.merge(pred_df[['item_nbr', 'store_nbr', 'date', 'unit_sales']], on = ['date', 'store_nbr', 'item_nbr'], how = 'left')
    test_df['unit_sales'] = test_df['unit_sales'].clip(lower = 0)
    test_df = test_df.fillna(0)

    return test_df