import requests
import os
import glob
import sys
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from fiona.crs import from_string
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import nearest_points, unary_union

import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def save_gdf_to_shp(gdf, dir, filename):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f"{filename}.shp")
    gdf.to_file(path)

def save_df_to_csv(df, dir, filename):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f"{filename}.csv")
    df.to_csv(path, index = False)

def scaler(df, opt='standard'):
    if opt=='standard':
        scale = StandardScaler()
    elif opt=='minmax':
        scale = MinMaxScaler()    
    scale.fit(df)
    return scale.transform(df)


data = pd.read_csv('data/train.csv')
train, test = train_test_split(data, test_size=0.2, random_state=42)

x_cols = ['gid','pop_man','pop_woman','tv_num','tv_px200','tv_px0','res_area','fac_area','near_hos','near_phos']
y_cols = ['oldman']

x_train = train[x_cols]
y_train = train[y_cols]
x_test = data[x_cols]#test[x_cols]
y_test = data[y_cols]#test[y_cols]

x_train = scaler(x_train, opt='standard')
y_train = scaler(y_train, opt='standard')

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

xgb_reg = xgb.XGBRegressor(
    booster='gbtree',
    colsample_bylevel=0.9,
    colsample_bytree=0.8,
    gamma=0,
    max_depth=10,
    min_child_weight=3,
    n_estimators=250,
    nthread=4,
    objective='reg:squarederror',
    random_state=42
)

fold_RMSE = []

for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, np.round(y_train))):
    print(f"Fold {fold + 1}/{n_splits}")
    
    X_train_t, X_val = x_train[train_idx], x_train[val_idx]
    y_train_t, y_val = y_train[train_idx], y_train[val_idx]
    
    xgb_reg.fit(
        X_train_t, y_train_t,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    y_val_pred = xgb_reg.predict(X_val)
    fmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Fold {fold + 1} RMSE: {fmse}")
    
    fold_RMSE.append(fmse)


print(f"MSE: {np.mean(fold_RMSE)}")

y_pred = xgb_reg.predict(x_test)
y_test = pd.DataFrame(y_test)
y_pred = pd.DataFrame(y_pred)

y_test.columns = ['actual']  
y_pred.columns = ['predicted']  

save_df_to_csv(y_test, 'result', 'target')
save_df_to_csv(y_pred, 'result', 'pred')

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")
