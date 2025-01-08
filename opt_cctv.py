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
from matplotlib import rcParams
from matplotlib import font_manager

from fiona.crs import from_string
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import nearest_points, unary_union

import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

font_manager.fontManager.addfont("./malgun.ttf")
rcParams['font.family'] = 'Malgun Gothic'

def save_gdf_to_shp(gdf, dir, filename):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f"{filename}.shp")
    gdf.to_file(path)

def save_df_to_csv(df, dir, filename):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f"{filename}.csv")
    df.to_csv(path, index = False)


pred = pd.read_csv('result/pred.csv')
train = gpd.read_file('data/train/merge4/merge4.shp')
cctv = gpd.read_file('data/dataframes/cctv/cctv.shp')
legal = gpd.read_file('data/raw_data/legal/LD_SIG_202307.shp')

pred['predicted'] = pred['predicted'].abs().astype(int)
train['pred'] = pred['predicted'].values

need_cctv = train[train['pred'] > 0]

save_gdf_to_shp(need_cctv, 'result/need_cctv', 'need_cctv')

buf_cctv = gpd.GeoDataFrame({'geometry': cctv.geometry.buffer(100)}, crs=cctv.crs) # buffer 값 수정으로 여러가지 결과 도출
cctv_area = unary_union(buf_cctv['geometry'])

cctv_opt = []

# CCTV 최적지 계산
for _, row in tqdm(need_cctv.iterrows(), total=len(need_cctv), desc="CCTV 최적지 계산"):
    poly = row.geometry
    dif_poly = poly.difference(cctv_area)

    if dif_poly.is_empty:
        continue

    opt = dif_poly.centroid
    if any(buf_cctv.contains(opt)):
        continue
    
    cctv_opt.append({'geometry': opt, 'pred': row['pred']})


cctv_opt = gpd.GeoDataFrame(cctv_opt, geometry='geometry', crs=train.crs)
save_gdf_to_shp(cctv_opt, 'result/opt/opt_100', 'opt_100') 



