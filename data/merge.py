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

# plt 한글 폰트 적용하기
font_manager.fontManager.addfont("./malgun.ttf")
rcParams['font.family'] = 'Malgun Gothic'

# 저장용 코드
def save_gdf_to_shp(gdf, dir, filename):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f"{filename}.shp")
    gdf.to_file(path)

def save_df_to_csv(df, dir, filename):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f"{filename}.csv")
    df.to_csv(path, index = False)

# train 데이터셋 만들기
# train은 population 으로 설정

# 1. <train과 cctv 통합>

train = gpd.read_file('dataframes/population/population.shp')

train['tv_num'] = 0
train['tv_px200'] = 0
train['tv_px0'] = 0

cctv = gpd.read_file('dataframes/cctv/cctv.shp')
cctv.loc[cctv['pixel'] < 200, 'pixel'] = 0
cctv.loc[cctv['pixel'] >= 200, 'pixel'] = 1

cctv_in_train = gpd.sjoin(cctv, train, how="inner", predicate="within")

for idx, row in tqdm(train.iterrows(), total=len(train), desc="Merge train-cctv"):
    idx_cctv = cctv_in_train[cctv_in_train['index_right'] == idx]
    train.at[idx, 'tv_num'] = idx_cctv['number'].sum()
    train.at[idx, 'tv_px0'] = (idx_cctv['pixel'] == 0).sum()
    train.at[idx, 'tv_px200'] = (idx_cctv['pixel'] == 1).sum()
    

print(train.head())
save_gdf_to_shp(train, 'train/merge1', 'merge1')


# 2. <train과 oldman 통합>

train = gpd.read_file('train/merge1/merge1.shp')
oldman = gpd.read_file('dataframes/oldman/oldman.shp')
train['oldman'] = 0

oldman_in_train = gpd.sjoin(oldman, train, how='inner', predicate="within")
oldman_in_train = oldman_in_train.groupby('index_right')['count'].sum()
train['oldman'] = oldman_in_train.reindex(train.index, fill_value=0)

save_gdf_to_shp(train, 'train/merge2', 'merge2')


# 3. <train과 land 통합>

train = gpd.read_file('train/merge2/merge2.shp')
land = gpd.read_file('dataframes/land/land.shp')

train['res_area'] = 0.0
train['traf_area'] = 0.0
train['fac_area'] = 0.0
train['train_index'] = train.index

intersect = gpd.overlay(train, land, how='intersection')
intersect['area'] = intersect.geometry.area
save_gdf_to_shp(intersect, 'temp', 'temp1')

for idx, row in tqdm(intersect.iterrows(), total=len(intersect), desc="토지 이용 현황도 계산..."):
    train_idx = row['train_index']  
    area = row['area']              
    cls = row['class']       
    
        # 'res' class인 경우에만 res_area 업데이트
    if cls == 'res':
        train.loc[train['train_index'] == train_idx, 'res_area'] += area
    # 'traf' class인 경우에만 traf_area 업데이트
    elif cls == 'traffic':
        train.loc[train['train_index'] == train_idx, 'traf_area'] += area
    # 'fac' class인 경우에만 fac_area 업데이트
    elif cls == 'facility':
        train.loc[train['train_index'] == train_idx, 'fac_area'] += area


train.drop(columns=['train_index'], inplace=True)
train['res_area'] = (train['res_area'] / 1e6).round(3) # epsg5179 기준 1km^2 = 100,000 으로 표기됨.
train['traf_area'] = (train['traf_area'] / 1e6).round(3)
train['fac_area'] = (train['fac_area'] / 1e6).round(3)

save_gdf_to_shp(train, 'train/merge3', 'merge3')


# 4. <병원, 보건소 통합>

train = gpd.read_file('train/merge3/merge3.shp')
hos = gpd.read_file('dataframes/hospital/hospital.shp')
phos = gpd.read_file('dataframes/pub_health/pub_health.shp')

train['near_hos'] = -1.0
train['near_phos'] = -1.0

for idx, row in tqdm(train.iterrows(), total=len(train), desc="가까운 병원 및 보건소 거리 계산..."):
    # hos
    for _, hos_row in  hos.iterrows():
        if row['geometry'].intersects(hos_row['geometry']):
            train.at[idx, 'near_hos'] = 0.0
            break

    if row['near_hos'] == -1 :
        min_hos = float('inf')
        for _, hos_row in hos.iterrows():
            dist = row['geometry'].distance(hos_row['geometry'])
            if dist < min_hos:
                min_hos = dist
        train.at[idx, 'near_hos'] = round(min_hos, 2)

    ## phos
    for _, phos_row in phos.iterrows():
        if row['geometry'].intersects(phos_row['geometry']):
            train.at[idx, 'near_phos'] = 0.0
            break

    if row['near_phos'] == -1 :
        min_phos = float('inf')
        for _, phos_row in phos.iterrows():
            dist = row['geometry'].distance(phos_row['geometry'])
            if dist < min_phos:
                min_phos = dist
        train.at[idx, 'near_phos'] = round(min_phos, 2)
    
save_gdf_to_shp(train, 'train/merge4', 'merge4')
train.drop(columns=['geometry'], inplace=True)
save_df_to_csv(train, 'train', 'train')
