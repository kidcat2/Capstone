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

def grid_visualize(population): # 1. 인구 수 격자 그리드 시각화
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    population.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.7)

    plt.title("Visualization of Legal Areas")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def pred_visualize(train, pred): # 2. 예측한 사고자 수 시각화
    pred['predicted'] = pred['predicted'].abs().astype(int)
    train['pred'] = pred['predicted'].values

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    train.plot(column='pred', cmap='Reds', legend=True, ax=ax)

    ax.set_title('사고 예측', fontsize=14)
    ax.set_axis_off()  
    ax.legend(loc='upper right', fontsize=12)
    plt.show()

def opt_visualize(legal, oldman, opt): # 3. 실제 사고자 수 - 예측 CCTV 최적지 시각화
    buf_opt = gpd.GeoDataFrame({'geometry': opt.geometry.buffer(30)}, crs=opt.crs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    legal.plot(ax=ax, color='lightgrey', edgecolor='black', label='전라북도')
    oldman.plot(ax=ax, color='blue', marker='o', markersize=3, label='사고지')
    buf_opt.boundary.plot(ax=ax, color='red', label='예측한 CCTV 최적지')

    plt.legend()
    plt.title("CCTV 최적지")
    plt.show()

def all_visualize(train, legal, oldman, pred, cctv): # 4. 전체 시각화
    pred['predicted'] = pred['predicted'].abs().astype(int)
    train['pred'] = pred['predicted'].values

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    train.plot(column='pred', cmap='Reds', legend=True, ax=ax)
    cctv.plot(ax=ax, color='lightgray', marker='o', markersize=3, label='실제 CCTV 위치')
    oldman.plot(ax=ax, color='green', marker='o', markersize=10, label='실제 사고 위치')
    legal.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1, label='법적 경계')

    ax.set_title('사고 예측', fontsize=14)
    ax.set_axis_off()  # 축 제거
    ax.legend(loc='upper right', fontsize=12)
    ax.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.1, 1))
    plt.show()


# 코드

# 1. 인구 수 격자 그리드 시각화 
# population = gpd.read_file('data/train/merge4/merge4.shp')
# grid_visualize(population)

# 2. 예측한 사고자 수 시각화
# train = gpd.read_file('data/train/merge4/merge4.shp')
# pred = pd.read_csv('result/pred.csv')
# pred_visualize(train, pred)

# 3. 실제 사고자 수 - 예측 CCTV 최적지 시각화
# oldman = gpd.read_file('data/dataframes/oldman/oldman.shp')
# legal = gpd.read_file('data/raw_data/legal/LD_SIG_202307.shp')
# opt = gpd.read_file('result/opt/opt_30/opt.shp')
# opt_visualize(legal, oldman, opt)

# 4. 전체 시각화
# pred = pd.read_csv('result/pred.csv')
# oldman = gpd.read_file('data/dataframes/oldman/oldman.shp')
# legal = gpd.read_file('data/raw_data/legal/LD_SIG_202307.shp')
# train = gpd.read_file('data/train/merge4/merge4.shp')
# cctv = gpd.read_file('data/dataframes/cctv/cctv.shp')





