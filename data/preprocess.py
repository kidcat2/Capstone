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

# plt 한글 폰트 
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


# 원본 데이터들 정리 및 데이터 프레임 생성

## 0. 좌표계 및 지역 번호 정리

#KLIS 좌표계 정의
epsg5173 = from_string("+proj=tmerc +lat_0=38 +lon_0=125.0028902777778 +k=1 +x_0=200000 +y_0=500000 +ellps=bessel +units=m +no_defs")  
epsg5174 = from_string("+proj=tmerc +lat_0=38 +lon_0=127.0028902777778 +k=1 +x_0=200000 +y_0=500000 +ellps=bessel +units=m +no_defs") 
epsg5175 = from_string("+proj=tmerc +lat_0=38 +lon_0=127.0028902777778 +k=1 +x_0=200000 +y_0=550000 +ellps=bessel +units=m +no_defs")  
epsg5176 = from_string("+proj=tmerc +lat_0=38 +lon_0=129.0028902777778 +k=1 +x_0=200000 +y_0=500000 +ellps=bessel +units=m +no_defs") 
epsg5177 = from_string("+proj=tmerc +lat_0=38 +lon_0=131.0028902777778 +k=1 +x_0=200000 +y_0=500000 +ellps=bessel +units=m +no_defs")  

# 지역 번호
gid_dict = {'Wansan' : 45111, 'Deokjin' : 45113, 'Gunsan' : 45130, 'Iksan' : 45140, 'Jeongeup' : 45180,
            'Namwon' : 45190, 'Gimje' : 45210, 'Wanju' : 45710, 'Jinan' : 45720, 'Muju' : 45730, 
            'Jangsu' : 45740, 'Imsil' : 45750, 'Sunchang' : 45770, 'Gochang' : 45790, 'Buan' : 45800}


### 1. <인구수 종합>

# 1-1. <격자 단위 인구수 종합>
pop_folderpath = 'raw_data/population'
pop_foldername = [f.name for f in os.scandir(pop_folderpath) if f.is_dir()]
population = []

for fname in tqdm(pop_foldername, total=len(pop_foldername), desc="인구 수 종합..."):
    gid = gid_dict[fname]

    f_man = gpd.read_file(pop_folderpath + '/' + fname + '/man/nlsp_021001011.shp', encoding='cp949')
    f_woman = gpd.read_file(pop_folderpath + '/' + fname + '/woman/nlsp_021001012.shp', encoding='cp949')

    f = f_man.merge(f_woman[['gid', 'val']], on='gid', how='left')
    f.drop(columns=['lbl'], inplace=True)
    f.rename(columns={'val_x': 'pop_man', 'val_y': 'pop_woman'}, inplace=True)
    f['gid'] = gid
    f['gid'] = f['gid'].astype(int)

    # 결측값 일부 처리
    f['pop_man'] = f['pop_man'].fillna(f['pop_woman'])
    f['pop_woman'] = f['pop_woman'].fillna(f['pop_man'])

    population.append(f)
    
population = pd.concat(population, ignore_index=True)
population = population.sort_values(by='gid').reset_index(drop=True)
population = population.to_crs(epsg=5179)

# 1-2. <시군구 법정경계 기준 인구수>
sigungu_path = 'raw_data/population_sigungu'
sigungu_name = [f.name for f in os.scandir(sigungu_path) if f.is_dir()]
sigungu_pop = []

for fname in tqdm(sigungu_name, total=len(sigungu_name), desc="시군구 기준 인구 수 종합..."):
    gid = gid_dict[fname]

    f_man = gpd.read_file(sigungu_path + '/' + fname + '/man/nlsp_002001011.shp', encoding='cp949')
    f_woman = gpd.read_file(sigungu_path + '/' + fname + '/woman/nlsp_002001012.shp', encoding='cp949')

    f = f_man.merge(f_woman[['gid', 'val']], on='gid', how='left')
    f.drop(columns=['lbl'], inplace=True)
    f.rename(columns={'val_x': 'pop_man', 'val_y': 'pop_woman'}, inplace=True)
    f['gid'] = gid
    f['gid'] = f['gid'].astype(int)

    sigungu_pop.append(f)
    
sigungu_pop = pd.concat(sigungu_pop, ignore_index=True)
sigungu_pop = sigungu_pop.sort_values(by='gid').reset_index(drop=True)
sigungu_pop = sigungu_pop.to_crs(epsg=5179)

# 1-3. <시군구 인구수를 활용한 격자 인구수 결측값 처리>
for _, row in sigungu_pop.iterrows():
    gid = row['gid'] 

    # gid에 속하는 지역 인구수 파악하기
    pop_notna = population[(population['gid'] == gid) & (population['pop_man'].notna())]
    pop_isna = population[(population['gid'] == gid) & (population['pop_man'].isna())]
    
    res_man = row['pop_man'] - pop_notna['pop_man'].sum()
    res_woman = row['pop_woman'] - pop_notna['pop_woman'].sum()

    man_ratio = res_man / pop_isna.shape[0]
    woman_ratio = res_woman / pop_isna.shape[0]

    population.loc[(population['gid'] == gid) & (population['pop_man'].isna()), 'pop_man'] = man_ratio.round()
    population.loc[(population['gid'] == gid) & (population['pop_woman'].isna()), 'pop_woman'] = woman_ratio.round()


save_gdf_to_shp(population, 'dataframes/population', 'population') # 저장


### 2. <토지 이용 현황도>

# 2-1. <토지 이용 현황도 종합>
Land_files = glob.glob(os.path.join('raw_data/land/Si_Gun_Gu/', '**', '*.shp'), recursive=True)
land = [] 

# tqdm(pop_foldername, total=len(pop_foldername), desc="인구 수 종합...")
for file in tqdm(Land_files, total=len(Land_files)):
    f = gpd.read_file(file)
    f.drop(columns=['ucb'], errors='ignore', inplace=True)
    land.append(f)
    
land = pd.concat(land, ignore_index=True)
land.crs = epsg5174
land = land.to_crs(epsg=5179)

# 2-2. <토지 분류 통합>
land['UCB'] = land['UCB'].astype(str)
land.rename(columns={'UCB': 'class'}, inplace=True)

land_ctos = {
    '1110': 'farm', '1120': 'farm', '1210': 'farm', '1220': 'farm', # 농지
    '2110': 'forest', '2120': 'forest', '2210': 'forest', '2220': 'forest', # 임지
    '2230': 'forest', '2310': 'forest', '2330': 'forest', '2320': 'forest', '2340': 'forest', # 임지
    '3110': 'res', '3120': 'res', '3130': 'res', '3140': 'res', # 주거지
    '3210': 'traffic', '3220': 'traffic', '3230': 'traffic', '3240': 'traffic', # 교통
    '3310': 'facility', '3320': 'facility', '3410': 'facility', '3420': 'facility', '3430': 'facility', '3540': 'facility', # 시설물
    '3440': 'facility', '3510': 'facility', '3520': 'facility', '3530': 'facility', '3550': 'facility', # 시설물
    '4110': 'water', '4120': 'water', '4210': 'water', '4310': 'water', '4320': 'water', '4410' : 'water' # 수계
}

land['class'] = land['class'].replace(land_ctos)

# 2-3. <결측값 처리>

# ratio : 전체에서 결측값을 제외한 분류별 비율 (시간이 너무 오래 걸래서 미리 계산)
ratio = { 'facility' : 0.013, 'farm' : 0.347 , 'forest' : 0.488, 'res' : 0.036, 'traffic' : 0.01 , 'water' : 0.102} 
land_nan = land[land['class'].isna()]
land_nanl = len(land_nan)

for k,v in ratio.items():
    ratio[k] = int(v * land_nanl)

for key, value in ratio.items() :
    nan_rows = land[land['class'].isna()]
    if value > len(nan_rows) : break
    
    sample_row = nan_rows.sample(n=value, random_state=42)
    land.loc[sample_row.index, 'class'] = key

land = land[~land['class'].isin([np.nan, 'farm', 'forest', 'water'])] # 최종

save_gdf_to_shp(land, 'dataframes/land', 'land') # 저장 

### 3. <CCTV>

# 3-1. cctv 종합
df = pd.read_excel('raw_data/cctv/Jeollabuk-do.xlsx')
df.drop(columns=['번호', '관리기관명', '소재지도로명주소', '소재지지번주소', '보관일수', '관리기관전화번호', '데이터기준일자', '촬영방면정보'], inplace=True) 
df.rename(columns={'설치연월' : 'date', '설치목적구분': 'purpose', '카메라대수' : 'number', 
                  '카메라화소수' : 'pixel', 'WGS84위도' : 'lat', 'WGS84경도' : 'lon'}, inplace=True)

df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.year
df['date'] = df['date'].fillna(0)

geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
cctv = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
cctv = cctv.to_crs("EPSG:5179")
cctv['date'] = cctv['date'].astype(int)

# 3-2. 목적(purpose) 이름 변경
purpose_mapping = {
    '교통단속': 'traffic',
    '교통정보수집': 'traffic_info',
    '기타': 'other',
    '다목적': 'multi',
    '생활방범': 'safety',
    '시설물관리': 'facility',
    '쓰레기단속': 'garbage',
    '어린이보호': 'child',
    '재난재해': 'disaster',
    '차량방범': 'car'
}

cctv['purpose'] = cctv['purpose'].replace(purpose_mapping)

save_gdf_to_shp(cctv, 'dataframes/cctv', 'cctv') # 저장 

### 4. <병원 현황>

hospital = gpd.read_file('raw_data/hospital/hospital.shp', encoding='cp949')
hospital.drop(columns=['번호', '병원_약국', '전화번호', '우편번호', 
                  '소재지주소', '입력주소', 'X', 'Y', 'CLSS', 
                  'PNU', '주소구분', '표준신주소', '표준구주소', 
                  '행정동코드', '행정동명', '법정동코드', '법정동명'], inplace=True)

hospital.crs = epsg5174
hospital = hospital.to_crs("EPSG:5179")
save_gdf_to_shp(hospital, 'dataframes/hospital', 'hospital') # 저장


### 5. <보건소 현황>
pub_health = gpd.read_file('raw_data/public-health/public-health.shp', encoding='cp949')
pub_health.drop(columns=['_순번', '시군명', '분류', '보건소명', '도로명주소', 
                         '전화번호', '입력주소', 'X', 'Y', 'CLSS', 'PNU', 
                         '주소구분', '표준신주소', '표준구주소', '우편번호', 
                         '행정동코드', '행정동명', '법정동코드', '법정동명',], inplace=True)

pub_health.crs = epsg5174
pub_health = pub_health.to_crs("EPSG:5179")
save_gdf_to_shp(pub_health, 'dataframes/pub_health', 'pub_health') # 저장

