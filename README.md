# [Capstone] 노인 사고 방지를 위한 CCTV 최적지 분석

공공 데이터를 기반으로 Python, QGIS, Machine Algorithm을 이용해 노인 사고 방지를 위한 CCTV 최적지 도출 프로그램

# Install
NVIDIA GeForce RTX 3060

QGIS, Python 3.9

geopands, shapely, maplotlib, 

```sh
conda create -n env python=3.9
conda install geopandas 
conda install shapely
conda install maplotlib
conda install openpyxl
```

# Folder

data : original data
dataframe : original data --> preprocessing --> .shp / .csv

```
┬─ save_model
│   ├─ 
│   └─ 
└─ data
    ├─ dataframe
    │   ├─ population
    │   │   └─ population.shp
    │   ├─ oldman
    │   │   └─ oldman.csv       
    │   └─ .....  
    ├─ population
    │   ├─ JeonJu-Deokjin-gu
    │   │   ├─ LD_SIG_202307.shp
    │   │   ├─ LD_SIG_202307.shx
    │   │   └─ ...
    │   └─ ... (City, County, District)
    ├─ legal
    │   └─ LD_SIG_202307.shp
    ├─ .....
    └─ explain.txt

```

# Train
```sh
python train.py
```

