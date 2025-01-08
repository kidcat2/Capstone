# [Capstone] 노인 사고 방지를 위한 CCTV 최적지 분석

전라북도 치매 노인 사고를 방지하기 위해 CCTV 최적지 도출

1. 전라북도를 100m x 100m 격자 단위로 분할 
2. 분할된 격자마다 나머지 데이터(cctv,hosptal..) 통합 후 최종 train.csv 생성. geopandas, GeoCoding을 이용
3. XGBoost 모델을 해당 데이터셋으로 training, StratifiedKFold을 이용해 정확도 향상
4. 모델로 예측한 CCTV 최적지와 실제 사고 위치를 시각화

- 실제 격자 수에 비해 사고자가 있는 격자가 매우 적은 데이터 불균형 현상을 해소할 필요가 있음.
- 데이터들의 결측값(인구 수에 많이 발생)을 해결해야 함.
- 기존 CCTV 감지 범위를 고려해야 함

---

### Install
NVIDIA GeForce RTX 3060

Python 3.9, geopands, shapely, matplotlib, 

```sh
conda create -n env python=3.9
conda install geopandas 
conda install shapely
conda install maplotlib
conda install openpyxl
```

### Folder

raw_data : 수집한 데이터

dataframes : raw_data --> 전처리 --> shp

train : XGBoost 모델 훈련을 위한 최종 데이터셋

result : 예측한 cctv 최적지 Geodataframe

opt : 기존 CCTV 감지 범위를 반경 30m,70m,100m로 설정한 후 최적지 도출 결과


```
┬─ result
│   ├─ need_cctv
│   │  └─ need_cctv.shp
│   └─ opt
│      ├─ opt_30
│      └─ opt_70
└─ data
    ├─ dataframes (shp)
    │   ├─ cctv
    │   │   ├─ cctv.shp
    │   │   └─...
    │   └─ ... (hospital, land, population)
    ├─ raw_data (csv, xlsx)
    │   ├─ cctv
    │    │   ├─ cctv.xlsx
    │   │   └─...
    │   └─ ... (hospital, land, population)
    ├─ train
    │   └─ LD_SIG_202307.shp 
    ├─ train.csv
    └─ malgun.ttf
 
```

### Data
모든 데이터셋의 기준은 전라북도이다.

1. cctv : cctv 개수, 위치, 성능
2. hospital, public-health : 병원, 보건소
3. land : 토지 이용 분류도
4. oldman : 년도별 노인 사고 사고자 수 (GT)
5. population : 고령 남녀별 인구 수
6. legal : 시군구 경계

**train.csv(격자 기준)** 
| Column | TYPE | 설명 |
|:---|:---|:---:|
| gid | INT | 지역(시군구) 고유 번호 |
| pop_man | FLOAT | 고령 남자 인구 수 | 
| pop_woman | FLOAT | 고령 여자 인구 수 | 
| tv_num | INT | cctv 개수 |
| tv_px200 | INT | 좋은 cctv 성능 개수 | 
| tv_px0 | INT | 안 좋은 cctv 성능 개수 | 
| oldman | INT | 노인 사고자 수 |
| XX_area | FLOAT |	XX 분류 토지 면적 |
| near_XX | FLOAT | 가장 가까운 XX 의 거리 |

### File

1. oldman_data.py : 년도별 사고자 수 데이터 요청 (url)
2. preprocess.py : 데이터 전처리 (결측값 제거 및 각종 전처리)
3. merge.py : 전처리된 각 데이터들을 통합해 train 데이터셋 생성
4. train.py : XGBoost Model training
5. opt_cctv.py : CCTV 최적지 도출
6. visualize.py : 필요한 부분 시각화 코드


### Train
```sh
python train.py
```

### Reference
- Reference.txt



