import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

plt.rc('font', family='Gothic')

#test데이터를 분석하여 그 정보를 활용하는 것은 Data Leakage에 해당하기에 train데이터만 사용합니다
train = pd.read_csv('./../../../music/train.csv')

#데이터의 가장 윗부분 5개 출력
print(train.head())

#데이터의 파라미터 정보 출력
print(train.info())

fig, ax = plt.subplots(figsize = (10, 5), dpi = 100)

sns.countplot(x = train['genre'])
plt.xticks(rotation = 45)
#plt.show()

"""
fig, axs = plt.subplots(figsize = (30, 10), ncols = 4, nrows = 3, dpi = 100)
lm_features = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']

for i,j in enumerate(lm_features):
    row = int(i/4)
    col = i%4

    sns.histplot(x = train[j], ax = axs[row][col])

"""
#plt.show()


#pop장르가 포함된 행을 모두 뽑아내기

pop_list = []

pop_list = train[train['genre'].str.contains('Pop')]

print(pop_list[0:20])

#각각의 파라미터를 분석
print(pop_list.info())
#270개
"""
fig, axs = plt.subplots(figsize = (30, 10), ncols = 4, nrows = 3, dpi = 100)
lm_features = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']

for i,j in enumerate(lm_features):
    row = int(i/4)
    col = i%4

    sns.histplot(x = pop_list[j], ax = axs[row][col])

plt.show()

"""
import random as rd

def making(category):
    avg = np.mean(pop_list[category])
    #limit = pop_list[category].min() * 100
    limit = avg * 100

    rand = round(rd.random(), 3) * 100
    parameter = (rand % avg) * 0.5
    
    if parameter % 2 == 1:
        data = avg + parameter
    else:
        data = avg - parameter

    return round(data, 4)

fake_pop = []
for i in range(len(pop_list)):
    fake_song = []
    #	danceability	energy	key	loudness	speechiness	acousticness	instrumentalness	liveness	valence	tempo	duration	genre 
    fake_song.append('fake')
    fake_song.append(making('danceability'))
    fake_song.append(making('energy'))
    fake_song.append(making('key'))
    fake_song.append(making('loudness'))
    fake_song.append(making('speechiness'))
    fake_song.append(making('acousticness'))
    fake_song.append(making('instrumentalness'))
    fake_song.append(making('liveness'))
    fake_song.append(making('valence'))
    fake_song.append(making('tempo'))
    fake_song.append(making('duration'))
    fake_song.append('Pop')

    fake_pop.append(fake_song)

ds_pop = pd.DataFrame(fake_pop, columns=[
    'ID', 'danceability', 'energy', 'key', 'loudness', 'speechiness',
    'acousticness',	'instrumentalness', 'liveness', 'valence', 
    'tempo', 'duration', 'genre'
])

print(ds_pop)
print(pop_list)

new_ds = train.append(pop_list, ignore_index = True)


print(new_ds)
new_ds.to_csv('./../../../music/new_train.csv', index = False)

"""
1) 카테고리 별로 분석 및 데이터 생성하는 함수 생성
2) 1)에서 만든 함수로 각각의 카테고리에서 데이터 생성
3) 생성한 데이터를 리스트에 담기
4) 3) 의 과정을 1000번 반복
5) 만든 리스트를 train데이터 프레임에 입력
6) 5)에서 만든 데이터 프레임을 csv파일로 저장

1)
1. 해당 카테고리의 평균 분석
2. 카테고리 최대 뽑기
3. 최대값을 정수로 바꾸기 위해 100을 곱함
4. 랜덤 수를 생성
5. 랜덤 수도 100을 곱해서 정수로 바꿈
6. 랜덤 
 
"""


