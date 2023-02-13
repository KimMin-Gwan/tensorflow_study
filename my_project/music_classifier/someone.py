import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

train = pd.read_csv("./../../../music/new_train.csv")
test = pd.read_csv("./../../../music/test.csv")

# X는 독립변수이므로 종속변수를 제거합니다. 또한 target 이외의 문자열 데이터를 제거합니다.
X = train.drop(["ID", "genre"], axis = 1)
# y는 종속변수로 값을 설정합니다.
y = train[['genre']]

# train에서와 마찬가지로 문자열이 포함된 특성은 제거합니다.
test = test.drop(["ID"], axis = 1)


# 학습데이터, 검증데이터 분리
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)

model = RandomForestClassifier(random_state = 42, n_estimators = 108)


# 학습데이터를 모델에 입력합니다.
model.fit(X_train,y_train)


val_pred = model.predict(X_valid)

# Macro f1 score을 사용하기 위해 average 인자 값을 "macro" 로 설정해줍니다.
print("현재 Macro F1 Score의 검증 점수는 {}입니다.".format(f1_score(val_pred, y_valid, average = "macro")))

# 최종 예측을 하기위해 test값을 입력합니다.
pred = model.predict(test)

# 제출 파일을 불러옵니다.
submission = pd.read_csv("./../../../music/sample_submission.csv")
submission["genre"] = pred
# 해당 파일을 다운로드 받아서 제출해주세요.
submission.to_csv("./submit4.csv", index = False)