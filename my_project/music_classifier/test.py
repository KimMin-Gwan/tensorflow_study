
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import os
from sklearn.metrics import f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

warnings.filterwarnings("ignore")

plt.rc('font', family = 'Gothic')

train = pd.read_csv('./../../../music/train.csv')
test = pd.read_csv('./../../../music/test.csv')

train.head()
train.info()
#데이터 크기 : 25383개
#Columns : 12개
#genre column 데이터타입 : object(범주형)
print(train)
print(train.isnull().sum()) #빈칸이 있는지 확인

list_pop = []

list_pop = train[train['genre'].str.contains('Pop')]

# ------------------장르 라벨 ---------------------------
y_raw = train[['genre']]
print("학습시킬 데이터 라벨 : ", list(np.unique(y_raw)))
num_classes = len(np.unique(y_raw))

label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_raw)

print("숫자로 바뀐 장르 라벨 : ", np.unique(y_train))

print(y_train)



#y_train = keras.utils.to_categorical(y_train, num_classes)
#print(y_train)
#exit(1)

#X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.2)
# --------------------------------------------------------

# ------------ 나머지 데이터 딕셔너리화 -------------------

#	danceability	energy	key	loudness	speechiness	acousticness	instrumentalness	liveness	valence	tempo	duration	genre 

ds_pop = []
for i, rows in list_pop.iterrows():
    ds_pop.append([ rows['danceability'],rows['energy'],rows['key'],rows['loudness'],rows['speechiness'],
                    rows['acousticness'],rows['instrumentalness'],
                    rows['liveness'],rows['valence'],rows['tempo'],rows['duration'],rows['genre']
                    ])
    
ds_pop = np.array(ds_pop)
ds_pop = tf.expand_dims(ds_pop, axis = 0)
ds_pop = tf.expand_dims(ds_pop, axis = 0)
print(ds_pop)

discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4 * 256, input_shape=(100,) ), 
    tf.keras.layers.Conv1DTranspose(256, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1DTranspose(128, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1DTranspose(64, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])


#간모델 -> 제네레이터와 디스크리미네이터도 하나의 레이어이다.
GAN = tf.keras.models.Sequential([
  generator,
  discriminator
])

discriminator.compile(optimizer = 'adam', loss = 'binary_crossentropy')
discriminator.trainable = False

GAN.compile( optimizer= 'adam', loss = 'binary_crossentropy')



x_data = ds_pop


batch = 32
epoch = 100
for j in range(epoch):

  print('now epoch {}'.format(j))

  for i in range(50000//batch):

    if i % 100 == 0:
      print('now batch is {}'.format(i))

    #진짜 사진 학습
    real_picture = x_data[ i * 128 : (i + 1) * 128 ]
    marking_one = np.ones(shape=(len(list_pop), 1)) #1로 마킹

    discriminator.train_on_batch(real_picture, marking_one) #학습 
    loss1 = discriminator.train_on_batch(real_picture, marking_one) #오차 측정

    #가짜 사진 학습
    random_num = np.random.uniform(-1, 1, size = (11, 100) ) #가짜 사진만들때 사용할 숫자
    fake_picture = generator.predict(random_num) #가짜사진 만들기
    marking_zero = np.zeros(shape=(len(list_pop), 1)) #0으로 마킹

    discriminator.train_on_batch(fake_picture, marking_zero) #학습
    loss2 = discriminator.train_on_batch(fake_picture, marking_zero) #오차 측정

    #편향을 줄이려면 위의 두 데이터를 섞어서 사용해도됨
    random_num = np.random.uniform(-1, 1, size = (128, 100) )
    marking_one = np.ones(shape=(len(list_pop), 1))
    GAN.train_on_batch(random_num, marking_one )
    loss3 = GAN.train_on_batch(random_num, marking_one)

  print (f'이번 epoch의 최종 loss는 discriminator {loss1 + loss2} GAN {loss3}')

print('end of precedure')
predict_pic()




#model = tf.keras.models.load_model('./model1/')
submission = pd.read_csv("./../../../music/sample_submission.csv")

ds_test_batch = ds_test.batch(32)

model.evaluate(ds_valid_batch)

predict = model.predict(ds_test_batch)
print(len(predict))


count = 0
index = []

for i in range(len(predict)):
    results = np.argsort(predict[count])[::-1]
    labels = label_encoder.inverse_transform(results)
    index.append(labels[0])

    count += 1


submission["genre"] = index
submission.to_csv("./submit.csv", index = False)



print("end of precedure")
