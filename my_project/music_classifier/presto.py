import pandas as pd
import numpy as np

data = pd.read_csv('./../../../music/train.csv')

from sklearn import preprocessing

y_raw = data[['genre']]
print("학습시킬 데이터 라벨 : ", list(np.unique(y_raw)))
num_classes = len(np.unique(y_raw))

label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_raw)

print("숫자로 바뀐 장르 라벨 : ", np.unique(y_train))

print(y_train)

import tensorflow as tf

#data.pop('genre')

ds = tf.data.Dataset.from_tensor_slices( ( dict(data) , y_train) ) 

feature_columns = []

feature_columns.append(tf.feature_column.numeric_column("danceability")) 
feature_columns.append(tf.feature_column.numeric_column("energy")) 
feature_columns.append(tf.feature_column.numeric_column("key"))
feature_columns.append(tf.feature_column.numeric_column("loudness"))
feature_columns.append(tf.feature_column.numeric_column("speechiness"))
feature_columns.append(tf.feature_column.numeric_column("acousticness"))
feature_columns.append(tf.feature_column.numeric_column("instrumentalness"))
feature_columns.append(tf.feature_column.numeric_column("liveness"))
feature_columns.append(tf.feature_column.numeric_column("valence"))
feature_columns.append(tf.feature_column.numeric_column("tempo"))
feature_columns.append(tf.feature_column.numeric_column("duration"))

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

ds_batch = ds.batch(16)

model.fit(ds_batch, shuffle=True, epochs=10)

print("end of precedure")