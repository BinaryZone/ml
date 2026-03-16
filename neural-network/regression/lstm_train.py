import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
import keras

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# load data
dataset = pd.read_csv('../../data/LBMA-GOLD.csv', index_col=[0])
# print(dataset)

# set train size
training_len = 1256 - 200

# get train dataset
training_set = dataset.iloc[0:training_len, [0]]
print(training_set)

# get test dataset
test_set = dataset.iloc[training_len:, [0]]

# normalize data
sc = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

# set train and test label
x_train = []
y_train = []

x_test = []
y_test = []

for i in range(5, len(train_set_scaled)):
    x_train.append(train_set_scaled[i - 5: i, 0])
    y_train.append(train_set_scaled[i, 0])

# print(x_train)
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], 5, 1))
# print(x_train)

for i in range(5, len(test_set_scaled)):
    x_test.append(test_set_scaled[i - 5: i, 0])
    y_test.append(test_set_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], 5, 1))

# build model
model = keras.Sequential()
model.add(LSTM(units=80, return_sequences=True, activation='relu'))
model.add(LSTM(units=100, return_sequences=False, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(units=1))

# compile model
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))

# train
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

# save model
model.save('lstm_model.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('model loss')
plt.legend()
plt.show()