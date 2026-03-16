import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from keras.models import load_model

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# load data
dataset = pd.read_csv('../../data/LBMA-GOLD.csv', index_col=[0])
# print(dataset)

# set train size
training_len = 1256 - 200

# get test dataset
test_set = dataset.iloc[training_len:, [0]]

# normalize data
sc = MinMaxScaler(feature_range=(0, 1))
test_set_scaled = sc.fit_transform(test_set)

x_test = []
y_test = []

for i in range(5, len(test_set_scaled)):
    x_test.append(test_set_scaled[i - 5: i, 0])
    y_test.append(test_set_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], 5, 1))

# load model
model = load_model('lstm_model.h5')

# predict
predicted = model.predict(x_test)

prediction = sc.inverse_transform(predicted)

real = sc.inverse_transform(test_set[5:])

rmse = sqrt(mean_squared_error(real, predicted))
mape = np.mean(np.abs((prediction - real) / prediction))

plt.plot(real, label='real')
plt.plot(prediction, label='prediction')
plt.title('LSTM Model')
plt.legend()
plt.show()