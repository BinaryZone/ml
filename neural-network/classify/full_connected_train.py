import keras
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.layers import Dense

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# load dataset
dataset = pd.read_csv('../../data/breast_cancer_data.csv')

# get X
X = dataset.iloc[:, : -1]

# get Y
Y = dataset['target']

# split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

# transfer to on hot vector
y_train_one = to_categorical(y_train, 2)
y_test_one = to_categorical(y_test, 2)

# normalization
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# build model
model = keras.Sequential()
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

# compile mode
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train_one, epochs=100, batch_size=32, verbose=2, validation_data=(x_test, y_test_one))
model.save("model.h5")

# draw loss photo
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('full connected neural network loss value')
plt.legend()
plt.show()

# draw accuracy photo
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('full connected neural network loss value')
plt.legend()
plt.show()