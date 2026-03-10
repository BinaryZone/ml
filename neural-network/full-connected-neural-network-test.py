import keras
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# load dataset
dataset = pd.read_csv('../data/breast_cancer_data.csv')

# get X
X = dataset.iloc[:, : -1]

# get Y
Y = dataset['target']

# split data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,  random_state=42)

# transfer to on hot vector
y_test_one = to_categorical(y_test, 2)

# normalization
sc = MinMaxScaler(feature_range=(0, 1))
x_test = sc.fit_transform(x_test)

# load model
model = load_model("model.h5")

# test
predict = model.predict(x_test)

y_pred = np.argmax(predict, axis=1)

# report
report = classification_report(y_test, y_pred, labels=[0, 1], target_names=['良性', '恶性'])
print(report)