import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# read data
dataset = pd.read_csv('../data/breast_cancer_data.csv')
print(dataset)

# get x
X = dataset.iloc[:, :-1]
print(X)

# get y
Y = dataset['target']
print(Y)

# split dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# Normalization
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)

# build model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# print param
print("w: ", lr.coef_)
print('b: ', lr.intercept_)

# predict
pre_result = lr.predict(x_test)
print(pre_result)

# print probe
pre_result_probe = lr.predict_proba(x_test)
print(pre_result_probe)

pre_list = pre_result_probe[:, 1]

# set thresholds
thresholds = 0.3

result = []
result_name = []

for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        result.append(1)
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性')

print(result)
print(result_name)

# print predict report
report = classification_report(y_test, result, labels=[0, 1], target_names=['恶性', '良性'])
print(report)