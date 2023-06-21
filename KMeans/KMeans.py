import pandas as pd

import numpy as np

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import *

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)
# scores = pd.read_csv('result/scores.csv')

x = np.array(train_data.iloc[1:, 1:])
y = np.array(train_data.iloc[1:, 0])

x_test = np.array(test_data.iloc[0:, :])

x_train, x_val, y_train, y_val = train_test_split(
    x, y, 
    train_size = 0.8,
    random_state = 0
)

model = KMeans(
    n_clusters = 10,
    n_init = 10,
    random_state = 0,
    max_iter = 10000
)

x_train, x_val, y_train, y_val = train_test_split(
    x, y, 
    train_size = 0.8,
    random_state = 0
)

model.fit(x_train, y_train)
class2label = {}

for label in range(0, 10):
    for i in range(x_train.shape[0]):
        if y_train[i] == label:
            class_ = model.predict(x_train[i, :].reshape(1, -1))
            if class_[0] not in class2label:
                class2label[class_[0]] = label
                break

y_predict = model.predict(x_val)
    
for i in range(y_predict.shape[0]):
    y_predict[i] = class2label[y_predict[i]]

tmp = accuracy_score(y_predict, y_val)
print("Accuracy score: {acc:.2f}%".format(acc = accuracy_score(y_predict, y_val) * 100))

# scores._set_value(0, "KMeans", tmp)
# scores.to_csv("result/scores.csv", index = False)

y_test = model.predict(x_test)

tmp = np.concatenate((np.arange(1, x_test.shape[0] + 1).reshape(-1, 1), y_test.reshape(-1, 1)), axis = 1)

result = pd.DataFrame(
    data = np.concatenate((np.arange(1, x_test.shape[0] + 1).reshape(-1, 1), y_test.reshape(-1, 1)), axis = 1),
    columns = ["ImageID", "Label"]
)

result.to_csv(SUBMISSION_FILE, index = False)