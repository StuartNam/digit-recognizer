import pandas as pd

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import *

"""
    LOAD AND PROCESS DATA
"""
train_dataset = pd.read_csv(TRAIN_FILE)

x = np.array(train_dataset.iloc[:, 1:]) / 255 * 2 - 1
y = np.array(train_dataset.iloc[:, 0])

x_train, x_val, y_train, y_val = train_test_split(
    x, y, 
    train_size = TRAIN_PROPORTION,
    random_state = 0
)

print("=============================")
print("DATA INSIGHT")
print("=============================")
print("- x_train: {}, y_train: {}".format(x_train.shape, y_train.shape))
print("- x_val: {}, y_val: {}".format(x_val.shape, y_val.shape))

test_dataset = pd.read_csv(TEST_FILE)
x_test = np.array(test_dataset) / 255 * 2 - 1

print("- x_test: {}".format(x_test.shape))
print()


"""
    SET UP MODEL
"""
model = SVC(
    random_state = 0
)

print("=============================")
print("MODEL")
print("=============================")
print("Support Vector Classifier")
# print("- n_neighbors = {}".format(NUM_NEIGHBORS))
print()


"""
    TRAIN MODEL!
"""
model.fit(x, y)


"""
    EVALUATE ON TRAIN AND VAL DATASET
"""
y_predicted_train = model.predict(x_train)
y_predicted_val = model.predict(x_val)

print("=============================")
print("EVALUATION")
print("=============================")
print("Accuracy on:")
print("- Train dataset: {}%".format(round(accuracy_score(y_train, y_predicted_train) * 100, 2)))
print("- Val dataset: {}%".format(round(accuracy_score(y_val, y_predicted_val) * 100, 2)))
print()

"""
    TEST MODEL
"""
y_predicted_test = model.predict(x_test).reshape(-1, 1)
y_predicted_test_no = np.arange(1, x_test.shape[0] + 1).reshape(-1, 1)

solution = np.concatenate([y_predicted_test_no, y_predicted_test], axis = 1)

solution_df = pd.DataFrame(
    data = solution,
    columns = ["ImageID", "Label"]
)

solution_df.to_csv(SUBMISSION_FILE, index = False)