import torch

import numpy as np
import pandas as pd

from config import *
from NNClassifier import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test dataset
test_dataset = pd.read_csv(TEST_FILE)
x = torch.tensor(np.array(test_dataset.iloc[0:, 1:])).reshape(-1, 1, 28, 28).to(device)

model = MNISTClassifier().to(device)
model.load_state_dict(torch.load(MODEL_STATE_DICT_FILE).to(device))

y_predicted = model(x)
y_choice = torch.argmax(y_predicted, dim = 1).numpy()
y_id = np.linspace(1, y_predicted.shape[0])

solution = np.concatenate((y_id, y_choice), axis = 1)
print(solution.shape)