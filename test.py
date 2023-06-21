import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from NNClassifier import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test dataset
test_dataset = pd.read_csv(TEST_FILE)
x = torch.tensor(np.array(test_dataset.iloc[0:, :])).reshape(-1, 1, 28, 28).to(device)
x = x / 255 * 2 - 1

x_batches = torch.split(x, 128)
y_predicted = torch.zeros(x.shape[0], 10).to(device)
print(y_predicted.shape)
model = MNISTClassifier().to(device)
for i in range(1, 3):
    model.load_state_dict(torch.load("result/model{}.pt".format(i)))

    model.eval()
    with torch.no_grad():
        y_predicted += torch.concatenate([model(x) for x in x_batches])

y_choice = torch.argmax(y_predicted, dim = 1).cpu().numpy().reshape(-1, 1)

plt.hist(y_choice, bins = 10)
plt.show()

y_id = np.arange(1, y_predicted.shape[0] + 1, 1, dtype = np.int64).reshape(-1, 1)

solution = np.concatenate((y_id, y_choice), axis = 1)

solution_df = pd.DataFrame(solution, columns = ["ImageID", "Label"])

solution_df.to_csv(SUBMISSION_FILE, index = False)

print(solution_df.iloc[:30, :])

sm_layer = nn.Softmax(0)
print(sm_layer(y_predicted[3]))
_, axes = plt.subplots(3, 10)
for i in range(30):
    x[i] = (x[i] - 1) / 2 * 255
    axes[i // 10, i % 10].imshow(x[i].cpu().reshape(28, 28), cmap = 'gray')

plt.show()