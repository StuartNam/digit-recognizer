import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import math

from DLModel import *
from config import *
from dataset import *
from trainer import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
train_data = pd.read_csv(TRAIN_FILE)

x = torch.tensor(np.array(train_data.iloc[0:, 1:])).reshape(-1, 1, 28, 28).to(device)

# - x: (42000, 1, 28, 28), scale from [0, 255] to [-1, 1]
x = x / 255 * 2 - 1

y = torch.tensor(np.array(train_data.iloc[0:, 0])).to(device)

# - y: (42000, 10), one hot encoded
#y = torch.nn.functional.one_hot(y, num_classes = 10).to(torch.float32)

dataset = MNISTTrainDataset(x, y)
# train_dataset, val_dataset = random_split(dataset, [TRAIN_SIZE, VAL_SIZE])

# train_dataloader = DataLoader(
#     train_dataset,
#     batch_size = BATCH_SIZE,
#     shuffle = True
# )

# Visualization
# fig, axes = plt.subplots(1, 10)
# for i, axes in enumerate(axes.flat):
#     axes.imshow(x[i], cmap = 'gray')

# plt.show()

# Model
model = MNISTClassifier()

# Loss
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = LRATE,
    weight_decay = 0
)

# Training loop
trainer = Trainer(
    model,
    dataset,
    loss_fn,
    optimizer
)

trainer.train(
    num_epochs = NUM_EPOCHS,
    lrate = LRATE
)