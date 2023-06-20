import torch
import torch.nn as nn

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from NNClassifier import *
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

dataset = MNISTTrainDataset(x, y)

# Model
model = MNISTClassifier().to(device)

# Loss
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = LRATE,
    weight_decay = 0
)

# Trainer
trainer = Trainer(
    model,
    dataset,
    loss_fn,
    optimizer
)

# Train!
trainer.train(
    num_epochs = NUM_EPOCHS,
    lrate = LRATE
)