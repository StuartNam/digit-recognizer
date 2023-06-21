import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

from model import *
from config import *
from dataset import *
from trainer import *

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
    LOAD AND PROCESS DATA
    - x
        . (42000, 1, 28, 28), torch.float32
        . Scale from [0, 255] to [-1, 1]
    - y
        . (42000, ), torch.int64

"""
train_data = pd.read_csv(TRAIN_FILE)
x = torch.tensor(np.array(train_data.iloc[0:, 1:])).reshape(-1, 1, 28, 28).to(device)
x = x / 255 * 2 - 1
y = torch.tensor(np.array(train_data.iloc[0:, 0])).to(device)
dataset = MNISTTrainDataset(x, y)


"""
    PREPARE MODEL
    - Import from model.py
"""
model = Autoencoder().to(device)


"""
    PREPARE LOSS FUNCTION
    - L2 loss
"""
loss_fn = nn.MSELoss()


"""
    PREPARE OPTIMIZER
    - Adam optimizer
"""
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = LRATE,
    weight_decay = WEIGHT_DECAY
)


"""
    PREPARE TRAINER
    - Help with training processs
    - Import from trainer.py
"""
trainer = Trainer(
    model,
    dataset,
    loss_fn,
    optimizer
)

"""
    TRAIN THE MODEL!
"""
trainer.train(
    num_epochs = NUM_EPOCHS,
    lrate = LRATE
)