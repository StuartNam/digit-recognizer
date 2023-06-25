import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from model import *
from config import *
from dataset import *
from trainer import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================#
#   DATA PREPARATION AND PRE-PROCESSING
#   - Load data
#   - Pre-process data
#   - Put data into Dataset object
# ========================================#
data = pd.read_csv(DATA_FOLDER + TRAIN_FILE)

x = np.array(data.iloc[0:, 1:])
x = torch.tensor(x)
x = x.reshape(-1, 1, 28, 28)
x = x.to(device)
x = x / 255 * 2 - 1

y = np.array(data.iloc[0:, 0])
y = torch.tensor(y)
y = y.to(device)

dataset = MNISTDataset(x, y)


# ========================================#
#   MODEL PREPARATION
#   - Import model from model.py
# ========================================#
model = ResNet(
    num_in_channels = 1,
    out_features = 10,
    depth = 30
)
model = model.to(device)

#=========================================#
#   LOSS FUNCTION PREPARATION
#   - Choose suitable loss function
#=========================================#
loss_fn = nn.CrossEntropyLoss()


#=========================================#
#   OPTIMIZER PREPARATION
#   - Choose suitable optimizer
#=========================================#
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = LRATE,
    weight_decay = WEIGHT_DECAY
)


#=========================================#
#   TRAINER PREPARATION
#   - Import from trainer.py
#   - Helps with training process including early stopping
#   - Helps with plotting and visualizing
#=========================================#
trainer = Trainer(
    model = model,
    dataset = dataset,
    loss_fn = loss_fn,
    optimizer = optimizer
)

"""
    TRAIN THE MODEL!
"""
trainer.train(
    num_epochs = NUM_EPOCHS
)