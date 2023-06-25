import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from model import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test dataset
test_dataset = None
test_x = None

test_y_predicted = None

model = DeepLearningModel().to(device)
model.load_state_dict(torch.load(MODEL_FOLDER + MODEL_STATE_DICT_FILE))

model.eval()
with torch.no_grad():
    test_y_predicted = model(test_x)

