import torch

#======================================================================================#
#   MODEL CONFIGURATION
#   - Model parameters
#======================================================================================#
DTYPE = torch.float32


#======================================================================================#
#   TRAINING CONFIGURATION
#   - Training hyperparameters
#======================================================================================#
TRAIN_SIZE = 33800
VAL_SIZE = 8200

NUM_EPOCHS = 100
BATCH_SIZE = 128
LRATE = 1e-3
WEIGHT_DECAY = 1e-4


#======================================================================================#
#   PROJECT CONFIGURATION
#   - File paths
#======================================================================================#
DATA_FOLDER = "data/"
RESULT_FOLDER = "result/"
MODEL_FOLDER = "model/"

TRAIN_FILE = "train.csv"
TEMPORARY_FILE = "temporary.pt"
MODEL_STATE_DICT_FILE = "model.pt"