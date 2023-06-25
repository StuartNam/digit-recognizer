import torch

# Training configuration
MODEL = ""

BATCH_SIZE = 128

LRATE = 1e-4

NUM_EPOCHS = 50

WEIGHT_DECAY = 0

TRAIN_SIZE = 33600

VAL_SIZE = 8400

NUM_FEATURES = 30
# Classifier configuration
TRAIN_PROPORTION = 0.9999

NUM_NEIGHBORS = 3

NUM_ESTIMATORS = 101

# General configuration
TRAIN_FILE = "../data/train.csv"

TEST_FILE = "../data/test.csv"

SUBMISSION_FILE = "result/submission.csv"

EVAL_FILE = "../eval.xlsx"

MODEL_STATE_DICT_FILE = "result/model.pt"

DTYPE = torch.float32

# Best
# n_neighbors = 5, num_features = 200