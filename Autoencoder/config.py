import torch

# Training configuration
MODEL = ""

BATCH_SIZE = 128

LRATE = 1e-4

NUM_EPOCHS = 200

WEIGHT_DECAY = 1e-4

TRAIN_SIZE = 33600

VAL_SIZE = 8400

# General configuration
# TRAIN_FILE = "../data/train.csv"

# TEST_FILE = "../data/test.csv"

# SUBMISSION_FILE = "result/submission.csv"

# EVAL_FILE = "../eval.xlsx"

# MODEL_STATE_DICT_FILE = "result/model.pt"

# DTYPE = torch.float32

TRAIN_FILE = "data/train.csv"

TEST_FILE = "data/test.csv"

SUBMISSION_FILE = "result/submission.csv"

EVAL_FILE = "eval.xlsx"

MODEL_STATE_DICT_FILE = "result/model.pt"

DTYPE = torch.float32