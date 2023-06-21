# General configuration
TRAIN_FILE = "data/train.csv"

TEST_FILE = "data/test.csv"

SUBMISSION_FILE = "result/submission.csv"

EVAL_FILE = "result/eval.xlsx"

MODEL_STATE_DICT_FILE = "result/model.pt"

# Training configuration
MODEL = ""
BATCH_SIZE = 128
LRATE = 1e-4
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-4
TRAIN_SIZE = 33600
VAL_SIZE = 8400

