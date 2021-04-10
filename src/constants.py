import enum

DATASET_NAMES = {
    "CAISO" : "dc_2.csv",
    "NYISO" : "dc_4.csv",
    "ISONE" : "dc_3.csv",
    "PJM" : "dc_1.csv"
}

LOOK_BACK = 288
TRAIN_FRAC = 0.8
VALIDATION_FRAC = 0.1
TEST_FRAC = 0.1
FILE_NAME_LSTM_MODEL = 'lstm_model.h5'
FILENAME_CHECKPOINT = 'best_model.h5'
FILENAME_HISTORY = 'history.pickle'
EPOCH_SIZE = 10
NUM_BATCHES = 288

class LSTM_MODELS(enum.Enum):
    ONE_HIDDEN_LAYER = 1
