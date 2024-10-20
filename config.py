import os

#Environment Var
GENERATED_IMAGES_DIR = os.getcwd() + r'\stylegan2-ada-pytorch\Results\generated_faces'
BASE_WORK_DIR = os.getcwd() + r'\tmp'

# Images information
INPUT_VECTOR_SIZE = 512#224
INPUT_DIMENSION = 3

# Model hyperparameters
EPOCH_AMOUNT = 100
BATCH_SIZE = 32
SPLIT_SIZE = 0.8
LOSS_FUNCTION = 'binary_crossentropy'
METRICS = ['binary_accuracy']