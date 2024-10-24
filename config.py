import os

#Environment Var
GENERATED_IMAGES_DIR = os.getcwd() + os.sep + 'stylegan2-ada-pytorch'+ os.sep +'Results'+ os.sep +'generated_faces'
BASE_WORK_DIR = os.getcwd() + os.sep + 'tmp'

# Images information
INPUT_VECTOR_SIZE = 512#224
INPUT_DIMENSION = 3
THETA_INCREMENT=10
THETA_AMOUNT=5
SPECIAL_DEGREES = [135, 315]

# Model hyperparameters
EPOCH_AMOUNT = 100
BATCH_SIZE = 32
SPLIT_SIZE = 0.8
LOSS_FUNCTION = 'binary_crossentropy'
METRICS = ['binary_accuracy']