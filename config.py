import net
#Environment Var
GENERATED_IMAGES_DIR = r'D:\CS\SummerProject\NeuralAdaptation\stylegan2-ada-pytorch\Results\generated_faces'
BASE_WORK_DIR = r'D:\CS\SummerProject\tmp'

# Images information
INPUT_VECTOR_SIZE = 512
INPUT_DIMENSION = 3

# Model hyperparameters
EPOCH_AMOUNT = 15
BATCH_SIZE = 32
SPLIT_SIZE = 0.7
LOSS_FUNCTION = 'categorical_crossentropy'
METRICS = ['accuracy']