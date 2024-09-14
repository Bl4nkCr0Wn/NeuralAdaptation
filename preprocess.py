import os
from shutil import copyfile
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def structure_raw_images(generated_images_dir):
    classes_dir = os.path.join(generated_images_dir, 'Classes')
    a_class_dir = os.path.join(classes_dir, 'A')
    b_class_dir = os.path.join(classes_dir, 'B')
    adaptation_dir = os.path.join(generated_images_dir, 'Adaptation' + os.sep + 'Adaptation')
    os.makedirs(classes_dir, exist_ok=True)
    os.makedirs(a_class_dir, exist_ok=True)
    os.makedirs(b_class_dir, exist_ok=True)
    os.makedirs(adaptation_dir, exist_ok=True)

    for file in os.listdir(generated_images_dir):
        if file.endswith('.png') and 'image_135' in file:
            copyfile(os.path.join(generated_images_dir, file), os.path.join(a_class_dir, file))
        elif file.endswith('.png') and 'image_315' in file:
            copyfile(os.path.join(generated_images_dir, file), os.path.join(b_class_dir, file))
        elif file.endswith('.png'):
            copyfile(os.path.join(generated_images_dir, file), os.path.join(adaptation_dir, file))

def _split_data(main_dir, training_dir, validation_dir, test_dir, split_size):
    """
    Splits the data into train validation and test sets

    Args:
    main_dir (string):  path containing the images
    training_dir (string):  path to be used for training
    validation_dir (string):  path to be used for validation
    test_dir (string):  path to be used for test
    split_size (float): size of the dataset to be used for training
    """
    files = []
    for file in os.listdir(main_dir):
        if os.path.getsize(os.path.join(main_dir, file)):  # check if the file's size isn't 0
            files.append(file)  # appends file name to a list

    shuffled_files = random.sample(files, len(files))  # shuffles the data
    split = int(split_size * len(shuffled_files))  # the training split casted into int for numeric rounding
    train = shuffled_files[:split]  # training split
    split_valid_test = int(split + (len(shuffled_files) - split) / 2)
    validation = shuffled_files[split:split_valid_test]  # validation split
    test = shuffled_files[split_valid_test:]

    for element in train:
        copyfile(os.path.join(main_dir, element),
                 os.path.join(training_dir, element))  # copy files into training directory

    for element in validation:
        copyfile(os.path.join(main_dir, element),
                 os.path.join(validation_dir, element))  # copy files into validation directory

    for element in test:
        copyfile(os.path.join(main_dir, element), os.path.join(test_dir, element))  # copy files into test directory

    print("Split sucessful!")

def create_data_generators(base_dir, generated_images_dir, split_size, input_size, batch_size):
    """
    Input dir expected to contain "Images" folder containing image classes in different folders, for example: $PATH/Images/class_A, $PATH/Images/class_B
    """
    # Define data path
    data_dir = os.path.join(generated_images_dir, 'Classes')
    base_work_dir = os.path.join(base_dir, 'tmp')
    training_dir = os.path.join(base_work_dir, 'training')
    validation_dir = os.path.join(base_work_dir, 'validation')
    test_dir = os.path.join(base_work_dir, 'test')
    class_names = ['A', 'B']

    os.makedirs(base_work_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for c in class_names:
        os.makedirs(os.path.join(training_dir, c), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)

    for c in class_names:
        _split_data(os.path.join(data_dir, c),
                   os.path.join(training_dir, c),
                   os.path.join(validation_dir, c),
                   os.path.join(test_dir, c),
                   split_size)
        print('Created {} split data with:\n'
              '{} training examples\n'
              '{} validation samples\n'
              '{} test samples.'.format(c, len(os.listdir(os.path.join(training_dir, c))),
                                        len(os.listdir(os.path.join(validation_dir, c))),
                                        len(os.listdir(os.path.join(test_dir, c)))))

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1. / 255.)

    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical')# class mode can be binary

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator, test_generator, class_names

def create_prediction_data_generator(generated_images_dir, input_size):
    adaptation_dir = os.path.join(generated_images_dir, 'Adaptation')
    adaptation_datagen = ImageDataGenerator(rescale=1. / 255.)
    adaptation_generator = adaptation_datagen.flow_from_directory(
        directory=adaptation_dir,
        target_size=(input_size, input_size),
        batch_size=1,
        class_mode=None,
        shuffle=False)

    return adaptation_generator