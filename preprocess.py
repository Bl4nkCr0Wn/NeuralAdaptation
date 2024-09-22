import os
from shutil import copyfile, rmtree
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class AdaptationData(object):
    CLASS_NAMES = ['A', 'B']

    def __init__(self, work_dir):
        self._work_dir = work_dir
        self._classes_dir = os.path.join(self._work_dir, 'Classes')
        self._adaptation_dir = os.path.join(self._work_dir, 'Adaptation' + os.sep + 'Adaptation')
        self._training_dir = os.path.join(self._work_dir, 'training')
        self._validation_dir = os.path.join(self._work_dir, 'validation')
        self._test_dir = os.path.join(self._work_dir, 'test')

    def structure_raw_images(self, generated_images_dir):
        a_class_dir = os.path.join(self._classes_dir, self.CLASS_NAMES[0])
        b_class_dir = os.path.join(self._classes_dir, self.CLASS_NAMES[1])
        try:
            rmtree(self._classes_dir)
            rmtree(self._adaptation_dir)
        except OSError:
            pass
        os.makedirs(self._classes_dir, exist_ok=True)
        os.makedirs(a_class_dir, exist_ok=True)
        os.makedirs(b_class_dir, exist_ok=True)
        os.makedirs(self._adaptation_dir, exist_ok=True)

        for file in os.listdir(generated_images_dir):
            if file.endswith('.png') and 'image_135' in file:
                copyfile(os.path.join(generated_images_dir, file), os.path.join(a_class_dir, file))
            elif file.endswith('.png') and 'image_315' in file:
                copyfile(os.path.join(generated_images_dir, file), os.path.join(b_class_dir, file))
            elif file.endswith('.png'):
                copyfile(os.path.join(generated_images_dir, file), os.path.join(self._adaptation_dir, file))

        print('Created {} class with {} images.\n'
              'Created {} class with {} images.\n'
              'Created {} class with {} images\n'.format(
                                        a_class_dir, len(os.listdir(a_class_dir)),
                                        b_class_dir, len(os.listdir(b_class_dir)),
                                        self._adaptation_dir, len(os.listdir(self._adaptation_dir))))

    def split_data(self, split_size):
        try:
            rmtree(self._training_dir)
            rmtree(self._validation_dir)
            rmtree(self._test_dir)
        except OSError:
            pass

        os.makedirs(self._training_dir, exist_ok=True)
        os.makedirs(self._validation_dir, exist_ok=True)
        os.makedirs(self._test_dir, exist_ok=True)

        for c in self.CLASS_NAMES:
            os.makedirs(os.path.join(self._training_dir, c), exist_ok=True)
            os.makedirs(os.path.join(self._validation_dir, c), exist_ok=True)
            os.makedirs(os.path.join(self._test_dir, c), exist_ok=True)

            files = os.listdir(os.path.join(self._classes_dir, c))
            shuffled_files = random.sample(files, len(files))  # shuffles the data
            split = int(split_size * len(shuffled_files))  # the training split casted into int for numeric rounding
            train = shuffled_files[:split]  # training split
            split_valid_test = int(split + (len(shuffled_files) - split) / 2)
            validation = shuffled_files[split:split_valid_test]  # validation split
            test = shuffled_files[split_valid_test:]

            for element in train:
                copyfile(os.path.join(self._classes_dir, c, element),
                         os.path.join(self._training_dir, c, element))  # copy files into training directory

            for element in validation:
                copyfile(os.path.join(self._classes_dir, c, element),
                         os.path.join(self._validation_dir, c, element))  # copy files into validation directory

            for element in test:
                copyfile(os.path.join(self._classes_dir, c, element),
                         os.path.join(self._test_dir, c, element))  # copy files into test directory

            print('Created {} split data with:\n'
                  '{} training examples\n'
                  '{} validation samples\n'
                  '{} test samples.'.format(c,
                                            len(os.listdir(os.path.join(self._training_dir, c))),
                                            len(os.listdir(os.path.join(self._validation_dir, c))),
                                            len(os.listdir(os.path.join(self._test_dir, c)))))

    def create_generators(self, input_size, batch_size):
        train_datagen = ImageDataGenerator(rescale=1./255)
        validation_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1. / 255.)

        train_generator = train_datagen.flow_from_directory(
            self._training_dir,
            target_size=(input_size, input_size),
            batch_size=batch_size,
            class_mode='categorical')# class mode can be binary

        validation_generator = validation_datagen.flow_from_directory(
            self._validation_dir,
            target_size=(input_size, input_size),
            batch_size=batch_size,
            class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
            self._test_dir,
            target_size=(input_size, input_size),
            batch_size=batch_size,
            class_mode='categorical')

        return train_generator, validation_generator, test_generator

    def create_adaptation_generator(self, input_size):
        adaptation_datagen = ImageDataGenerator(rescale=1. / 255.)
        adaptation_generator = adaptation_datagen.flow_from_directory(
            directory=self._adaptation_dir,
            target_size=(input_size, input_size),
            batch_size=1,
            class_mode=None,
            shuffle=False)

        return adaptation_generator
