from numpy import array
import pandas as pd
from tensorboard.summary.v1 import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import re

import preprocess
import simulation_net
import visualization
import config

def stage1():
    ################################# Stage One
    # preprocess data
    preprocess.structure_raw_images(config.GENERATED_IMAGES_DIR)

    train_generator, validation_generator,\
        test_generator, class_names = preprocess.create_data_generators(config.BASE_WORK_DIR,
                                                                        config.GENERATED_IMAGES_DIR,
                                                                        config.SPLIT_SIZE,
                                                                        config.INPUT_VECTOR_SIZE,
                                                                        config.BATCH_SIZE)

    # prepare model architecture
    model = simulation_net.alexnet(input_shape=(config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
                                   num_classes=len(class_names))
    model.compile(optimizer=Adam(), loss=config.LOSS_FUNCTION, metrics=config.METRICS)

    # train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=config.EPOCH_AMOUNT
    )

    model.save('alexnet_face_classifier_1.h5')
    model.evaluate(test_generator)
    results = pd.DataFrame(history.history)
    print(results.tail())
    return model


def stage2():
    ################################# Stage Two
    model = load_model('alexnet_face_classifier_1.h5')

    # Index moving degree images by degree
    adaptation_generator = preprocess.create_prediction_data_generator(config.GENERATED_IMAGES_DIR, config.INPUT_VECTOR_SIZE)
    images_by_degree = {}
    for i in range(len(adaptation_generator)):
        x = adaptation_generator.next()
        filename = adaptation_generator.filenames[i]
        idx = filename.find('image_') + len('image_')
        file_degree = int(re.search(r'\d+', filename[idx:]).group())
        images_by_degree[file_degree] = x

    # Train each image class alternatively
    degree_sequence = []
    for i in range(1, 62):# jumps of 1 (in experimentation, after exactly 60 degrees the families flip and at 61 they get fixed on class A)
        degree_sequence.append(135 + i)
        degree_sequence.append((315 + i)%360)

    for key in degree_sequence:
        x = images_by_degree[key]
        prediction = model.predict(x)[0]
        print('Class {} for degree {}'.format(prediction, key))
        y = array([prediction])
        model.fit(x, y, epochs=1)

    return model

def main():
    # stage1()

    stage2()
    return


if __name__ == '__main__':
    main()
