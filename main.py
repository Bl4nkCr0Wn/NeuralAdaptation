
from numpy import array
import pandas as pd
from tensorflow.keras.models import load_model
import re

import preprocess
import net
import config

def prepare_new_data():
    data = preprocess.AdaptationData(config.BASE_WORK_DIR)
    data.structure_raw_images(config.GENERATED_IMAGES_DIR)
    data.split_data(config.SPLIT_SIZE)
    return data

def load_data():
    data = preprocess.AdaptationData(config.BASE_WORK_DIR)
    return data

def train_new_model(data):
    train_generator, validation_generator,\
        test_generator = data.create_generators(config.INPUT_VECTOR_SIZE, config.BATCH_SIZE)

    # prepare model architecture
    model = net.AdaptationNet.create_model((config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
                                           len(data.CLASS_NAMES),
                                           config.LOSS_FUNCTION,
                                           config.METRICS)

    # train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=config.EPOCH_AMOUNT
    )

    model.evaluate(test_generator)
    results = pd.DataFrame(history.history)
    return model, results

def rotate_fit(model, data, angle_range, angle_range_start = 1):
    # Index moving degree images by degree
    adaptation_generator = data.create_adaptation_generator(config.INPUT_VECTOR_SIZE)
    images_by_degree = {}
    for i in range(len(adaptation_generator)):
        x = adaptation_generator.next()
        filename = adaptation_generator.filenames[i]
        idx = filename.find('image_') + len('image_')
        file_degree = int(re.search(r'\d+', filename[idx:]).group())
        images_by_degree[file_degree] = x

    # Train each image class alternatively
    degree_sequence = []
    for i in range(angle_range_start, angle_range_start + angle_range):
        degree_sequence.append(135 + i)
        degree_sequence.append((315 + i)%360)

    for key in degree_sequence:
        x = images_by_degree[key]
        prediction = model.predict(x)[0]
        print('Class {} for degree {}'.format(prediction, key))
        y = array([prediction])
        model.fit(x, y, epochs=1)

    return model

def test_rotated_model(model, data):
    # Test original face classes classification after rotation.
    train_generator, validation_generator, \
        test_generator = data.create_generators(config.INPUT_VECTOR_SIZE, config.BATCH_SIZE)
    print('Evaluate model on training data:')
    model.evaluate(train_generator)

def main():
    # data = prepare_new_data()
    # model, history = train_new_model(data)
    # model.save('alexnet_face_classifier_5.h5')

    data = load_data()
    model = load_model('alexnet_face_classifier_5.h5')

    ranges = range(1, 90, 15)
    for angle in ranges:
        model = rotate_fit(model, data, 15, angle)
        test_rotated_model(model, data)


    # model.save('rotated_alexnet_face_classifier.h5')

    return

if __name__ == '__main__':
    main()
