import numpy as np
from numpy import array
import pandas as pd
from tensorflow.keras.models import load_model

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

def train_new_model(data, pretrained = False):
    train_generator, validation_generator,\
        test_generator = data.create_generators(config.INPUT_VECTOR_SIZE, config.BATCH_SIZE)

    if not pretrained:
        # prepare model architecture
        model = net.AdaptationNet.create_model((config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
                                               len(data.CLASS_NAMES),
                                               config.LOSS_FUNCTION,
                                               config.METRICS)
    else:
        model = net.AdaptationNet.create_pretrained_model((config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
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


# def self_supervised_rotate_fit(model, data, angle_range, angle_range_start = 1):
#     # Index moving degree images by degree
#     adaptation_generator = data.create_adaptation_generator(config.INPUT_VECTOR_SIZE)
#     images_by_degree = preprocess.get_degree_images_dictionary(adaptation_generator)
#
#     # Train each image class alternatively
#     degree_sequence = []
#     for i in range(angle_range_start, angle_range_start + angle_range):
#         degree_sequence.append(135 + i)
#         degree_sequence.append((315 + i)%360)
#
#     for key in degree_sequence:
#         x = images_by_degree[key]
#         prediction = model.predict(x)[0]
#         print('Class {} for degree {}'.format(prediction, key))
#         y = array([prediction])
#         model.fit(x, y, epochs=1)
#
#     return model

def test_rotated_model(model, data):
    # Test original face classes classification after rotation.
    train_generator, validation_generator, \
        test_generator = data.create_generators(config.INPUT_VECTOR_SIZE, config.BATCH_SIZE)
    # print('Evaluate model on training data:')
    # model.evaluate(train_generator)
    print('Evaluate model on test data:')
    return model.evaluate(test_generator)

# def self_supervision_validation_fit(model, data):
#     train_generator, validation_generator, \
#         test_generator = data.create_generators(config.INPUT_VECTOR_SIZE, 1)
#
#     for i in range(len(validation_generator)):
#         print(f'Step {i}')
#         x, y = validation_generator.next()
#         prediction = model.predict(x)[0]
#         y = array([prediction])
#         model.fit(x, y, epochs=1)
#
#     return model

def supervised_rotate_fit(model, data, angle_range, angle_range_start = 1):
    # Index moving degree images by degree
    adaptation_generator = data.create_adaptation_generator(config.INPUT_VECTOR_SIZE)
    # images_by_degree = preprocess.get_degree_images_dictionary(adaptation_generator)

    # Train each image class alternatively
    degree_sequence = []
    for i in range(angle_range_start, angle_range_start + angle_range):
        degree_sequence.append(135 + i)
        degree_sequence.append((315 + i)%360)

    fit = array([[1.0, 0.0]])# A class
    alternate_fit = array([[0.0, 1.0]])# B class
    for i in range(0, len(degree_sequence), 2):
        images_by_degree = preprocess.get_images_by_degree(adaptation_generator, [degree_sequence[i], degree_sequence[i+1]])
        zipped = zip(images_by_degree[degree_sequence[i]], images_by_degree[degree_sequence[i+1]])
        alternated_img = [item for pair in zipped for item in pair]
        x = np.concatenate(alternated_img, axis=0)
        alternated_prediction = [fit, alternate_fit] * int(len(alternated_img) / 2)
        y = np.concatenate(alternated_prediction, axis=0)
        # x = images_by_degree[degree_sequence[i]] + images_by_degree[degree_sequence[i+1]]
        # x = np.concatenate(x, axis=0)
        # y = ([fit,] * int(len(x)/2)) + ([alternate_fit,] * int(len(x)/2))
        # y = np.concatenate(y, axis=0)
        print('Fitting {}'.format([degree_sequence[i], degree_sequence[i+1]]))
        print('Predicted values are: {}'.format(model.predict(x)))
        model.fit(x, y, epochs=1)

    return model

def main():
    # data = prepare_new_data()
    data = load_data()

    model, history = train_new_model(data)
    # model, history = train_new_model(data, pretrained=True)
    model.save('alexnet_face_classifier_with_regularization.h5')
    # model = load_model('alexnet_face_classifier_1.h5')

    # model = supervised_rotate_fit(model, data, 179, 1)
    # loss, acc = test_rotated_model(model, data)

    # angle_range = 15
    # ranges = range(1, 179, angle_range)
    # for angle in ranges:
    #     # model = self_supervised_rotate_fit(model, data, 15, angle)
    #     if angle + angle_range > 180:
    #         angle_range = 180 - angle
    #     model = supervised_rotate_fit(model, data, angle_range, angle)
    #     loss, acc = test_rotated_model(model, data)
    #     if acc < 0.5:
    #         print(f'Flipping at {angle}')
    #
    # model.save('supervised_rotated_alexnet_face_classifier_5.h5')
    return

if __name__ == '__main__':
    main()
