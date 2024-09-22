import os
from numpy import array
import pandas as pd
from tensorflow.keras.models import load_model
import re
import cv2
import numpy as np

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

def rotate_fit(model, data, angle):
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
    for i in range(1, angle):
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
    # Stage 3 - test original face classes classification after rotation.
    train_generator, validation_generator, \
        test_generator = data.create_generators(config.INPUT_VECTOR_SIZE, config.BATCH_SIZE)
    print('Evaluate model on training data:')
    model.evaluate(train_generator)

    # A_res = {}
    # for file in os.listdir(class_A_path):
    #     res = predict_specific(model, os.path.join(class_A_path, file))
    #     res = tuple(res.tolist()[0])
    #     if res in A_res:
    #         A_res[res] += 1
    #     else:
    #         A_res[res] = 1
    #
    # print('Results for A class:')
    # print(A_res)
    #
    # B_res = {}
    # for file in os.listdir(class_B_path):
    #     res = predict_specific(model, os.path.join(class_B_path, file))
    #     res = tuple(res.tolist()[0])
    #     if res in B_res:
    #         B_res[res] += 1
    #     else:
    #         B_res[res] = 1
    #
    # print('Results for A class:')
    # print(A_res)
    # print('Results for B class:')
    # print(B_res)

    # Try training on original families and see if the model fixates
    # model = load_model('alexnet_face_classifier_1.h5')
    # for fileA, fileB in zip(os.listdir(class_A_path), os.listdir(class_B_path)):
    #     resA, imgA = predict_specific(model, os.path.join(class_A_path, fileA), True)
    #     resB, imgB = predict_specific(model, os.path.join(class_B_path, fileB), True)
    #     resA_tup = tuple(resA.tolist()[0])
    #     resB_tup = tuple(resB.tolist()[0])
    #     # model.fit(imgA, array([resA[0]]), epochs=1)
    #     # model.fit(imgB, array([resB[0]]), epochs=1)
    #     print('Class A: {} Class B: {}. (files: {}, {})'.format(resA_tup, resB_tup, fileA, fileB))
    #     if resA_tup == resB_tup:
    #         print('Prediction changed!')
    #         os.system('PAUSE')
'''
def predict_specific(model, image_path, img_return=False):
    img = cv2.imread(image_path)
    img = img.astype(np.float32)
    img = img * (1.0 / 255.0)
    img = np.array((img,))

    if img_return:
        return model.predict(img), img

    return model.predict(img)
'''

def main():
    data = prepare_new_data()
    model, history = train_new_model(data)
    model.save('alexnet_face_classifier.h5')

    # data = load_data()
    # model = load_model('alexnet_face_classifier.h5')

    # model = rotate_fit(model, data, 90)
    # model.save('rotated_alexnet_face_classifier.h5')

    return

if __name__ == '__main__':
    main()
