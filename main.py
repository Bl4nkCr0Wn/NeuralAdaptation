import numpy as np
from numpy import array
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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

def train_new_model(data, model):
    train_generator, validation_generator,\
        test_generator = data.create_generators(config.INPUT_VECTOR_SIZE, config.BATCH_SIZE)

    # train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=config.EPOCH_AMOUNT,
        callbacks=[EarlyStopping(patience=10, monitor='val_binary_accuracy', restore_best_weights=True)]
    )

    model.evaluate(test_generator)
    results = pd.DataFrame(history.history)
    return model, results

def test_model(model, data):
    # Test original face classes classification
    train_generator, validation_generator, \
        test_generator = data.create_generators(config.INPUT_VECTOR_SIZE, config.BATCH_SIZE)
    print('Evaluate model on test data:')
    return model.evaluate(test_generator)

def self_supervised_rotate_fit(model, data, angle_range, angle_range_start = 1):
    # Index moving degree images by degree
    adaptation_generator = data.create_adaptation_generator(config.INPUT_VECTOR_SIZE)
    # images_by_degree = preprocess.get_degree_images_dictionary(adaptation_generator)

    # Train each image class alternatively
    degree_sequence = []
    for i in range(angle_range_start, angle_range_start + angle_range):
        degree_sequence.append(135 + i)
        degree_sequence.append((315 + i)%360)

    for i in range(0, len(degree_sequence), 2):
        images_by_degree = preprocess.get_images_by_degree(adaptation_generator, [degree_sequence[i], degree_sequence[i+1]])
        zipped = zip(images_by_degree[degree_sequence[i]], images_by_degree[degree_sequence[i+1]])
        alternated_img = [item for pair in zipped for item in pair]
        x = np.concatenate(alternated_img, axis=0)

        print('Fitting {}'.format([degree_sequence[i], degree_sequence[i+1]]))
        y = model.predict(x)
        print('Predicted values are: {}'.format(y))
        model.fit(x, y, epochs=10)

    return model

def semi_supervised_rotate_fit(model, data, angle_range, angle_range_start = 1):
    # Index moving degree images by degree
    adaptation_generator = data.create_adaptation_generator(config.INPUT_VECTOR_SIZE)

    # Train each image class alternatively
    degree_sequence = []
    for i in range(angle_range_start, angle_range_start + angle_range):
        degree_sequence.append(135 + i)
        degree_sequence.append((315 + i)%360)

    for i in range(0, len(degree_sequence), 2):
        images_by_degree = preprocess.get_images_by_degree(adaptation_generator, [degree_sequence[i], degree_sequence[i+1]])
        zipped = zip(images_by_degree[degree_sequence[i]], images_by_degree[degree_sequence[i+1]])
        alternated_img = [item for pair in zipped for item in pair]
        x = np.concatenate(alternated_img, axis=0)

        print('Fitting {}'.format([degree_sequence[i], degree_sequence[i+1]]))
        for i in range(10):
            y = model.predict(x)
            print('Predicted values are: {}'.format(y))
            classes = np.argmax(y, axis=1)
            class_probs = [prob[cls] for cls, prob in zip(classes, y)]
            mask = np.array([1 if prob >= 0.99 else 0 for prob in class_probs])
            print('Mask is: {}'.format(mask))
            print ('Classes are: {}'.format(classes))
            print('augmenting probs..')
            y = []
            for c in classes:
                if c == 0:
                    y.append([1.0, 0.0])
                else:
                    y.append([0.0, 1.0])
            y = array(y)
            model.fit(x[mask], y[mask], epochs=1)#10

    return model

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
        print('Fitting {}'.format([degree_sequence[i], degree_sequence[i+1]]))
        print('Predicted values are: {}'.format(model.predict(x)))
        model.fit(x, y, epochs=10)

    return model

def test_model_by_class(model, data):
    dg,vg,tg = data.create_generators(config.INPUT_VECTOR_SIZE, config.BATCH_SIZE)

    pred = []
    test = []
    for x, y in tg:
        y_pred = model.predict(x)
        pred.extend(np.argmax(y_pred, axis=1))
        test.extend(np.argmax(y, axis=1))
        if len(test) >= tg.samples:
            break
    test = np.array(test)
    pred = np.array(pred)
    mask = test == 0
    print('Accuracy on class 0: {}'.format(accuracy_score(test[mask], pred[mask])))
    mask = test == 1
    print('Accuracy on class 1: {}'.format(accuracy_score(test[mask], pred[mask])))

def main():
    RUN_NAME = 'alexnet'
    data = prepare_new_data()
    # data = load_data()

    # prepare model architecture
    # model = net.AdaptationNet.create_pretrained_model((config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
    #                                                   len(data.CLASS_NAMES),
    #                                                   config.LOSS_FUNCTION,
    #                                                   config.METRICS)

    # model = net.AdaptationNet.create_regularized_alexnet(
    #     (config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
    #     len(data.CLASS_NAMES),
    #     config.LOSS_FUNCTION,
    #     config.METRICS)

    model = net.AdaptationNet.create_alexnet((config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
                                           len(data.CLASS_NAMES),
                                           config.LOSS_FUNCTION,
                                           config.METRICS)

    # resnet requires added preprocessing function to ImageDataGenerator
    # model = net.AdaptationNet.create_resnet(
    #     (config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
    #     len(data.CLASS_NAMES),
    #     config.LOSS_FUNCTION,
    #     config.METRICS)

    model, history = train_new_model(data, model)
    model.save(RUN_NAME+'_face_classifier.h5')
    history.to_csv(RUN_NAME + '_train_history.csv', index=False)

    # model = load_model('augmented_alexnet_face_classifier.h5')
    test_model_by_class(model, data)
    test_model(model, data)


    history.loc[:, ['loss', 'val_loss']].plot()
    history.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    plt.show()

    angle_range = 15
    ranges = range(1, 179, angle_range)
    for angle in ranges:
        if angle + angle_range > 180:
            angle_range = 180 - angle
        model = supervised_rotate_fit(model, data, angle_range, angle)
        # model = self_supervised_rotate_fit(model, data, angle_range, angle)
        # model = semi_supervised_rotate_fit(model, data, angle_range, angle)
        test_model_by_class(model, data)

    model.save(RUN_NAME + '_supervised_rotation.h5')
    return

if __name__ == '__main__':
    main()
