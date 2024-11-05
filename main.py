import gc
import math

import numpy as np
from numpy import array
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Dense
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

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
        callbacks=[EarlyStopping(patience=3, monitor='val_loss', min_delta=0.1, restore_best_weights=True)]
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

def self_supervised_rotate_fit(model, data, input_size, degree):
    # Train each image class alternatively
    degree_sequence = [config.SPECIAL_DEGREES[0] + degree, (config.SPECIAL_DEGREES[1] + degree)%360]
    print('Fitting {}'.format([degree_sequence[0], degree_sequence[1]]))

    x = preprocess.get_images_by_degree(data, input_size,[degree_sequence[0], degree_sequence[1]], config.THETA_AMOUNT)
    x = zip(x[degree_sequence[0]], x[degree_sequence[1]])
    x = [item for pair in x for item in pair]
    x = np.concatenate(x, axis=0)

    y = model.predict(x)
    print('Predicted: {}'.format(y))
    classes = [1 if p > 0.5 else 0 for p in y]# np.argmax(y, axis=1)
    # print('augmenting probs..')
    y = []
    wrong = 0
    for i, c in enumerate(classes):
        if c == 0:
            # y.append([1.0, 0.0])
            if (i % 2) == 1:
                wrong += 1
        else:
            # y.append([0.0, 1.0])
            if (i % 2) == 0:
                wrong += 1
    y = classes
    y = array(y)
    print('Classes are (count of wrong classification): {}'.format(wrong))

    split = len(x)//5#config.SPLIT_SIZE
    y_val = np.concatenate([array([0.0]), array([1.0])] * int(len(x) / 2), axis=0)

    # Generate new weight plane for classification
    for layer in model.layers:
        if isinstance(layer, Dense):
        # for v in layer.trainable_variables:
            # Modify the weights with noise from a some distribution
            def uniform_noise(weights):
                # Get the shape of the kernel weights
                weight_shape = weights.shape
                # Calculate fan_in and fan_out
                fan_in = weight_shape[0]  # Number of input units
                if len(weight_shape) == 2:
                    fan_out = weight_shape[1]  # Number of output units
                else:
                    fan_out = 0
                limit = math.sqrt(6 / (fan_in + fan_out))
                noise_weights = tf.random.uniform(shape=weights.shape, minval=-limit, maxval=limit)
                return weights + noise_weights

            weights, biases = layer.get_weights()
            # Add noise to weights and biases
            weights = uniform_noise(weights)
            biases = uniform_noise(biases)
            # Set the modified weights and biases
            layer.set_weights([weights, biases])

            # K.set_value(layer.weights[0], new_weights)
            # if 'kernel' in v.name or 'bias' in v.name:
            #     v.assign(tf.keras.initializers.GlorotUniform()(v.shape))

    model.fit(x[split:], y[split:],validation_data = (x[:split], y_val[:split]) , epochs=config.EPOCH_AMOUNT,
              callbacks=[EarlyStopping(patience=3, monitor='val_loss',min_delta=0.1, restore_best_weights=True)])
    del x, y, classes
    # return model

def supervised_rotate_fit(model, data, input_size, degree_generator=range(1, 180, 1)):
    # Train each image class alternatively
    raise Exception("not adjusted for binary classifier")
    degree_sequence = []
    for i in degree_generator:
        degree_sequence.append(135 + i)
        degree_sequence.append((315 + i)%360)

    fit = array([[1.0, 0.0]])# A class
    alternate_fit = array([[0.0, 1.0]])# B class
    for i in range(0, len(degree_sequence), 2):
        images = preprocess.get_images_by_degree(data, input_size,[degree_sequence[i], degree_sequence[i+1]])
        images = zip(images[degree_sequence[i]], images[degree_sequence[i+1]])
        images = [item for pair in images for item in pair]
        x = np.concatenate(images, axis=0)
        images = None # release memory

        y = [fit, alternate_fit] * int(len(images) / 2)
        y = np.concatenate(y, axis=0)
        print('Fitting {}'.format([degree_sequence[i], degree_sequence[i+1]]))
        print('Predicted values are: {}'.format(model.predict(x)))
        model.fit(x, y, epochs=10)# can it be 1 epoch given enough examples?

def test_model_by_class(model, data):
    dg,vg,tg = data.create_generators(config.INPUT_VECTOR_SIZE, config.BATCH_SIZE)

    pred = []
    test = []
    for x, y in tg:
        y_pred = model.predict(x)
        print(y_pred)
        pred.extend([1 if p > 0.5 else 0 for p in y_pred])# pred.extend(np.argmax(y_pred, axis=1))
        test.extend(y)# test.extend(np.argmax(y, axis=1))
        if len(test) >= tg.samples:
            break
    test = np.array(test)
    pred = np.array(pred)
    mask = test == 0
    classA_score = accuracy_score(test[mask], pred[mask])
    print('Accuracy on class 0: {}'.format(classA_score))
    mask = test == 1
    classB_score = accuracy_score(test[mask], pred[mask])
    print('Accuracy on class 1: {}'.format(classB_score))
    return classA_score, classB_score

def calc_dividing_plane(model, data, input_size):
    history = {'degree' : [], 'classA' : [], 'classB' : []}
    for degree in range(0, 360, config.THETA_INCREMENT):
        degree = (config.SPECIAL_DEGREES[0] + degree) % 360
        print('Fitting {}'.format(degree))
        images_by_degree = preprocess.get_images_by_degree(data, input_size,[degree], config.THETA_AMOUNT//5)
        x = np.concatenate(images_by_degree[degree], axis=0)
        y = model.predict(x)
        classes = np.array([1 if p > 0.5 else 0 for p in y])# classes = np.argmax(y, axis=1)
        history['degree'].append(degree)
        history['classA'].append(np.sum(classes == 0))
        history['classB'].append(np.sum(classes == 1))
    return pd.DataFrame(history)

def show_plane(dividing_plane, name):
    plt.scatter(dividing_plane.degree, dividing_plane.classA, label='Class_A')
    plt.scatter(dividing_plane.degree, dividing_plane.classB, label='Class_B')
    plt.title(name)
    plt.xlabel('Degree')
    plt.ylabel('Classifications')
    plt.savefig(name + '.png')
    plt.legend()
    plt.show()

def main():
    RUN_NAME = 'regularized_binary_alexnet_dataset_XL'
    # data = prepare_new_data()
    data = load_data()

    # model = net.AdaptationNet.create_regularized_custom_alexnet(
    #     (config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
    #     len(data.CLASS_NAMES),
    #     config.LOSS_FUNCTION,
    #     config.METRICS)
    #
    # model, history = train_new_model(data, model)
    # model.save(RUN_NAME+'_face_classifier.h5')
    # history.to_csv(RUN_NAME + '_train_history.csv', index=False)
    # history.loc[:, ['loss', 'val_loss']].plot()
    # history.loc[:, ['accuracy', 'val_accuracy']].plot()
    # plt.show()

    model = load_model(RUN_NAME+'_face_classifier.h5', custom_objects={'custom_loss' : net.custom_loss})
    # test_model_by_class(model, data)

    # res = calc_dividing_plane(model, data,
    #                           config.INPUT_VECTOR_SIZE)
    # show_plane(res, RUN_NAME + '_dividing_plane_angle_' + str(0))

    # prepare model for semi rotation
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            layer.trainable = False

    ROTATION_TYPE = 'self'#, 'supervised'
    angles = range(config.THETA_INCREMENT, 181, config.THETA_INCREMENT)
    for angle in angles:
        success = False
        while not success:
            try:
                gc.collect()
                self_supervised_rotate_fit(model, data, config.INPUT_VECTOR_SIZE, angle)
                success = True
            except Exception as e:
                print('Failed', e)

        # if angle > 90:
        success = False
        while not success:
            try:
                gc.collect()
                res = calc_dividing_plane(model, data, config.INPUT_VECTOR_SIZE)
                gc.collect()
                show_plane(res, RUN_NAME + '_dividing_plane_angle_' + str(angle))
                success = True
            except Exception as e:
                print('Failed', e)
        # model.save(RUN_NAME + '_' + ROTATION_TYPE + '_rotation_' + str(angle) + '.h5')

    test_model_by_class(model, data)
    model.save(RUN_NAME + '_'+ ROTATION_TYPE + '_rotation.h5')
    return

if __name__ == '__main__':
    main()
