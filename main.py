from numpy import array
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import preprocess
import simulation_net
import visualization
import config


def main():
    # ################################# Stage One
    # # preprocess data
    # train_generator, validation_generator,\
    #     test_generator, class_names = preprocess.create_data_generators(config.BASE_DIR,
    #                                                                       config.SPLIT_SIZE,
    #                                                                       config.INPUT_VECTOR_SIZE,
    #                                                                       config.BATCH_SIZE)
    #
    # # prepare model architecture
    # model = simulation_net.alexnet(input_shape=(config.INPUT_VECTOR_SIZE, config.INPUT_VECTOR_SIZE, config.INPUT_DIMENSION),
    #                                num_classes=len(class_names))
    # model.compile(optimizer=Adam(), loss=config.LOSS_FUNCTION, metrics=config.METRICS)
    #
    # # train model
    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=train_generator.samples // train_generator.batch_size,
    #     validation_data=validation_generator,
    #     validation_steps=validation_generator.samples // validation_generator.batch_size,
    #     epochs=config.EPOCH_AMOUNT
    # )
    #
    # model.save('alexnet_face_classifier.h5')
    # model.evaluate(test_generator)
    # results = pd.DataFrame(history.history)
    # print(results.tail())

    model = load_model('alexnet_face_classifier.h5')

    ################################# Stage Two

    adaptation_generator = preprocess.create_prediction_data_generator(config.BASE_DIR, config.INPUT_VECTOR_SIZE)
    for i in range(len(adaptation_generator)):
        x = adaptation_generator.next()
        filename = adaptation_generator.filenames[i]
        prediction = model.predict(x)[0]
        print('Class {} for image {}'.format(prediction, filename))
        y = array([prediction])
        model.fit(x, y, epochs=1)

    # The last model didnt learn the images well enough and reached a 0.52 on test set
    return


'''


    while: 
        test_gen = ImageDataGenerator(rescale=1. / 255.)
        train_generator = train_datagen.flow_from_directory(
            training_dir,
            target_size=(input_size, input_size),
            batch_size=batch_size,
            class_mode='categorical')# class mode can be binary

        model.predict(generator)

        model.fit(generator,)

    ...
'''

if __name__ == '__main__':
    main()

