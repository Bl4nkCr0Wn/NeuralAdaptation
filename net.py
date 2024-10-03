from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

class AdaptationNet(object):
    @staticmethod
    def create_model(input_shape, num_classes, loss_function, metrics):
        model = _alex_net(input_shape, num_classes)

        def l2_regularized_loss(y_true, y_pred):
            loss = tf.keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred)
            # regularization_loss = tf.reduce_sum(tf.square(model.trainable_weights))
            regularization_loss = tf.add_n([ tf.nn.l2_loss(v) for v in model.trainable_weights])
            return loss + 0.5 * regularization_loss

        model.compile(optimizer=Adam(), loss=l2_regularized_loss, metrics=metrics)
        return model

    @staticmethod
    def create_pretrained_model(input_shape, num_classes, loss_function, metrics):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output

        # trainable Fully Connected Layer
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)# NOTE: compared to untrained network, here there is no dropout

        # Output Layer
        x = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs=base_model.input, outputs=x)
        model.compile(optimizer=Adam(), loss=loss_function, metrics=metrics)
        return model

def _alex_net(input_shape, num_classes):
    """
    Build ANN architecture
    :param input_shape: (x size, y size, dimension amount)
    :param num_classes: integer of unique classes
    :return: model architecture
    """
    model = models.Sequential()

    # 1st Convolutional Layer
    model.add(layers.Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    # 4th Convolutional Layer
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())

    # 5th Convolutional Layer
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Flatten
    model.add(layers.Flatten())

    # 1st Fully Connected Layer
    model.add(layers.Dense(4096))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    # 2nd Fully Connected Layer
    model.add(layers.Dense(4096))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))

    return model