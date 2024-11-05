import math

from keras.initializers.initializers_v2 import RandomNormal
from keras.losses import binary_crossentropy
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0, ResNet101
import tensorflow as tf

def custom_loss(y_true, y_pred):
    bce_loss = binary_crossentropy(y_true, y_pred)
    ''' conditions:
    model needs to predict right.
    it needs to learn from right predictions == loss != zero
    
    '''
    # distance_loss = tf.abs(0.5 - tf.reduce_mean(tf.abs(y_pred)))

    # L2 regularization for all layers
    #l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in model.trainable_weights]) * l2_lambda

    total_loss = bce_loss# + distance_loss# + l2_loss
    return total_loss

class AdaptationNet(object):
    @staticmethod
    def create_alexnet(input_shape, num_classes, loss_function, metrics):
        model = AdaptationNet._alexnet(input_shape, num_classes)
        model.compile(optimizer=Adam(), loss=loss_function, metrics=metrics)
        return model

    @staticmethod
    def create_regularized_custom_alexnet(input_shape, num_classes, loss_function, metrics):
        model = AdaptationNet._regularized_custom_alexnet(input_shape, num_classes)
        model.compile(optimizer=Adam(), loss=custom_loss, metrics=metrics)
        return model

    @staticmethod
    def create_regularized_alexnet(input_shape, num_classes, loss_function, metrics):
        model = AdaptationNet._regularized_alexnet(input_shape, num_classes)
        model.compile(optimizer=Adam(), loss=loss_function, metrics=metrics)
        return model

    @staticmethod
    def create_resnet(input_shape, num_classes, loss_function, metrics):
        '''unknown'''
        model = AdaptationNet._resnet101(input_shape, num_classes)
        model.compile(optimizer=Adam(), loss=loss_function, metrics=metrics)
        return model

    @staticmethod
    def create_pretrained_model(input_shape, num_classes, loss_function, metrics):
        ''' Wasnt able to learn the classes '''
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

    @staticmethod
    def _alexnet(input_shape, num_classes):
        """
        Build ANN architecture
        :param input_shape: (x size, y size, dimension amount)
        :param num_classes: integer of unique classes
        :return: model architecture
        """
        model = models.Sequential()

        # 1st Convolutional Layer
        model.add(
            layers.Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid'))
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

    @staticmethod
    def _regularized_custom_alexnet(input_shape, num_classes):
        """
        Build ANN architecture
        :param input_shape: (x size, y size, dimension amount)
        :param num_classes: integer of unique classes
        :return: model architecture
        """
        model = models.Sequential()

        # 1st Convolutional Layer
        model.add(
            layers.Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid', ))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', ))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', ))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # 4th Convolutional Layer
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', ))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # 5th Convolutional Layer
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', ))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        # Flatten
        model.add(layers.Flatten())

        # 1st Fully Connected Layer
        model.add(layers.Dense(4096, ))
        model.add(layers.Activation('relu'))
        # model.add(layers.Dropout(0.5))

        # 2nd Fully Connected Layer
        model.add(layers.Dense(4096, ))
        model.add(layers.Activation('relu'))
        # model.add(layers.Dropout(0.5))

        # Output Layer
        model.add(layers.Dense(1, ))
        model.add(layers.Activation('sigmoid'))

        # Define a function to calculate L2 strength based on number of weights
        def get_l2_strength(model_layer):
            num_weights = model_layer.count_params()  # Total number of weights in the layer
            if num_weights > 0:  # Only apply regularization if there are trainable weights
                # Access the weights of the layer
                weights = model_layer.weights[0]
                if isinstance(model_layer, layers.Dense):
                    # Get the shape of the kernel weights
                    weight_shape = weights.shape
                    # Calculate fan_in and fan_out
                    fan_in = weight_shape[0]  # Number of input units
                    fan_out = weight_shape[1]  # Number of output units
                else: #Conv layer
                    kernel_size = model_layer.kernel_size
                    # Calculate fan_in and fan_out
                    fan_in = kernel_size[0] * kernel_size[1] * weights.shape[-2]
                    fan_out = kernel_size[0] * kernel_size[1] * weights.shape[-1]

                limit = math.sqrt(6 / (fan_in + fan_out))
                return num_weights * (2 / 3) * (limit ** 2)# Calculation of Glorot distribution expectation over N weights
            return

        # Dynamically add L2 regularization to each layer
        for i, layer in enumerate(model.layers):
            if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
                l2_strength = get_l2_strength(layer)
                if l2_strength > 0:
                    # Rebuild the layer with L2 regularization
                    model.layers[i] = layer.__class__(
                        **{**layer.get_config(), 'kernel_regularizer': regularizers.l2(l2_strength)}
                    )

        return model

    @staticmethod
    def _regularized_alexnet(input_shape, num_classes):
        """
        Build ANN architecture
        :param input_shape: (x size, y size, dimension amount)
        :param num_classes: integer of unique classes
        :return: model architecture
        """
        model = models.Sequential()

        # 1st Convolutional Layer
        model.add(
            layers.Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid',))
                          # kernel_regularizer = regularizers.l2(L2_strength)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same',))
                                # kernel_regularizer = regularizers.l2(L2_strength)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',))
                                # kernel_regularizer = regularizers.l2(L2_strength)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # 4th Convolutional Layer
        model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',))
                                # kernel_regularizer = regularizers.l2(L2_strength)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # 5th Convolutional Layer
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',))
                                # kernel_regularizer = regularizers.l2(L2_strength)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

        # Flatten
        model.add(layers.Flatten())

        # 1st Fully Connected Layer
        model.add(layers.Dense(4096,))# kernel_regularizer = regularizers.l2(L2_strength)))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))

        # 2nd Fully Connected Layer
        model.add(layers.Dense(4096,))# kernel_regularizer = regularizers.l2(L2_strength)))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))

        # Output Layer
        model.add(layers.Dense(num_classes))# kernel_regularizer = regularizers.l2(L2_strength)))
        model.add(layers.Activation('softmax'))


        # Define a function to calculate L2 strength based on number of weights
        def get_l2_strength(model_layer):
            num_weights = model_layer.count_params()  # Total number of weights in the layer
            if num_weights > 0:  # Only apply regularization if there are trainable weights
                return 1.0 / num_weights
            return 
            
        # Dynamically add L2 regularization to each layer
        for i, layer in enumerate(model.layers):
            if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
                l2_strength = get_l2_strength(layer)
                if l2_strength > 0:
                  # Rebuild the layer with L2 regularization
                  model.layers[i] = layer.__class__(
                      **{**layer.get_config(), 'kernel_regularizer': regularizers.l2(math.sqrt(l2_strength))}
                  )

        return model

    @staticmethod
    def _resnet101(input_shape, num_classes):
        return ResNet101(weights=None, input_shape=input_shape, classes=num_classes)