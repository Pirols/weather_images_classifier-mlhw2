from tensorflow import keras as K

def build_model(input_shape, output_size, weights='imagenet', dense_size=64, regularization_factor=0.001, dropout_rate=0.3, base_learning_rate=0.001):
    """
        PARAMETERS:
            -input_shape: The shape of the model's input, e.g. (299, 299, 3)
            -output_size: The number of output classes the model will be trained to classify
            -weights[Default:'imagenet']: Should be either imagenet or None for pretrained or nonpretrained xception model.
            -dense_size[Default: 64]: The size of the first dense layer
            -regularization_factor[Default: 0.001]: The parameters of the L2 regularization on the first dense layer
            -dropout_rate[Default: 0.3]: The rate of the dropout layers
            -base_learning_rate[Default: 0.001]: The initial learning rate of the rmsprop optimizer with rho=0.9

        RETURNS:
            Returns a keras model built joining the following models/layers:
                1) Xception model(either pretrained or randomly initialised)
                2) Dense layer with L2
                3) Dropout layer
                4) Pooling layer
                5) Dropout layer
                6) Output - dense - layer

        RAISES:
            -
    """

    base_model = K.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)

    # Complete the model with an average pooling and a final dense layer
    model = K.Sequential([
        base_model,
        K.layers.Dense(dense_size, kernel_regularizer=K.regularizers.l2(regularization_factor), bias_regularizer=K.regularizers.l2(regularization_factor), activation='relu'),
        K.layers.Dropout(dropout_rate),
        K.layers.GlobalAveragePooling2D(),
        K.layers.Dropout(dropout_rate),
        K.layers.Dense(output_size, activation='softmax')
    ])

    model.compile(optimizer=K.optimizers.RMSprop(learning_rate=base_learning_rate, rho=0.9),
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])

    return model
