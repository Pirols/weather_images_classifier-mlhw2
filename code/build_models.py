import tensorflow as tf

def build_pretrained_model(img_size, output_size, fine_tune_at):
    """
        build_pretrained_model(img_size, fine_tune_at)
        takes as INPUTS:
            -img_size: the size(resolution) of the input images
            -output_size: the number of output classes the model will be trained to classify
            -fine_tune_at: The first fine_tune_at layers of tf.keras.applications.MobileNetV2 will be frozen and won't be updated during backtracking
        DOES:
            Builds and returns a model composed of 3 components:
                1) a pretrained tf.keras.applications.MobileNetV2
                2) an additional layer tf.keras.layers.GlobalAveragePooling2D()
                3) an output layer: tf.keras.layers.Dense(output_size)
        and OUTPUTS:
            The model obtained concatenating 1 2 and 3 according to this order.
    """

    IMG_SHAPE = (img_size, img_size, 3) # (Height, Width, RGB)

    # Create the base model from the pre-trained model MobileNet V2
    pretrained_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                        include_top=False,
                                                        weights='imagenet')

    try:
        for layer in pretrained_model.layers[:fine_tune_at]:
            layer.trainable = False
    except IndexError:
        print("fine_tune_at parameter exceeded the number of layers of tf.keras.applications.MobileNetV2.layers={}".
                format(len(pretrained_model.layers)))
        return -1

    # Complete the model with an average pooling and a final dense layer
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(output_size)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model


def build_model(img_size, output_size):
    """
        build_model(img_size, output_size)
        takes as INPUTS:
            -img_size: the size(resolution) of the input images
            -output_size: the number of output classes the model will be trained to classify
        DOES:
            Builds and returns a model composed of 3 components:
                1) an uninitialised tf.keras.applications.MobileNetV2
                2) an additional layer tf.keras.layers.GlobalAveragePooling2D()
                3) an output layer: tf.keras.layers.Dense(output_size)
        and OUTPUTS:
            The model obtained concatenating 1 2 and 3 according to this order.
    """

    IMG_SHAPE = (img_size, img_size, 3) # (Height, Width, RGB)

    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights=None)

    # Complete the model with an average pooling and a final dense layer
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(output_size)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model
