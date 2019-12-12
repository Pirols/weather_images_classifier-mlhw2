from model import build_model
from plot import save_plot

# Used to count the number of samples
import os

# Used to save histories to files
import pickle

# Used to perform argmin
import numpy as np

# Used to train the models
from tensorflow import keras as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# used to mute some warnings
import warnings

if __name__ == "__main__":

    # constants
    training_dataset = './datasets/MWI-Dataset-1.1_2000'
    validation_dataset = './datasets/MWI-Dataset-1.2.5'
    testing_dataset = './datasets/MWI-Dataset-1.2.4'

    # CONSTANTS
    # Very high epochs since I will use early stopping callback
    EPOCHS = 150
    BATCH_SIZE = 50
    PATIENCE = 3
    IMG_SIZE = 299
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    OUTPUT_CLASSES = 4

    subdirs = {'/HAZE', '/RAINY', '/SNOWY', '/SUNNY'}
    TRAINING_SAMPLES   = sum(len(os.listdir(training_dataset   + subdir)) for subdir in subdirs)
    VALIDATION_SAMPLES = sum(len(os.listdir(validation_dataset + subdir)) for subdir in subdirs)
    TESTING_SAMPLES    = sum(len(os.listdir(testing_dataset    + subdir)) for subdir in subdirs)

    TRAINING_STEPS     = TRAINING_SAMPLES   // BATCH_SIZE + (TRAINING_SAMPLES   % BATCH_SIZE != 0)
    VALIDATION_STEPS   = VALIDATION_SAMPLES // BATCH_SIZE + (VALIDATION_SAMPLES % BATCH_SIZE != 0)
    TESTING_STEPS      = TESTING_SAMPLES    // BATCH_SIZE + (TESTING_SAMPLES    % BATCH_SIZE != 0)

    # tunable hyperparameters
    learning_rates = [0.0001, 0.001]
    dropout_rates = [0.3, 0.5]
    regularizer_rates = [0.001, 0.01]
    dense_sizes = [64, 128]
    fine_tune_ats = [0, -1, -2]

    # setting up training and validation generators
    train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(training_dataset,
                                                                            target_size=(IMG_SIZE, IMG_SIZE),
                                                                            color_mode="rgb",
                                                                            batch_size=BATCH_SIZE,
                                                                            class_mode="categorical",
                                                                            shuffle=True)

    val_generator   = ImageDataGenerator(rescale=1./255).flow_from_directory(validation_dataset,
                                                                            target_size=(IMG_SIZE, IMG_SIZE),
                                                                            color_mode="rgb",
                                                                            batch_size=BATCH_SIZE,
                                                                            class_mode="categorical",
                                                                            shuffle=True)

    # dictionaries used to store the histories of training of both the models over the gridsearch run
    histories = dict()

    # setup early stopping callback
    es_val_loss = K.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)

    # needed to avoid displaying many warnings due to some non-picture files 
    warnings.filterwarnings("ignore")

    ### Performing a very simple GridSearch ###
    for weights in ['imagenet', None]:
        for lr in learning_rates:
            for drop in dropout_rates:
                for reg in regularizer_rates:
                    for size in dense_sizes:
                        for fine_tune_at in fine_tune_ats:
                            
                            # build the  model
                            model = build_model(INPUT_SHAPE, 
                                                OUTPUT_CLASSES,
                                                fine_tune_at=fine_tune_at,
                                                weights=weights,
                                                dense_size=size,
                                                regularization_factor=reg,
                                                dropout_rate=drop,
                                                base_learning_rate=lr)

                            # train the model
                            histories[(weights, lr, drop, reg, size, fine_tune_at)] = model.fit_generator(train_generator,
                                                                                            steps_per_epoch=TRAINING_STEPS,
                                                                                            epochs=EPOCHS,
                                                                                            validation_data=val_generator, 
                                                                                            validation_steps=VALIDATION_STEPS,
                                                                                            callbacks=[es_val_loss])

                            # storying histories for later usages
                            with open('./histories/'+'_'.join(str(label) for label in [weights, lr, drop, reg, size, fine_tune_at]), 'wb') as fd:
                                pickle.dump(histories[(weights, lr, drop, reg, size, fine_tune_at)].history, fd)

                            if not weights:
                                print('Trained from scratch with: lr={}, dropout={}, regularization term={}, dense size={}'.
                                    format(lr, drop, reg, size))
                                # fine_tune_at doesn't affect the non-pretrained model
                                break
                            else:
                                print('Trained from pretrained with: lr={}, dropout={}, regularization term={}, dense size={}, fine_tune_at={}'.
                                    format(lr, drop, reg, size, fine_tune_at))

    # Evaluating the results
    best_comb = (weights, lr, drop, reg, size, fine_tune_at)
    minimum = min(histories[(weights, lr, drop, reg, size, fine_tune_at)].history['val_loss'])

    for comb, history in histories.items():
        if minimum > min(history.history['val_loss']):
            minimum = min(history.history['val_loss'])
            best_comb = comb

        # save plots
        save_path = './weather_image_classifier-mlhw2_report/images' + '_'.join(str(comb[i]) for i in range(6)) + '.pdf'
        save_plot(save_path, history.history)


    print('The best combination of hyperparameters is found to be: {}'.format(best_comb))
    print('The corresponding metrics are: val_loss={} and val_categorical_accuracy={}'.
        format(minimum, histories[best_comb].history['val_categorical_accuracy'][np.argmin(histories[best_comb].history['val_loss'])]))

    weights, lr, drop, reg, size, fine_tune_at = best_comb

    ### Train the best model ###
    best_model = build_model(INPUT_SHAPE, 
                             OUTPUT_CLASSES,
                             fine_tune_at=fine_tune_at,
                             weights=weights,
                             dense_size=size,
                             regularization_factor=reg,
                             dropout_rate=drop,
                             base_learning_rate=lr)

    checkpointer = K.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                               monitor='val_loss',
                                               save_best_only=True)

    best_model.fit_generator(train_generator,
                            steps_per_epoch=TRAINING_STEPS,
                            epochs=EPOCHS,
                            validation_data=val_generator, 
                            validation_steps=VALIDATION_STEPS,
                            callbacks=[es_val_loss, checkpointer])

    # re-enabling warnings
    warnings.filterwarnings("default")
