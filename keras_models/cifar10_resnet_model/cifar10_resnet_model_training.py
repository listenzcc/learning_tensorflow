# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from pprint import pprint
import random
import os

# %%
batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True

# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Normalize data.
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Subtract pixel mean is enabled
X_train_mean = np.mean(X_train, axis=0)
X_train -= X_train_mean
X_test -= X_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

datagen.fit(X_train)

# %%
input_shape = X_train.shape[1:]
Input = keras.layers.Input
Conv2D = keras.layers.Conv2D
Activation = keras.layers.Activation
BatchNormalization = keras.layers.BatchNormalization
MaxPooling2D = keras.layers.MaxPool2D
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout
Dense = keras.layers.Dense

inputs = Input(shape=input_shape)
a = Conv2D(16, (3, 3), padding='same')(inputs)
a = BatchNormalization()(a)
a = Activation('relu')(a)

b = Conv2D(16, (3, 3))(a)
a = Conv2D(16, (3, 3))(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)
a = keras.layers.add([a, b])
a = MaxPooling2D(pool_size=(2, 2))(a)
a = Dropout(0.25)(a)

b = Conv2D(16, (3, 3))(a)
a = Conv2D(32, (3, 3), padding='same')(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)
a = Conv2D(16, (3, 3))(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)
a = keras.layers.add([a, b])
a = MaxPooling2D(pool_size=(2, 2))(a)
a = Dropout(0.25)(a)

a = Flatten()(a)
a = Dense(512)(a)
a = Activation('relu')(a)
a = Dropout(0.5)(a)
a = Dense(num_classes)(a)
a = Activation('softmax')(a)

model = keras.Model(inputs=inputs, outputs=a)

# %%
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                               cooldown=0,
                                               patience=5,
                                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    workers=32,
                    callbacks=callbacks)


# %%
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# %%
model_dir, model_name = os.getcwd(), 'cifar10_resnet_model_200.h5'
model.save(os.path.join(model_dir, model_name))

# %%
