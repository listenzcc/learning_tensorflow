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

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.style.use('seaborn-dark')
fig = plt.figure(figsize=(10, 10))
for i in range(9):
    j = random.choice(range(len(X_train)))
    ax = fig.add_subplot(3, 3, i+1)
    ax.imshow(X_train[j], interpolation='bilinear')
    ax.set_title('{}'.format(classes[y_train[j][0]]))
    ax.axis('off')
fig.tight_layout()

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# %%
# Input image dimensions.
input_shape = X_train.shape[1:]

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
fig = plt.figure(figsize=(5, 5))
for i in range(3):
    ax1 = fig.add_subplot(3, 2, i*2+1)
    ax2 = fig.add_subplot(3, 2, i*2+2)
    
    j = random.choice(range(len(X_train)))
    x = X_train[j]
    y = datagen.random_transform(x)
    
    ax1.imshow(x[:, :, 0], interpolation='bilinear')
    ax1.set_title("Raw")
    ax1.axis('off')
    
    ax2.imshow(y[:, :, 0], interpolation='bilinear')
    ax2.set_title("Transform")
    ax2.axis('off')
fig.tight_layout()

# %%
Input = keras.layers.InputLayer
Conv2D = keras.layers.Conv2D
Activation = keras.layers.Activation
MaxPooling2D = keras.layers.MaxPool2D
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout
Dense = keras.layers.Dense

model = keras.models.Sequential()

model.add(Conv2D(16, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(48, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

# %%
opt_coarse = keras.optimizers.RMSprop()
opt_fine = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt_fine,
              metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=batch_size),
                    epochs=20,
                    validation_data=(X_test, y_test),
                    workers=32)
print('-' * 80)

model.compile(loss='categorical_crossentropy',
              optimizer=opt_fine,
              metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=batch_size),
                    epochs=20,
                    validation_data=(X_test, y_test),
                    workers=32)
print('-' * 80)


# %%
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# %%
model_dir, model_name = os.getcwd(), 'cifar10_simple_model.h5'
model.save(os.path.join(model_dir, model_name))

# %%
model

# %%
