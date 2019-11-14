# %%
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

from pprint import pprint

import random


# %%
print(keras.backend.backend(), keras.__version__, tf.__version__)


# %%
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# %%
print(plt.style.available)
plt.style.use('seaborn-dark')
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
plt.tight_layout()


# %%
nb_classes = 10

def normalize(raw):
    return (raw.astype(np.float32) - 127) / 127

dataset = dict(
    X_train = normalize(X_train.reshape(len(X_train), -1)),
    X_test = normalize(X_test.reshape(len(X_test), -1)),
    y_train = np_utils.to_categorical(y_train, nb_classes),
    y_test = np_utils.to_categorical(y_test, nb_classes),
)
pprint(dataset)


# %%
model = Sequential()

model.add(Dense(units=512, activation='relu', input_dim=784))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=nb_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# %%
model.fit(dataset['X_train'], dataset['y_train'], epochs=2, batch_size=64, verbose=1, validation_split=0.05)


# %%
loss, accuracy = model.evaluate(dataset['X_test'], dataset['y_test'])
print('Test loss:', loss)
print('Accuracy:', accuracy)


# %%
plt.style.use('seaborn-dark')
for i in range(9):
    plt.subplot(3,3,i+1)
    j = random.choice(range(len(dataset['X_test'])))
    x = dataset['X_test'][j]
    y = np.argmax(dataset['y_test'][j])
    cls = model.predict_classes(x[np.newaxis, :])
    plt.imshow(x.reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Guess {} of {}".format(cls[0], y))
plt.tight_layout()


# %%
filepath = 'mnist_simple_model.h5'
model.save(filepath)

# %%