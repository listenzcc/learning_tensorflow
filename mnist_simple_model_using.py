# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.utils import np_utils
import random
from pprint import pprint


# %%
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

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
filepath = 'mnist_simple_model.h5'
model = keras.models.load_model(filepath)


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


