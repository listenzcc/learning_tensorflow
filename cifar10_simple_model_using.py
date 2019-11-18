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

# %%
model_dir, model_name = os.getcwd(), 'cifar10_simple_mode.h5'
model = keras.models.load_model(os.path.join(model_dir, model_name))
model

# %%
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape

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

# %%
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# %%
help(model.predict_proba)

# %%
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.style.use('seaborn-dark')
for _ in range(3):
    j = random.choice(range(len(X_test)))
    x = X_test[j]
    y = y_test[j]
    c = model.predict_proba(x[np.newaxis, :])[0]
    x = ((x + X_train_mean) * 255).astype(np.uint32)
    fig = plt.figure(figsize=(6, 3))
    axes = fig.subplots(1, 2)
    axes[0].imshow(x, interpolation='bilinear')
    axes[0].set_title("True: {} Guess: {}".format(np.argmax(y), np.argmax(c)))
    axes[1].barh(classes, c)
    axes[1].barh(classes[np.argmax(c)], c[np.argmax(c)])
    fig.tight_layout()

# %%
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

# %%
