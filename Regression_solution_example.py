# %%
import tensorflow as tf
import numpy as np


# %%
# Ground truth session

# Parameters, ground-truth of W and b
W_true = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2).astype(np.float32)
b_true = np.array([7, 8, 9]).reshape(3, 1).astype(np.float32)

# y = W * x + b
def linear_forward(x, W=None, b=None, forward=False, W_true=W_true, b_true=b_true):
    # Fast forward for ground-truth
    if forward:
        return tf.matmul(tf.Variable(W_true), x) + tf.Variable(b_true)
    # Compute regression with input x and current W and b
    return tf.matmul(W, x) + b

# Datas
NUM_EXAMPLES = 2000
x_data = np.random.normal(size=(2, NUM_EXAMPLES)).astype(np.float32)

# Random fetch method
def random_fetch(x_data=x_data, batchsize=10):
    # Shuffle on column dimension
    np.random.shuffle(np.transpose(x_data))
    # Cut data with batchsize
    x = x_data[:, :batchsize]
    # Generate noise
    noise = np.random.normal(size=(3, batchsize)).astype(np.float32)
    # Return x and y
    return x, linear_forward(x, forward=True) + noise


# %%
random_fetch()


# %%
# Module building session

# LinearModule building
class LinearModel(tf.keras.Model):
    # Cross init
    def __init__(self):
        super(LinearModel, self).__init__()
        # Variables that are trainable
        self.W = tf.Variable(tf.random.uniform((3, 2)), name='weight')
        self.b = tf.Variable(tf.random.uniform((3, 1)), name='bias')
    # 'call method' of LinearModel
    def call(self, inputs):
        return linear_forward(inputs, self.W, self.b)

# Computation of loss function to be optimized
def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

# Computation of gradient
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    # Return current loss_value and gradient
    return loss_value, tape.gradient(loss_value, [model.W, model.b])


# %%
# Training session

# Init LinearModuel as model
model = LinearModel()
# Init optimizer
optimizer = tf.optimizers.Adam(0.1)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Training 1000 times
for i in range(1000):
    # Random fetch x and y
    x, y = random_fetch(batchsize=100)
    # Compute loss_value and grads
    loss_value, grads = grad(model, x, y)
    # Apply gradients
    optimizer.apply_gradients(zip(grads, [model.W, model.b]))
    # Print loss each 100 steps
    if i % 100 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss_value))

# Print trained W and b
print(model.W, model.b)


# %%


