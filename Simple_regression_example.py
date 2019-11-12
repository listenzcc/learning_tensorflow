# %%
import tensorflow as tf
import numpy as np


# %%
class Linear(tf.keras.Model):
  def __init__(self):
    super(Linear, self).__init__()
    self.W = tf.Variable(tf.random.uniform((2, 3)), name='weight')
    self.b = tf.Variable(tf.random.uniform((2, 1)), name='bias')
  def call(self, inputs):
    return tf.matmul(self.W, inputs) + self.b

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([3, NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
W = tf.Variable(np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3).astype(np.float32))
b = tf.Variable(np.array([7, 8]).reshape(2, 1).astype(np.float32))
training_outputs = tf.matmul(W, training_inputs) + b + noise

# The loss function to be optimized
def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.b])


# %%
model = Linear()
optimizer = tf.optimizers.Adam(0.1)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.b]))
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))


# %%
print(model.W, model.b)


# %%


