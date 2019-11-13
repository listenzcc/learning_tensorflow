
# %%
import tensorflow as tf


# %%
# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")  # const(2)

# create TensorFlow variables
b = tf.Variable(2.0, name='b')  # b = 2
c = tf.Variable(1.0, name='c')  # c = 1

# now create some operations
d = tf.add(b, c, name='d')  # d = b + c, 3
e = tf.add(c, const, name='e')  # e = c + const(2), 3
a = tf.multiply(d, e, name='a')  # a = d x e, 9

print(a)


# %%


