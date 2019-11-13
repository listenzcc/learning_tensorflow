# Some quick understandable practice of tensorflow

***

## TensorFlow Intro

<img src="logo.png" alt="TensorFlow LOGO" width="20%" align="left" hspace="20" vspace="5">

> [TensorFlow](https://tensorflow.google.cn/guide) is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

Tensorflow architecture works in three parts:  

- Preprocessing the data
- Build the model
- Train and estimate the model

### Componnets of TensorFlow

Tensor

![Tensor PDF](Tensors_TM2002211716.pdf)

> Tensorflow's name is directly derived from its core framework: Tensor. In Tensorflow, all the computations involve tensors. A tensor is a vector or matrix of n-dimensions that represents all types of data. All values in a tensor hold identical data type with a known (or partially known) shape. The shape of the data is the dimensionality of the matrix or array.
>
> A tensor can be originated from the input data or the result of a computation. In TensorFlow, all the operations are conducted inside a graph. The graph is a set of computation that takes place successively. Each operation is called an op node and are connected to each other.
>
> The graph outlines the ops and connections between the nodes. However, it does not display the values. The edge of the nodes is the tensor, i.e., a way to populate the operation with data.

Graphs

> TensorFlow makes use of a graph framework. The graph gathers and describes all the series computations done during the training. The graph has lots of advantages:
>
> - It was done to run on multiple CPUs or GPUs and even mobile operating system
> - The portability of the graph allows to preserve the computations for immediate or later use. The graph can be saved to be executed in the future.
> - All the computations in the graph are done by connecting tensors together
>   - A tensor has a node and an edge. The node carries the mathematical operation and produces an endpoints outputs. The edges the edges explain the input/output relationships between nodes.

***

## Quick start example

An example for simple calculation.

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

***

## Simple regression example

An example for solving regression problem.

### Model creation

    class Linear(tf.keras.Model):
        def __init__(self):
            super(Linear, self).__init__()
            self.W = tf.Variable(tf.random.uniform((2, 3)), name='weight')
            self.b = tf.Variable(tf.random.uniform((2, 1)), name='bias')
        def call(self, inputs):
            return tf.matmul(self.W, inputs) + self.b

### A toy dataset of points around W * x + b

    NUM_EXAMPLES = 2000
    training_inputs = tf.random.normal([3, NUM_EXAMPLES])
    noise = tf.random.normal([NUM_EXAMPLES])
    W = tf.Variable(np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3).astype(np.float32))
    b = tf.Variable(np.array([7, 8]).reshape(2, 1).astype(np.float32))
    training_outputs = tf.matmul(W, training_inputs) + b + noise

### Loss function definition

    def loss(model, inputs, targets):
        error = model(inputs) - targets
        return tf.reduce_mean(tf.square(error))

### Gradient computation

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets)
        return tape.gradient(loss_value, [model.W, model.b])

### Model and Optimizer initilization

    model = Linear()
    optimizer = tf.optimizers.Adam(0.1)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

### Training

    for i in range(300):
        grads = grad(model, training_inputs, training_outputs)
        optimizer.apply_gradients(zip(grads, [model.W, model.b]))
        if i % 20 == 0:
            print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

### Solution printing

    print(model.W, model.b)

***

## Regression solution example

It contains more format model deploying and training processes.

### Ground truth session

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

### Module building session

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

### Training session

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