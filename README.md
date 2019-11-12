# Some quick understandable practice of tensorflow

***

## TensorFlow Intro

> TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

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