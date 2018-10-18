# Introduction

# Background

The basic principles behind automatic differentiation rely on calculating the derivative of a function by splitting the calculation into a number of parts, and can be approached using a number of equivalent methodologies. The simplest illustration of the forward mode of AD is taking the chain rule and treating a tricky function as a composite function of a series of elementary operations. The product of each derivative, building up through each of the elementary functions, gives a computationally simple and accurate method for evaluating difficult derivatives.

For functions that map ![equation](http://latex.codecogs.com/gif.latex?R%5Em) to $R^n$, we can see that this method of computing the derivative is equivalent to computing the Jacobian. This can be computationally illustrated through:

$$D_{p}x = \sum_{j=1}^{m}{\dfrac{\partial x}{\partial x_{j}}p_{j}}$$

where we take x as our m-dimensional input vector, and $p$ as a seed vector that is all 0's except for a 1 at the dimension we desire to compute for the Jacobian, thus filling in each of the entries.

Finally, for the implementation in our software package, we take advantage of the construction of dual numbers, which is an algebra with the following construction:

Given any number x, rewrite it as: $x = x+\epsilon x'$, where $\epsilon$ has the property such that $\epsilon^2 = 0$. This construction is extremely useful because it enables the automatic computation of derivatives, provided the initial derivative at any given x upon instantiation, simply by expanding the formula and computing algebraically the equivalent dual number solution. For the purposes of our implementation of the forward mode of automatic differentiation, we will use dual numbers to compute the appropriate derivatives at each step.

# How to Use AutoDiff

# Software Organization

Our directory will look like:
```
cs207-Finalproject\
  autodiff\
    __init__.py
    node.py
    operators.py
    visualization.py
    utils.py
  tests\
    test_node.py
    test_operators.py
    test_visualization.py
    test_utils.py
  examples\
    root_finder.py
    optimization.py
    ...
  docs\
    milestone1.md
    milestone2.md
  .gitignore
  .travis.yml
  LICENSE
  setup.py
  setup.cfg
  README.md
  requirements.txt
```

The key modules and their basic functionalities are:
* node.py: Defines the core structure of forward automatic differentiation. This includes operator overloading.
* operators.py: Defines operators that can be applied to nodes.
* visualization.py: Visualizes the computational graph or table for forward automatic differentation.

The test suite will live in the `tests/` directory, which we will be maintain by using  TravisCI for continuous integration and Coveralls for verifying test coverage. The package will be distributed through PyPI.

# Implementation
We will be implementing the automatic differentiation by using `Node` instances and its subclasses, which are defined in `Operators` module. We can also visualize the algorithm using `Visualization` module.

## What are the core data structures?
We will be using a tuple for our `Node` module as it core data structure. It will store both the value and the derivative/gradient of the node (we will be using a numpy array to store the gradients).

## What classes will you implement
We will be implementing the `Node` class first. We will then extend the `Node` class for each operator, which will form a subclass. We also plan to implement a visualization class.

## What method and name attributes will your classes have?

`Node`
The Node class is the core structure. It is the basis of all classes in `Operators` module. Below are the method and name attributes of the `Node` class.

```
Class Node:
  Attributes:
    value: Value of the node
    der: Derivative/Gradients of the node
  Methods:
    evalute(self): Returns the value and the derivative of the Node
    __str__(self): Returns the string representation of the Node
    __eq__(self, other):
    __neg__(self):
    __add__(self, other):
    __radd__(self, other):
    __sub__(self, other):
    __rsub__(self,other):
    __mul__(self, other):
    __rmul__(self, other)
    __power__(self, other):
```

`Operators`
There are many classes in `Operators` module, one for each elementary function. They are subclasses of th `Node` class, and they will override the `evaluate` method.

## What external dependencies will you rely on?

We will rely heavily on `Numpy` and other matrix/math libraries, which will be specified in both `setup.py` and `requirements.py`.

We will specifically leverage [matrix operations](https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.html) and [universal functions](https://docs.scipy.org/doc/numpy-1.15.1/reference/ufuncs.html) from `Numpy`.

## How will you deal with elementary functions like sin and exp?

Elementary functions including trigonometric functions, logarithmic functions, and exponential functions, will be accounted for in `Operators.py`, which are subclasses of `Node`.




