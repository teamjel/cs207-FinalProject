# Introduction
Differentiation is one of the most important operations in science.  Finding extrema of functions and determining zeros of functions are central to optimization.  Numerically solving differential equations forms a cornerstone of modern science and engineering and is intimately linked with predictive science.
A very frequent occurrence in science requires the scientist to find the zeros of a function ![equation](http://latex.codecogs.com/gif.latex?f%5Cleft%28x%5Cright%29).  The input to the function is a m- dimensional vector and the function returns an n- dimensional vector.  We denote this mathematically as ![equation](http://latex.codecogs.com/gif.latex?f%5Cleft%28x%5Cright%29): ![equation](http://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5E%7Bm%7D%20%5Cmapsto%20%5Cmathbb%7BR%7D%5E%7Bn%7D).  This expression is read:  the function ![equation](http://latex.codecogs.com/gif.latex?f%5Cleft%28x%5Cright%29) maps ![equation](http://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5E%7Bm%7D%20%5Cmapsto%20%5Cmathbb%7BR%7D%5E%7Bn%7D).
In CS207, we explored the finite difference method, but we also computed a symbolic derivative.  The finite difference approach is nice because it is quick and easy.  However, it suffers from accuracy and stability problems.  On the other hand, symbolic derivatives can be evaluated to machine precision, but can be costly to evaluate.
Automatic differentiation (AD) overcomes both of these deficiencies. It is less costly than symbolic differentiation while evaluating derivatives to machine precision.  There are two modes of automatic differentiation: forward and reverse.  This library will be primarily concerned with the forward mode. (Lecture 9)


# Background

The basic principles behind automatic differentiation rely on calculating the derivative of a function by splitting the calculation into a number of parts, and can be approached using a number of equivalent methodologies. The simplest illustration of the forward mode of AD is taking the chain rule and treating a tricky function as a composite function of a series of elementary operations. The product of each derivative, building up through each of the elementary functions, gives a computationally simple and accurate method for evaluating difficult derivatives.

For functions that map ![equation](http://latex.codecogs.com/gif.latex?R%5Em) to ![equation](http://latex.codecogs.com/gif.latex?R%5En), we can see that this method of computing the derivative is equivalent to computing the Jacobian. This can be computationally illustrated through:

![equation](http://latex.codecogs.com/gif.latex?D_%7Bp%7Dx%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7Bm%7D%7B%5Cdfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20x_%7Bj%7D%7Dp_%7Bj%7D%7D)

where we take x as our m-dimensional input vector, and $p$ as a seed vector that is all 0's except for a 1 at the dimension we desire to compute for the Jacobian, thus filling in each of the entries.

Finally, for the implementation in our software package, we take advantage of the construction of dual numbers, which is an algebra with the following construction:

Given any number x, rewrite it as: ![equation](http://latex.codecogs.com/gif.latex?x%20%3D%20x&plus;%5Cepsilon%20x%27), where ![equation](http://latex.codecogs.com/gif.latex?%5Cepsilon) has the property such that ![equation](http://latex.codecogs.com/gif.latex?%5Cepsilon%5E2%20%3D%200). This construction is extremely useful because it enables the automatic computation of derivatives, provided the initial derivative at any given x upon instantiation, simply by expanding the formula and computing algebraically the equivalent dual number solution. For the purposes of our implementation of the forward mode of automatic differentiation, we will use dual numbers to compute the appropriate derivatives at each step.

# How to Use AutoDiff

How to Use PackageName: The user will obtain our package through TestPyPI. The user will first install the package then import it into their project file. To instantiate AD objects, the user should provide the type of function as an argument, an alpha, or coefficient for the function, and a third argument if appropriate for the function.

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




