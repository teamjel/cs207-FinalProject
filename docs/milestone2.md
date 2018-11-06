# Introduction
Differentiation is one of the most important operations in science.  Finding extrema of functions and determining zeros of functions are central to optimization.  Numerically solving differential equations forms a cornerstone of modern science and engineering and is intimately linked with predictive science.

A very frequent occurrence in science requires the scientist to find the zeros of a function ![equation](http://latex.codecogs.com/gif.latex?f%5Cleft%28x%5Cright%29).  The input to the function is a m- dimensional vector and the function returns an n- dimensional vector.  We denote this mathematically as ![equation](http://latex.codecogs.com/gif.latex?f%5Cleft%28x%5Cright%29): ![equation](http://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5E%7Bm%7D%20%5Cmapsto%20%5Cmathbb%7BR%7D%5E%7Bn%7D).  This expression is read:  the function ![equation](http://latex.codecogs.com/gif.latex?f%5Cleft%28x%5Cright%29) maps ![equation](http://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5E%7Bm%7D%20%5Cmapsto%20%5Cmathbb%7BR%7D%5E%7Bn%7D).
In CS207, we explored the finite difference method, but we also computed a symbolic derivative.  The finite difference approach is nice because it is quick and easy.  However, it suffers from accuracy and stability problems.  On the other hand, symbolic derivatives can be evaluated to machine precision, but can be costly to evaluate.
Automatic differentiation (AD) overcomes both of these deficiencies. It is less costly than symbolic differentiation while evaluating derivatives to machine precision.  There are two modes of automatic differentiation: forward and reverse.  This library will be primarily concerned with the forward mode. (Lecture 9)

# How to Use Package (TODO)
To use, first create a new virtual environment in order to develop with the package without polluting the global environment with dependencies. To do so, install virtualenv with the command 'sudo easy_install virtualenv'. Next, go to the top level of your project directory and create a new virtual environment with the command 'virtualenv [name]'. To activate the environment, type the command '[name] env/bin/activate'. Thus far, you have set up and activated your dev environment and can begin interacting with the AutoDiff package. To install the package, type in the command line 'python3 -m pip install --index-url https://test.pypi.org/simple/ [TODO: name of package]'. You should see an output like so:

Collecting [TODO]
  Downloading https://test-files.pythonhosted.org/packages/.../[TODO]-0.0.1-py3-none-any.whl
Installing collected packages: [TODO]
Successfully installed [TODO]-0.0.1 

Congratulations, you have installed [TODO]. To check that the installation was successful, run the python interpreter by typing 'python'. Import the module and do a simple operation such as printing out the name of the package like so:

>>> import [TODO]
>>> [TODO].name
'[TODO]'

If your screen looks like the above, you have successfully installed [TODO]!

Include a basic demo for the user. Come up with a simple function to differentiate and walk the user through the steps needed to accomplish that task.

# Background

The basic principles behind automatic differentiation rely on calculating the derivative of a function by splitting the calculation into a number of parts, and can be approached using a number of equivalent methodologies. The simplest illustration of the forward mode of AD is taking the chain rule and treating a tricky function as a composite function of a series of elementary operations. The product of each derivative, building up through each of the elementary functions, gives a computationally simple and accurate method for evaluating difficult derivatives.

For functions that map ![equation](http://latex.codecogs.com/gif.latex?R%5Em) to ![equation](http://latex.codecogs.com/gif.latex?R%5En), we can see that this method of computing the derivative is equivalent to computing the Jacobian. This can be computationally illustrated through:

![equation](http://latex.codecogs.com/gif.latex?D_%7Bp%7Dx%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7Bm%7D%7B%5Cdfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20x_%7Bj%7D%7Dp_%7Bj%7D%7D)

where we take x as our m-dimensional input vector, and ![equation](http://latex.codecogs.com/gif.latex?p) as a seed vector that is all 0's except for a 1 at the dimension we desire to compute for the Jacobian, thus filling in each of the entries.

Finally, for the implementation in our software package, we take advantage of the construction of dual numbers, which is an algebra with the following construction:

Given any number x, rewrite it as: ![equation](http://latex.codecogs.com/gif.latex?x%20%3D%20x&plus;%5Cepsilon%20x%27), where ![equation](http://latex.codecogs.com/gif.latex?%5Cepsilon) has the property such that ![equation](http://latex.codecogs.com/gif.latex?%5Cepsilon%5E2%20%3D%200). This construction is extremely useful because it enables the automatic computation of derivatives, provided the initial derivative at any given x upon instantiation, simply by expanding the formula and computing algebraically the equivalent dual number solution. For the purposes of our implementation of the forward mode of automatic differentiation, we will use dual numbers to compute the appropriate derivatives at each step.


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

# How to Use AutoDiff

The user will obtain our package through PyPI. The user will first install the package then import it into their project file.
Example use:

```Python
import autodiff as AD

x1 = AD.node()
x2 = AD.node()

y = AD.exp(x1) + x2

x1.set_value(0)
x2.set_value(3)

print(y.evaluate) # (4, [0, 1]): the first element in the tuple represents the value, and the second represents the gradients
```

The user can instantiate multiple nodes and apply any operators outlined in the `Operators` module. The value and the gradients of the node can be accessed by using the `evaluate` method.

# Implementation
We will be implementing the automatic differentiation by using `Node` instances and its subclasses, which are defined in `Operators` module. We can also visualize the algorithm using `Visualization` module.

## What are the core data structures?
We will be using a custom Dual Numbers implementation, which serves the purpose of both storing the value at each node and propagating the derivative through simplified calculations.

## What classes will you implement
We will be implementing the `Node` class first. We will then extend the `Node` class for each operator, which will form a subclass. Each node will contain overrides for all operations we support, and every operation will use a class method to return another node object as the result. In the creation of the node object, we will append a reference to previous nodes, and thus implicitly create the computational graph through this linking process. We also plan to implement a visualization class, utilizing the saved graphs.

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

# Future
- Use PyPy to improve performace
- Reverse Mode
- Visualization
- Matrix
- 



