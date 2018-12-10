# Introduction
Differentiation is one of the most important operations in science.  Finding extrema of functions and determining zeros of functions are central to optimization.  Numerically solving differential equations forms a cornerstone of modern science and engineering and is intimately linked with predictive science.

A very frequent occurrence in science requires the scientist to find the zeros of a function ![equation](http://latex.codecogs.com/gif.latex?f%5Cleft%28x%5Cright%29).  The input to the function is a m- dimensional vector and the function returns an n- dimensional vector.  We denote this mathematically as ![equation](http://latex.codecogs.com/gif.latex?f%5Cleft%28x%5Cright%29): ![equation](http://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5E%7Bm%7D%20%5Cmapsto%20%5Cmathbb%7BR%7D%5E%7Bn%7D).  This expression is read:  the function ![equation](http://latex.codecogs.com/gif.latex?f%5Cleft%28x%5Cright%29) maps ![equation](http://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5E%7Bm%7D%20%5Cmapsto%20%5Cmathbb%7BR%7D%5E%7Bn%7D).
In CS207, we explored the finite difference method, but we also computed a symbolic derivative.  The finite difference approach is nice because it is quick and easy.  However, it suffers from accuracy and stability problems.  On the other hand, symbolic derivatives can be evaluated to machine precision, but can be costly to evaluate.
Automatic differentiation (AD) overcomes both of these deficiencies. It is less costly than symbolic differentiation while evaluating derivatives to machine precision.  There are two modes of automatic differentiation: forward and reverse.  This library will be primarily concerned with the forward mode. (Lecture 9)

# How to Use Package
To use, first create a new virtual environment in order to develop with the package without polluting the global environment with dependencies. To do so, install virtualenv with the command `sudo easy_install virtualenv`. Next, go to the top level of your project directory and create a new virtual environment with the command `virtualenv [name]`. To activate the environment, type the command `[name] env/bin/activate`. Thus far, you have set up and activated your dev environment and can begin interacting with the AutoDiff package. To install the package, type in the command line 'python3 -m pip install -i https://test.pypi.org/simple/ autodiff-jel'. You should see an output like so:

Collecting `autodiff-jel`
  Downloading https://test-files.pythonhosted.org/packages/.../autodiff-jel-0.0.1-py3-none-any.whl
Installing collected packages: autodiff-
Successfully installed autodiff-jel-0.0.1

Congratulations, you have installed autodiff-jel. To check that the installation was successful, run the python interpreter by typing 'python'. Import the module and do a simple operation such as printing out the name of the package like so:

```Python
>>> import autodiff as AD
>>> import numpy as np
>>> AD.name
'autodiff_jel'
```

If your screen looks like the above, you have successfully installed `autodiff`!

Let's now go through a demo. Let us use automatic differentiation on the function sin(x).
We first need to set up the variable and the equation.

```Python
x = AD.Variable("x")
y = AD.sin(x)
```

Next we need to assign a value to each variable during the evaluation call.

```Python
y(x=np.pi)
```

Lastly, we can now get the derivative of the function like so and specify the variable of the partial derivative:

```Python
print (y.derivative()["x"])
-1
```

Congratulations, you can now begin automatically differentiating away!

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
from autodiff import Variable, sin, cos, log, exp

a = Variable("a")
b = Variable("b")
c = Variable("c")
d = Variable("d")
y = cos((-a)**2/c) - 4*sin(b) * log(exp(d) + 1, 10)

y(a = 2, b = 3, c = -1, d = 4)

print(round(y.value(),2))
-1.64

print(round(y.derivative()["a"], 2))
3.03

print(y)
Node(Function = 'Subtraction', Value = -1.638695338498409, Derivative = {'b': 6.910386575432481, 'd': -0.24074123364509895, 'c': 3.027209981231713, 'a': 3.027209981231713})
```

The user can instantiate multiple nodes and apply any operators outlined in the `Operators` module. The value and the gradients of the node can be accessed by using the `value` and `derivative` methods respectively. Our implementation is meant to be as intuitive as possible, allowing natural manipulation of formula expressions through extensive use of python magic methods, and built in functions for handling the most common math functions. In addition, every node saves its values upon computation at any given point, allowing for more extensive analysis at different points and the capability of implementing a visualization module (to be completed for the next milestone).

# Implementation
Our implementation centers around the use of the class `Node`, which is an abstract class defining a single operation. Nodes may be `Constant`s and `Variable`s, which are reflexive functions that simply return their value. Every node is equivalently a dual number store, as it contains both the real value part, and the dual differentiated part at that point in the graph. Nodes are built upon one another by the `children` attribute, which contain all the lower-level nodes that are involved in the computation of the current node. This implementation seeks to elegantly reconstruct the basis of automatic differentiation, the computational graph, in implementing both the forward and reverse modes (and potential for visualization), and thus necessitates the storage of the derivative values as they are propagated through the graph.

## What are the core data structures?
The core data structures `Variable`s, which are symbolic at initialization, and are given a value at computation. Computation is invoked by calling any node directly with either a dictionary or a direct keyword list, where the keys to both are simply the names of the variables at instantiation. The variables needed for any given node are only those which are involved directly in the computation up until that node, meaning the recursive list of all variables involved in that node's children. When computation is called, both the values, and the partial derivatives (for every variable involved), are propagated from the Variables to the node from which computation is called, for the forward mode. The data is stored automatically through use of a decorator factory that makes defining any new operations extremely simple, and pain-free by requiring only numerical computation in subclass implementations of new functions, and no handling of the internals of our implementation.

## What classes will you implement
We implement the base Node class representing a function, which necessitates subclassing and specifically the overriding of the `eval` and `diff` methods. These methods, when combined with the provided `node_decorator`, will automatically pass `(values)` and `Cvalues, diffs)` to `eval` and `diff` respectively, which are lists of the values and immediate derivatives of all children nodes. This means that any user-subclassed custom functions will only need to numerically handle the value computation and dual-number based derivative computation and return that output, and the rest of the implementation will work. Furthermore, we use a seed-based derivative system where partials are computed one at a time (essentially passing all requisite variables a one-hot kind of vector in their derivatives to compute one partial), meaning that implementations of the derivative can remain univariate in output, simplifying computation.

## What method and name attributes will your classes have?

`Node`

The Node class is the core structure. It is the basis of all classes in `Operators` module. Below are the method and name attributes of the `Node` class.

```
class Node:
  Attributes:
    _value: Value at the current node; holds values for the most recent computation
    _derivative: Derivative/Gradients of the node in dictionary form
    _variables: All variables involved in the computation of this node
    _cur_var: Marker for determining the current partial being computed when iterating through all seed values (in computing full Jacobian)
    children: A list of all children nodes which are involved in this computation
    type: String describing the type of computation or node this is
  Methods:
    ### Class methods ###
    @classmethod
    make_constant(cls, value): Class method for constructing a Constant node

    @classmethod
    make_node(cls, node, *values): Important class method that takes in a new Node instance, and properly instantiates it with children from the unpacked values argument list (which can include both numeric values and nodes)

    ### Magic Methods ###
    __call__(self): Convenience wrapper for calling the compute function, which computes the node value and derivatives at given point
    __repr__(self): Representation of node with values, derivatives, and type of function
    __add__(self, value): Constructs an Addition node
    __radd__(self, value): ^
    __neg__(self): Constructs a Negation node
    __sub__(self, value): Constructs a Subtraction node
    __rsub__(self, value): ^
    __mul__(self, value): Constructs a Multiplication node
    __rmul__(self, value): ^
    __truediv__(self, value): Constructs a Division node
    __rtruediv__(self, value): ^
    __pow__(self, value): Constructs a Power node
    __rpow__(self, value): ^

    ### Attribute Methods ###
    value(self): Function for returning the value at the current node
    derivative(self): Function for returning the derivatives at the current node
    set_value(self, value): Set a value
    set_derivative(self, value): Set a derivative
    set_children(self, *children): Give current node children

    ### Variable Methods ###
    update_variables(self): Called when constructing a new node. This computes the minimal set of variables involved among the children, and sets the current node's variables reference appropriately
    set_variables(self, input_dict): Called at computation, and sets all variables to the given values defined by input_dict. Note that the same minimal set of variables is referenced by all nodes that use it, so this function can be called from anywhere further in the computational graph
    update_cur_var(self): Find the current partial by looking at which variable has been seeded properly. This is necessary as a way to let other nodes not directly calling the compute method what variable the current partial is with regard to.
    iterate_seeds(self): A generator that is responsible for iterating among all partials necessary at the current node to find the full gradient

    ### Computation Methods ###
    compute(self, *args, **kwargs): Method that initates the full computation through all children by taking in either an input dictionary (such as {'x': 4, 'y': 3}), or keyword pairs (such as (x=4, y=3)) with the variables referenced by the name they were instantiated with. Returns self once all values are updated.
    eval(self, values): A method to be overriden by subclasses, which will implement the actual calculation of the value itself by the function this node is responsible for. Usage of the decorator node_decorate greatly simplifies this implementation, see specifics below
    diff(self, values, diffs): Like eval, to be overriden and implemented with the dual-number solution to the derivative of the current function.

class node_decorate:
  This is a decorator implemented as a class (can be implemented as a function, but less elegant) that serves as a decorator factory for methods that need to be overrided: eval, and diff. This class allows all subclasses of Node to only worry about implementing a purely numerical method for computation of values and derivatives, and will handle all the logistics necessary to both save those values at each point in the graph automatically, and properly expose and propagate data as necessary to the function.

class Variable:
  Subclass of Node that implements a simple Variable. Basis of all computation in forward-mode.
class Constant:
  Convenience class constructed automatically when a constant shows up in computation.
class Addition, Subtraction ...
  Subclasses of Node that implement elementary functions.
```

`Operators`

Classes here contain additional Node types that define common operations such as sin, log, exp, etc. These also contain the constructor for these nodes - recall that Node is simply the symbolic representation of a function in the graph, and not an actual computation directly initializable by the user. For that, a more familiar and intuitive approach is provided by the built in functions (denoted with lowercase letters) sin, log, exp, etc. that will take in either numeric values or nodes, and output the appropriate respective Sin, Log, Exp nodes.

## What external dependencies will you rely on?

We rely mostly on numpy for efficient computation; other requirements will be specified as they arise (when we implement visualization, for example).

We will specifically leverage [matrix operations](https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.html) and [universal functions](https://docs.scipy.org/doc/numpy-1.15.1/reference/ufuncs.html) from `Numpy`.

## How will you deal with elementary functions like sin and exp?

Elementary functions including trigonometric functions, logarithmic functions, and exponential functions, will be accounted for in `Operators.py`, which are subclasses of `Node`. These will be naturally handled and can be user overriden by direct import, so that they can use intuitive expressions like `sin(x+y/4)`

# Future

Below are a few things we would like to implement:
- Allow vector inputs
- Use PyPy to improve performace
- Implement Reverse Mode
- Visualization (Computation Graph, Computation Table)



