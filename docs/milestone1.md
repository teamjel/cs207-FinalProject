# Introduction

# Background

The basic principles behind automatic differentiation rely on calculating the derivative of a function by splitting the calculation into a number of parts, and can be approached using a number of equivalent methodologies. The simplest illustration of the forward mode of AD is taking the chain rule and treating a tricky function as a composite function of a series of elementary operations. The product of each derivative, building up through each of the elementary functions, gives a computationally simple and accurate method for evaluating difficult derivatives.

For functions that map $R^m$ to $R^n$, we can see that this method of computing the derivative is equivalent to computing the Jacobian. This can be computationally illustrated through:

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

The test suite will live in the `tests/` directory, which we will be maintain by using  TravisCI for continuous integration and Coveralls for test coverage. The package will be distributed through PyPI.

# Implementation