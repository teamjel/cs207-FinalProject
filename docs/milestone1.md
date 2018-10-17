# Introduction

# Background

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