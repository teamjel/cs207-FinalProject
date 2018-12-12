import pytest
from examples.fractals import *
from autodiff.node import *
# Test newton's method
def test_newtons_method():
  x = Variable("x")
  f = x + 1
  assert(newtons_method(f, -1)[0] == -1.0)
  assert(round(newtons_method(f,0.5, "Finite")[0],2) == -1.0)
