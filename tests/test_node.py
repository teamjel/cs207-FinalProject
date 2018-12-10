import pytest
from autodiff.node import *

# Test node overloading unary ops.
def test_unary_node():
    a = Variable("a")
    assert (isinstance(-a, Negation))
    b = Variable("b")
    assert (isinstance(b**2, Power))
    assert (isinstance(2**b, Power))
    assert (str(b) == "Node(Function = 'Variable', Value = None, Derivative = {}, name = b)")

def test_unary_node_errors():
    with pytest.raises(NotImplementedError):
        c = Node()
        c.eval(None)
    with pytest.raises(NotImplementedError):
        c = Node()
        c.diff(None, None)
    with pytest.raises(ValueError):
        a = Variable()
    with pytest.raises(UnboundLocalError):
        a = Variable(a)
    with pytest.raises(ValueError):
        a = Variable(4)
    with pytest.raises(TypeError):
        a = Variable("a", "b")
    with pytest.raises(ValueError):
        a = Variable([])
    with pytest.raises(NoValueError):
        a = Variable("a")
        a.eval()
    with pytest.raises(NoValueError):
        a = Variable("a")
        a.set_derivative(None)
        a.diff()
    a = Variable("a")
    with pytest.raises(NameError):
        b = a**t

# Test node overloading binary ops.
def test_binary_node():
    a = Variable("a")
    b = Variable("b")
    assert (isinstance(a + b, Addition))
    assert (isinstance(a + 2, Addition))
    assert (isinstance(2 + b, Addition))
    assert (isinstance(a - b, Subtraction))
    assert (isinstance(a - 2, Subtraction))
    assert (isinstance(2 - b, Subtraction))
    assert (isinstance(a * b, Multiplication))
    assert (isinstance(2 * b, Multiplication))
    assert (isinstance(a * 2, Multiplication))
    assert (isinstance(a / b, Division))
    assert (isinstance(a / 2, Division))
    assert (isinstance(2 / b, Division))