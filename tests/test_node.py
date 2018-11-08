import pytest
import autodiff as AD

# Test node overloading unary ops.
def test_unary_node():
    a = AD.Var()
    assert (-a == AD.Neg(a))
    b = AD.Var()
    assert (b**2 == AD.Pow(b, 2))

# Test node overloading binary ops.
def test_binary_node():
    a = AD.Var()
    b = AD.Var()
    assert (a + b == AD.Add(a, b))
    assert (a - b == AD.Subtract(a, b))
    assert (a * b == AD.Multiply(a, b))
    assert (a / b == AD.Divide(a, b))