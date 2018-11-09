import pytest
import math
from autodiff.node import *
from autodiff.operators import *

# Test constant.
def test_constant_results():
    const = Node.make_constant(4)
    assert (const.derivative() == 0)
    assert (const.value() == 4)
    assert (const.eval() == 4)
    const.set_value(1)
    assert (const.derivative() == 0)
    assert (const.value() == 1)
    assert (const.eval() == 1)
    print(const)

def test_constant_errors():
    const = Node.make_constant(4)
    assert (isinstance(const, Constant))
    # with pytest.raises(TypeError):
    #     const.derivative("hi")
    with pytest.raises(AttributeError):
        const(4)

# Test unary operators: negation, log, sin, cos, tan, power, exponential.
def test_unary_result():
    # function: a, derivative: 1
    a = Variable("a")
    assert (a(a = 3).value() == 3)
    assert (a.derivative() == 1)
    # function: -b, derivative: -1
    b = Variable("b")
    negB = -b
    assert (negB(b = 3).value() == -3)
    assert (negB.derivative()["b"] == -1)
    # function: log(c), derivative: 1/c
    c = Variable("c")
    logC = log(c)
    assert (int(logC(c = math.exp(1)).value()) == 1)
    assert (round(logC.derivative()["c"], 2) == round(1/math.exp(1), 2))
    assert (round(logC(c = 2).value(), 2) == 0.69)
    assert (round(logC.derivative()["c"], 2) == 0.50)
    log10C = log(c, 10)
    assert (round(log10C(c = 2).value(), 2) == 0.30)
    assert (round(log10C.derivative()["c"], 2) == 0.22)
    # function: sin(d), derivative: cos(d)
    d = Variable("d")
    sinD = sin(d)
    assert (int(sinD(d = math.pi).value()) == 0)
    assert (int(sinD.derivative()["d"]) == -1)
    assert (int(sinD(d = math.pi/2).value()) == 1)
    assert (int(sinD.derivative()["d"]) == 0)
    # function: cos(d), derivative: -sin(d)
    e = Variable("e")
    cosE = cos(e)
    assert (int(cosE(e = math.pi).value()) == -1)
    assert (int(cosE.derivative()["e"]) == 0)
    assert (int(cosE(e = math.pi/2).value()) == 0)
    assert (int(cosE.derivative()["e"]) == -1)
    # function: sqrt(f), derivative: 1/2(x)^(-1/2)
    f = Variable("f")
    sqrtF = sqrt(f)
    assert (int(sqrtF(f = 4).value()) == 2)
    assert (round(sqrtF.derivative()["f"], 3) == 0.25)
    # function: g**3, derivative: 3*x^2
    g = Variable("g")
    powerG = g**3
    assert (powerG(g = 3).value() == 27)
    assert (powerG.derivative()["g"] == 27)
    assert (powerG(g = 2).value() == 8)
    assert (powerG.derivative()["g"] == 12)
    # function: e^h, derivative: e^h
    h = Variable("h")
    expH = exp(h)
    assert (round(expH(h = 2).value(), 2) == 7.39)
    assert (round(expH.derivative()["h"], 2) == 7.39)
    
def test_unary_errors():
    # function: a, derivative: 1
    a = Variable("a")
    assert (isinstance(a, Variable))
    # Error when differentiating before evaluating [TODO: which error?]
    # with pytest.raises(NoValueError):
    #     a.derivative()
    # with pytest.raises(TypeError):
    #     a(a = "hi")
    with pytest.raises(TypeError):
        a(b = 3)
    with pytest.raises(TypeError):
        a(a = None)
    with pytest.raises(TypeError):
        a.derivative(a)
    # function: -b, derivative: -1
    b = Variable("b")
    negB = -b
    # with pytest.raises(NoValueError):
    #     negB.derivative()
    with pytest.raises(TypeError):
        negB(b = "hi")
    with pytest.raises(TypeError):
        negB(c = 3)
    with pytest.raises(TypeError):
        negB.derivative(b)
    # function: log(c), derivative: 1/c
    c = Variable("c")
    logC = log(c)
    # with pytest.raises(NoValueError):
    #     logC.derivative()
    with pytest.raises(TypeError):
        logC(c = "hi")
    with pytest.raises(TypeError):
        logC(d = 5)
    with pytest.raises(TypeError):
        logC.derivative(c)
    with pytest.raises(ZeroDivisionError):
        logC(c = 0)
    # function: sin(d), derivative: cos(d)
    d = Variable("d")
    sinD = sin(d)
    with pytest.raises(TypeError):
        sinD.derivative(d)
    with pytest.raises(TypeError):
        sinD(d = "hi")
    with pytest.raises(TypeError):
        sinD(e = 5)
    # with pytest.raises(NoValueError):
    #     sinD.derivative()
    # function: cos(d), derivative: -sin(d)
    e = Variable("e")
    cosE = cos(e)
    with pytest.raises(TypeError):
        cosE.derivative(e)
    with pytest.raises(TypeError):
        cosE(e = "hi")
    with pytest.raises(TypeError):
        cosE(f = 4)
    # with pytest.raises(NoValueError):
    #     cosE.derivative()
    # function: g**3, derivative: 3*x^2
    g = Variable("g")
    powerG = g**3
    with pytest.raises(Exception):
        powerG.derivative(g)
    with pytest.raises(TypeError):
        powerG(g = "hi")
    with pytest.raises(TypeError):
        powerG(f = 4)
    # with pytest.raises(NoValueError):
    #     powerG.derivative()
    # function: 3^h, derivative: 3^h*log(3)
    h = Variable("h")
    expH = exp(3)
    with pytest.raises(Exception):
        expH.derivative(h)
    with pytest.raises(TypeError):
        expH(h = "hi")
    with pytest.raises(TypeError):
        expH(i = 3)
    # with pytest.raises(NoValueError):
    #     expH.derivative()

# Test binary operators: addition, subtraction, multiplication, division. 
def test_binary_result():
    a = Variable("a")
    b = Variable("b")
    c = a + b
    assert (c(a = 2, b = 3).value() == 5)
    assert (c.derivative()["a"] == 1)
    assert (c.derivative()["b"] == 1)
    e = a - b
    assert (e(a = 2, b = 3).value() == -1)
    assert (e.derivative()["a"] == 1)
    assert (e.derivative()["b"] == -1)
    g = a * b 
    assert (g(a = 3, b = 2).value() == 6)
    assert (g.derivative()["a"] == 2)
    assert (g.derivative()["b"] == 3)
    i = a / b 
    assert (int(i(a = 6, b = 2).value()) == 3)
    assert (round(i.derivative()["a"], 2) == 0.50)
    assert (round(i.derivative()["b"], 2) == -1.50)

def test_binary_errors():
    a = Variable("a")
    b = Variable("b")
    c = a + b
    with pytest.raises(TypeError):
        c.derivative(c)
    with pytest.raises(TypeError):
        c(a = "hi", b = 2)
    with pytest.raises(TypeError):
        c(b = 2)
    with pytest.raises(TypeError):
        c(a = 2, b = 3, c = 1)
    # with pytest.raises(NoValueError):
    #     c.derivative()
    with pytest.raises(UnboundLocalError):
        c(2,3)
    e = a - b
    with pytest.raises(TypeError):
        e.derivative(e)
    with pytest.raises(TypeError):
        e(a = "hi", b = 2)
    with pytest.raises(TypeError):
        e(b = 2)
    with pytest.raises(TypeError):
        e(a = 2, b = 3, c = 1)
    # with pytest.raises(NoValueError):
    #     e.derivative()
    with pytest.raises(UnboundLocalError):
        e(2,3)
    g = a * b 
    with pytest.raises(TypeError):
        g.derivative(g)
    with pytest.raises(TypeError):
        g(a = "hi", b = 2)
    with pytest.raises(TypeError):
        g(b = 2)
    with pytest.raises(TypeError):
        g(a = 2, b = 3, c = 1)
    # with pytest.raises(NoValueError):
    #     g.derivative()
    with pytest.raises(UnboundLocalError):
        g(2,3)
    i = a / b 
    with pytest.raises(TypeError):
        i.derivative(i)
    with pytest.raises(TypeError):
        i(a = "hi", b = 2)
    with pytest.raises(TypeError):
        i(b = 1)
    with pytest.raises(TypeError):
        i(a = 2, b = 3, c = 1)
    with pytest.raises(ZeroDivisionError):
        i(a = 2, b = 0)
    # with pytest.raises(NoValueError):
    #     i.derivative()
    with pytest.raises(UnboundLocalError):
        i(2,3)

# Test composite function: y = cos((-a)^2 / c) - 4*sin(b) * log_10(e^d + 1)'''
def test_composition_result():
    a = Variable("a")
    b = Variable("b")
    c = Variable("c")
    d = Variable("d")
    y = cos((-a)**2/c) - 4*sin(b) * log(exp(d) + 1, 10)
    assert (round(y(a = 2, b = 3, c = -1, d = 4).value(), 2) == -1.64)
    # partial derivative dy/da: -(2 a sin(a^2/c))/c
    assert (round(y.derivative()["a"], 2) == 3.03)
    # partial derivative dy/db: -(4 cos(b) log(1 + e^d))/log(10)
    assert (round(y.derivative()["b"], 2) == 6.91)
    # partial derivative dy/dc: (a^2 sin(a^2/c))/c^2
    assert (round(y.derivative()["c"], 2) == 3.03)
    # partial derivative dy/dd: -(4 e^d log(e) sin(b))/((1 + e^d) log(10))
    assert (round(y.derivative()["d"], 2) == -0.24)

def test_composition_errors():
    a = Variable("a")
    b = Variable("b")
    c = Variable("c")
    d = Variable("d")
    e = Variable("e")
    y = cos((-a)**2/c) - 4*sin(b) * log(exp(d) + 1, 10)
    with pytest.raises(TypeError):
        y.derivative(y)
    with pytest.raises(TypeError):
        y(a = "hi", b = 3, c = -1, d = 4)
    with pytest.raises(TypeError):
        y(a = 2, b = "hi", c = -1, d = 4)
    with pytest.raises(TypeError):
        y(a = 2, b = 3, c = "hi", d = 4)
    with pytest.raises(TypeError):
        y(a = 2, b = 3, c = -1, d = "hi")
    with pytest.raises(TypeError):
        y(a = 2, b = 3, c = -1)
    with pytest.raises(UnboundLocalError):
        y(2, 3,-1, 4)
    # with pytest.raises(NoValueError):
    #     y.derivative()
    # with pytest.raises(ZeroDivisionError):
    #     y(a = 2, b = 0, c = 0, d = 5)