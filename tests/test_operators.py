import pytest
import numpy as np
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
    # function: sin(d), derivative: cos(d)
    d = Variable("d")
    sinD = sin(d)
    assert (int(sinD(d = np.pi).value()) == 0)
    assert (int(sinD.derivative()["d"]) == -1)
    assert (int(sinD(d = np.pi/2).value()) == 1)
    assert (int(sinD.derivative()["d"]) == 0)
    # function: cos(d), derivative: -sin(d)
    e = Variable("e")
    cosE = cos(e)
    assert (int(cosE(e = np.pi).value()) == -1)
    assert (int(cosE.derivative()["e"]) == 0)
    assert (int(cosE(e = np.pi/2).value()) == 0)
    assert (int(cosE.derivative()["e"]) == -1)
    # function: sqrt(f), derivative: 1/2(x)^(-1/2)
    f = Variable("f")
    sqrtF = sqrt(f)
    assert (int(sqrtF(f = 4).value()) == 2)
    assert (round(sqrtF.derivative()["f"], 3) == 0.25)
    # function: e^h, derivative: e^h
    h = Variable("h")
    expH = exp(h)
    assert (round(expH(h = 2).value(), 2) == 7.39)
    assert (round(expH.derivative()["h"], 2) == 7.39)
    i = Variable("i")
    tani = tan(i)
    assert(int(tani(i = np.pi).value()) == 0)
    assert(int(tani.derivative()["i"]) == 1)
    j = Variable("j")
    arcsinj = arcsin(j)
    assert(round(arcsinj(j = 0.5).value(), 2) == 0.52)
    assert(round(arcsinj.derivative()["j"], 2) == 1.15)
    k = Variable("k")
    arccosk = arccos(k)
    assert(round(arccosk(k = 0.5).value(), 2) == 1.05)
    assert(round(arccosk.derivative()["k"], 2) == -1.15)
    l = Variable("l")
    arctanl = arctan(l)
    assert(round(arctanl(l = 0.5).value(), 2) == 0.46)
    assert(round(arctanl.derivative()["l"], 2) == 0.80)
    m = Variable("m")
    sinhm = sinh(m)
    assert(round(sinhm(m = 0.5).value(), 2) == 0.52)
    assert(round(sinhm.derivative()["m"], 2) == 1.13)
    n = Variable("n")
    coshn = cosh(n)
    assert(round(coshn(n = 0.5).value(), 2) == 1.13)
    assert(round(coshn.derivative()["n"], 2) == 0.52)
    o = Variable("o")
    tanho = tanh(o)
    assert(round(tanho(o = 0.5).value(), 2) == 0.46)
    assert(round(tanho.derivative()["o"], 2) == 0.79)
    p = Variable("p")
    logisticp = logistic(p)
    assert(round(logisticp(p = 0.5).value(), 2) == 0.62)
    assert(round(logisticp.derivative()["p"], 2) == 0.24)

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
    with pytest.raises(TypeError):
        negB(b = "hi")
    with pytest.raises(TypeError):
        negB(c = 3)
    with pytest.raises(TypeError):
        negB.derivative(b)
    # function: sin(d), derivative: cos(d)
    d = Variable("d")
    sinD = sin(d)
    with pytest.raises(TypeError):
        sinD.derivative(d)
    with pytest.raises(TypeError):
        sinD(d = "hi")
    with pytest.raises(TypeError):
        sinD(e = 5)
    # function: cos(d), derivative: -sin(d)
    e = Variable("e")
    cosE = cos(e)
    with pytest.raises(TypeError):
        cosE.derivative(e)
    with pytest.raises(TypeError):
        cosE(e = "hi")
    with pytest.raises(TypeError):
        cosE(f = 4)
    f = Variable("f")
    tanf = tan(f)
    with pytest.raises(ZeroDivisionError):
        tanf(f = np.pi/2)
    # function: 3^h, derivative: 3^h*log(3)
    h = Variable("h")
    expH = exp(3)
    with pytest.raises(Exception):
        expH.derivative(h)
    with pytest.raises(TypeError):
        expH(h = "hi")
    with pytest.raises(TypeError):
        expH(i = 3)
    i = Variable("i")
    arcsini = arcsin(i)
    with pytest.raises(ZeroDivisionError):
        arcsini(i = 1)
    j = Variable("j")
    arccosj = arccos(j)
    with pytest.raises(ZeroDivisionError):
        arccosj(j = 1)


def test_unary_vector_result():
    # function: a, derivative: 1
    test_arr = np.array([1, 1.5])
    # a = Variable("a")
    # assert (a(a = test_arr).value() == [0, 1, 1.5])
    # assert (a.derivative() == [1, 1, 1])
    # function: -b, derivative: -1
    b = Variable("b")
    negB = -b
    assert (np.array_equal(negB(b = test_arr).value(), np.array([-1, -1.5])))
    assert (np.array_equal(negB.derivative()["b"], np.array([-1, -1])))
    # function: sin(d), derivative: cos(d)
    d = Variable("d")
    sinD = sin(d)
    assert (np.array_equal(np.round(sinD(d = test_arr).value(), 2), np.array([0.84, 1])))
    assert (np.array_equal(np.round(sinD.derivative()["d"], 2), np.array([0.54, 0.07])))
    # function: cos(d), derivative: -sin(d)
    e = Variable("e")
    cosE = cos(e)
    assert (np.array_equal(np.round(cosE(e = test_arr).value(), 2), np.array([0.54, 0.07])))
    assert (np.array_equal(np.round(cosE.derivative()["e"], 2), np.array([-0.84, -1])))
    # function: sqrt(f), derivative: 1/2(x)^(-1/2)
    f = Variable("f")
    sqrtF = sqrt(f)
    assert (np.array_equal(np.round(sqrtF(f = test_arr).value(), 2), np.array([1, 1.22])))
    assert (np.array_equal(np.round(sqrtF.derivative()["f"], 2), np.array([0.5, 0.41])))
    # function: e^h, derivative: e^h
    h = Variable("h")
    expH = exp(h)
    assert (np.array_equal(np.round(expH(h = test_arr).value(), 2), np.array([2.72, 4.48])))
    assert (np.array_equal(np.round(expH.derivative()["h"], 2), np.array([2.72, 4.48])))
    i = Variable("i")
    tani = tan(i)
    assert (np.array_equal(np.round(tani(i = test_arr).value(), 2), np.array([1.56, 14.10])))
    assert (np.array_equal(np.round(tani.derivative()["i"], 2), np.array([3.43, 199.85])))
    j = Variable("j")
    test_arr_arc = np.array([-0.5, 0.5])
    arcsinj = arcsin(j)
    assert (np.array_equal(np.round(arcsinj(j = test_arr_arc).value(), 2), np.array([-0.52, 0.52])))
    assert (np.array_equal(np.round(arcsinj.derivative()["j"], 2), np.array([1.15, 1.15])))
    k = Variable("k")
    arccosk = arccos(k)
    assert (np.array_equal(np.round(arccosk(k = test_arr_arc).value(), 2), np.array([2.09, 1.05])))
    assert (np.array_equal(np.round(arccosk.derivative()["k"], 2), np.array([-1.15, -1.15])))
    l = Variable("l")
    arctanl = arctan(l)
    assert (np.array_equal(np.round(arctanl(l = test_arr_arc).value(), 2), np.array([-0.46, 0.46])))
    assert (np.array_equal(np.round(arctanl.derivative()["l"], 2), np.array([0.80, 0.80])))
    m = Variable("m")
    sinhm = sinh(m)
    assert (np.array_equal(np.round(sinhm(m = test_arr_arc).value(), 2), np.array([-0.52, 0.52])))
    assert (np.array_equal(np.round(sinhm.derivative()["m"], 2), np.array([1.13, 1.13])))
    n = Variable("n")
    coshn = cosh(n)
    assert (np.array_equal(np.round(coshn(n = test_arr_arc).value(), 2), np.array([1.13, 1.13])))
    assert (np.array_equal(np.round(coshn.derivative()["n"], 2), np.array([-0.52, 0.52])))
    o = Variable("o")
    tanho = tanh(o)
    assert (np.array_equal(np.round(tanho(o = test_arr_arc).value(), 2), np.array([-0.46, 0.46])))
    assert (np.array_equal(np.round(tanho.derivative()["o"], 2), np.array([0.79, 0.79])))
    p = Variable("p")
    logisticp = logistic(p)
    assert (np.array_equal(np.round(logisticp(p = test_arr_arc).value(), 2), np.array([0.38, 0.62])))
    assert (np.array_equal(np.round(logisticp.derivative()["p"], 2), np.array([0.24, 0.24])))


# Test binary operators: addition, subtraction, multiplication, division, logarithm, power
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
    # function: log(c), derivative: 1/c
    c = Variable("c")
    logC = log(c)
    assert (int(logC(c = np.exp(1)).value()) == 1)
    assert (round(logC.derivative()["c"], 2) == round(1/np.exp(1), 2))
    assert (round(logC(c = 2).value(), 2) == 0.69)
    assert (round(logC.derivative()["c"], 2) == 0.50)
    log10C = log(c, 10)
    assert (round(log10C(c = 2).value(), 2) == 0.30)
    assert (round(log10C.derivative()["c"], 2) == 0.22)
    # function: g**3, derivative: 3*x^2
    g = Variable("g")
    powerG = g**3
    assert (powerG(g = 3).value() == 27)
    assert (powerG.derivative()["g"] == 27)
    assert (powerG(g = 2).value() == 8)
    assert (powerG.derivative()["g"] == 12)
    h = Variable("h")
    powerGH = g**h
    assert(powerGH(g = 3, h = 2).value() == 9)
    assert(powerGH.derivative()["g"] == 6)
    assert(round(powerGH.derivative()["h"], 2) == 9.89)

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
    with pytest.raises(UnboundLocalError):
        i(2,3)
    # function: log(c), derivative: 1/c
    c = Variable("c")
    logC = log(c)
    with pytest.raises(TypeError):
        logC(c = "hi")
    with pytest.raises(TypeError):
        logC(d = 5)
    with pytest.raises(TypeError):
        logC.derivative(c)
    with pytest.raises(ZeroDivisionError):
        logC(c = 0)
    # function: g**3, derivative: 3*x^2
    g = Variable("g")
    powerG = g**3
    with pytest.raises(Exception):
        powerG.derivative(g)
    with pytest.raises(TypeError):
        powerG(g = "hi")
    with pytest.raises(TypeError):
        powerG(f = 4)

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
    # with pytest.raises(ZeroDivisionError):
    #     y(a = 2, b = 0, c = 0, d = 5)