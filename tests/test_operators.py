import pytest
import math
import autodiff as AD

# Test constant.
def test_constant_results():
    const = AD.Constant()
    assert (const.differentiate() == 0)
    assert (const.get_value() == const.val)
    assert (const.eval() == const.val)

def test_constant_types():
    const = AD.Constant()
    with pytest.raises(TypeError):
        const.differentiate("hi")
    with pytest.raises(TypeError):
        const.eval("hi")

def test_constant_zerocoeff():
    pass        
    
# Test unary operators: negation, log, sin, cos, tan, power, exponential.
def test_unary_result():
    # function: a, derivative: 1
    a = AD.Var()
    assert (a.eval(3) == 3)
    assert (a.differentiate(a) == 1)
    # function: -b, derivative: -1
    b = AD.Var()
    negB = -b
    assert (negB.eval(3) == -3)
    assert (negB.differentiate(b) == -1)
    # function: log(c), derivative: 1/c
    c = AD.Var()
    logC = AD.Log(c)
    assert (logC.eval(math.exp(1)) == 1)
    assert (round(logC.differentiate(c), 2) == round(1/math.exp(1), 2))
    assert (round(logC.eval(2), 2) == 0.69)
    assert (round(logC.differentiate(c), 2) == 0.50)
    # function: sin(d), derivative: cos(d)
    d = AD.Var()
    sinD = AD.Sin(d)
    assert (sinD.eval(math.pi) == 0)
    assert (sinD.differentiate(d) == -1)
    assert (sinD.eval(math.pi/2) == 1)
    assert (sinD.differentiate(d) == 0)
    # function: cos(d), derivative: -sin(d)
    e = AD.Var()
    cosE = AD.Cos(e)
    assert (cosE.eval(math.pi) == -1)
    assert (cosE.differentiate(e) == 0)
    assert (cosE.eval(math.pi/2) == 0)
    assert (cosE.differentiate(e) == -1)
    # function: tan(f), derivative: sec^2(f)
    f = AD.Var()
    tanF = AD.Tan(f)
    assert (tanF.eval(math.pi) == 0)
    assert (tanF.differentiate(f) == 1)
    # function: g**3, derivative: 3*x^2
    g = AD.Var()
    powerG = g**3
    assert (powerG.eval(3) == 27)
    assert (powerG.differentiate(g) == 27)
    assert (powerG.eval(2) == 8)
    assert (powerG.differentiate(g) == 12)
    # function: 3^h, derivative: 3^h*log(3)
    h = AD.Var()
    expH = AD.Exp(h, 3)
    assert (expH.eval(2) == 9)
    assert (round(expH.differentiate(h), 2) == 9.89)
    
def test_unary_errors():
    # function: a, derivative: 1
    a = AD.Var()
    # Error when differentiating before evaluating [TODO: which error?]
    with pytest.raises(Exception):
        a.differentiate(a)
    with pytest.raises(TypeError):
        a.eval("hi")
    with pytest.raises(TypeError):
        a.differentiate()
    # function: -b, derivative: -1
    b = AD.Var()
    negB = -b
    with pytest.raises(Exception):
        negB.differentiate(b)
    with pytest.raises(TypeError):
        negB.eval("hi")
    with pytest.raises(TypeError):
        negB.differentiate()
    # function: log(c), derivative: 1/c
    c = AD.Var()
    logC = AD.Log(c)
    with pytest.raises(Exception):
        logC.differentiate(c)
    with pytest.raises(TypeError):
        logC.eval("hi")
    with pytest.raises(TypeError):
        logC.differentiatev()
    with pytest.raises(ValueError):
        logC.eval(0)
    # function: sin(d), derivative: cos(d)
    d = AD.Var()
    sinD = AD.Sin(d)
    with pytest.raises(Exception):
        sinD.differentiate(d)
    with pytest.raises(TypeError):
        sinD.eval("hi")
    with pytest.raises(TypeError):
        sinD.differentiate()
    # function: cos(d), derivative: -sin(d)
    e = AD.Var()
    cosE = AD.Cos(e)
    with pytest.raises(Exception):
        cosE.differentiate(e)
    with pytest.raises(TypeError):
        cosE.eval("hi")
    with pytest.raises(TypeError):
        cosE.differentiate()
    # function: tan(f), derivative: sec^2(f)
    f = AD.Var()
    tanF = AD.Tan(f)
    with pytest.raises(Exception):
        tanF.differentiate(f)
    with pytest.raises(TypeError):
        tanF.eval("hi")
    with pytest.raises(TypeError):
        tanF.differentiate()
    with pytest.raises(ValueError):
        tanF.eval(math.pi/2)
    # function: g**3, derivative: 3*x^2
    g = AD.Var()
    powerG = g**3
    with pytest.raises(Exception):
        powerG.differentiate(g)
    with pytest.raises(TypeError):
        powerG.eval("hi")
    with pytest.raises(TypeError):
        powerG.differentiate()
    # function: 3^h, derivative: 3^h*log(3)
    h = AD.Var()
    expH = AD.Exp(h, 3)
    with pytest.raises(Exception):
        expH.differentiate(h)
    with pytest.raises(TypeError):
        expH.eval("hi")
    with pytest.raises(TypeError):
        expH.differentiate()

# Test binary operators: addition, subtraction, multiplication, division. 
def test_binary_result():
    a = AD.Var()
    b = AD.Var()
    c = a + b
    assert (c.eval({a: 2, b: 3}) == 5)
    assert (c.differentiate(c) == 0)
    assert (c.differentiate(a) == 1)
    assert (c.differentiate(b) == 1)
    e = a - b
    assert (e.eval({a: 2, b: 3}) == -1)
    assert (e.differentiate(e) == 0)
    assert (e.differentiate(a) == 1)
    assert (e.differentiate(b) == -1)
    g = a * b 
    assert (g.eval({a: 3, b: 2}) == 6)
    assert (g.differentiate(g) == 0)
    assert (g.differentiate(a) == 2)
    assert (g.differentiate(b) == 3)
    i = a / b 
    assert (i.eval({a: 6, b: 2}) == 3)
    assert (i.differentiate(i) == 0)
    assert (math.round(i.differentiate(a), 2) == 0.50)
    assert (math.round(i.differentiate(b), 2) == -1.50)

def test_binary_errors():
    a = AD.Var()
    b = AD.Var()
    c = a + b
    with pytest.raises(Exception):
        c.differentiate(c)
    with pytest.raises(TypeError):
        c.eval("hi")
    with pytest.raises(TypeError):
        c.eval(1)
    with pytest.raises(TypeError):
        c.differentiate()
    with pytest.raises(TypeError):
        c.differentiate(z)
    e = a - b
    with pytest.raises(Exception):
        e.differentiate(e)
    with pytest.raises(TypeError):
        e.eval("hi")
    with pytest.raises(TypeError):
        e.eval(1)
    with pytest.raises(TypeError):
        e.differentiate()
    with pytest.raises(TypeError):
        e.differentiate(z)
    g = a * b 
    with pytest.raises(Exception):
        g.differentiate(g)
    with pytest.raises(TypeError):
        g.eval("hi")
    with pytest.raises(TypeError):
        g.eval(1)
    with pytest.raises(TypeError):
        g.differentiate()
    with pytest.raises(TypeError):
        g.differentiate(z)
    i = a / b 
    with pytest.raises(Exception):
        i.differentiate(i)
    with pytest.raises(TypeError):
        i.eval("hi")
    with pytest.raises(TypeError):
        i.eval(1)
    with pytest.raises(TypeError):
        i.differentiate()
    with pytest.raises(TypeError):
        i.differentiate(z)
    with pytest.raises(ValueError):
        i.eval({a:2, b:0})

# Test composite function: y = (-a)^2 / cos(c) - sin(b) * 4^d + 1 '''
def test_composition_result():
    pass

def test_composition_errors():
    with pytest.raises(TypeError):
        roots.quad_roots("", "green", "hi")