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
    assert (negB.differentiate(b ) == -1)
    # function: log(c), derivative: 1/c
    c = AD.Var()
    logC = AD.Log(c)
    assert (logC.eval(math.exp(1)) == 1)
    assert (round(logC.differentiate(), 2) == round(1/math.exp(1), 2))
    assert (round(logC.eval(2), 2) == 0.69)
    assert (round(logC.differentiate(), 2) == 0.50)
    # function: sin(d), derivative: cos(d)
    d = AD.Var()
    sinD = AD.Sin(d)
    assert (sinD.eval(math.pi) == 0)
    assert (sinD.differentiate() == -1)
    assert (sinD.eval(math.pi/2) == 1)
    assert (sinD.differentiate() == 0)
    # function: cos(d), derivative: -sin(d)
    e = AD.Var()
    cosE = AD.Cos(e)
    assert (cosE.eval(math.pi) == -1)
    assert (cosE.differentiate() == 0)
    assert (cosE.eval(math.pi/2) == 0)
    assert (cosE.differentiate() == -1)
    # function: tan(f), derivative: sec^2(f)
    f = AD.Var()
    tanF = AD.Tan(f)
    assert (tanF.eval(math.pi) == 0)
    assert (tanF.differentiate() == 1)
    # function: g**3, derivative: 3*x^2
    g = AD.Var()
    powerG = g**3
    assert (powerG.eval(3) == 27)
    assert (powerG.differentiate() == 27)
    assert (powerG.eval(2) == 8)
    assert (powerG.differentiate() == 12)
    # function: 3^h, derivative: 3^h*log(3)
    h = AD.Var()
    expH = AD.Exp(h, 3)
    assert (expH.eval(2) == 9)
    assert (round(expH.differentiate(), 2) == 9.89)
    
def test_unary_errors():
    # function: a, derivative: 1
    a = AD.Var()
    # Error when differentiating before evaluating [TODO: which error?]
    with pytest.raises(Exception):
        a.differentiate()
    with pytest.raises(TypeError):
        a.eval("hi")
    # function: -b, derivative: -1
    b = AD.Var()
    negB = -b
    with pytest.raises(Exception):
        negB.differentiate()
    with pytest.raises(TypeError):
        negB.eval("hi")
    # function: log(c), derivative: 1/c
    c = AD.Var()
    logC = AD.Log(c)
    with pytest.raises(Exception):
        logC.differentiate()
    with pytest.raises(TypeError):
        logC.eval("hi")
    with pytest.raises(ValueError):
        logC.eval(0)
    # function: sin(d), derivative: cos(d)
    d = AD.Var()
    sinD = AD.Sin(d)
    with pytest.raises(Exception):
        sinD.differentiate()
    with pytest.raises(TypeError):
        sinD.eval("hi")
    # function: cos(d), derivative: -sin(d)
    e = AD.Var()
    cosE = AD.Cos(e)
    with pytest.raises(Exception):
        cosE.differentiate()
    with pytest.raises(TypeError):
        cosE.eval("hi")
    # function: tan(f), derivative: sec^2(f)
    f = AD.Var()
    tanF = AD.Tan(f)
    with pytest.raises(Exception):
        tanF.differentiate()
    with pytest.raises(TypeError):
        tanF.eval("hi")
    with pytest.raises(ValueError):
        tanF.eval(math.pi/2)
    # function: g**3, derivative: 3*x^2
    g = AD.Var()
    powerG = g**3
    with pytest.raises(Exception):
        powerG.differentiate()
    with pytest.raises(TypeError):
        powerG.eval("hi")
    # function: 3^h, derivative: 3^h*log(3)
    h = AD.Var()
    expH = AD.Exp(h, 3)
    with pytest.raises(Exception):
        expH.differentiate()
    with pytest.raises(TypeError):
        expH.eval("hi")

# Test binary operators: addition, subtraction, multiplication, division. 
def test_binary_result():
    assert node.quad_roots(1.0, 1.0, -12.0) == ((3+0j), (-4+0j))

def test_binary_types():
    with pytest.raises(TypeError):
        roots.quad_roots("", "green", "hi")

def test_binary_zerocoeff():
    with pytest.raises(ValueError):
        roots.quad_roots(a=0.0)

# Test composite function: y = (-a)^2 / cos(c) - sin(b) * 4^d + 1 '''
def test_composition_result():
    pass

def test_composition_types():
    with pytest.raises(TypeError):
        roots.quad_roots("", "green", "hi")

def test_composition_zerocoeff():
    with pytest.raises(ValueError):
        roots.quad_roots(a=0.0)