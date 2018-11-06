import pytest
import autodiff as AD

# Test constant.
def test_constant_results():
    const = AD.Constant()
    assert (const.derivative() == 0)
    assert (const.get_value() == const.val)
    assert (const.eval() == const.val)

def test_constant_types():
    with pytest.raises(TypeError):
        const.derivative("hi")
    with pytest.raises(TypeError):
        const.get_value("hi")
    with pytest.raises(TypeError):
        const.eval("hi")

def test_constant_zerocoeff():
    pass        
    
# Test unary operators: negation, log, sin, cos, tan, exponential, power.
def test_unary_result():
    a = AD.Var()
    assert (a.derivative() == 1)
    assert (a.get_value() == const.val)
    assert (a.eval() == const.val)  
    b = AD.Var()
    c = AD.Var()
    d = AD.Var()
    assert node.quad_roots(1.0, 1.0, -12.0) == ((3+0j), (-4+0j))

def test_unary_types():
    with pytest.raises(TypeError):
        roots.quad_roots("", "green", "hi")

def test_unary_zerocoeff():
    with pytest.raises(ValueError):
        roots.quad_roots(a=0.0)

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