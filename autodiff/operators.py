import numpy as np
from node import Node

class Constant(Node):
  def __init__(self, value, name=None):
    super().__init__(value=value, name=name)

  def diff(self, name):
    return 0

  def eval(self):
    return self._value

class Log(Node):
  def __init__(self, node, base=np.e, name=None):
    super().__init__(name=name)
    self.node = node if isinstance(node, Node) else Constant(node)
    self.base = base if isinstance(base, Node) else Constant(base)

  def der(self, name):
    self.eval()
    g = np.log(self.node.value)
    g_prime = (self.node.der(name) / self.node.value)
    h = np.log(self.base.value)
    h_prme = (self.base.der(name) / self.base.value)
    return (g_prime * h - g * h_prime) / (h ** 2)

  def eval(self):
    self._value = np.log(self.node.value) / np.log(self.base.value)

class Neg(Node):
  def __init__(self, node, name=None)
    super().__init__(name=name)
    self.node = node

  def diff(self, name):
    return -self.node.diff(name)

  def eval(self):
    self._value = -self.node.value

class Exp(Node):
  def __init__(self, node, name=None):
    super().__init__(name=name)
    self.node = node if isinstance(node, Node) else Constant(node)

  def diff(self, name):
    self.eval()
    return self.value * self.node.diff(name)

  def eval(self):
    self._value = np.exp(self.node.value)

class Sqrt(Node):
  def __init__(self, node, name=None):
    super().__init__(name=name)
    self.node = node if isinstance(node, Node) else Constant(node)

  def diff(self, name):
    self.eval()
    return 1/(2 * self.val) * self.node.diff(name)

  def eval(self):
    self._value = np.sqrt(self.node.value)

class Sin(Node):
  def __init__(self, node, name=None):
    super().__init__(name=name)
    self.node = node if isinstance(node, Node) else Constant(Node)

  def diff(self, name):
    self.eval()
    return np.cos(self.node.value) * self.node.der(name)

  def eval(self):
    self._value = np.sin(self.node.value)

class Cos(Node):
  def __init__(self, node, name=None):
    super().__init__(name=name)
    self.node = node if isinstance(node, Node) else Constant(Node)

  def diff(self, name):
    return -np.sin(self.node.value) * self.node.der(name)

  def eval(self):
    self._value = np.cos(self.node.value)

