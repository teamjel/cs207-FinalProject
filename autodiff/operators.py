class Log(Node):
  def __init__(self, node, base, name=""):
    super().__init__(name=name)
    self.node = node
    self.base = base

  def der(self, name):
    self.eval()
    g = np.log(self.node.value)
    g_prime = (self.node.der(name) / self.node.value)
    h = np.log(self.base.value)
    h_prme = (self.base.der(name) / self.base.value)
    return (g_prime * h - g * h_prime) / (h **2)

  def eval(self):
    self._value = np.log(self.node.value) / np.log(self.base.value)


class Exp(Node):
  def __init__(self, node, name=None):
    super().__init__(name=name)
    self.

class Sin(Node):
  def __init__(self, node, name=""):
    super().__init__(name=name)
    self.node = node

  def der(self, name):
    self.eval()
    return np.cos(self.node.value) * self.node.der(name)

  def eval(self):
    self._value = np.sin(self.node.value)

class Cos(Node):
  def __init__(self, node, name=""):
    super().__init__(name=name)
    self.node = node

  def der(self, name):
    return -np.sin(self.node.value) * self.node.der(name)

  def eval(self):
    self._value = np.cos(self.node.value)

class Tan(Node):
  def __init__(self, node, name=""):
    super().__init__(name=name)
    self.node = node

  def der(self, name):
    return -np.sin(self.node.value) * self.node.der(name)

  def eval(self):
    self._value = np.cos(self.node.value)

class Neg(Node):
  def __init__(self, node, name=None)
    super().__init__(name=name)
    self.node = node

  def der(self, node):
    return -self.node.der()

  def eval(self):
    self.value = -self.node.value
