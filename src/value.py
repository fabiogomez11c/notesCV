import math

class Value:
  def __init__(self, value: float, child = None) -> None:
    self.value = value # the value of the node
    self._grad = 1 # just the local gradient
    self.grad_value = 0  # mean to have the gradient of the whole graph
    self._child = child  # the child node
  
  def __add__(self, other):
    self._grad = 1
    return Value(self.value + other.value, child=self)
  
  def __sub__(self, other):
    self._grad = 1
    return Value(self.value - other.value, child=self)
  
  def __mul__(self, other):
    self._grad = other.value
    return Value(self.value * other.value, child=self)

  def sigmoid(self):
    temp_value = 1 / (1 + math.exp(-self.value))
    self._grad = temp_value * (1 - temp_value)
    return Value(temp_value, child=self)
  
  def binary_loss(self, true_label):
    loss = true_label * math.log(self.value) + (1 - true_label) * math.log(1 - self.value)
    self._grad = -true_label / self.value + (1 - true_label) / (1 - self.value)
    return Value(-loss, child=self)

  def backward(self, parent_grad = 1):
    self.grad_value = self._grad * parent_grad
    if self._child is not None:
      self._child.backward(self.grad_value)


