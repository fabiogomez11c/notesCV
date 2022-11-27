import math

class Value:
  def __init__(self, value: int) -> None:
    self.value = value
  
  def __add__(self, other):
    return Value(self.value + other.value)
  
  def __sub__(self, other):
    return Value(self.value - other.value)
  
  def __mul__(self, other):
    return Value(self.value * other.value)

  def sigmoid(self):
    return Value(1 / (1 + math.exp(-self.value)))
  
  def binary_loss(self, true_label):
    loss = true_label * math.log(self.value) + (1 - true_label) * math.log(1 - self.value)
    return Value(-loss)









