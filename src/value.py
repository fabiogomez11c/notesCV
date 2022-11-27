
class Value:
  def __init__(self, value: int) -> None:
    self.value = value
  
  def __add__(self, other):
    return Value(self.value + other.value)
  
  def __sub__(self, other):
    return Value(self.value - other.value)
  
  def __mul__(self, other):
    return Value(self.value * other.value)

class Layer:
  pass












