from src.value import Value

class TestValue:
  value = Value(1)

  def test_init(self):
    assert self.value.value == 1
  
  def test_add(self):
    value_to_add = Value(10)
    new_value = self.value + value_to_add
    assert isinstance(new_value, Value)
    assert new_value.value == 11
  
  def test_remove(self):
    value_to_remove = Value(5)
    new_value = self.value - value_to_remove
    assert isinstance(new_value, Value)
    assert new_value.value == -4
  
  def test_multiply(self):
    value_to_multiply = Value(5)
    new_value = self.value * value_to_multiply
    assert isinstance(new_value, Value)
    assert new_value.value == 5

