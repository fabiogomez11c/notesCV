import pytest
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

  def test_sigmoid(self):
    new_value = self.value.sigmoid()
    assert isinstance(new_value, Value)
    assert new_value.value == pytest.approx(0.7310585786300049)
  
  def test_binary_loss(self):
    pre_loss = Value(0.5)
    # check if the true label is 1
    binary_loss = pre_loss.binary_loss(1)
    assert isinstance(binary_loss, Value)
    assert binary_loss.value == pytest.approx(0.6931471805599453)
    assert binary_loss.value != pytest.approx(0.5)
    # check if true value is 0
    binary_loss = pre_loss.binary_loss(0)
    assert isinstance(binary_loss, Value)
    assert binary_loss.value == pytest.approx(0.6931471805599453)
    assert binary_loss.value != pytest.approx(0.5)

    pre_loss = Value(0.1)
    # check if the true label is 1
    binary_loss = pre_loss.binary_loss(1)
    assert isinstance(binary_loss, Value)
    assert binary_loss.value == pytest.approx(2.3025850929940455)
    assert binary_loss.value != pytest.approx(0.5)
    # check if the true label is 0
    binary_loss = pre_loss.binary_loss(0)
    assert isinstance(binary_loss, Value)
    assert binary_loss.value == pytest.approx(0.10536051565782628)
    assert binary_loss.value != pytest.approx(0.5)

    pre_loss = Value(0.9)
    # check if the true label is 1
    binary_loss = pre_loss.binary_loss(1)
    assert isinstance(binary_loss, Value)
    assert binary_loss.value == pytest.approx(0.10536051565782628)
    assert binary_loss.value != pytest.approx(0.5)
    # check if the true label is 0
    binary_loss = pre_loss.binary_loss(0)
    assert isinstance(binary_loss, Value)
    assert binary_loss.value == pytest.approx(2.302585092994046)
    assert binary_loss.value != pytest.approx(0.5)
  
  def test_backpropagation(self):
    x_var = Value(1)
    w_var = Value(0.5)
    b_var = Value(0.1)
    wx = w_var * x_var
    z_var = wx + b_var
    activation = z_var.sigmoid()
    y_true = Value(1)
    loss = activation.binary_loss(y_true.value)
    loss.backward()

    # compute the same loss function with tiny increment h
    h_var = Value(0.00001)
    x_h = Value(1)
    w_h = Value(w_var.value + h_var.value)
    b_h = Value(0.1)
    w_hx_h = w_h * x_h
    z_h = w_hx_h + b_h
    activation_h = z_h.sigmoid()
    y_true = Value(1)
    loss_h = activation_h.binary_loss(y_true.value)
    gradient = (loss_h.value - loss.value) / h_var.value

    assert gradient == pytest.approx(w_var.grad_value, 0.0001)
  
  def test_update(self):
    x_var = Value(1)
    w_var = Value(0.5)
    b_var = Value(0.1)
    wx = w_var * x_var
    z_var = wx + b_var
    activation = z_var.sigmoid()
    y_true = Value(1)
    loss = activation.binary_loss(y_true.value)
    loss.backward()
