class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y
    return out

  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x
    return dx, dy

class AddLayer:
  def forward(self, x, y):
    out = x + y
    return out

  def backward(self, dout):
    dx = dout
    dy = dout
    return dx, dy



apple = 100
apple_num = 2
tax = 1.1

orange = 150
orange_num = 3

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
aggregate_layer = AddLayer()
mul_tax_layer = MulLayer()


apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
subtotal = aggregate_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(subtotal, tax)

print(price)

# backward
dprice = 1
dsubtotal, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = aggregate_layer.backward(dsubtotal)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

print('apple:', dapple, dapple_num)
print('orange:', dorange, dorange_num)
print('tax:', dtax)
print('subtotal:', dsubtotal)