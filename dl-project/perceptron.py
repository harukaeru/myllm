def AND(x1, x2):
  print((x1, x2))
  w1, w2, bias = 0.5, 0.5, -0.7 
  return 0 if bias + x1 * w1 + x2 * w2 <= 0 else 1


print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))