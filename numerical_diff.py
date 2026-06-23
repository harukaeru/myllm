def func1(x):
  return 0.01 * (x ** 2) + 0.1 * x


def numerical_diff(func, x):
  h = 1e-4
  return (func(x + h) - func(x - h)) / (2 * h)

print(numerical_diff(func1, 5))
print(numerical_diff(func1, 10))