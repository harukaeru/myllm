import numpy as np


def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x)
  for idx in range(x.size):
    point = x[idx]

    x[idx] = point + h
    val0 = f(x)

    x[idx] = point - h
    val1 = f(x)

    grad[idx] = (val0 - val1) / (2 * h)
    x[idx] = point
  return grad

a = np.array([3.0, 4.0])
print(numerical_gradient(lambda x: np.sum(x ** 2), a))