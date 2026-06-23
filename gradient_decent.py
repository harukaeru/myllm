import numpy as np

def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x)
  for idx in range(x.size):
    point = x[idx]
    x[idx] = point + h
    fxh1 = f(x)

    x[idx] = point - h
    fxh2 = f(x)
    grad[idx] = (fxh1 - fxh2) / (2 * h)
    x[idx] = point
  return grad

  
def gradient_descent(f, init_x, lr=0.1, step_num=100):
  x = init_x
  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad
  return x

def function_2(x):
  return np.sum(x ** 2)


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x))