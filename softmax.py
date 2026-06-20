import numpy as np


a = np.array([100000, 2.9, 4.0])

def softmax(inp):
  m = np.max(inp)
  print(inp)
  inp = inp - m
  print('aft', inp)
  expd = np.exp(inp)
  summed = np.sum(expd)
  return expd / summed

print(softmax(a))