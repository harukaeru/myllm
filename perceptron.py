import numpy as np

def AND(inp):
  print(inp)
  weight = np.array([0.5, 0.5])
  bias = -0.7
  return 0 if np.sum(inp * weight) + bias <= 0 else 1

def NAND(inp):
  print(inp)
  weight = np.array([-0.5, -0.5])
  bias = 0.7
  return 0 if np.sum(inp * weight) + bias <= 0 else 1

def OR(inp):
  print(inp)
  weight = np.array([1, 1])
  bias = -0.7
  return 0 if np.sum(inp * weight) + bias <= 0 else 1

def XOR(inp):
  print('--', inp)
  one11 = OR(inp)
  one00 = NAND(inp)
  return AND(np.array([one00, one11]))





print(XOR(np.array([0, 0])))
print(XOR(np.array([0, 1])))
print(XOR(np.array([1, 0])))
print(XOR(np.array([1, 1])))