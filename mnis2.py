import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from mnist import load_mnist
from PIL import Image

def img_show(img):
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.show()


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
  c = np.max(x)
  x -= c
  return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
  delta = 1e-7
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  return -np.sum(t * np.log(y  + delta)) / batch_size

# def softmax(x):
#     x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
#     return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def get_data():
  (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
  return x_train, t_train

x, t = get_data()
print(x.shape)
print(t.shape)

train_size = x.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x[batch_mask]
t_batch = t[batch_mask]
print(x_batch)
print(t_batch)