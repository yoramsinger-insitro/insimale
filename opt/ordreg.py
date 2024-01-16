# Ordinal logit regression

import numpy as np
import math

# Ordinal regression loss (uniform class-confusion cost)
def ordreg_loss(X, Y, w, b):
  Z = (X @ w).reshape(-1, 1) - b.reshape(1, -1)
  return np.mean(np.log1p(np.exp(-Z * Y)))

# Ordinal regression gradient (for uniform class-confusion cost)
def ordreg_grad(X, Y, w, b):
  Z = X @ w.reshape(X.shape[1], 1) - b.reshape(1, len(b))
  Q = Y / (1 + np.exp(Y * Z))
  d_w = np.sum(Q.transpose() @ X, axis=0) / len(Y)
  d_b = -np.mean(Q, axis=0)
  return d_w, d_b

def to_ord(y):
  C = 2 * (np.tri(max(y) + 1)[:,1:]) - 1
  return C[y]

# Ordinal regression learning using adagrad
def ordreg_learn(

def ordlearn()
if __name__ == "__main__":
  n, d, k = 1000, 7, 3 # #examples, #dimensions, #ordinals
  X = np.random.randn(n, d) / np.sqrt(d)
  w = np.random.randn(d) / np.sqrt(d)
  f = X @ w
  b = np.histogram(f, k)[1][1:]
  c = np.sum((f.reshape(-1, 1) - b.reshape(1, -1)) > 0, axis=1)
  Y = to_ord(c)

  b = b[:-1]

  loss = ordloss(X, Y, w, b)
  d_w, d_b = ordgrad(X, Y, w, b)
  print('\nLoss: ', loss.round(4))
  print(d_w.round(3), d_b.round(3))

  loss = ordloss(X, Y, 100 * w, 100 * b)
  d_w, d_b = ordgrad(X, Y, 100 * w, 100 * b)
  print('\nLoss: ', loss.round(3))
  print(d_w.round(3), d_b.round(3))

  loss = ordloss(X, Y, 10 + w, b)
  d_w, d_b = ordgrad(X, Y, 10 + w, b)
  print('\nLoss: ', loss.round(3))
  print(d_w.round(3), d_b.round(3))

  w = np.random.randn(d) / np.sqrt(d)
  loss = ordloss(X, Y, w, b)
  d_w, d_b = ordgrad(X, Y, w, b)
  print('\nLoss: ', loss.round(3))
  print(d_w.round(3), d_b.round(3))
