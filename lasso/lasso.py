#
# Primal-dual solver for least squares regression with L1 & L2 penalties and
# optional positivity constraint. Each sample may be associated with weight.
#

import numpy as np
np.printoptions(precision=4, threshold=20)

# Exposed on purpose but no need to change
accuracy = 1e-6
max_iter = 10
max_epch = 10000

# Hack to save parameter passing
global pos
pos = True

# Keep these as lambda expressions for efficiency or re-implement mindfully
pos_shrk = lambda v, l1, l2: np.maximum(v - l1, 0) / l2
gen_shrk = lambda v, l1, l2: np.sign(v) * pos_shrk(np.abs(v), l1, l2)
shrk = lambda v, l1, l2: pos_shrk(v, l1, l2) if pos else gen_shrk(v, l1, l2)
prox = lambda v, th: np.sign(v) * np.maximum(np.abs(v) - th, 0)
ltwo = lambda v: np.sum(v * v)
lone = lambda v: np.sum(np.abs(v))

# Schwarzenegger test based on current and previous values
schwarz = lambda c, p: abs(c - p) / (abs(c) + abs(p) + accuracy ** 2) < accuracy

def solve_dual_univar(x, y, u, l1r, l2r, a0=0):
  """
    Solve approximaltely a single dual ascent step.
    Input:
      x: single example (vector)
      y: single target (scalar)
      u: pre-shrinkage representation of primal solution
    Output: scalar a -- optimal dual step
  """
  pa = a0
  sk = l2r / (l2r + ltwo(x))
  for k in range(1, max_iter+1):
    w = shrk(-u - pa * x, l1r, l2r)
    a = (1 - sk) * pa + sk * (np.dot(w, x) - y)
    sk *= k / (k + 1)
    if schwarz(a, pa): break
    pa = a
  return a

def primal_obj(X, y, w, l1r, l2r):
  """
    Primal of so called Elastic Net:
      ||X w - y||^2 + l2r * ||w||^2 + l1r * ||w||_1
  """
  return 0.5 * ltwo(X @ w - y) + l1r * lone(w) + 0.5 * l2r * ltwo(w)

def dual_obj(X, y, a, l1r, l2r):
  """
    Dual of elastic net:
      -1/2 <a, a+y> - l2r * || shrk(-a X, l1r) ||^2
      where shrk(z, l1) = max(0, z-l1r) + min(0, z+l2r)
  """
  return -0.5 * np.dot(a, a + y) - l2r * ltwo(shrk(-a @ X, l1r, l2r))

def duality_gap(X, y, w, a, l1r, l2r):
  """
    Direct calculation of duality gap"
  """
  pm = 0.5 * ltwo(X @ w - y) + l1r * lone(w)
  am = 0.5 * np.dot(a, a + y)
  return am + pm + l2r * ltwo(w)

def scale_decayed_data(X, y, decay):
  m, n = X.shape
  q = np.sqrt(receding_weights(m, decay))
  return X * q.reshape(m, 1), y * q

def scale_data(X, y, q):
  """
    Replace weighted sample with proportionally sclaed inputs and targets.
  """
  m, n = X.shape
  tq = np.sqrt(q / lone(q))
  return X * tq.reshape(m, 1), y * tq

def yasso_lasso(X, y, l1r, l2r, q=[], u=[], a=[]):
  """
    Efficient solver for ``Elastic Nets'' with true sparsity.
    Input:
      X: data matrix (#examples x #features)
      y: targets, vector of size #examples
      u: pre-shrinkage representation of primal solution
    Output:
      w: optimal primal solution
      u: pre-shrunk primal solution
      a: optimal dual solution
  """
  m, n = X.shape
  if len(q) > 0:
    X, y = scale_data(X, y, q)
  if u == []:
    a = -y
    u = X.T @ a
  dobj = []
  for e in range(1, max_epch+1):
    prm = np.random.permutation(m)
    for i in prm:
      xi, yi = X[i,:], y[i]
      u -= a[i] * xi
      a[i] = solve_dual_univar(xi, yi, u, l1r, l2r, a[i])
      u += a[i] * xi
    dobj.append(dual_obj(X, y, a, l1r, l2r))
    if e > 2 and schwarz(dobj[-1], dobj[-2]): break
    if e % 10 == 0: print(e, ':', (m * dobj[-1]).round(4))
  w = shrk(-a @ X, l1r, l2r)
  return w, u, a
  
def gen_data(m, n, sparsity, noise):
  X = np.random.randn(m, n) / np.sqrt(n)
  nz = round(sparsity * n)
  w = np.random.randn(n) / np.sqrt(n - nz)
  w[np.argsort(np.abs(w))[:nz]] = 0
  if pos: w = np.abs(w)
  y = (X @ w) * (1 + noise * np.random.randn(m))
  return X, y, w
  
if __name__ == "__main__":
  m, n, sparsity, noise = 100, 200, 0.9, 0.001
  l1r, l2r = 0.05/m, 0.01/m
  X, y, w = gen_data(m, n, sparsity, noise)
  q = np.random.rand(m)
  X, y, = scale_data(X, y, q)
  wh, u, a = yasso_lasso(X, y, l1r, l2r)
  wh = shrk(-a @ X, l1r, l2r)
  print('\n')
  print('False zeros: ', (sum((w!=0) * (wh==0)) * (100 / m)).round(0), '%')
  print('Unrecovered: ', (sum((w==0) * (wh!=0)) * (100 / m)).round(0), '%')
