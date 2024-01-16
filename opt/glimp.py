# Generalized Linear Models (GLIM) for Multivariate Prediction

from numpy import exp, log, log1p, maximum, mean, minimum, sign
from numpy import ones, ones_like
from numpy import sum as np_sum
from sampler import Sampler

class Objective:

  def available_objectives(self):
    return list(self.objectives.keys())

  def __init__(self):
    self.objectives = dict()

  def objc(self, data):
    pass

  def grad(self, data):
    pass

  def transfer(self, y):
    return y

  def nweights(self, data):
    s = data[3] if len(data) > 3 else 1
    t = np_sum(data[2]) if len(data) > 2 else data[0].shape[0]
    return (s / t) * data[2]

# -----------------------------------------------------------------------------

# Mean-Squared-Error
class MSE(Objective):
  def __init__(self):
    super().__init__()
    self.objectives['MSE'] = self

  def objc(self, data):
    y, p, q = data[0], data[1], self.nweights(data)
    return 0.5 * np_sum(q * (p - y)**2)

  def grad(self, data):
    y, p, q = data[0], data[1], self.nweights(data)
    return q * (p - y)

# -----------------------------------------------------------------------------

# Smooth L1
class SL1(Objective):
  def __init__(self, eps=0, scl=1):
    super().__init__()
    self.objectives['SL1'] = self
    self.eps = eps * scl
    self.scl = scl
    self.boundry = 100
    self.sftp = lambda v, b: log1p(exp(minimum(v, b))) + maximum(v - b, 0)
    self.bias = self.sftp(-self.eps, self.boundry) / self.scl

  def objc(self, data):
    y, p, q = data[0], data[1], self.nweights(data)
    z = self.scl * (p - y)
    l1 = np_sum(q * self.sftp(z - self.eps, self.boundry))
    l2 = np_sum(q * self.sftp(-z - self.eps, self.boundry))
    return (l1 + l2) / (2 * self.scl) - self.bias

  def grad(self, data):
    y, p, q = data[0], data[1], self.nweights(data)
    ssig = lambda v, b: 1 / (1 + exp(minimum(v, b)))
    z = self.scl * (p - y)
    g1 = q * ssig(self.eps - z, self.boundry)
    g2 = q * ssig(self.eps + z, self.boundry)
    return (g1 - g2) / 2

# -----------------------------------------------------------------------------

# Vector (Multivariate) Cross-Entropy (single example)
class VCE(Objective):
  def __init__(self):
    super().__init__()
    self.objectives['VCE'] = self
    eps = self.eps = 1e-20
    self.smooth = lambda v: (1-eps) * v + eps / len(v)

  def objc(self, data):
    y, p = self.smooth(data[0]), self.smooth(data[1])
    return np_sum(y * log(y / p))

  def grad(self, data):
    y, p = self.smooth(data[0]), self.smooth(data[1])
    return -y / p
    
  def transfer(self, y):
    p = exp(y - max(y))
    return p / np_sum(p)

# -----------------------------------------------------------------------------

# Softmax: -log(P[y|x]) **** TODO(ys) : build test
class Softmax(Objective):
  def __init__(self):
    super().__init__()
    self.objectives['Softmax'] = self
    eps = self.eps = 1e-20
    self.smooth = lambda v, k: (1 - eps) * v + eps / k

  def objc(self, data):
    y, p, q = data[0], data[1], self.nweights(data)
    py = self.smooth(p[np.arange(len(y)),y], p.shape[-1])
    return -np_sum(q * np.log(py))

  def grad(self, data):
    y, p, q = data[0], data[1], self.nweights(data)
    return np.reshape(q, (-1, 1)) * (p - np.eye(p.shape[-1])[y])
    
  def transfer(self, y):
    p = exp(y - max(y, axis=1, keepdims=True))
    return self.smooth(p / np_sum(p, axis=1, keepdims=True))

# -----------------------------------------------------------------------------

# Binary Cross-Entropy (multi samples)
class BCE(Objective):
  def __init__(self):
    super().__init__()
    self.objectives['VCE'] = self
    eps = self.eps = 1e-20
    self.smooth = lambda v: (1 - eps) * v + eps / 2
    
  def objc(self, data):
    y, p = self.smooth(data[0]), self.smooth(data[1]), self.nweights(data)
    return np_sum(q * (y * log(y / p) + (1 - y) * log((1 - y) / (1 - p))))
    
  def grad(self, data):
    y, p = self.smooth(data[0]), self.smooth(data[1])
    q = self.nweights(data)
    return q * ((1 - y) / (1 - p) - y / p)
    
  def transfer(self, y):
    return 1 / (1 + exp(-maximum(y, 2*log(self.eps))))

# -----------------------------------------------------------------------------

# Huber Loss
class HUB(Objective):
  def __init__(self, eps=0, scl=1):
    super().__init__()
    self.objectives['HUB'] = self
    self.eps = eps
    self.scl = scl
    self.ons = lambda y, p: maximum(abs(y - p) - self.eps, 0)

  def objc(self, data):
    y, p, q = data[0], data[1], self.nweights(data)
    z = self.ons(y, p)
    s = self.scl
    return np_sum(q * (1/(2*s) * (z<=s) * z**2 + (z>s) * (z-s/2)))

  def grad(self, data):
    y, p, q = data[0], data[1], self.nweights(data)
    z = self.ons(y, p)
    return q * sign(p - y) * minimum(z/self.scl, 1)

# -----------------------------------------------------------------------------
#               Superclass for Setting Sampler with Specific Loss
# -----------------------------------------------------------------------------

class Glimper:
  def __init__(self, obj_param, smp_param, *data):
    X, Y = data[0], data[1]
    Q = ones_like(Y) if len(data) == 2 else data[2]
    # Set gradient gradient sampler
    self.grd_sampler = Sampler(*smp_param, X, Y, Q)
    # Set objective full sampler:
    #   [1: no downsampling, F: no permutation, 0: no skip, smp_param[-1]]
    self.obj_sampler = Sampler(1.0, False, 0, smp_param[-1], X, Y, Q)
    self.set_obj(obj_param)

  def set_obj(self, objp):
    self.obj_list = {'MSE':MSE, 'SL1':SL1, 'VCE':VCE, 'BCE':BCE, 'HUB':HUB}
    assert(self.obj_list.get(objp[0])), 'Undefined objective'
    self.Obj = (self.obj_list[objp[0]])(*objp[1:])

  # Pred-transfer and Post-grad are set for univariate regrssion.
  def pred_transfer(self, X, W):
    return self.Obj.transfer(X @ W)
  def post_grad(self, G, X):
    return G @ X
   
  # Stochastic gradient at its abstract glory
  def grad(self, W):
      batch = self.grd_sampler.sample()
      if len(batch) == 0: return []
      Xs, Ys, Qs = batch
      Ps = self.pred_transfer(Xs, W)
      Gs = self.Obj.grad([Ys, Ps, Qs])
      return self.post_grad(Gs, Xs)

  # Deterministic objective for paramter vector W -- Linear Regression
  def objc(self, W):
    batch = self.obj_sampler.sample()
    if len(batch) == 0: return self.objc(W) # In case of end-of-data
    Xs, Ys, Qs = batch
    Ps = self.pred_transfer(Xs, W)
    return self.Obj.objc([Ys, Ps, Qs])

# -----------------------------------------------------------------------------

if __name__ == "__main__":
  """ Simple Visual Test """
  import numpy as np
  l = [MSE(), SL1(1.0, 10.), HUB(0.1, 1.0)]
  r, g = [[], [], []], [[], [], []]
  xs = np.arange(-10, 10, 0.01)
  for x in xs:
    for i in range(3):
      data = np.array([[x], [0], [10]])
      r[i].append(l[i].objc(data))
      g[i].append(l[i].grad(data))
  plt.figure('losses')
  plt.plot(xs, r[0], xs, r[1], xs, r[2])
  plt.figure('grads')
  plt.plot(xs, g[0], xs, g[1], xs, g[2])
