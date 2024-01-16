#
# Adagrad optimizer for Regularized Objectives
#

import numpy as np

relu = lambda v: np.maximum(v, 0)
shrk = lambda v, s: relu(np.abs(v) - s) * np.sign(v)
clop = lambda v, c: np.clip(v, -c, c)
lone = lambda v: np.sum(np.abs(v))
ltwo = lambda v: np.sum(v * v)

class AdaGrad:

  def __init__(self, params, Ws):
    self.__set_defaults(params)
    self.W = np.copy(Ws[0])
    self.W1 = Ws[1] if len(Ws[1]) > 0 else 0
    self.W2 = Ws[2] if len(Ws[1]) > 0 else 0
    self.GG = 1e-12 * np.ones_like(self.W)   # To prevent 0/0
    self.AG = np.zeros_like(self.W) if self._p['da_mode'] else []

  def __set_defaults(self, params):
    gets = lambda str, r=0 : params.str if params.get(str) else r
    default_params = {
      'l1r': 0, 'l2r': 0.01, 'l1c': 0, 'lmx': 0, 'eta': 1, 'eps': 1e-4,
      'rfreq': 100, 'epochs': 1000, 'da_mode': False
    }
    for k in default_params.keys():
      if params.get(k): default_params[k] = params[k]
    self._p = default_params
    if self._p['lmx'] == 0: self._p['lmx'] = 1e30 # To void any clipping
    print('\n', 75*'=')
    print(' Running AdaGrad with:')
    for k, v in self._p.items():
      print('  ', k, ':', v)
    print('', 75*'=')

  def __enough(self, loss):
    eps = self._p['eps']
    if len(loss) <= 2 or eps == 0:
      return False
    if (loss[-2] - loss[-1]) / (loss[1] - loss[-1]) < eps:
      return True
    return False
  
  def __PreProx(self, Gt, eta_t):
    if self.AG != []:
      self.AG += Gt
      self.W = -eta_t * self.AG
    else:
      self.W = self.W - eta_t * Gt
    
  def __LpProx(self, eta_t):
    p = self._p
    l1r, l2r, lmx = eta_t * p['l1r'], eta_t * p['l2r'], p['lmx']
    V0 = self.W  - self.W1
    V2 = self.W2 - self.W1
    V = V0 + l2r * V2
    self.W = clop(shrk(V, l1r) / (1 + l2r) + self.W1, lmx)

  # Scaled projection -- project v onto <a,|v|> <= c (a > 0)
  # Algorithm is from:
  #   ``Adaptive Subgradient Methods'', JMLR, Vol. 12, 2011.
  def __p_scaled_ball1(self, v, a, c):
    if np.dot(a, np.abs(v)) <= c:
      return v
    # No descending order in numpy
    idx = np.argsort(-np.abs(v) / a)
    u = (np.abs(v) /a)[idx]
    b = a[idx] ** 2
    rho = 1 + np.argwhere(np.cumsum(b * u) - u * np.cumsum(b) < c).max()
    theta = (np.dot(b[:rho], u[:rho]) - c) / np.sum(b[:rho])
    return np.sign(v) * relu(np.abs(v) - theta * a)

  def Regobj(self, W):
    o = self.objective(W)
    o += self._p['l1r'] * lone(W - self.W1)
    o += 0.5 * self._p['l2r'] * ltwo(W - self.W2)
    return o

  def optimize(self, objective, gradient, epochs=1000):
    self.objective = objective
    loss = [self.Regobj(self.W)]
    if epochs: self._p['epochs'] = epochs
    print("  Iter   Error\n  ----   -----")
    for e in range(self._p['epochs']):
      Gt = gradient(self.W)
      while len(Gt) > 0:
        # Incorporate L2 regularization to the gradient
        if self._p['l2r'] > 0:
          Gt += self._p['l2r'] * self.W
        # Update accumulated gradient outer product
        self.GG = self.GG + Gt * Gt
        eta_t = self._p['eta'] / np.sqrt(self.GG)
        # In dual averaging mode accumulated gradient is updated as well
        # and update is performed on AG
        self.__PreProx(Gt, eta_t)
        # Perform all proximal operators except for L1 contrained projection
        self.__LpProx(eta_t)
        # Perform L1 projection (use w/o any other regularization !)
        if self._p['l1c'] > 0.:
          r_eta = np.sqrt(eta_t)
          self.W = r_eta * _p_scaled_ball1(self.W / r_eta, r_eta, self._p['l1c'])
        # Get a new gradient
        Gt = gradient(self.W)
      o = self.Regobj(self.W)
      loss.append(o)
      if ((e + 1) % self._p['rfreq']) == 0: print("%4d: %8.3f" % (e + 1, o))
      if self.__enough(loss): break
    print("%4d: %8.3f" % (e + 1, o))
    print('', 75*'=')
    return self.W, loss
