#
# Abstract Optimizer Class 
#

import numpy as np

relu = lambda v: np.maximum(v, 0)
shrk = lambda v, s: relu(np.abs(v) - s) * np.sign(v)
clop = lambda v, c: np.clip(v, -c, c)
lone = lambda v: np.sum(np.abs(v))
ltwo = lambda v: np.sum(v * v)

class GradOptimizer:

  def __init__(self, Ws, params, name="optimizer"):
    self.SetDefaults(params, name)
    self.W  = Ws[0]
    self.W1 = Ws[1] if len(Ws[1]) else 0
    self.W2 = Ws[2] if len(Ws[2]) else 0

  def Get():
    return self.W

  def Set(W):
    self.W = W

  def vprint(self, *args):
    if self.params['verbose']: print(*args)

  def SetDefaults(self, params, name):
    self.name = name
    self.params = params.copy()
    self.params['verbose'] = True #### Turn Off if needed ###
    if not self.params.get('eta'): self.params['eta'] = 1
    if self.params['lmx'] == 0: self.params['lmx'] = 1e30 # To void any clipping
    if self.params.get('verbose'):
      print('\n', 75*'=')
      print(' Running', name, 'with:')
      for k, v in self.params.items(): print('  ', k, ':', v)
      print('', 75*'=')
      if not self.params.get('rfreq'): self.params['rfreq'] = 10
    else:
      self.params['verbose'] = False
      self.params['rfreq'] = int(1e9)

  def Terminate(self, loss):
    eps = self.params['eps']
    if len(loss) <= 2 or eps == 0:
      return False
    if (loss[-2] - loss[-1]) / (loss[1] - loss[-1]) < eps:
      return True
    return False

  def RegObj(self, W):
    obj = self.objective(W)
    obj += self.params['l1r'] * lone(W - self.W1)
    obj += 0.5 * self.params['l2r'] * ltwo(W - self.W2)
    return obj

  # Calculate condition for optimality using the full sample gradient
  def MaxGradDeviation(self, gradient):
    G, n = np.zeros_like(self.W), 0
    Gt = gradient(self.W)
    while len(Gt) > 0:
      G += Gt
      n += 1
      Gt = gradient(self.W)
    G = G / n
    G += self.params['l2r'] * (self.W - self.W2)
    G += self.params['l1r'] * np.sign(self.W - self.W1)
    c = np.abs(self.W - self.W1) > self.params['eps']
    return np.max(c * G + (1 - c) * relu(G))

  def LpProx(self, eta_t):
    p = self.params
    l1r, l2r, lmx = eta_t * p['l1r'], eta_t * p['l2r'], p['lmx']
    V0 = self.W  - self.W1
    V2 = self.W2 - self.W1
    V = V0 + l2r * V2
    return clop(shrk(V, l1r) / (1 + l2r) + self.W1, lmx)

  def LearningRate(self, G):
    pass

  def L1Projection(self, eta_t):
    pass

  def Go(self, objective, gradient, epochs=100):
    self.objective = objective
    loss = [self.RegObj(self.W)]
    self.vprint("  Iter  Objective  Opt-Cond\n  ----  ---------  --------")
    for e in range(epochs):
      Gt = gradient(self.W)
      while len(Gt) > 0:
        eta_t = self.LearningRate(Gt)
        self.W = self.W - eta_t * Gt
        self.W = self.LpProx(eta_t)
        Gt = gradient(self.W)
      o = self.RegObj(self.W)
      loss.append(o)
      if ((e + 1) % self.params['rfreq']) == 0:
        max_dev = self.MaxGradDeviation(gradient)
        self.vprint("%4d:    %8.5f  %8.5f" % (e + 1, o, max_dev))
        if max_dev < self.params['eps']: break
    return self.W, loss

# -----------------------------------------------------------------------------

class SGD(GradOptimizer):
  def __init__(self, Ws, params):
    super().__init__(Ws, params, "SGD")
    if not self.params['eta']:
      self.params['eta'] = 1
    self.step = 0

  def LearningRate(self, G):
    self.step = self.step + 1
    if self.params['l2r'] > 0:
      return self.params['eta'] / (self.params['l2r'] * self.step)
    else:
      return self.params['eta'] / np.sqrt(self.step)

# -----------------------------------------------------------------------------

class AdaGrad(GradOptimizer):
  def __init__(self, Ws, params):
    super().__init__(Ws, params, "AdaGrad")
    self.GG = 1e-12 * np.ones_like(self.W)

  def LearningRate(self, G):
    self.GG = self.GG + G * G
    return self.params['eta'] / np.sqrt(self.GG)

# -----------------------------------------------------------------------------

class AccGrad(GradOptimizer):
  def __init__(self, Ws, params):
    super().__init__(Ws, params, "AccGrad")
    self.step = lambda s: (1 + math.sqrt(1 + 4*s*s)) / 2
    self.Wp = np.copy(W)
    self.s0 = 1

  def LearningRate(self, G):
    s1 = self.step(self.s0)
    m = (self.s0 - 1) / s1
    Z = (1 + m) * self.W - m * self.Wp
    self.Wp = self.W
    self.W = Z
    self.s0 = s1
    return self.params['eta']
