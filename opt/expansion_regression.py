#---------------------------------------------------#
#  Expand (Lift) Input And Regress w.r.t Target     #
#---------------------------------------------------#

import numpy as np
np.set_printoptions(precision=4)
from glimp import Glimper
from optimizer import AdaGrad


obj_p = ['HUB', 1e-3, 0.5]         # loss and optional parameters
obj_p = ['MSE']
smp_p = [1.0, True, 0, 0]          # batch_size, rnd_prm, skip, axis
opt_p = {'l1r': 1e-5, 'l2r': 1e-8, 'lmx': 100, 'eta': 1.0, 'eps': 1e-3}

# Lift the original dimension by performing elements operators from the
# list 'ops' and then take all outer products to model second order poly
def LiftData(X, ops):
  ex_ops = np.append(ops, lambda x: x)
  def lift_vec(v):
    x = np.append(np.concatenate([o(v) for o in ex_ops]), 1)
    return np.outer(x, x)[np.triu_indices(len(x))]
  return np.apply_along_axis(lift_vec, 1, X)
  
# Linear Regression:
#
#  Set your objective above from (or add yours to) glimp.py.
#  The code uses as default HUB.
#
#  data is X, Y, Q [Q is optional and used for weighted regression]
# 
#  Sampling method for gradient. Setting above is:
#   sub-sample of 10% of data, randomly permute data, example axis is 0
#
#  For L1 or L2 regularization w.r.t non-zero vectors set W1, W2 o.w. W1=W2=0
#
def LiftedLinearRegression(X, Y, Q=[]):
  ops = [np.tanh]
  Xe = LiftData(X, ops)
  d, xd = X.shape[1], Xe.shape[1]
  opt_p['l1r'] *= (xd / d) ; opt_p['l2r'] *= np.sqrt(xd / d)
  print('Lifted to', Xe.shape[1], 'dimensions ( from', X.shape[1], ')')
  if len(Q) > 0:
    gl = Glimper(obj_p, smp_p, Xe, Y, Q)
  else:
    gl = Glimper(obj_p, smp_p, Xe, Y)
  Ws = [np.zeros(Xe.shape[1]), [], []]
  learner = AdaGrad(Ws, opt_p)
  W, L = learner.Go(gl.objc, gl.grad)
  Yh = Xe @ W
  return W, Yh

if __name__ == "__main__":
  n, d, noise = 1000, 100, 0.1
  X = np.random.randn(n, d) / np.sqrt(d)
  Y = X[:,1] * X[:,0] + np.tanh(X[:,2]) - np.tanh(X[:,0]) * np.tanh(X[:,3])
  W, Yh = LiftedLinearRegression(X, Y)
  print('\nSupport of W:', sum(W != 0))
  print('MSE(Yh-Y)/STD(Y):', (np.var(Yh - Y) / np.std(Y)).round(4))

