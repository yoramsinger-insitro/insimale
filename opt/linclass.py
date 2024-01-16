#
# Linear classificatuiion with flexible loss usage
#

import numpy as np
from glimp import Glimper
from optimizer import AdaGrad
from datagen import regression_data

"""
  Example and defaults for objectives, sampler parameters, and optimizer
  parameters
"""
obj_prms = ['HUB', 1e-3, 0.5]         # loss and optional parameters
smp_prms = [0.1, False, 0, 0]         # batch_size, rnd_prm, skip, axis
opt_prms = {'l1r': 1e-3, 'l2r': 1e-3, 'lmx': 100, 'eta': 0.4, 'eps': 1e-3}
all_prms = { 'obj_params':obj_prms, 'smp_params':smp_prms, 'opt_params':opt_prms }

def GeneraLinearRegression(setting, Ws, *data):
  """
    Generalized Linear Regression:

    Set your objective above from, (or add yours to) glimp.py.
    The code uses as default HUB which is the Huber loss.

    data is a tupple X, Y, Q [Q is optional and used for weighted regression]

    Default sampling method for above parameters is:
      sub-sample of 10% of data, randomly permute data, example axis is 0
  """
  gets = lambda s: setting[s] if setting.get(s) else all_prms[s]
  gets = lambda s: setting[s] if setting.get(s) else all_prms[s]
  obj_p = gets('obj_params')
  smp_p = gets('smp_params')
  opt_p = gets('opt_params')
  gl = Glimper(obj_p, smp_p, *data)
  learner = AdaGrad(Ws, opt_p)
  W, L = learner.Go(gl.objc, gl.grad)
  return W, L, learner.RegObj

if __name__ == "__main__":
  n, d, noise = 1000, 100, 0.1
  [X, Y, Q], W = regression_data(n=n, d=d, noise=noise, gQ=0.1, sparsity=0)
  for o in (['MSE'], ['SL1', 0., 10.0], ['HUB', 1e-3, 0.5]):
    print('\n\n//////////', o[0], '//////////')
    setting = all_prms.copy()
    setting['obj_params'] = o
    Ws = [np.zeros_like(W), [], []]
    W1, L1, regobj1 = GeneraLinearRegression(setting, Ws, X, Y)
    W2, L2, regobj2 = GeneraLinearRegression(setting, Ws, X, Y, Q)
    e0 = regobj1(W).round(5)
    e1 = regobj1(W1).round(5)
    e2 = regobj2(W2).round(5)
    if (e0 <= max(e1, e2)):
      print(e0, e1, e2)
      assert(e0 > max(e1, e2)), 'Oy vey!!!'
  print('\n\n')
  print('  <<<<<<<<<<<<>>>>>>>>>>>>')
  print('  <<< Passed all tests >>>')
  print('  <<<<<<<<<<<<>>>>>>>>>>>>')
