import numpy as np

#
# Check conditions for optimality for a continously differential loss
# function with 1 and 2 norm regularization.
#
def optcond(g, w, l1, l2, g0=[]):
  s = np.sign(w)
  v = g + l2 * w + l1 * s
  v[s==0] = np.maximum(np.abs(v[s==0]) - l1, 0)
  if len(g0) == 0: g0 = np.ones_like(g)
  err = []
  for n in (1, 2, np.inf):
    err.append(np.linalg.norm(v, n) / np.linalg.norm(g0, n))
  return np.allclose(err, 0)

if __name__ == "__main__":
  a = np.random.randn(10000)
  l1, l2 = 0.5, 0.1
  g0 = np.minimum(-a, l2 * a + np.sign(a) * l1)
  o = np.maximum(np.abs(a) - l1, 0) * np.sign(a) / (1 + l2)
  assert optcond(o - a, o, l1, l2, g0)
