import numpy as np

dbg = False

def input_gen(n, d, gQ, sparsity):
  X = np.random.randn(n, d) / np.sqrt(d)
  Q = (1 - gQ) * np.ones(n) + gQ * np.random.uniform(0, 2, n)
  Q = Q / np.sum(Q)
  return X, Q

vec_gen = lambda d, sparsity: \
  np.random.randn(d) / np.sqrt(d) * np.random.binomial(1, 1-sparsity, d)
  
rpredict = lambda X, W: X @ W
classify = lambda X, W: np.sign(rpredict(X, W))
ordinals = lambda X, W, b: np.sum(rpredict(X, W) < np.array(b).reshape((len(b), 1)), axis=0)
mclasses = lambda X, W: np.argmax(rpredict(X, W), axis=1)

def ordinal_noise(k, n, noise):
  m = np.append(1-noise, np.ones(k-1) * noise / (k - 1))
  E = np.random.multinomial(1, m, n) @ np.arange(0, k)
  if dbg: print('Error rate:     ', round(100 * np.mean(E != 0), 1), '%')
  return E
 
def regression_data(n=1000, d=10, noise=0.1, gQ=0, sparsity=0):
  X, Q = input_gen(n, d, gQ, sparsity)
  W = vec_gen(d, sparsity)
  Y = rpredict(X, W)
  if noise > 0:
    E = noise * np.random.randn(n)
    if dbg:
      print('Signal to Noise:', \
        round(10 * np.log10(np.sum(Y*Y) / np.sum(E*E)), 1), 'dB')
    Y += E
  return [X, Y, Q], W
 
def classification_data(n=1000, d=10, noise=0.1, gQ=0, sparsity=0):
  X, Q = input_gen(n, d, gQ, sparsity)
  W = vec_gen(d, sparsity)
  Y = classify(X, W)
  if noise > 0:
    E = 2 * np.random.binomial(1, 1-noise, n) - 1
    if dbg: print('Error rate:     ', 100 * np.mean(E < 0), '%')
    Y *= E
  return [X, Y, Q], W
 
def ordinal_data(n=1000, d=10, k=5, noise=0.1, gQ=0, sparsity=0):
  X, Q = input_gen(n, d, gQ, sparsity)
  W = vec_gen(d, sparsity)
  R = X @ W
  b = [np.quantile(R, t) for t in np.arange(1/k, 1, 1/k)]
  Y = ordinals(X, W, b)
  if noise > 0: Y = np.mod(Y + ordinal_noise(k, n, noise), k)
  return [X, Y, Q], [W, b]

def multiclass_data(n=1000, d=10, k=5, noise=0.1, gQ=0, sparsity=0):
  X, Q = input_gen(n, d, gQ, 0)
  W = np.random.randn(d, round(d * (1-sparsity)))
  _, W = np.linalg.eig(W @ W.T)
  W = W[0:k].T
  Y = mclasses(X, W)
  if noise > 0: Y = np.mod(Y + ordinal_noise(k, n, noise), k)
  return [X, Y, Q], W

if __name__ == "__main__":
  dbg = True
  data, W = regression_data(); X, Y, Q = data
  R = rpredict(X, W); E = Y - R
  print('Signal to Noise:', round(10 * np.log10(np.sum(R*R) / np.sum(E*E)), 1), 'dB')
  data, W = classification_data(); X, Y, Q = data
  R = classify(X, W); E = Y - R
  print('Error rate:     ', 100 * np.mean(E != 0), '%')
  data, [W, b] = ordinal_data(); X, Y, Q = data
  R = ordinals(X, W, b); E = Y - R
  print('Error rate:     ', 100 * np.mean(E != 0), '%')
  data, W = multiclass_data();  X, Y, Q = data
  R = mclasses(X, W); E = Y - R
  print('Error rate:     ', 100 * np.mean(E != 0), '%')
