import numpy as np
import torch as torch
import torch.nn as nn
devname = 'gpu'
devname = 'cpu'
if devname == 'gpu':
  dev = torch.device('cuda:0')
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
  dev = torch.device('cpu')
np.set_printoptions(precision=4)

dbgp = False

#------------------------------------------------------------------------------#

class Params:
  def __init__(self, n, mode='ranking'):
    self.n = n
    self.mode = mode
    self.lossfun = domloss if mode == 'ranking' else binloss
    self.max_epochs = 100
    self.print_freq = 0.01
    self.batch_frac = 0.1
    self.batch_size = int(self.batch_frac * n)
    self.print_epoc = int(self.print_freq * self.max_epochs)
    self.rich_weigh = True
    self.eps = 1e-7
    self.lr = 0.1

  def stats(self):
    print(30*'_', '\n\nLearning parameters:')
    print(' ', self.mode, 'mode')
    print(' ', self.max_epochs, 'max epochs')
    print(' ', self.batch_size, 'batch sizes out of', self.n)
    print(30*'_', '\n')

#------------------------------------------------------------------------------#

def binloss(y, h):
  return torch.mean(torch.log1p(torch.exp(-h * y)))

def domloss(y, h):
  T = torch.exp(torch.tile(h.flatten(), (len(h), 1)) - h.reshape((-1, 1)))
  P = (y.reshape((-1, 1)) > y.flatten()).float()
  return torch.mean(torch.log1p(torch.sum(T * P, axis=1)))

#------------------------------------------------------------------------------#

def learn2rank(net, Dtrain, params=None):
  p = Params(len(Dtrain))
  p.stats()
  opt = torch.optim.Adagrad(net.parameters(), lr=p.lr)

  def eval_update(D, update=True):
    acc, loss, q = 0, 0, 1
    for X, y in D:
      if p.rich_weigh: q = torch.sum(y == 1)
      loss += p.lossfun(y, net(X))
      acc += q
    loss = loss / acc
    if update:
      net.zero_grad()
      loss.backward()
      opt.step()
    return loss.detach().cpu().numpy()

  losses = [eval_update(Dtrain, update=False)]

  for e in range(1, p.max_epochs + 1):
    np.random.shuffle(Dtrain)
    epoch_loss = 0
    for i in range(0, len(Dtrain), p.batch_size):
      epoch_loss += eval_update(Dtrain[i:i+p.batch_size])
    epoch_loss /= i
    if e % p.print_epoc == 0:
      print("%4d: %8.5f" % (e, epoch_loss))
    losses.append(epoch_loss)
    if len(losses) > 5 and (losses[-2] - losses[-1]) / losses[1] < p.eps:
      break
  return losses

#------------------------------------------------------------------------------#

def data_gen(net, sizes, rels, dim, skew=True):
  def null_gen(X):
    return torch.mean(X, axis=0)

  def skewmat(dim):
    S = np.random.randn(dim, 2*dim)
    _, S = np.linalg.eigh(S @ S.T)
    d = 1 / np.arange(1, dim+1)
    return torch.tensor(S @ np.diag(d / np.mean(d)), dtype=torch.float32)

  def grp_gen(n, k):
    X = torch.randn(n+1, dim)
    if skew: X = X @ skewmat(dim)
    X[-1] = null_gen(X[:-1])
    y = -torch.ones(n+1); y[-1] = 0
    if k > 0:
      _, ind = torch.sort(net(X[:-1]), descending=True)
      y[ind[:k]] = 1
    if dbgp: print(y, '\n', net(X))
    return X, y

  return [grp_gen(sizes[i], rels[i]) for i in range(len(sizes))]

#------------------------------------------------------------------------------#

class Linet(nn.Module):
  def __init__(self, d, s=1):
    super(Linet, self).__init__()
    self.w = nn.Linear(d, 1, device=dev)
    if s != 1: self.w.weight.data *= s

  def forward(self, X):
    # flatten all dimensions except the batch dimension
    return self.w(X).flatten()

class Denet(nn.Module):
  def __init__(self, dims, s=1):
    super(Denet, self).__init__()
    depth = len(dims) - 1
    self.W = nn.ModuleList([nn.Linear(dims[i], dims[i+1], device=dev) for i in range(depth)])
    if s != 1:
      for w in self.W: w.weight.data *= s

  def forward(self, X):
    # flatten all dimensions except the batch dimension
    H = X
    for Wn in self.W:
      H = nn.functional.leaky_relu(Wn(H))
    return H.flatten()

#------------------------------------------------------------------------------#

def check_losses(n=100, l=10, mk=3):
  mlos = lambda net, D, f: \
    torch.mean(torch.tensor([f(y, net(X)) for X,y in D])).item()
  sizes = l * np.ones(n, dtype=int)
  rels = np.random.randint(mk+1, size=n)
  d = 20; dims = [d, 10, 5, 1]
  nets = [Linet(d), Linet(d, s=3), Denet(dims), Denet(dims, s=3)]
  for s in [False, True]:
    for n in [0, 2]:
      D = data_gen(nets[n], sizes, rels, d, skew=s)
      for loss in [binloss, domloss]:
        if dbgp:
          print(s, n, loss, mlos(nets[n], D, loss), mlos(nets[n+1], D, loss))
        assert mlos(nets[n], D, loss) < mlos(nets[n+1], D, loss)
  pass
 
def check_l2r(n=1000, l=10, mk=3):
  sizes = l * np.ones(n, dtype=int)
  rels = np.random.randint(mk+1, size=n)
  d = 20; dims = [d, 10, 5, 1]
  lnet = Linet(d)
  Dl = data_gen(lnet, sizes, rels, d, skew=True)
  net = Linet(d, s=0)
  losses_lin = learn2rank(net, Dl)

  dnet = Denet(dims)
  Dn = data_gen(dnet, sizes, rels, d, skew=True)
  net = Denet(dims)
  losses_dep = learn2rank(net, Dn)
  return losses_lin, losses_dep

#------------------------------------------------------------------------------#

if __name__ == "__main__":
  if dbgp: losses = check_losses()
  llin, ldep = check_l2r()
