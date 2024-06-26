import os
import time
import pydot

import torch as tc
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from webbrowser import open as wopen

class PNode():

  def __init__(self, depth, dim, name=""):
    self.depth = depth
    self.name = name
    self.dim = dim
    if depth > 0:
      self.predict = nn.Linear(dim, 1)
      self.predict.bias.data.fill_(0.)
    else:
      self.predict = lambda x: 0.
    self.subtree = []

  def _logsumexp(self, Xs):
    ls = 0
    if len(self.subtree) > 0:
      ls = tc.logsumexp(tc.cat([n._logsumexp(Xs) for n in self.subtree], 1), 1)
    return self.predict(Xs) + ls
  
  def _pathpred(self, Xs, Ys):
    def _pathpred_single(x, y):
      if len(self.subtree) > 0:
        ps = self.subtree[y[0]]._pathpred_single(y[1:])
      return self.predict(x) + ps
    return tc.cat([_pathpred_single(x, y) for x,y in zip(Xs, Ys)], 1)

  def logprob(self, Xs, Ys):
    return self._logsumexp(Xs) - self._pathpred(Xs, Ys)

  def softmax(self, Xs, Ys):
    pass

  def spawn(self, width, names=[]):
    lnames = names if len(names) > 0 else [self.name + str(i) for i in range(1, i+1)]
    self.subtree = [PNode(self.depth + 1, self.dim, ln) for ln in lnames]
    pass

  def node(self, y):
    return self if len(y) == 0 else self.subtree[y[0]].node(y[1:])

  def path(self, y):
    return self.name + "->" + self.subtree[y[0]].name(y[1:])
    
  def dotplot(self, g):
    ns = pydot.Node(self.name)
    for n in self.subtree:
      nc = n.dotplot(g)
      g.add_edge(pydot.Edge(ns, nc))
    return ns

def plotree(root):
  pfile = '/tmp/ptree.pdf'
  g = pydot.Dot(graph_type='graph')
  _ = root.dotplot(g)
  _ = g.write_pdf(pfile)
  return wopen('file://'+'/tmp/ptree.pdf')

if __name__ == "__main__":
  root = PNode(0, 10, "root")
  root.spawn(3, ['a', 'b', 'c'])
  for i in range(3):
    n = root.node([i])
    n.spawn(2+i)
  plotree(root)
