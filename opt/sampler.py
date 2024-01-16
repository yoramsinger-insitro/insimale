# Data closures for sampling and objective + gradient closure

from numpy.random import permutation 
from numpy import take, arange

# Batch size can be provided in two way:
#   i. absolute: integer number of examples (>= 1)
#  ii. relative: fraction of total examples (< 1.0)
#
# rndprm:
#   0 or False: no permutation -- forward sequential
#   1 or True : random permutation
#   -1        : no permutation -- backward sequential 

class Sampler:

  def __init__(self, batch_size, rndprm, skip, axis, *data):
    ne = self.num_examples = data[0].shape[axis]
    self.prm = permutation(ne) if rndprm == True else arange(ne)
    if type(batch_size) == float:
      assert(batch_size <= 1.0), "Fractional batch size cannot be > 1"
      self.batch_size = round(batch_size * ne)
      self.skip = self.batch_size if skip == 0 else round(skip * ne)
    else:
      assert(batch_size >= 1), "Absolute batch_size must be greater than zero"
      self.batch_size = batch_size
      self.skip = self.batch_size if skip == 0 else skip
    assert(self.batch_size >= self.skip), "No skipping more than batch size"
    self.rndprm, self.axis, self.data, self.head = rndprm, axis, data, 0

  def __indices(self, lcl_batch_size=0):
    if self.head >= self.num_examples:
      self.head = 0
      return []
    if self.head == 0 and self.rndprm == True:
      self.prm = permutation(self.num_examples)
    bs = self.batch_size if lcl_batch_size == 0 else lcl_batch_size
    s = self.head
    t = min(self.head + bs, self.num_examples)
    self.head += self.skip
    return self.prm[s:t] 

  def sample(self, lcl_batch_size=0):
    ids = self.__indices(lcl_batch_size)
    if len(ids) == 0: return []
    if self.rndprm == -1: ids = -(ids+1)
    return [take(d, ids, axis=self.axis) for d in self.data]
