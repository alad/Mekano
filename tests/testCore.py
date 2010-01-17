import mekano as mk

from nose.tools import *
from math import sqrt
from random import seed, random

def setup():
  global af, a1, a2, a3, a4, av1, av2

  seed(10225)

  af = mk.AtomFactory("global")
  a1 = af["one"]
  a2 = af["two"]
  a3 = af["one"]
  a4 = af["three"]

  av1 = mk.AtomVector()
  av1.set(a1, 10)
  av1[a2] = 15
  av1.set(a3, 10)

  av2 = mk.AtomVector()
  av2.set(a1, 20)

def testAtomFactory():
  assert af(1) == "one"
  af.locked = True
  assert_raises(KeyError, lambda : af["()FD*$$"])
  af.locked = False
  
def testAtoms():
  assert a1 == a3
  assert a1 is a3
  assert a1 != a2

def testAtomVectors():
  global af, a1, a2, a3, a4, av1, av2
  assert len(av1) == 2
  av1[a1] = 30.0
  assert_equals(av1[a1], 30.0)
  assert_equals(av1 * av2, 600.0)
  assert len(av2) == 1
  # += for non-existant key
  av2[a4] += 10.0
  # length
  assert len(av2) == 2
  # delete
  del av2[a4]
  assert len(av2) == 1
  # cosine len
  assert av1.CosineLen() == sqrt(1125.0)
  # normalization
  cl = av1.CosineLen()
  av1_norm = av1/cl
  print av1_norm.CosineLen()
  assert av1_norm.CosineLen() == 1.0
  print av1

  # no change to original vector
  assert av1.CosineLen() == sqrt(1125.0)
  # now change the original vec
  id_av1 = id(av1)
  av1 /= cl
  assert av1.CosineLen() == 1.0
  # make sure it was done in-place
  assert id(av1) == id_av1

def testAtomVectorInit():
  d = {1:1.0, 2:2.5, 3:1.0}
  a = mk.AtomVector(d)
  assert len(a) == 3

  # Init from another AtomVector
  b = mk.AtomVector(a)
  assert len(b) == 3
  assert a.CosineLen() == b.CosineLen()

def testAtomVectorIterators():
  d = {1:1.0, 2:2.5, 3:1.0}
  a = mk.AtomVector(d)

  # Two different ways of getting keys
  k1 = [k for k in a.iterkeys()]
  k2 = [k for k in a]
  assert k1 == k2
  assert k1 == [1,2,3]

  # Items iterator
  i1 = [(k,v) for k,v in a.iteritems()]
  assert i1 == d.items()
  
def testAtomVectorNames():
  av = mk.AtomVector(name = "xx")
  av[1] = 2.0
  assert av.name == "xx"

  # Copy means copy name too
  avcopy = av.copy()
  av.name = "xxx"
#  assert avcopy.name == "xx"

  # In-place sum preserves name
  av += avcopy
  assert av.name == "xxx"

  # Normal sum does not copy names
  avsum = av + avcopy
  assert avsum.name == ""

@timed(.02)
def testAtomVectorMult():
  d1 = dict([(int(300*random()), random()) for i in xrange(150)])
  d2 = dict([(int(300*random()), random()) for i in xrange(1000)])
  a1 = mk.AtomVector()
  a2 = mk.AtomVector()

  for k,v in d1.items():
    a1[k] = v

  for k,v in d2.items():
    a2[k] = v

  for i in xrange(1000):
    p = a1*a2
  
def testAtomVectorCopy():
  av1_copy = av1.copy()
  av1_copy[a1] = 100.0
  assert av1_copy[a1] != av1[a1]

def testAtomVectorFromString():
  a = mk.AtomVector.fromstring("   1:1.0 2:-7.434   6:1323    ")
  b = mk.AtomVector({1:1.0, 2:-7.434, 6:1323})

  assert list(a.iterkeys()) == list(b.iterkeys())
  assert a.CosineLen() == b.CosineLen()
    
def testDataset():
  ds = mk.Dataset("")
  ds.add(av1, [1,2])
  ds.add(av2, [2,3])
  assert len(ds.docs) == len(ds.labels) == 2
  ds.digest()
  assert len(ds.labelset) == 3

def testBinarization():
  ds = mk.Dataset("")
  ds.add(av1, [1,2])
  ds.add(av2, [2,3])

  ds.digest()
  bset = ds.binarize()
  for label in bset:
    print "===>", label
    print bset[label]
    print "-----------"

def testCorpusStatistics():
  ii = mk.InvertedIndex()
  ii.add(av1)
  ii.add(av2)
  assert ii.getDF(a1) == 2
  assert ii.getDF(a2) == 1
  assert ii.getN() == 2

def testWeightVectors():
  ii = mk.InvertedIndex()
  ii.add(av1)
  ii.add(av2)
  wv = mk.WeightVectors(ii)
  wav1 = wv[av1]
  wav2 = wv[av2]
  print wav1.CosineLen()
  assert wav1.CosineLen() == 1.0
  
