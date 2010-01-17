import mekano as mk

from nose.tools import *

def setup():
  global ds

  ds = mk.Dataset.from_rainbow("rainbow.txt")

def testRainbowReading():
  assert len(ds.docs) == 4
  assert len(ds.labelset) == 2
  assert len(ds.tokenfactory.atom_to_obj) == 15


def testLTC():
  ds.makeWeighted()
##   In [19]: ds.docs
##   Out[19]:
##     [[1:0.481,2:0.481,3:0.481,4:0.481,5:0.274],
##      [6:0.500,7:0.500,8:0.500,9:0.500],
##      [10:0.500,11:0.500,12:0.500,13:0.500],
##      [5:0.373,14:0.656,15:0.656]]
    
  assert_almost_equal(ds.docs[0][5], 0.27378505410680321)
  
  
