import mekano as mk

from nose.tools import *

def setup():
    pass

def testThresholding():
    # We will do it for one label only,
    # since tresholdder treats all labels separately in any case.
    t = mk.ml.thresholder.Thresholder("test")
    t.add(set([1]), {1:.75})
    t.add(set([]), {1:.3})
    t.add(set([1]), {1:.6})
    t.add(set([]), {1:.7})
    t.add(set([]), {1:.1})
    t.add(set([]), {1:.4})
    t.add(set([]), {1:.2})
    t.add(set([]), {1:.15})
    t.add(set([1]), {1:.8})
    
    thresholds = t.findthresholds([1])
    assert thresholds[1] == 0.6
