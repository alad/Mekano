import sys
import mekano as mk

def testLogisticRegression():
    lr = mk.ml.LogisticRegressionClassifier(mu=2.0, LAMBDA=.1)
    ds = mk.Dataset()
    a1 = mk.AtomVector()
    a1[1] = 2
    a1[3] = 4.0
    a1[5] = -3
    a2 = mk.AtomVector()
    a2[3] = 8.7
    a2[1] = 1.2
    a2[5] = 1.3
    ds.add(a1, False)
    ds.add(a2, True)

    lr.train(ds)
    print lr.w
    print lr.b

    print sys.getrefcount(lr.w)

