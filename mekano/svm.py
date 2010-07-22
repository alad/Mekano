from __future__ import with_statement

import tempfile
import os

from Config import Parameters
from AtomVector import AtomVector
from Errors import *
from MultiClassifier import MultiClassifier

class SVMClassifier:
    """SVM Wrapper

        >>> svm = SVMClassifier(modelfile)
        >>> svm.score(av)
    """

    def __init__(self, modelfile = None):
        self.w = AtomVector()
        self.b = 0.0
        self.sv = 0
        self.params = Parameters()
        self.params.c = 1.0
        self.params.tmp = "/tmp/"
        self.params.binary = "svm_perf_learn"
        if modelfile is not None:
            self.readmodelfile(modelfile)

    def readmodelfile(self, modelfile):
        with open(modelfile) as fin:
            line = fin.readline()
            if "SVM" not in line:
                raise InvalidInput("Not an SVM model file!")
            [fin.readline() for i in xrange(8)]
            line = fin.readline()
            sv = int(line.split()[0])
            line = fin.readline()
            b = float(line.split()[0])
            for i in xrange(sv-1):
                a = fin.readline().rstrip().split()
                alpha_y = float(a[0])
                av = AtomVector()
                for pairs in a[1:]:
                    if "#" in pairs: break
                    a, v = map(float, pairs.split(":"))
                    av[a] = v*alpha_y
                self.w.addvector(av)

            self.sv = sv
            self.b = b

    def train(self, ds):
        params = self.params
        assert(ds.isBinary())
        fout = tempfile.NamedTemporaryFile(suffix='svm', dir=params.tmp)
        ds.toSVM(fout)
        fout.file.flush()
        modelfilename = fout.name + ".model"
        _run("%s %s %s %s > /dev/null 2>&1" % (params.binary, to_svm_params(params), fout.name, modelfilename))
        fout.close()
        self.readmodelfile(modelfilename)
        os.remove(modelfilename)

    def __repr__(self):
        return "<SVMClassfier len(w)=%d  b=%7.4f  #sv=%d>" % (len(self.w), self.b, self.sv)

    def score(self, av):
        # keep the shorter vector on the left side for faster dot products!
        return (av * self.w) - self.b

class SVMMultiClassifier:
    def __init__(self):
        self.mc = MultiClassifier()
        self.params = Parameters()
        self.labelset = set()
        self.params.c = 1.0
        self.params.tmp = "/tmp/"
        self.params.binary = "svm_perf_learn"
        
    def train(self, ds):
        params = self.params
        bds = ds.binarize()
        positions = None
        print "SVMMultiClassifier: Training with %d docs, %d labels" % (len(ds.docs), len(bds))
        for label in bds:
            if positions is None:
                fout = tempfile.NamedTemporaryFile(suffix='svm', dir=params.tmp)
                positions = bds[label].toSVM(fout)
            else:
                bds[label].toSVMSubsequent(fout, positions)
            fout.file.flush()
            modelfilename = "%s-%s.model" % (fout.name, label)
            _run("%s %s %s %s > /dev/null 2>&1" % (params.binary, to_svm_params(params), fout.name, modelfilename))
            self.mc.add(label, SVMClassifier(modelfilename))
            os.remove(modelfilename)
        fout.close()
        self.labelset = set(bds)

    def __repr__(self):
        return "<SVMMultiClassfier: %d labels>" % len(self.labelset)

    def score(self, av):
        return self.mc.score(av)

def _run(cmd):
    print "running:", cmd
    retcode = os.system(cmd)
    if retcode != 0:
        print "Return code=", retcode
        raise Exception("SVM Error")

def to_svm_params(params):
    svm_params = set(["c", "j"])
    return " ".join("-%s %s" % (k,v) for k,v in params.params.iteritems() if k in svm_params)

            
