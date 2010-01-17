cimport AtomVector
cimport CorpusStats
from Errors import *

cdef extern from "math.h":
    double log(double x)
    
cdef class WeightVectors:
    """For creating LTC vectors.

    wv = WeightVectors(cs) 
    Creates a weight-vector cache linked to the given corpus stats.

    cs.add(unweighted_vector) 
    weighted_vector = wv[unweighted_vector]
    assert(weighted_vector.CosineLen() == 1.0)

    """
    
    cdef double n
    cdef CorpusStats.CorpusStats cs
    cdef object cache
    cdef int maintaincache
    cdef int n_access
    cdef int n_hits
    
    def __init__(self, CorpusStats.CorpusStats cs, cache=False):
        self.cs = cs
        self.cache = {}
        self.n_access = 0
        self.n_hits = 0
        if cache:
            self.maintaincache = 1
        else:
            self.maintaincache = 0

    def __getitem__(self, AtomVector.AtomVector vec):
        cdef double n
        cdef int a
        cdef double v
        cdef AtomVector.dictitr itr, end
        cdef AtomVector.AtomVector wav
        cdef int df

        self.n_access += 1
        if self.maintaincache == 1 and vec in self.cache:
            self.n_hits += 1
            return self.cache[vec]
        else:
            wav = AtomVector.AtomVector()
            n = self.cs.getN()

            itr = vec.mydict.begin()
            end = vec.mydict.end()
            while(itr.neq(end)):
                a = itr.first
                v = itr.second
                df = self.cs.getDF(a)
                if df > 0:
                    wav.set(a, (1.0+log(v))*log((1.0+n)/df))
                itr.advance()
            wav.Normalize()
            if self.maintaincache == 1:
                self.cache[vec] = wav
            return wav

    def __repr__(self):
        return "<WeightVectors: #access:%d #hits:%d %s>" % (self.n_access, self.n_hits, repr(self.cs))
