from mekano.Dataset import Dataset
cimport mekano.AtomVector
import sys

cdef extern from "math.h":
    double exp(double x)
    double fabs(double x)

cdef extern from "LRHelper.h":
    object ctrain(object docs, object labels, object mu, double LAMBDA, int maxiter, double epsilon, double c)


cdef class LogisticRegressionClassifier(object):
    """A logistic regression classifier trained using CG. Supports prior mean on 'w'.
     
    lr = LogisticRegressionClassifier()
    lr.maxiter = 100
    lr.epsilon = 1e-8
    lr.c = 1.0             # How much more to weigh the +ve examples than -ve examples.
    lr.train(ds)           # see help for train()
    s = lr.score(av)
        
    """
    
    cdef public double b
    cdef public double LAMBDA
    cdef public double epsilon
    cdef public double c
    cdef public int maxiter
    cdef public mekano.AtomVector.AtomVector w
    cdef public object mu
    
    def __cinit__(self, maxiter=10, epsilon=1e-8, mu=0.0, double LAMBDA=0.1, double c=1.0):
        self.w = mekano.AtomVector.AtomVector()
        self.b = 0.0
        self.c = c
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.LAMBDA = LAMBDA
    
    def __init__(self, maxiter=10, epsilon=1e-8, mu=0.0, LAMBDA=0.1, c=1.0):
        """
		maxiter in utility08 was 10. May be it was too low? We set it to 100.
		mu can be a double value, or a list of values (at least as many as #features+1)
        lambda is a double value that controls the strength of the prior.
		"""

        assert type(mu) in [list, int, float], "mu must be list or a number"
        self.mu = mu
    
    def train(self, ds):
        """Train the model.
          
        ds is a Dataset which contains 'docs' as a list AtomVectors, 
            and 'labels' as a list of true/false objects.
        """
        
        assert type(ds.docs) is list, "ds.docs must be a list"
        assert type(ds.labels) is list, "ds.labels must be a list"
        w, b = ctrain(ds.docs, ds.labels, self.mu, self.LAMBDA, self.maxiter, self.epsilon, self.c)
        i = 0

        cdef double v

        self.w = mekano.AtomVector.AtomVector()
        for v in w:
            if fabs(v) > 1e-5:
                self.w[i+1] = v
            i += 1
        self.b = b
    
    def score(self, mekano.AtomVector.AtomVector av):
        cdef double s = self.b + av.dot(self.w)
        return 1.0/(1.0+exp(-s))
    
    def __repr__(self):
        return "<LogisticRegressionClassifier len(w)=%d b=%5.3f>" % (len(self.w), self.b)
    
    
