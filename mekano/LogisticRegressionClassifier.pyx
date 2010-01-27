from Dataset import Dataset
from AtomVector import AtomVector
import sys

cdef extern from "math.h":
    double exp(double x)
    
cdef extern from "LRHelper.h":
    object ctrain(object docs, object labels, object mu, double LAMBDA, int maxiter, double epsilon)
    
class LogisticRegressionClassifier(object):
    """A logistic regression classifier trained using CG. Supports prior mean on 'w'.
     
    lr = LogisticRegressionClassifier()
    lr.maxiter = 100
    lr.epsilon = 1e-6
    lr.train(ds)           # see help for train()
    s = lr.score(av)
        
    """
    
    def __cinit__(self):
        pass
        
    def __init__(self):
        self.maxiter = 100
        self.epsilon = 1e-6
        self.w = AtomVector()
        self.b = 0.0
    
    def train(self, ds, mu=0.0, LAMBDA=0.1):
        """Train the model.
         
        ds is a Dataset which contains 'docs' as a list AtomVectors, 
            and 'labels' as a list of true/false objects.
        mu can be a double value, or a list of values (at least as many as #features+1)
        lambda is a double value that controls the strength of the prior.
        """
        
        w, b = ctrain(ds.docs, ds.labels, mu, LAMBDA, self.maxiter, self.epsilon)
        i = 0
        for v in w:
            self.w[i+1] = v
            i += 1
        self.b = b

    def score(self, av):
        cdef double s = self.b + self.w*av
        return 1.0/(1.0+exp(-s))

        
 