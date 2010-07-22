from __future__ import with_statement
from ..atoms import AtomVector
from ..Errors import *

class MultiClassifier:
    """
    mc = MultiClassifier()
    
    Manages a set of classifiers for a multi-class/multi-label
    classification problem.
    There are two ways to create a MultiClassifier.
    First, manually create the classifier for each label, then
    add it to MultiClasifier:
    
        >>> mc.add(label, classifier)
    
    Each classifier should provide a score function.

    scores = mc.score(av)
    'scores' is a map of label:score
    
    Second, provide a BaseClassifier class and a multi-labeled dataset
    and optimal params to be passed to BaseClassifier, in which case
    MultiClassifier with binarize the dataset and do the training:
    
        >>> mc = mekano.MultiClassifier.create(mekano.LogisticRegressionClassifier, trainset, LAMBDA=0.1, c=1.0)
    
    """

    def __init__(self):
        self.classifiers = {}
        self.labelset = set()
    
    def add(self, label, classifier):
        self.classifiers[label] = classifier
        self.labelset.add(label)
    
    def __getitem__(self, key):
        return self.classifiers[key]
    
    def score(self, av):
        """Score a vector av
        
        Returns a map of label:score
        """
        return dict([(l,c.score(av)) for l,c in self.classifiers.iteritems()])
    
    def __repr__(self):
        return "<MultiClassfier: %d classifiers>" % (len(self.classifiers))
    
    @staticmethod
    def create(BaseClassifier, ds, **params):
        """Create a MultiClassifier from a base classifier class and multi-labeled dataset.
        
        'params' are optional parameters to pass to the BaseClassifier constructor.
        """
        ret = MultiClassifier()
        bds = ds.binarize()
        for label, bd in bds.iteritems():
            print "MultiClassifier: Training for", label
            baseClassifier = BaseClassifier(**params)
            baseClassifier.train(bd)
            ret.add(label, baseClassifier)
        return ret
    
                
