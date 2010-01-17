from __future__ import with_statement
from AtomVector import AtomVector
from Errors import *

class MultiClassifier:
    """
    mc = MultiClassifier()

    Manages a set of classifiers for a multi-class/multi-label
    classification problem.

    mc.add(label, classifier)
    Each classifier should provide a score function.

    scores = mc.score(av)
    'scores' is a map of label:score
    """

    def __init__(self):
        self.classifiers = {}

    def add(self, label, classifier):
        self.classifiers[label] = classifier

    def __repr__(self):
        return "<MultiClassfier: %d classifiers>" % (len(self.classifiers))

    def score(self, av):
        """Score a vector av
        
        Returns a map of label:score
        """
        return dict([(l,c.score(av)) for l,c in self.classifiers.iteritems()])

    # def thresholded_set(self, av, thres = 0.0):
    #     """Returns labels whose prediction scores are >= thres
    #     """
    #     return set([l for l,s in self.score(av).iteritems() if s >= thres])
    # [Useless function: need different thresholds for different cats]
