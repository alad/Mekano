"""Evaluation tools"""

from itertools import izip
from numpy import mean
from ml.utils import *

class DegenerateMetric(Exception):
    pass

cdef class ConfusionMatrix:
    cdef public int tp, tn, fp, fn

    def __cinit__(self):
        self.tp = self.tn = self.fp = self.fn = 0

    def recall(self):
        cdef int nr = self.tp + self.fn
        if nr == 0:
            raise DegenerateMetric()

        return <double>self.tp/nr

    def precision(self):
        cdef int np = self.tp + self.fp
        if np == 0:
            raise DegenerateMetric()

        return <double>self.tp/np

    def f1(self):
        cdef int nr = self.tp + self.fn
        if nr == 0:
            raise DegenerateMetric, self.__repr__()

        return 2.0*self.tp/(2.0*self.tp + self.fp + self.fn)

    def add(self, int truth, int prediction):
        if truth and prediction: self.tp += 1
        elif truth and not prediction: self.fn += 1
        elif not truth and prediction: self.fp += 1
        else: self.tn += 1

    def __repr__(self):
        return "<ConfusionMatrix: tp=%-4d fn=%-4d fp=%-4d tn=%-4d>" % (self.tp, self.fn, self.fp, self.tn)

    def clear(self):
        self.tp = self.tn = self.fp = self.fn = 0


class ConfusionEvaluator():
    """
    Multi-label evaluator for confusion matrix based metrics.

        >>> ac = ConfusionEvaluator(labels)
        >>> ac.add(truthset, predictionset)
        >>> print ac.f1macro()
        >>> print ac.f1micro()
    """

    def __init__(self, labels):
        self.labels = labels
        self.cm_map = dict([(l,ConfusionMatrix()) for l in labels])
        self.setcorrect = 0
        self.n = 0

    def add(self, truthset, predset):
        cdef int truth, pred
        for label in self.labels:
            if label in truthset: truth = 1
            else: truth = 0
            if label in predset: pred = 1
            else: pred = 0
            self.cm_map[label].add(truth, pred)
        self.n += 1
        if set(truthset) == set(predset):
            self.setcorrect += 1
    
    def addbatch(self, truths, predictions):
        for truth, prediction in izip(truths, predictions):
            self.add(truth, prediction)

#  def f1s(self):
#    return dict([(l,cm.f1()) for l,cm in self.cm_map.iteritems()])

    def f1macro(self):
        n = 0
        f1 = 0.0
        for label in self.labels:
            try:
                f1 += self.cm_map[label].f1()
                n += 1
            except DegenerateMetric:
                pass
        return f1/n

    def f1micro(self):
        cdef int tp = 0, fp = 0, fn = 0
        for cm in self.cm_map.values():
            try:
                tp += cm.tp
                fp += cm.fp
                fn += cm.fn
            except DegenerateMetric:
                pass
        if tp == 0: raise DegenerateMetric, "tp was zero"
        return 2.0*tp/(2.0*tp + fp + fn)


    def setaccuracy(self):
        return float(self.setcorrect)/self.n

    def __repr__(self):
        ret = []
        for label, cm in self.cm_map.iteritems():
            try:
                f1 = "%7.4f" % cm.f1()
            except DegenerateMetric:
                f1 = "-------"
            ret.append("%25s %s    F1: %s" % (label, cm, f1))
        ret.append("-"*80)
        ret.append("F1-micro: %7.4f" % (self.f1micro()))
        ret.append("F1-macro: %7.4f" % (self.f1macro()))
        return "\n".join(ret)


def getEditDistancefor(truth, prediction):
    """
    Edit distance between two sets, i.e., C{len(A-B) + len(B-A)}
    """
    addition = len(truth-prediction)
    deletion = len(prediction-truth)
    return addition + deletion

def getMRRfor(truth, prediction):
    """
    prediction should be a list
    """
    rr = 0.0
    rank = 0
    for item in prediction:
        rank += 1
        if item in truth:
            rr += 1.0/rank
    return rr/len(truth)

def getAPfor(truth, prediction):
    """
    prediction should be a list
    """
    ap = 0.0
    rank = 0.0
    relitems = 0
    for item in prediction:
        rank += 1
        # ascertain relevance
        if item in truth:
            relitems += 1
            ap += float(relitems)/rank
    return ap/len(truth)


def evaluateDataset(classifier, ds, thresholds):
    ds.digest()
    e = ConfusionEvaluator(classifier.labelset)
    decs = decideAll(classifier, ds.docs, thresholds)
    e.addbatch(ds.labels, decs)
    return e
    