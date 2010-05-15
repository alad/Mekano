from itertools import izip
from Evaluator import ConfusionMatrix, DegenerateMetric
from ClassifierDriver import *

class Thresholder:
    """
    Threshold finder for multi-label predictions

    t = Thresholder(name)

    e.add(truth, prediction)
    Really generic. 'truth' is a set, prediction is a list of (label, score)

    """
    
    # The lower threshold for SCut.FBR method
    fbr = 0.1

    def __init__(self, name=""):
        self.name = name
        self.truths = []
        self.predictions = []

    def add(self, truth, prediction):
        self.truths.append(truth)
        self.predictions.append(prediction)
    
    def addbatch(self, truths, predictions):
        self.truths = truths
        self.predictions = predictions

    def findthresholds(self, labelset):
        ret = {}
        cm = ConfusionMatrix()
        # do it independently for each label {
        for label in labelset:
            data = []
            pos = 0
            neg = 0
            # for each document {
            for truth, prediction in izip(self.truths, self.predictions):
                binarylabel = label in truth
                if binarylabel: pos += 1
                else: neg += 1
                try:
                    data.append((prediction.get(label, -100.0), binarylabel))
                except KeyError:
                    raise Exception, "Threshold error!"
            # } for each doc
            data = sorted(data, reverse=True)
            if len(data) == 0: 
                print "Thresholding: Data length was zero. Continuing."
                continue

            # now we have (prediction score, binary truth) tuples
            # default accuracy if threshold > highest score
            cm.tp = cm.fp = 0
            cm.fn = pos
            cm.tn = neg
            # If we can't beat fbr F1, then the best threshold is 
            # equal to the score of the top-ranking document.
            bestf1 = Thresholder.fbr
            bestf1_thres = data[0][0]
            acc = neg                         # no need for denominator
            bestacc = acc
            bestthreshold = data[0][0]
            for pair in data:
                if pair[1]:
                    acc += 1
                    cm.tp += 1
                    cm.fn -= 1
                else:
                    acc -= 1
                    cm.fp += 1
                    cm.tn -= 1
                if acc > bestacc:
                    bestacc = acc
                    bestthreshold = pair[0]
                try:
                    f1 = cm.f1()
                except DegenerateMetric:
                    f1 = 0.0
                if f1 > bestf1:
                    bestf1 = f1
                    bestf1_thres = pair[0]
            ret[label] = bestf1_thres
        # } for each label
        return ret
    
def findThresholdsForDataset(classifier, ds):
    ds.digest()
    thres = Thresholder()
    preds = scoreAll(classifier, ds.docs)
    thres.addbatch(ds.labels, preds)
    # We use classifier.labelset since it might be smaller than ds.labelset
    return thres.findthresholds(classifier.labelset)

