
def scoreAll(classifier, docs):
    return [classifier.score(d) for d in docs]

def decideAll(classifier, docs, thresholds):
    return [applyThresholds(classifier.score(d), thresholds) for d in docs]
    
def applyThresholds(scores, thresholds):
    """Apply thresholds to the set of scores.
    Both inputs are dictinaries.
    
    Output is a set.
    """
    
    return set([cat for cat, score in scores.iteritems() if score >= thresholds[cat]])

