"""Handy functions to produce scores and decisions for a list of documents.

For tuning thresholds, see L{Thresholder}.
"""

def scoreAll(classifier, docs):
    """Score all docs using a classifier.
    
    @param classifier   : A binary classifier
    @param docs         : A list of docs of type L{AtomVector}
    @return             : A list of scores corresponding to the `docs`
    """
    return [classifier.score(d) for d in docs]

def decideAll(classifier, docs, thresholds):
    """Makes decisions based on given thresholds for all docs using a multiclassifier.
    
    @param classifier   : A L{MultiClassifier}
    @param docs         : List of docs
    @param thresholds   : A label:float dictionary
    @return             : A list of sets of positive labels
    """
    return [applyThresholds(classifier.score(d), thresholds) for d in docs]


def applyThresholds(scores, thresholds):
    """Apply thresholds to a label:score dictionary corresponding to a single doc.

    @param scores       : A label:float dictionary of L{MultiClassifier}-produced scores
    @param thresholds   : A label:float dictionary of thresholds
    @return             : A set of positive labels
    """
    
    return set([cat for cat, score in scores.iteritems() if score >= thresholds[cat]])

