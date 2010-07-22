from itertools import izip
from atoms import *
from textual import WordNumberRegexTokenizer
from io import SMARTParser


# TODO:
# Fix the semantics of labels. They should always be sets.
# Do not allow True and False. They can't even be added to dictionaries
# as keys without surprises.

# seems this class can benefit from a generator function,
# esp. one that loads from disk on-demand.
class Dataset:
    """
    A Dataset with 'documents' and 'labels'

    Documents should be L{AtomVector}-like i.e. they should be
    iterable, yielding C{(a,v)} pairs.

        >>> ds = Dataset("reuters")
        >>> ds.add(doc, labels)
        >>> ds.digest()
    
    @todo: Fix semantics of labels.
    """
    
    def __init__(self, name=""):
        # a human readable name
        self.name = name
        # the set of all labels; this should have one label if this is
        # a binary label dataset
        self.labelset = set()
        # the list of AtomVectors-like objects
        self.docs = []
        # the list of Labels corresponding to data[]
        self.labels = []
        self.digested = True
        
        # Not all datasets have these
        self.cs = None
        self.catfactory = None
        self.tokenfactory = None
    

    def __iter__(self):
        """Iterate over (doc, labels) tuples.

        """
        return izip(self.docs, self.labels)
    

    def add(self, doc, labels):
        """
        Add a (doc, labels) pair to the dataset

        'labels' can be either a sequence (e.g. [1,2,5],
        or a single value (e.g. True or False)
        """
        self.docs.append(doc)
        self.labels.append(labels)
        self.digested = False
    

    def digest(self, force=False):
        """Analyze the data and generate an internal list of labels.
        
        Useful for binarizing etc.
        """
        if self.digested and force==False: return

        self.labelset = set()
        for labels in self.labels:
            if hasattr(labels, "__iter__"):
                for label in labels:
                    self.labelset.add(label)
            elif labels:
                self.labelset.add(labels)
        self.digested = True
    

    def isBinary(self):
        self.digest()
        return len(self.labelset) == 1
    

    def getCategoryCounts(self):
        """Get a dictionary of labels and respective document counts.

        This is an O(n) operation!
        """
        self.digest()
        counts = dict([(l,0) for l in self.labelset])
        for labels in self.labels:
            for label in labels:
                counts[label] += 1

        return counts
    

    def __repr__(self):
        self.digest()
        return "<Dataset '%s', %d docs, %d labels>" % (self.name, len(self.docs), len(self.labelset))

    def toMultiClassSVM(self, fout):
        """Write a multi-class dataset to fout in SVM format.

        This can be directly consumed by LIBSVM.
        """
        for doc, labels in self:
            svm_label = labels[0]
            fout.write("%s %s\n" % (svm_label,
                                    " ".join(["%d:%-7.4f" % (a,v) for a,v in sorted(doc.iteritems())])))

    def toSVM(self, fout):
        """Write a binary dataset to fout in SVM format.

        Returns the byte positions of the labels, which can be used
        by L{toSVMSubsequent}() to overwrite the labels with something
        else.
        """
        assert(self.isBinary())
        positions = []
        for doc, label in self:
            if label: svm_label = "+1"
            else: svm_label = "-1"
            positions.append(fout.tell())
            fout.write("%s %s\n" % (svm_label,
                                    " ".join(["%d:%-7.4f" % (a,v) for a,v in sorted(doc.iteritems())])))
        return positions

    def toSVMSubsequent(self, fout, positions):
        assert(self.isBinary())
        i = 0
        for doc, label in self:
            position = positions[i]
            i += 1
            if label: svm_label = "+1"
            else: svm_label = "-1"
            fout.seek(position)
            fout.write(svm_label)

    def toSMART(self, fout):
        if self.catfactory is None or self.tokenfactory is None:
            raise Exception("Dataset must have catfactory and tokenfactory")

        for doc, labels in self:
            fout.write(".I %s\n" % doc.name)
            fout.write(".C\n")
            fout.write("; ".join(["%s 1" % self.catfactory.get_object(a) for a in labels]))
            fout.write("\n")
            fout.write(".T\n\n")
            fout.write(".W\n")
            fout.write(" ".join([" ".join([self.tokenfactory.get_object(a)] * int(v)) for a,v in doc.iteritems()]))
            fout.write("\n")

    def binarize(self):
        """Create and return binary datasets.

        @return: A C{{k:v}} dictionary where k is a category name, and v is a binary dataset.
        """

        self.digest()
         
        assert not self.isBinary(), "Dataset is already binary"

        name = self.name
        all_labels = self.labelset

        # create a dictionary of datasets that we will return
        ret = dict([(l,Dataset("%s.%s" % (name, str(l)))) for l in all_labels])

        for doc, doclabels in self:
            doclabels = set(doclabels)
            for label in all_labels:
                if label in doclabels:
                    ret[label].labels.append(True)
                else:
                    ret[label].labels.append(False)

        for ds in ret.values():
            # docs are shared!
            ds.docs = self.docs
            ds.digest(force=True)
            ds.catfactory = self.catfactory
            ds.tokenfactory = self.tokenfactory
            ds.cs = self.cs

        return ret

    def makeWeighted(self, cs = None):
        """Convert to a weighted (e.g. LTC) dataset
        
        @param cs       : An optional L{CorpusStats} object, otherwise it will be created
                          an associated with the dataset.

        """

        if cs is None:
            cs = CorpusStats()

        for doc, doclabels in self:
            cs.add(doc)

        wvc = WeightVectors(cs)
        for i in range(len(self.docs)):
            self.docs[i] = wvc[self.docs[i]]

        self.cs = cs
    
    def subset(self, count):
        """Creates count subsets of the dataset.
        
        Subsetting is performed using round-robin.

        @param count    : Number of subsets to create
        @return         : A list of datasets
        """
        n = len(self.docs)
        docs_per_set = int(n/count)
        if docs_per_set < 1:
            raise Exception, "#subsets > #docs"
        subsets = [Dataset("%s-%d" % (self.name, i+1)) for i in range(count)]
        j = 0 # subset chooser
        for i in range(n):
            j = i % count
            subsets[j].add(self.docs[i], self.labels[i])

        for ds in subsets:
            ds.digest()
            ds.catfactory = self.catfactory
            ds.tokenfactory = self.tokenfactory
            ds.cs = self.cs
            
        return subsets

    def kfold(self, count):
        """Create cross-validation folds.
        
        The dataset is broken into `count` pieces, each fold (i.e. train-test pair)
        is created by assigning 1 piece to `train`, and `count-1` pieces to `test`.
        
        @param count        : Number of folds
        @return             : A list of [train,test] datasets
        """
        subsets = self.subset(count)
        folds = [[Dataset(), Dataset()] for i in range(count)]
        for i in range(count):
            for j in range(count):
                if i == j:
                    folds[i][1] = subsets[j]
                else:
                    folds[i][0] += subsets[j]
        return folds
        
    def __add__(self, other):
        """Add two datasets.
        
        If both datasets are non-empty, then they must be 'compatible',
        i.e., share the same factories and corpus stats.
        
        The resulting dataset combines the docs and labels, and inherits
        the factories and corpus stats of the non-empty parent dataset.

        If both parents were L{digest}ed, the resulting dataset is also digested.
        """
        result = Dataset()
        # do not add incompatible datasets, unless one of them is empty.
        if len(self.docs) > 0 and len(other.docs) > 0:
            if self.catfactory != other.catfactory or self.tokenfactory != other.tokenfactory or self.cs != other.cs:
                raise Exception("Incompatible datasets")
        
        if len(self.docs) > 0:
            reference_ds = self
        else:
            reference_ds = other
            
        result.docs = self.docs + other.docs
        result.labels = self.labels + other.labels
        if self.digested and other.digested:
            result.labelset = self.labelset | other.labelset
            result.digested = True
        else:
            result.digested = False

        result.catfactory = reference_ds.catfactory
        result.tokenfactory = reference_ds.tokenfactory
        result.cs = reference_ds.cs
        
        result.name = self.name + "+" + other.name
    
        return result

    @staticmethod    
    def fromSMART(filename, linkto=None):
        ds = Dataset(filename)
        if linkto is None:
            catfactory = AtomFactory("cats")
            tokenfactory = AtomFactory("tokens")
        else:
            catfactory = linkto.catfactory
            tokenfactory = linkto.tokenfactory

        def handler(docid, cats, text):
            catatoms = [catfactory[c] for c in cats]
            av = AtomVector()
            for token in WordNumberRegexTokenizer(text):
                tokenatom = tokenfactory[token]
                av[tokenatom] += 1
            ds.add(av, catatoms)

        with open(filename) as fin:
            sp = SMARTParser(fin, handler, ["T", "W"])
            sp.parse()

        ds.digest()    
        ds.catfactory = catfactory
        ds.tokenfactory = tokenfactory
        return ds
        
    @staticmethod    
    def from_rainbow(filename, linkto=None):
        """Create a dataset from rainbow's output.

        $ rainbow -d model --index 20news/train/*
        $ rainbow -d model --print-matrix=siw > train.txt

            >>> ds = from_rainbow("train.txt")

        C{ds.catfactory} holds the L{AtomFactory} for category names.
        C{ds.tokenfactory} holds the L{AtomFactory} for the tokens.

        A test set should share its factories with a training set.
        Therefore, read is like so:

            >>> ds2 = from_rainbow("testfile.txt", linkto = ds)
        
        @param filename     : File containing rainbow's output
        @param linkto       : Another dataset whose L{AtomFactory} we should borrow.
        @return             : A brand new dataset.
        """

        # cdef AtomVector.AtomVector av
        # cdef int i, l, atom
        # cdef double count

        ds = Dataset(filename)
        if linkto is None:
            catfactory = AtomFactory("cats")
            tokenfactory = AtomFactory("tokens")
        else:
            catfactory = linkto.catfactory
            tokenfactory = linkto.tokenfactory
        fin = open(filename, "r")
        for line in fin:
            a = line.split(None, 2)
            catatom = catfactory[a[1]]
            a0 = a[0]
            p = a0.rfind("/")
            if p != -1:
                docname = a0[p+1:]
            else:
                docname = a0
            a = a[2].split()
            l = len(a)
            av = AtomVector(name=docname)
            #for i from 0 <= i < l by 2:
            for i in range(0,l,2):
                atom = tokenfactory[a[i]]
                count = float(a[i+1])
                av.set(atom, count)
            ds.add(av,[catatom])
        ds.digest()
        ds.catfactory = catfactory
        ds.tokenfactory = tokenfactory
        return ds



