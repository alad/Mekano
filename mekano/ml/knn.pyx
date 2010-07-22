from python_dict cimport *
cimport mekano.AtomVector
cimport mekano.InvertedIndex
cimport mekano.AtomVectorStore
import cPickle

cdef extern from "math.h":
    double log(double x)

cdef extern from "CUtils.h":
    ctypedef struct IntVector "IntVector":
        int size()
        int ele "operator[]" (int i)
        void push_back (int e)

    IntVector* new_IntVector "new" ()
    void del_IntVector "delete" (IntVector*)

    ctypedef struct NeighborSet "NeighborSet":
        int size()
        void clear()
        void insert(void*)
    
    NeighborSet* new_ns "new NeighborSet" ()
    void del_ns "delete" (NeighborSet* ns)
    int contains_ns (NeighborSet* ns, void *ele)
        
    ctypedef struct NeighborPair "NeighborPair":
        void* neighbor "first"
        double score "second"
    
    ctypedef struct NeighborVector "NeighborVector":
        int size()
        void clear()
        void reserve(int)
        NeighborPair ele "operator[]"(int)
    
    NeighborVector* new_nv "new NeighborVector" ()
    void del_nv "delete" (NeighborVector* nv)
    void add_neighbor(NeighborVector* nv, void* neighbor, double score)
    void partial_sort_nv(NeighborVector* nv, int K)

def KNNClassifier_fromfile(filename):
    with open(filename) as fin:
        r = cPickle.load(fin)
    return r

cdef class KNNClassifier:
    """
    knn = KNNClassifier()

    A generic KNN classifier with no a-priori
    notion of documents. It simply stores all
    vectors at training time.

    knn.add(vec)
    :
    sc = knn.score(vec, labels)
    """

    cdef mekano.InvertedIndex.InvertedIndex ii
    cdef mekano.AtomVector.AtomVector idf
    cdef double N
    cdef mekano.AtomVector.AtomVector scores
    cdef NeighborVector* nv
    cdef NeighborSet* ns
    cdef int K

    def __cinit__(self):
        self.ii = mekano.InvertedIndex.InvertedIndex()
        self.idf = mekano.AtomVector.AtomVector()
        self.nv = new_nv()
        self.ns = new_ns()
        self.K = 10

    def __init__(self):
        pass

    def __dealloc__(self):
        del_nv(self.nv)
        del_ns(self.ns)
    
    def __reduce__(self):
        return KNNClassifier, (), self.__getstate__(), None, None
    
    def __getstate__(self):
        return (self.K, self.ii, self.idf)
    
    def __setstate__(self, s):
        self.K, self.ii, self.idf = s

    def add(self, vec):
        self.ii.add(vec)

    def setK(self, int K):
        self.K = K
        
    def finish(self):
        """
        pre-calculates the IDF of all atoms
        """
        cdef double n, df
        cdef int atom

        n = self.ii.getN() + 1
        self.N = <double>n
        for atom in self.ii.atoms():
            df = <double>self.ii.getDF(atom)
            self.idf[atom] = log((n+0.0)/df)

    def score(self, mekano.AtomVector.AtomVector vec, labels):
        cdef int a, i
        cdef double N, v
        cdef double idf_a = 0.0
        cdef mekano.AtomVector.dictitr itr, end
        cdef mekano.AtomVectorStore.AtomVectorStore avs
        cdef mekano.AtomVector.AtomVector neighbor
        cdef double vote
        
        self.ns.clear()

        N = self.N
        itr = vec.mydict.begin()
        end = vec.mydict.end()
        while(not itr.eq(end)):
            a = itr.first
            v = itr.second
            itr.advance()
            idf_a = self.idf.get(a)
            if idf_a < 1.5: continue
            avs = self.ii.getii(a)
            for 0 <= i < avs.N:
                neighbor = avs.getAt(i)
                if not contains_ns(self.ns, <void*> neighbor):
                    self.ns.insert(<void*>neighbor)
                    add_neighbor(self.nv, <void*>neighbor, <double> neighbor.dot(vec))

        cdef int n_neighbors = self.ns.size()
        self.nv.clear()
        self.nv.reserve(n_neighbors)
        
        cdef int K = self.K
        
        if K > n_neighbors:
            K = n_neighbors
        
        print n_neighbors
        partial_sort_nv(self.nv, K)

        scores = {} 
        cdef int l
        cdef double s
        
        for label in labels:
            l = <int> label
            s = 0.0
            for 0 <= i < K:
                neighbor = <mekano.AtomVector.AtomVector> self.nv.ele(i).neighbor
                vote = <double> self.nv.ele(i).score
                if neighbor.contains(l):
                    s += vote
            scores[label] = s
        
        return scores
    
    def getii(self):
        return self.ii
        
    def save(self, filename):
        with open(filename, "w") as fout:
            cPickle.dump(self, fout, -1)

    fromfile = staticmethod(KNNClassifier_fromfile)


# TODO
# IDF
# tf?
