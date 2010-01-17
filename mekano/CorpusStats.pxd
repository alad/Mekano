cimport AtomVector

cdef class CorpusStats:
    cdef AtomVector.AtomVector df
    cdef int N
    
    cpdef add(self, AtomVector.AtomVector av)
    cpdef int getDF(self, int a)
    cpdef int getN(self)
    
    
