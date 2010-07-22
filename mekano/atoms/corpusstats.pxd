cimport atomvector

cdef class CorpusStats:
    cdef atomvector.AtomVector df
    cdef int N
    
    cpdef add(self, atomvector.AtomVector av)
    cpdef int getDF(self, int a)
    cpdef int getN(self)
    
    
