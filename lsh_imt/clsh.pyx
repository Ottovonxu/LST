from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
import cython

cdef extern from "LSH.h":
    cdef cppclass LSH:
        LSH(int, int, int, int) except +
        void insert(int*, int) except +
        void insert_multi(int*, int) except +
        void insert_multi_label(int*, int*, int) except +
        unordered_set[int] query(int*) except +
        vector[int] query_multi(int*, int) except +
        vector[int] query_multi_label(int*, int, int) except +
        void query_batch(int*, long*, int, int) except +
        void clear()
        void count()

cdef class pyLSH:
    cdef LSH* c_lsh

    def __cinit__(self, int MAX_SIZE, int K, int L, int THREADS):
        self.c_lsh = new LSH(MAX_SIZE, K, L, THREADS)

    def __dealloc__(self):
        del self.c_lsh

    @cython.boundscheck(False)
    def insert(self, np.ndarray[int, ndim=1, mode="c"] fp, int item_id):
        self.c_lsh.insert(&fp[0], item_id)

    @cython.boundscheck(False)
    def insert_multi(self, np.ndarray[int, ndim=2, mode="c"] fp, int N):
        self.c_lsh.insert_multi(&fp[0, 0], N)

    @cython.boundscheck(False)
    def insert_multi_label(self, np.ndarray[int, ndim=2, mode="c"] fp, np.ndarray[int, ndim=1, mode="c"] labels, int N):
        self.c_lsh.insert_multi_label(&fp[0, 0], &labels[0], N)

    @cython.boundscheck(False)
    def query(self, np.ndarray[int, ndim=1, mode="c"] fp):
        return self.c_lsh.query(&fp[0])

    @cython.boundscheck(False)
    def query_multi(self, np.ndarray[int, ndim=2, mode="c"] fp, int N):
        return self.c_lsh.query_multi(&fp[0, 0], N)

    @cython.boundscheck(False)
    def query_multi_label(self, np.ndarray[int, ndim=2, mode="c"] fp, int N, int c):
        return self.c_lsh.query_multi_label(&fp[0, 0], N, c)

    @cython.boundscheck(False)
    def query_batch(self, np.ndarray[int, ndim=2, mode="c"] fp, np.ndarray[long, ndim=2, mode="c"] result, int N, int SIZE):
        self.c_lsh.query_batch(&fp[0, 0], &result[0, 0], N, SIZE)

    def clear(self):
        self.c_lsh.clear()

    def count(self):
        self.c_lsh.count()
