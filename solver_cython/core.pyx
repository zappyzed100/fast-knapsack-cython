# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp

def test_speed(cnp.int32_t[:] values):
    cdef int i
    cdef long total = 0
    cdef int n = values.shape[0]
    
    for i in range(n):
        total += values[i]
    
    return total