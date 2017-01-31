"""Cython wrapper for _omp_smoosh.c extension.
Cython generates C code that makes passing NumPy arrays
in between Python <-> C easy and copy-free."""

import numpy as np
cimport numpy as np


img_type = np.uint8
ctypedef np.uint8_t img_t
cdef size_t data_size = sizeof(img_t)


cdef extern from "_omp_smoosh.h":
    # our _omp_smoosh_2d function from _omp_smoosh.c
    _omp_smoosh_2d(img_t *img, img_t *foreground,
                   int rows, int cols, int max_depth)


def smoosh(img, foreground, max_depth):
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[0]
    cdef int pixels = img.size / 3
    assert pixels == foreground.size
    rgb_flat = img.reshape((pixels, 3))
    foreground_flat = foreground.reshape((pixels,))
    _smoosh_2d(rgb_flat, foreground_flat, rows, cols, max_depth)


cdef void _smoosh_2d(img_t[:, ::1] img, img_t[:, ::1] foreground,
                     int rows, int cols, int max_depth):
    _omp_smoosh_2d(img, foreground, rows, cols, max_depth)



