import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def smoosh(np.ndarray[np.uint8_t, ndim=3] img,
           np.ndarray[np.int_t, ndim=1] lstart,
           np.ndarray[np.int_t, ndim=1] rstart,
           np.ndarray[np.uint8_t, ndim=2] foreground,
           int depth):
    cdef int nrows = img.shape[0]
    cdef int ncols = img.shape[1]
    cdef int chans = img.shape[2]
    cdef int midline = ncols / 2
    cdef int meet = 0
    cdef int travel = 0
    cdef int collapse_max = 0
    cdef int to_collapse = 0
    cdef int i, j, jnext, k, left, right
    cdef int ls, rs

    for i in range(nrows):
        ls = lstart[i]
        rs = rstart[i]
        travel = (rs + ls) / 2
        right = midline + rs
        left = midline - ls
        meet = left + (right - left) / 2
        if travel < depth:
            collapse_max = to_collapse = depth - travel
        else:
            travel = depth
            collapse_max = to_collapse = 0

        # collapse from center to left edge
        j = meet
        jnext = j - travel
        while jnext > midline - depth:
            if to_collapse and not foreground[i, jnext]:
                to_collapse -= 1
            else:
                for k in range(chans):
                    img[i, j, k] = img[i, jnext, k]
                j -= 1
            jnext -= 1
        while jnext:
            for k in range(chans):
                img[i, j, k] = img[i, jnext, k]
            j -= 1
            jnext -= 1
        for j in range(j, -1, -1):
            for k in range(chans):
                img[i, j, k] = 255

        # collapse from center to right
        to_collapse = collapse_max
        j = meet
        jnext = j + travel
        while jnext < midline + depth:
            if to_collapse and not foreground[i, jnext]:
                to_collapse -= 1
            else:
                for k in range(chans):
                    img[i, j, k] = img[i, jnext, k]
                j += 1
            jnext += 1
        while jnext < ncols:
            for k in range(chans):
                img[i, j, k] = img[i, jnext, k]
            j += 1
            jnext += 1
        for j in range(j, ncols):
            for k in range(chans):
                img[i, j, k] = 255


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def smoosh_overlap(
        np.ndarray[np.uint8_t, ndim=3] img,
        np.ndarray[np.uint8_t, ndim=2] foreground,
        np.ndarray[np.uint8_t, ndim=1] is_left_overlap,
        int depth):
    cdef int nrows = img.shape[0]
    cdef int ncols = img.shape[1]
    cdef int chans = img.shape[2]
    cdef int meet = 0
    cdef int travel = 0
    cdef int collapse_max = 0
    cdef int to_collapse = 0
    cdef int i, j, jnext, k, midline, left, right
    cdef int ls, rs
    midline = ncols / 2

    for i in range(nrows):
        if not is_left_overlap[i]:
            foreground[i] = foreground[i, ::-1]
            img[i] = img[i, ::-1]

    for i in range(nrows):
        travel = 0
        for j in range(midline, midline + depth):
            if not foreground[i, j]:  # background gap found
                break
        if j == midline + depth - 1:
            is_left_overlap[i] = not is_left_overlap[i]
            foreground[i] = foreground[i, ::-1]
            img[i] = img[i, ::-1]
            for j in range(midline, midline + depth):
                if not foreground[i, j]:  # background gap found
                    break
        for jnext in range(j+3, j + depth + 3):
            if foreground[i, jnext]:  # end of background gap found
                break
            else:
                travel += 1
        travel += 3
        travel /= 2
        meet = j + travel
        if travel < depth:
            collapse_max = to_collapse = depth - travel
        else:
            travel = depth
            collapse_max = to_collapse = 0

        # collapse from center to left edge
        j = meet
        jnext = j - travel
        while jnext > midline - depth:
            if to_collapse and not foreground[i, jnext]:
                to_collapse -= 1
            else:
                for k in range(chans):
                    img[i, j, k] = img[i, jnext, k]
                j -= 1
            jnext -= 1
        while jnext:
            for k in range(chans):
                img[i, j, k] = img[i, jnext, k]
            j -= 1
            jnext -= 1
        for j in range(j, -1, -1):
            for k in range(chans):
                img[i, j, k] = 255

        # collapse from center to right
        to_collapse = collapse_max
        j = meet
        jnext = j + travel
        while jnext < midline + depth:
            if to_collapse and not foreground[i, jnext]:
                to_collapse -= 1
            else:
                for k in range(chans):
                    img[i, j, k] = img[i, jnext, k]
                j += 1
            jnext += 1
        while jnext < ncols:
            for k in range(chans):
                img[i, j, k] = img[i, jnext, k]
            j += 1
            jnext += 1
        for j in range(j, ncols):
            for k in range(chans):
                img[i, j, k] = 255

    for i in range(nrows):
        if not is_left_overlap[i]:
            foreground[i] = foreground[i, ::-1]
            img[i] = img[i, ::-1]

