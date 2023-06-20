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
    void _omp_smoosh_2d(img_t *img, img_t *foreground,
                        int rows, int cols, int max_depth,
                        int background_value)


def smoosh(img, foreground, max_depth, background_value=0xFFFFFF):
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]
    cdef int pixels = img.size / 3
    assert pixels == foreground.size
    rgb_flat = img.reshape((img.size,))
    foreground_flat = foreground.reshape((pixels,))
    _smoosh_2d(rgb_flat, foreground_flat, rows, cols, max_depth,
               background_value)


cdef _smoosh_2d(img_t[::1] img, img_t[::1] foreground,
                     int rows, int cols, int max_depth,
                     int background_value):
    _omp_smoosh_2d(&img[0], &foreground[0], rows, cols, max_depth,
                   background_value)

cdef _native_smoosh_2d(img_t[::1] img, img_t[::1] foreground,
                     int rows, int cols, int max_depth,
                     int background_value):

    # indices for our 2D background mask array
    # we'll multiply by 3 to stride across our RGB image array
    cdef int stride = cols;  # row stride for foreground mask array
    cdef int middle = stride / 2;
    cdef int absolute_max
    if max_depth < cols/2:
        absolute_max = max_depth
    else:
        absolute_max = cols/2

    # R, G, and B channels anded/shifted from the background_value int
    # We'll use this backgound color to fill in vacated space
    cdef int bg_rgb[3]
    bg_rgb[:] = [
        background_value & 0xFF,
        (background_value & 0xFF00) >> 8,
        (background_value & 0xFF0000) >> 16
    ]

    # The OpenMP loop over all image row indices
    cdef int i;
#pragma omp parallel for schedule(static)
    for i in range(rows):
    #for (i = 0; i < rows; i++) {
        # left and right "inner" cursors starting from middle
        l_in = middle;
        r_in = middle + 1;
        fg_offs = i * stride;
        img_offs = fg_offs * 3;

        # Now we find where the foreground on each side meet
        j;
        #for (j = l_in; j > middle-absolute_max*2; j--) {
        #for (j = l_in; j > absolute_max; j--) {
        for j in range(l_in, absolute_max, -1):
            if (foreground[fg_offs+j]):
                break;
        l_out = j;  # left outer cursor is set
        #for (j = r_in; j < middle+absolute_max*2; j++) {
        #for (j = r_in; j < cols-1-absolute_max; j++) {
        for j in range(r_in, cols-1-absolute_max):
            if (foreground[fg_offs+j]):
                break;
        r_out = j;  # right outer cursor is set

        # L and R inner cursors are at the meeting pobetween the subjects
        l_in = int(round(l_out + (r_out - l_out)/2));
        r_in = l_in + 1;
        r_in_orig = r_in;
        l_in_orig = l_in;

        # Find start indices of background pixels on LHS and RHS
        #for (j = l_in; j >= 0; j--) {
        for j in range(l_in, -1, -1):
            if not foreground[fg_offs+j]:
                break;  # we found the background on the left
        l_in = j;
        if l_in > l_out + absolute_max:
            l_in = l_out + absolute_max;
        #for (j = r_in; j < cols-1; j++) {
        for j in range(r_in, cols-1):
            if not foreground[fg_offs+j]:
                break;  # we found the background on the right
        r_in = j;
        if (r_in < r_out - absolute_max):
            r_in = r_out - absolute_max;

        # Collapse background pixels, not to exceed absolute_max
        #k;  # RGB color channel index
        #z;  # temp cursor index
        l_in_last = l_in;
        l_out_last = l_out;
        l_crushed = 0;
        while l_in and ((l_in-l_out) < absolute_max):
            # Step 1: move outer pixel inward
            #for (k = 0; k < 3; k++) {
            for k in range(3):
                img[img_offs+l_in*3+k] = img[img_offs+l_out*3+k];

            # Step 2: clear/set mask on outer/inner cursors
            foreground[fg_offs+l_in] = 1;  # inner cursor is now foreground
            foreground[fg_offs+l_out] = 0;  # outer cursor is now background

            # Step 3: Decrement until inner cursor has selected background
            #for (z = l_in; z >= 0; z--) {
            for z in range(l_in, -1, -1):
                if not foreground[fg_offs+z]:
                    break;
            l_in_last = l_in;
            l_in = z;

            # Step 4: Decrement until outer cursor has selected foreground
            # for (z = l_out; z >= 0; z--) {
            for z in range(l_out, -1, -1):
                if (foreground[fg_offs+z]):
                    break;
            l_out_last = l_out;
            l_out = z;
            l_crushed = 1;

        # Correct for overshoot due to lack of foreground
        if not l_crushed:
            l_in_last = l_in_orig;
            l_out_last = l_in_orig - absolute_max;

        # Move remaining pixels on left
        l_dist = l_in_last - l_out_last;
        #for (j = l_out_last; j >= 0; j--) {
        for j in range(l_out_last, -1, -1):
            #for (k = 0; k < 3; k++) {
            for k in range(3):
                img[img_offs + (j+l_dist)*3 + k] = (
                    img[img_offs + j*3 + k]
                )

        # Clear vacated space on left edge
        #for (j = l_dist; j >= 0; j--) {
        for j in range(l_dist, -1, -1):
            #for (k = 0; k < 3; k++) {
            for k in range(3):
                img[img_offs + j*3 + k] = bg_rgb[k];

        # Crush the right half of the image
        r_in_last = r_in;
        r_out_last = r_out;
        r_crushed = 0;
        while ((r_in < cols-1) and (r_out-r_in < absolute_max)):
            # Step 1: move outer pixel inward
            #for (k = 0; k < 3; k++) {
            for k in range(3):
                img[img_offs+r_in*3+k] = img[img_offs+r_out*3+k];

            # Step 2: clear/set mask on outer/inner cursors
            foreground[fg_offs+r_in] = 1;  # inner cursor is now foreground
            foreground[fg_offs+r_out] = 0;  # outer cursor is now background

            # Step 3: Increment until inner cursor has selected background
            #for (z = r_in; z <= cols-1; z++) {
            for z in range(r_in, cols):
                if not foreground[fg_offs+z]:
                    break;
            r_in_last = r_in;
            r_in = z;

            # Step 4: Increment until outer cursor has selected foreground
            #for (z = r_out; z <= cols-1; z++) {
            for z in range(r_out, cols):
                if (foreground[fg_offs+z]):
                    break;
            r_out_last = r_out;
            r_out = z;
            r_crushed = 1;

        # Correct for overshoot due to lack of foreground
        if not r_crushed:
            r_in_last = r_in_orig;
            r_out_last = r_in_orig + absolute_max;

        # Move remaining pixels on right
        r_dist = r_out_last - r_in_last;
        #for (j = r_out_last; j <= cols-1; j++) {
        for j in range(r_out_last, cols):
            #for (k = 0; k < 3; k++) {
            for k in range(3):
                img[img_offs + (j-r_dist)*3 + k] = (
                    img[img_offs + j*3 + k])

        # Clear vacated space on left edge
#        for (j = cols-1-r_dist; j <= cols-1; j++) {
        for j in range(cols-1-r_dist, cols):
#            for (k = 0; k < 3; k++) {
            for k in range(3):
                img[img_offs + j*3 + k] = bg_rgb[k];

