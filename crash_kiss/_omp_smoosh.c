// C implementation of the "smoosh" operation made quicker with OpenMP
// The "smoosh" is a process where the left and right halves of an image
// are pushed together; foreground area is preserved and background area
// is lost.

#include <math.h>
#include <stdint.h>
#include <stlib.h>
#include <omp.h>


void _omp_smoosh_2d(uint8_t *img, uint8_t *foreground,
                    int rows, int cols, int max_depth);


void _omp_smoosh_2d(uint8_t *img, uint8_t *foreground,
                    int rows, int cols, int max_depth) {
    // indices for our 2D background mask array
    // we'll multiply by 3 to stride across our RGB image array
    const int middle = cols / 2;
    const int stride = cols;  // row stride for foreground mask array
    const int absolute_max = min(cols / 2, max_depth);

    int i;
#pragma omp parallel for
    for (i = 0; i < rows; i++) {
        // left and right "inner" cursors starting from middle
        int l_in = stride*i + middle;
        int r_in = stride*i + middle + 1;
        // total number of background pixels collapsed
        int crushed = 0;

        // First, we find the start indices of the background
        // on the left and right of the center line
        int j;
        for (j = l_in; j >= 0; j--) {
            if (!foreground[j]) {
                break;  // we found the background on the left
        l_in = j;
        for (j = r_in; j < cols; j++) {
            if (!foreground[j]) {
                break;  // we found the background on the right
        r_in = j;
        if (r_in == cols && l_in == 0) {
            // we have no foreground area in this row
            continue;
        }
        // left and right "outer" cursors start just beyond inner cursors
        int l_out = l_in - 1;
        int r_out = r_in + 1;

        // Collapse background pixels, not to exceed max_depth
        while (crushed < absolute_max && (l_in ) {
            if (foreground[l_out]) {
                // Left outer cursor has selected a foreground pixel
                l_out--;
                foreground[l_in] = 1;
            } else {
                // Left inner cursor has selected a background pixel
                crushed++;
            }
            if (foreground[ri]) {
                // Right inner cursor has selected a foreground pixel
                r_in++;
            } else {
                // Right inner cursor has selected a background pixel
                crushed++;
            }
            l_out--;
            r_out++;
        }
    }
}

