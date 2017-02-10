// C implementation of the "smoosh" operation made quicker with OpenMP
// The "smoosh" is a process where the left and right halves of an image
// are pushed together; foreground area is preserved and background area
// is lost.

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


/*
void _omp_smoosh_2d(uint8_t *img, uint8_t *foreground,
                    int rows, int cols, int max_depth,
                    int background_value);
*/


void _omp_smoosh_2d(uint8_t *img, uint8_t *foreground,
                    int rows, int cols, int max_depth,
                    int background_value) {
    // indices for our 2D background mask array
    // we'll multiply by 3 to stride across our RGB image array
    const int stride = cols;  // row stride for foreground mask array
    const int middle = stride / 2;
    //const int absolute_max = max_depth;
    const int absolute_max = max_depth < cols/2 ? max_depth : cols/2;
    const int bg_rgb[] = {background_value & 0xFF,
                          (background_value & 0xFF00) >> 8,
                          (background_value & 0xFF0000) >> 16};

    int i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < rows; i++) {
        // left and right "inner" cursors starting from middle
        int l_in = middle;
        int r_in = middle + 1;
        int fg_offs = i * stride;
        int img_offs = fg_offs * 3;

        // Now we find where the foreground on each side meet
        int j;
        //for (j = l_in; j > l_in - absolute_max; j--) {
        for (j = l_in; j > absolute_max; j--) {
            if (foreground[fg_offs+j]) // found the first foreground pixel on the left
                break;
        }
        int l_out = j;  // left outer cursor is set
        //for (j = r_in; j < r_in + absolute_max; j++) {
        for (j = r_in; j < cols-1-absolute_max; j++) {
            if (foreground[fg_offs+j]) // found the first foreground pixel on the right
                break;
        }
        int r_out = j;  // right outer cursor is set
        // L and R inner cursors are at the meeting point between the subjects

        l_in = l_out + (r_out - l_out)/2;
        r_in = l_in + 1;

        // First, we find the start indices of the background
        // on the left and right of the meeting point
        for (j = l_in; j >= 0; j--) {
            if (!foreground[fg_offs+j])
                break;  // we found the background on the left
        }
        l_in = j;
        if (l_in > l_out + absolute_max)
            l_in = l_out + absolute_max;
        //l_in = fmin(j, l_out+absolute_max);
        for (j = r_in; j < cols-1; j++) {
            if (!foreground[fg_offs+j])
                break;  // we found the background on the right
        }
        r_in = j;
        if (r_in < r_out - absolute_max)
            r_in = r_out - absolute_max;
        //r_in = fmax(j, r_out-absolute_max);

        // Collapse background pixels, not to exceed absolute_max
        int k;  // RGB color channel index
        int z;  // temp cursor index

        // Crush the left half of the image
        int l_in_last = l_in;
        int l_out_last = l_out;
        while (l_in && (l_in-l_out < absolute_max)) {
            // Step 1: move outer pixel inward
            for (k = 0; k < 3; k++) {
                img[img_offs+l_in*3+k] = img[img_offs+l_out*3+k];
            }

            // Step 2: clear/set mask on outer/inner cursors
            foreground[fg_offs+l_in] = 1;  // inner cursor is now foreground
            foreground[fg_offs+l_out] = 0;  // outer cursor is now background

            // Step 3: Decrement until inner cursor has selected background
            for (z = l_in; z >= 0; z--) {
                if (!foreground[fg_offs+z])
                    break;
            }
            l_in_last = l_in;
            l_in = z;

            // Step 4: Decrement until outer cursor has selected foreground
            for (z = l_out; z >= 0; z--) {
                if (foreground[fg_offs+z])
                    break;
            }
            l_out_last = l_out;
            l_out = z;
        }

        // Move remaining pixels on left
        if (l_out_last == absolute_max) {
            l_out_last = middle - absolute_max;
            l_in_last = middle;
        }
        int l_dist = l_in_last - l_out_last;
        for (j = l_out_last; j >= 0; j--) {
            for (k = 0; k < 3; k++) {
                img[img_offs + (j+l_dist)*3 + k] = 
                    img[img_offs + j*3 + k];
            }
        }

        // Clear vacated space on left edge
        for (j = l_dist; j >= 0; j--) {
            for (k = 0; k < 3; k++) {
                img[img_offs + j*3 + k] = bg_rgb[k];
            }
        }

        // Crush the right half of the image
        int r_in_last = r_in;
        int r_out_last = r_out;
        while ((r_in < cols-1) && (r_out-r_in < absolute_max)) {
            // Step 1: move outer pixel inward
            for (k = 0; k < 3; k++) {
                img[img_offs+r_in*3+k] = img[img_offs+r_out*3+k];
            }

            // Step 2: clear/set mask on outer/inner cursors
            foreground[fg_offs+r_in] = 1;  // inner cursor is now foreground
            foreground[fg_offs+r_out] = 0;  // outer cursor is now background

            // Step 3: Increment until inner cursor has selected background
            for (z = r_in; z <= cols-1; z++) {
                if (!foreground[fg_offs+z])
                    break;
            }
            r_in_last = r_in;
            r_in = z;

            // Step 4: Increment until outer cursor has selected foreground
            for (z = r_out; z <= cols-1; z++) {
                if (foreground[fg_offs+z])
                    break;
            }
            r_out_last = r_out;
            r_out = z;
        }

        // Move remaining pixels on right
        if (r_out_last == cols - absolute_max - 1) {
            r_out_last = middle + absolute_max;
            r_in_last = middle;
        }
        int r_dist = r_out_last - r_in_last;
        for (j = r_out_last; j <= cols-1; j++) {
            for (k = 0; k < 3; k++) {
                img[img_offs + (j-r_dist)*3 + k] = 
                    img[img_offs + j*3 + k];
            }
        }

        // Clear vacated space on left edge
        for (j = cols-1-r_dist; j <= cols-1; j++) {
            for (k = 0; k < 3; k++) {
                img[img_offs + j*3 + k] = bg_rgb[k];
            }
        }
    }
}

