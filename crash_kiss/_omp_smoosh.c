// C implementation of the "smoosh" operation made quicker with OpenMP
// The "smoosh" is a process where the left and right halves of an image
// are pushed together; foreground area is preserved and background area
// is lost.

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
//#include <omp.h>


void _omp_smoosh_2d(uint8_t *img, uint8_t *foreground,
                    int rows, int cols, int max_depth,
                    int background_value) {

    // indices for our 2D background mask array
    // we'll multiply by 3 to stride across our RGB image array
    const int stride = cols;  // row stride for foreground mask array
    const int middle = stride / 2;
    const int absolute_max = max_depth;

    // R, G, and B channels anded/shifted from the background_value int
    // We'll use this backgound color to fill in vacated space
    const int bg_rgb[] =  {background_value & 0xFF,
                          (background_value & 0xFF00) >> 8,
                          (background_value & 0xFF0000) >> 16};

    // The OpenMP loop over all image row indices
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
        for (j = l_in; j > middle-absolute_max*2; j--) {
            if (foreground[fg_offs+j])
                break;
        }
        int l_out = j;  // left outer cursor is set
        for (j = r_in; j < middle+absolute_max*2; j++) {
            if (foreground[fg_offs+j])
                break;
        }
        int r_out = j;  // right outer cursor is set

        // L and R inner cursors are at the meeting point between the subjects
        l_in = l_out + (r_out - l_out)/2;
        r_in = l_in + 1;
        int r_in_orig = r_in;
        int l_in_orig = l_in;

        // Find LHS inner index
        for (j = l_in; j >= l_out; j--) {
            if (!foreground[fg_offs+j])
                break;  // we found the background on the left
        }
        l_in = j;

        // Find RHS inner index
        for (j = r_in; j < r_out; j++) {
            if (!foreground[fg_offs+j])
                break;  // we found the background on the right
        }
        r_in = j;

        // Collapse background pixels, not to exceed absolute_max
        int k;  // RGB color channel index
        int z;  // temp cursor index
        int l_in_last = l_in;
        int l_out_last = l_out;
        int l_crushed = 0;
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
            l_crushed = 1;
        }

        // Correct for overshoot due to lack of foreground
        if (!l_crushed) {
            l_in_last = l_in_orig;
            l_out_last = l_in_orig - absolute_max;
        }

        // Move remaining pixels on left
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
        int r_crushed = 0;
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
            r_crushed = 1;
        }

        // Correct for overshoot due to lack of foreground
        if (!r_crushed) {
            r_in_last = r_in_orig;
            r_out_last = r_in_orig + absolute_max;

        }

        // Move remaining pixels on right
        int r_dist = r_out_last - r_in_last;
        for (j = r_out_last; j <= cols-1; j++) {
            for (k = 0; k < 3; k++) {
                img[img_offs + (j-r_dist)*3 + k] = 
                    img[img_offs + j*3 + k];
            }
        }

        // Clear vacated space on right edge
        for (j = cols-1-r_dist; j <= cols-1; j++) {
            for (k = 0; k < 3; k++) {
                img[img_offs + j*3 + k] = bg_rgb[k];
            }
        }
    }
}

