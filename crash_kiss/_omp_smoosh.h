// Header for OpenMP-optimized "smoosh" operation.
// This is used by the Cython wrapper (_omp_smoosh_2d.pyx)
// to provide a copy-less extension to Python/NumPy.

#include <stdint.h>


void _omp_smoosh_2d(uint8_t *img, uint8_t *foreground,
                    int rows, int cols, int max_depth,
                    int background_value);

