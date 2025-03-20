#include <iostream>

namespace computation {
void matrixTranspose(float *h_in, float *h_out, int width, int height);
void matrixGlobalTranspose(float *h_in, float *h_out, int width, int height);
void tranposeComparison(float *h_in, float *h_out_shared, float *h_out_global,
                        int width, int height);
}; // namespace computation
