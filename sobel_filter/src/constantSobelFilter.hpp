// constants.h
#pragma once

namespace Constants {
void launchKernel(float h_filter[], float *h_input, float *h_output, int width,
                  int height);
} // namespace Constants
