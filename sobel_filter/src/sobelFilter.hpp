#pragma once

namespace SobelFilter {
void launchKernel(float h_filter[], float *h_input, float *h_output, int width,
                  int height);
} // namespace SobelFilter
