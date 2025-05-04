// constants.h
#pragma once

namespace Texture {
void launchKernel(float h_filter[], float *h_input, float *h_output, int width,
                  int height);
} // namespace Texture
