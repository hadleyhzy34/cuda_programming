#include "src/Utility.cpp"
#include "src/transpose.hpp"
#include <cstdio>

int main() {
  int width = 10240;
  int height = 10240;
  size_t matrixSize = width * height * sizeof(float);

  float *h_in = (float *)malloc(matrixSize);
  float *h_out_shared = (float *)malloc(matrixSize);
  float *h_out_global = (float *)malloc(matrixSize);

  // Initialize input matrix
  for (int i = 0; i < width * height; ++i) {
    h_in[i] = (float)i;
  }

  computation::tranposeComparison(h_in, h_out_shared, h_out_global, width,
                                  height);
  // std::function<void(float *, float *, int, int)> matrixTranspose =
  //     computation::matrixTranspose;
  //
  // measureTime(computation::matrixTranspose, h_in, h_out_shared, width,
  // height);
  //
  // // verify result
  // for (int i = 0; i < width * height; ++i) {
  //   int row = i / width;
  //   int col = i % width;
  //   if (std::abs(h_out_shared[i] - h_in[col * width + row]) >= 1e-6) {
  //     std::cout << "result not same at position: " << i << std::endl;
  //     std::cout << i << " " << row << " " << col << " " << h_out_shared[i]
  //               << " " << h_in[col * width + row] << std::endl;
  //     break;
  //   }
  // }
  //
  // std::cout << "shared memory mechanism works\n";

  measureTime(computation::matrixGlobalTranspose, h_in, h_out_global, width,
              height);
  //
  // verify result
  for (int i = 0; i < width * height; ++i) {
    int row = i / width;
    int col = i % width;
    if (std::abs(h_out_global[i] - h_in[col * width + row]) >= 1e-6) {
      std::cout << "result not same at position: " << i << std::endl;
      std::cout << i << " " << row << " " << col << " " << h_out_global[i]
                << " " << h_in[col * width + row] << std::endl;
      break;
    }
  }

  std::cout << "global memory mechanism works\n";

  measureTime(computation::matrixTranspose, h_in, h_out_shared, width, height);

  // verify result
  for (int i = 0; i < width * height; ++i) {
    int row = i / width;
    int col = i % width;
    if (std::abs(h_out_shared[i] - h_in[col * width + row]) >= 1e-6) {
      std::cout << "result not same at position: " << i << std::endl;
      std::cout << i << " " << row << " " << col << " " << h_out_shared[i]
                << " " << h_in[col * width + row] << std::endl;
      break;
    }
  }

  std::cout << "shared memory mechanism works\n";

  free(h_in);
  free(h_out_shared);
  free(h_out_global);

  return 0;
}
