#include <functional>
#include <iostream>

void measureTime(const std::function<void(float *, float *, int, int)> &f,
                 float *h_in, float *h_out, int width, int height) {
  // start running time
  clock_t start = clock();

  f(h_in, h_out, width, height);

  printf("Time taken for current approach: %.2fs\n",
         (double)(clock() - start) / CLOCKS_PER_SEC);
}
