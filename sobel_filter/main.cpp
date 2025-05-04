#include "src/constantSobelFilter.hpp"
#include "src/sobelFilter.hpp"
#include "src/textureSobelFilter.hpp"
#include <ctime>
#include <functional>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>

void measureTime(
    std::string path,
    const std::function<void(float[], float *, float *, int, int)> &func) {
  // Load image using OpenCV
  cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cerr << "Error: Could not load image!" << std::endl;
    return;
  }

  // Convert image to float (0-1.0) for CUDA processing
  cv::Mat imageFloat;
  image.convertTo(imageFloat, CV_32F, 1.0 / 255.0);
  int width = imageFloat.cols;
  int height = imageFloat.rows;

  // Copy Sobel filter to constant memory
  float h_sobelX[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1}; // Sobel X filter

  // Copy result back to host
  cv::Mat outputFloat(height, width, CV_32F);
  cv::Mat output;

  clock_t start = clock();
  func(h_sobelX, imageFloat.ptr<float>(), outputFloat.ptr<float>(), width,
       height);

  printf("current function running time is: %f\n",
         (float)(clock() - start) / CLOCKS_PER_SEC);

  // Convert back to 8-bit for display
  outputFloat.convertTo(output, CV_8U, 255.0);

  // // Show results
  // cv::imshow("Original", image);
  // cv::imshow("Sobel Edge Detection (CUDA)", output);
  // cv::waitKey(0);
}

int main(int argc, char *argv[]) {
  // std::string path = "/home/hadley/Pictures/Vd-Orig.png";
  std::string path = "/home/hadley/Pictures/test.png";
  // auto func0 = Constants::launchKernel;
  // auto func1 = Texture::launchKernel;
  auto func2 = SobelFilter::launchKernel;
  // auto func0 = Constants::launchKernel;
  auto func1 = Texture::launchKernel;
  auto func0 = Constants::launchKernel;
  // auto func1 = SobelFilter::launchKernel;
  measureTime(path, func0);
  measureTime(path, func1);
  measureTime(path, func2);
  return 0;
}
