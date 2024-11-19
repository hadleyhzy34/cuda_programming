#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef __UTILITY_H__
#define __UTILITY_H__

#define HANDLE_ERROR(err) (Utility::HandleError(err, __FILE__, __LINE__))

namespace Utility {
// static void HandleError(int err, const char *file, int line);

void HandleError(cudaError_t err, const char *file, int line);

void printDevice();
}; // namespace Utility
#endif
