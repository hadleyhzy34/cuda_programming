# cuda_programming

## basic commands

* compilation and debugging:

``` terminal
nvcc file_name -o excutable
```


## definition && declaration
1.


## memory transfer
be careful that you cannot get access to memory space without using kernel/function from cuda libs, otherwise it would generate segmentation fault error

in order to get access of memory allocated to array using GPU, two methods:
1.transfer from device back to host and copy data back cpu allocated array
2.use __global__ cuda libs to get access of gpu allocated array

//memory transfer back to host
```c++
cudaMemcpy(gpu_output,c_gpu_output,byte_size,cudaMemcpyDeviceToHost);
```

```c++
__global__ void sum_array_gpu(int *a,int *b,int *c,int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
    //printf("gid : %d, a[gid] : %d, b[gid] : %d, c[gid] : %d\n", gid, a[gid], b[gid], c[gid]);
}
```
## cuda error handling


## time calculation

```c++
clock_t gpu_start,gpu_end;
gpu_start = clock();
...
gpu_end = clock();
printf("%s\n", (double)((double)(gpu_end - gpu_start)/CLOCKS_PER_SEC));

## problem
* inline and using `define` to make error exception

```


## device properties

### device instance initialization

```c++
int deviceCount = 0;
     cudaGetDeviceCount(&deviceCount);
 
     if(deviceCount == 0)
     {
         printf("no cuda support device found\n");
     }else{
         printf("number of cuda device supported: %d\n", deviceCount);
     }
```

### device properties initialization

```c++
int devNo = 0;
     cudaDeviceProp iProp;
     cudaGetDeviceProperties(&iProp, devNo);
 
     printf("device %d: %s\n", devNo, iProp.name);
```


### device properties query

```c++
printf("number of multiprocessors: %d\n",iProp.multiProcessorCount);
     printf("total amount of global memory: %4.2f GB \n",(iProp.totalGlobalMem/(1024.0*1024.0*1024.0
     printf("total amount of constant memory: %4.2f KB\n",iProp.totalConstMem / 1024.0);
     printf("  Total number of registers available per block: %d\n",
         iProp.regsPerBlock);
     printf("  Warp size:                                     %d\n",
         iProp.warpSize);
     printf("  Maximum number of threads per block:           %d\n",
         iProp.maxThreadsPerBlock);
     printf("  Maximum number of threads per multiprocessor:  %d\n",
         iProp.maxThreadsPerMultiProcessor);
     printf("  Maximum number of warps per multiprocessor:    %d\n",
         iProp.maxThreadsPerMultiProcessor / 32);
     printf("  Maximum Grid size                         :    (%d,%d,%d)\n",
         iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);
     printf("  Maximum block dimension                   :    (%d,%d,%d)\n",
         iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
```


