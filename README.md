# cuda_programming

## basic commands

* compilation and debugging:

``` terminal
nvcc file_name -o excutable
```


## definition && declaration
1.

basic steps of a cuda program:
* initialize memory in host
* transfer the memory from host to device
* launch the kernel from host
* wait until kernel execution finish
* transfer the memory from device to host
* reclaim the memory




## memory transfer
be careful that you cannot get access to memory space without using kernel/function from cuda libs, otherwise it would generate segmentation fault error

in order to get access of memory allocated to array using GPU, two methods:
1.transfer from device back to host and copy data back cpu allocated array
2.use __global__ cuda libs to get access of gpu allocated array

//memory transfer back to host
```c++
cudaMemcpy(gpu_output,c_gpu_output,byte_size,cudaMemcpyDeviceToHost);
```

## asunchronous operation

### synchronous and asynchronous

The send, receive, and reply operations may be synchronous or asynchronous. A synchronous operation blocks a process till the operation completes. An asynchronous operation is non-blocking and only initiates the operation. The caller could discover completion by some other mechanism discussed later. (Does it make sense to have an asynchronous RPC send?) 

so when we need the results of a kernel execution we have to explicitly wait in the host using cudaDeviceSynchronize function, after the kernel call, this is to ensure the device code completes before the main code returns. Kernel launches are asynchronous, meaning the host does not wait for the kernel to return before continuing on.

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


## CUDA architecture: single instruction multiple threads

block<->streaming multiprocessor
threads<->number of cores per streaming multiprocessor

### relations between block, warps, threads

check this folder execution file: /home/swarm/developments/cuda_programming/warps

tutorial: https://www.bilibili.com/video/BV147411c7Fq?p=20

### threads in the same warps to execute different instructions

* first thought:

```c++
if(tid < 16){
    //if block
}else{
    //else block
}
```
* difficulties:
GPU:SIMT(single instruction multiple threads)
every threads in one warp has to execute same instruction


































