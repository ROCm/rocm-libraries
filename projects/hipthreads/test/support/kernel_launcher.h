#ifndef __KERNEL_LAUNCHER_H__
#define __KERNEL_LAUNCHER_H__

#include "hip/hip_runtime.h"
#include "hip_helpers.h"

__global__ void gmain();

int main(int, char**) {
    hipLaunchKernelGGL(gmain, dim3(1), dim3(1), 0, nullptr);
    __HIP_CHECK__(hipGetLastError());
    __HIP_CHECK__(hipDeviceSynchronize());
    return 0;
}

#endif // __KERNEL_LAUNCHER_H__
