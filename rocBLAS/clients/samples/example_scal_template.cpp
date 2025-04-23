/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "client_utility.hpp"
#include "rocblas.hpp"
#include "rocblas_init.hpp"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

/* ============================================================================================ */

int main()
{

    rocblas_int    N      = 10240;
    rocblas_status status = rocblas_status_success;
    float          alpha  = 10.0;

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory, plz follow this practice
    std::vector<float> hx(N);
    std::vector<float> hz(N);
    float*             dx;

    double gpu_time_used;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // allocate memory on device
    hipMalloc(&dx, N * sizeof(float));

    // Initial Data on CPU
    srand(1);
    rocblas_init(hx.data(), 1, N, 1);

    // copy vector is easy in STL; hz = hx: save a copy in hz which will be output of CPU BLAS
    hz = hx;

    hipMemcpy(dx, hx.data(), sizeof(float) * N, hipMemcpyHostToDevice);

    printf("N        rocblas(us)     \n");

    gpu_time_used = get_time_us_sync_device(); // in microseconds

    /* =====================================================================
         ROCBLAS  C++ template interface
    =================================================================== */

    status = rocblas_scal<float>(handle, N, &alpha, dx, 1);
    if(status != rocblas_status_success)
    {
        return status;
    }

    gpu_time_used = get_time_us_sync_device() - gpu_time_used;

    // copy output from device to CPU
    hipMemcpy(hx.data(), dx, sizeof(float) * N, hipMemcpyDeviceToHost);

    // verify rocblas_scal result
    for(rocblas_int i = 0; i < N; i++)
    {
        if(hz[i] * alpha != hx[i])
        {
            printf("error in element %d: CPU=%f, GPU=%f ", i, hz[i] * alpha, hx[i]);
            break;
        }
    }

    printf("%d    %8.2f        \n", (int)N, gpu_time_used);

    hipFree(dx);
    rocblas_destroy_handle(handle);
    return rocblas_status_success;
}
