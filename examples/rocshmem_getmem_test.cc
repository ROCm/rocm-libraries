/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

/*
hipcc -c -fgpu-rdc -x hip rocshmem_getmem_test.cc \
  -I/opt/rocm/include \
  -I$ROCSHMEM_INSTALL_DIR/include \
  -I$OPENMPI_UCX_INSTALL_DIR/include/

hipcc -fgpu-rdc --hip-link rocshmem_getmem_test.o -o rocshmem_getmem_test \
  $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a \
  $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 ./rocshmem_getmem_test
*/

#include <iostream>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

#define CHECK_HIP(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, error);                             \
        }                                                                 \
    }

using namespace rocshmem;

__global__ void simple_getmem_test(int *src, int *dst, size_t nelem)
{
    rocshmem_wg_init();

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId == 0) {
        int rank = rocshmem_my_pe();
        int peer =  rank ? 0 : 1;
        rocshmem_getmem(dst, src, nelem * sizeof(int), peer);
        rocshmem_quiet();
    }

    __syncthreads();
    rocshmem_wg_finalize();
}

#define MAX_ELEM 256

int main (int argc, char **argv)
{
    int rank = rocshmem_my_pe();
    int ndevices, my_device = 0;
    CHECK_HIP(hipGetDeviceCount(&ndevices));
    my_device = rank % ndevices;
    CHECK_HIP(hipSetDevice(my_device));
    int nelem = MAX_ELEM;

    if (argc > 1) {
        nelem = atoi(argv[1]);
    }

    rocshmem_init();
    int npes =  rocshmem_n_pes();
    int *src = (int *)rocshmem_malloc(nelem * sizeof(int));
    int *dst = (int *)rocshmem_malloc(nelem * sizeof(int));
    if (NULL == src || NULL == dst) {
        std::cout << "Error allocating memory from symmetric heap" << std::endl;
        std::cout << "source: " << src << ", dest: " << dst << ", size: "
          << sizeof(int) * nelem << std::endl;
        rocshmem_global_exit(1);
    }

    for (int i=0; i<nelem; i++) {
        src[i] = 0;
        dst[i] = 1;
    }
    CHECK_HIP(hipDeviceSynchronize());

    int threadsPerBlock=256;
    simple_getmem_test<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(src, dst, nelem);
    rocshmem_barrier_all();
    CHECK_HIP(hipDeviceSynchronize());

    bool pass = true;
    for (int i=0; i<nelem; i++) {
        if (dst[i] != 0) {
            pass = false;
#if VERBOSE
            printf("[%d] Error in element %d expected 0 got %d\n", rank, i, dst[i]);
#endif
        }
    }
    printf("Test %s \t %s\n", argv[0], pass ? "[PASS]" : "[FAIL]");

    rocshmem_free(src);
    rocshmem_free(dst);
    rocshmem_finalize();
    return 0;
}
