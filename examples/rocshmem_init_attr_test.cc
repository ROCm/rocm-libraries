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
hipcc -c -fgpu-rdc -x hip rocshmem_init_attr_test.cc \
  -I/opt/rocm/include \
  -I$ROCSHMEM_INSTALL_DIR/include \
  -I$OPENMPI_UCX_INSTALL_DIR/include/

hipcc -fgpu-rdc --hip-link rocshmem_init_attr_test.o -o rocshmem_init_attr_test \
  $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a \
  $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

ROCSHMEM_MAX_NUM_CONTEXTS=2 mpirun -np 2 ./rocshmem_init_attr_test
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

int main (int argc, char **argv)
{
    int world_rank, world_nranks;
    int ret;
    rocshmem_uniqueid_t uid;
    rocshmem_init_attr_t attr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &world_nranks);

    // Create two disjoint groups of processes, each
    // one creating a unique rocshmem environment independent
    // of the other group
    MPI_Comm newcomm;
    int color = world_rank %2;
    int rank, nranks;

    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &newcomm);
    MPI_Comm_rank (newcomm, &rank);
    MPI_Comm_size (newcomm, &nranks);

    if (rank == 0) {
      ret = rocshmem_get_uniqueid (&uid);
      if (ret != ROCSHMEM_SUCCESS) {
        std::cout << rank << ": Error in rocshmem_get_uniqueid. Aborting.\n";
        MPI_Abort (MPI_COMM_WORLD, ret);
      }
    }

    MPI_Bcast (&uid, sizeof(rocshmem_uniqueid_t), MPI_BYTE, 0, newcomm);
    ret = rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << rank << ": Error in rocshmem_set_attr_uniqueid_args. Aborting.\n";
      MPI_Abort (MPI_COMM_WORLD, ret);
    }
    
    ret = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
    if (ret != ROCSHMEM_SUCCESS) {
      std::cout << rank << ": Error in rocshmem_init_attr. Aborting.\n";
      MPI_Abort (MPI_COMM_WORLD, ret);
    }

    std::cout << rank << ": rocshmem_init_attr SUCCESS\n";

    rocshmem_finalize();
    MPI_Comm_free (&newcomm);
    MPI_Finalize();
    return 0;
}
