/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "ipc_policy.hpp"

#include <mpi.h>

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "backend_bc.hpp"
#include "context_incl.hpp"
#include "util.hpp"

namespace rocshmem {

__host__ void IpcOnImpl::ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases,
                                     MPI_Comm thread_comm) {
  /*
   * Create an MPI communicator that deals only with local processes.
   */
  MPI_Comm shmcomm;
  MPI_Comm_split_type(thread_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shmcomm);

  /*
   * Figure out how many local process there are.
   */
  int Shm_size;
  MPI_Comm_size(shmcomm, &Shm_size);
  shm_size = Shm_size;

  /*
   * Figure out how this process' rank among local processes.
   */
  MPI_Comm_rank(shmcomm, &shm_rank);

  /*
   * Allocate a host-side c-array to hold the IPC handles.
   */
  void *ipc_mem_handle_uncast = malloc(shm_size * sizeof(hipIpcMemHandle_t));
  hipIpcMemHandle_t *vec_ipc_handle =
      reinterpret_cast<hipIpcMemHandle_t *>(ipc_mem_handle_uncast);

  /*
   * Call into the hip runtime to get an IPC handle for my symmetric
   * heap and store that IPC handle into the host-side c-array which was
   * just allocated.
   */
  char *base_heap = heap_bases[my_pe];
  CHECK_HIP(hipIpcGetMemHandle(&vec_ipc_handle[shm_rank], base_heap));

  /*
   * Do an all-to-all exchange with each local processing element to
   * share the symmetric heap IPC handles.
   */
  MPI_Allgather(MPI_IN_PLACE, sizeof(hipIpcMemHandle_t), MPI_CHAR,
                vec_ipc_handle, sizeof(hipIpcMemHandle_t), MPI_CHAR, shmcomm);

  /*
   * Allocate device-side array to hold the IPC symmetric heap base
   * addresses.
   */
  char **ipc_base;
  CHECK_HIP(hipMalloc(reinterpret_cast<void **>(&ipc_base),
                      shm_size * sizeof(char **)));

  /*
   * For all local processing elements, initialize the device-side array
   * with the IPC symmetric heap base addresses.
   */
  for (int i = 0; i < shm_size; i++) {
    if (i != shm_rank) {
      void **ipc_base_uncast = reinterpret_cast<void **>(&ipc_base[i]);
      CHECK_HIP(hipIpcOpenMemHandle(ipc_base_uncast, vec_ipc_handle[i],
                                    hipIpcMemLazyEnablePeerAccess));
    } else {
      ipc_base[i] = base_heap;
    }
  }

  /*
   * Set member variables used by subsequent method calls.
   */
  ipc_bases = ipc_base;

  /*
   * Free the host-side memory used to exchange the symmetric heap base
   * addresses.
   */
  free(vec_ipc_handle);

  if (0 == rocshmem_env_config.ro_disable_ipc) {
    int thread_comm_rank;

    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&pes_with_ipc_avail), shm_size * sizeof(int)));

    MPI_Comm_rank(thread_comm, &thread_comm_rank);
    MPI_Allgather(&thread_comm_rank, 1, MPI_INT, pes_with_ipc_avail, 1, MPI_INT, shmcomm);
  }
}

__host__ void IpcOnImpl::ipcHostStop() {
  for (int i = 0; i < shm_size; i++) {
    if (i != shm_rank) {
      CHECK_HIP(hipIpcCloseMemHandle(ipc_bases[i]));
    }
  }
  CHECK_HIP(hipFree(ipc_bases));

  if (nullptr != pes_with_ipc_avail) {
    CHECK_HIP(hipFree(pes_with_ipc_avail));
  }
}

__device__ void IpcOnImpl::ipcCopy(void *dst, void *src, size_t size) {
  memcpy(dst, src, size);
}

__device__ void IpcOnImpl::ipcCopy_wave(void *dst, void *src, size_t size) {
  memcpy_wave(dst, src, size);
}

__device__ void IpcOnImpl::ipcCopy_wg(void *dst, void *src, size_t size) {
  memcpy_wg(dst, src, size);
}

}  // namespace rocshmem
