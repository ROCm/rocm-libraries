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

#include "shmem_ptr_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void ShmemPtrTest(char *r_buf, int *available) {
  rocshmem_wg_init();

  if (hipThreadIdx_x == 0) {
    char *local_addr = r_buf + 4;
    void *remote_addr = rocshmem_ptr((void *)local_addr, 1);
    if (remote_addr != NULL) {
      *available = 1;
      ((char *)remote_addr)[0] = '1';
    }
  }

  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
ShmemPtrTester::ShmemPtrTester(TesterArguments args) : Tester(args) {
  CHECK_HIP(hipMalloc((void **)&_available, sizeof(int)));
  r_buf = (char *)rocshmem_malloc(args.max_msg_size);
}

ShmemPtrTester::~ShmemPtrTester() {
  CHECK_HIP(hipFree(_available));
  rocshmem_free(r_buf);
}

void ShmemPtrTester::resetBuffers(uint64_t size) {
  memset(r_buf, '0', args.max_msg_size);
  memset(_available, 0, sizeof(int));
}

void ShmemPtrTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                  uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(ShmemPtrTest, gridSize, blockSize, shared_bytes, stream,
                     r_buf, _available);

  num_msgs = 0;
  num_timed_msgs = 0;
}

void ShmemPtrTester::verifyResults(uint64_t size) {
  if (args.myid == 0) {
    if (*_available == 0) {
      fprintf(stderr, "SHMEM_PTR NOT AVAILBLE \n");
    }
  } else {
    if (r_buf[4] != '1') {
      fprintf(stderr, "Data validation error \n");
      fprintf(stderr, "Got %c, Expected %c\n", r_buf[4], '1');
    }
  }
}
