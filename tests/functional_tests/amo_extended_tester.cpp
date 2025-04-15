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

#include "amo_extended_tester.hpp"

#include <iostream>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/* Declare the global kernel template with a generic implementation */
template <typename T>
__global__ void AMOExtendedTest(int loop, int skip, long long int *start_time,
                                long long int *end_time, char *r_buf,
                                T *s_buf, T *ret_val, TestType type,
                                ShmemContextType ctx_type) {
  return;
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T>
AMOExtendedTester<T>::AMOExtendedTester(TesterArguments args) : Tester(args) {
  CHECK_HIP(hipMalloc((void **)&_ret_val, args.max_msg_size * args.num_wgs));
  _r_buf = (char *)rocshmem_malloc(args.max_msg_size);
  _s_buf = (T *)rocshmem_malloc(args.max_msg_size * args.num_wgs);
}

template <typename T>
AMOExtendedTester<T>::~AMOExtendedTester() {
  rocshmem_free(_r_buf);
  CHECK_HIP(hipFree(_ret_val));
}

template <typename T>
void AMOExtendedTester<T>::resetBuffers(uint64_t size) {
  memset(_r_buf, 0, args.max_msg_size);
  memset(_ret_val, 0, args.max_msg_size * args.num_wgs);
  memset(_s_buf, 0, args.max_msg_size * args.num_wgs);
}

template <typename T>
void AMOExtendedTester<T>::launchKernel(dim3 gridsize, dim3 blocksize, int loop,
                                        uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(AMOExtendedTest, gridsize, blocksize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, _r_buf, _s_buf,
                     _ret_val, _type, _shmem_context);

  _gridSize = gridsize;
  num_msgs = (loop + args.skip) * gridsize.x;
  num_timed_msgs = loop;
}

template <typename T>
void AMOExtendedTester<T>::verifyResults(uint64_t size) {
  T ret;
  if (args.myid == 0) {
    T expected_val = 0;

    switch (_type) {
      case AMO_FetchTestType:
        expected_val = 0;
        break;
      case AMO_SetTestType:
        expected_val = 44;
        break;
      case AMO_SwapTestType:
        expected_val = num_msgs / 2;
        break;
      default:
        break;
    }

    int fetch_op =
        (_type == AMO_FetchTestType || _type == AMO_SwapTestType) ? 1 : 0;

    if (fetch_op == 1) {
      ret = *std::max_element(_ret_val, _ret_val + args.num_wgs);
    } else {
      ret = *std::max_element(_s_buf, _s_buf + args.num_wgs);
    }
    if (ret != expected_val) {
      std::cerr << "data validation error\n";
      std::cerr << "got " << ret << ", expected " << expected_val << std::endl;
      exit(-1);
    }
  }
}

#define AMO_EXTENDED_DEF_GEN(T, TNAME)                                        \
  template <>                                                                 \
  __global__ void AMOExtendedTest<T>(                                         \
      int loop, int skip, long long int *start_time,                          \
      long long int *end_time, char *r_buf, T *s_buf, T *ret_val,             \
      TestType type, ShmemContextType ctx_type) {                             \
    __shared__ rocshmem_ctx_t ctx;                                            \
    int wg_id = get_flat_grid_id();                                           \
    rocshmem_wg_init();                                                       \
    rocshmem_wg_ctx_create(ctx_type, &ctx);                                   \
    if (hipThreadIdx_x == 0) {                                                \
      T ret = 0;                                                              \
      T cond = 0;                                                             \
      for (int i = 0; i < loop + skip; i++) {                                 \
        if (i == skip) {                                                      \
          start_time[wg_id] = wall_clock64();                                 \
        }                                                                     \
        switch (type) {                                                       \
          case AMO_FetchTestType:                                             \
            ret = rocshmem_ctx_##TNAME##_atomic_fetch(ctx, (T *)r_buf, 1);    \
            break;                                                            \
          case AMO_SetTestType:                                               \
            rocshmem_ctx_##TNAME##_atomic_set(ctx, (T *)r_buf, 44, 1);        \
            break;                                                            \
          case AMO_SwapTestType:                                              \
            ret = rocshmem_ctx_##TNAME##_atomic_swap(ctx, (T *)r_buf,         \
                                                      ret + 1, 1);            \
            break;                                                            \
          default:                                                            \
            break;                                                            \
        }                                                                     \
      }                                                                       \
      rocshmem_ctx_quiet(ctx);                                                \
      end_time[wg_id] = wall_clock64();                                       \
      ret_val[wg_id] = ret;                                                   \
      rocshmem_ctx_getmem(ctx, &s_buf[wg_id], r_buf, sizeof(T), 1);           \
    }                                                                         \
    rocshmem_wg_ctx_destroy(&ctx);                                            \
    rocshmem_wg_finalize();                                                   \
  }                                                                           \
  template class AMOExtendedTester<T>;

AMO_EXTENDED_DEF_GEN(float, float)
AMO_EXTENDED_DEF_GEN(double, double)
AMO_EXTENDED_DEF_GEN(int, int)
AMO_EXTENDED_DEF_GEN(long, long)
AMO_EXTENDED_DEF_GEN(long long, longlong)
AMO_EXTENDED_DEF_GEN(unsigned int, uint)
AMO_EXTENDED_DEF_GEN(unsigned long, ulong)
AMO_EXTENDED_DEF_GEN(unsigned long long, ulonglong)
