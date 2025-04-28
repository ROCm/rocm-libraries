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

#ifndef _TESTER_HPP_
#define _TESTER_HPP_

#include <rocshmem/rocshmem.hpp>
#include <vector>
#include <climits>

#include "tester_arguments.hpp"
#include "../src/util.hpp"
#include "verify_results_kernels.hpp"

/******************************************************************************
 * TESTER CLASS TYPES
 *****************************************************************************/
enum TestType {
  GetTestType = 0,
  GetNBITestType = 1,
  PutTestType = 2,
  PutNBITestType = 3,
  AMO_FAddTestType = 4,
  AMO_FIncTestType = 5,
  AMO_FetchTestType = 6,
  AMO_FCswapTestType = 7,
  AMO_AddTestType = 8,
  AMO_IncTestType = 9,
  AMO_CswapTestType = 10,
  InitTestType = 11,
  PingPongTestType = 12,
  RandomAccessTestType = 13,
  BarrierAllTestType = 14,
  SyncAllTestType = 15,
  SyncTestType = 16,
  CollectTestType = 17,
  TeamFCollectTestType = 18,
  TeamAllToAllTestType = 19,
  AllToAllsTestType = 20,
  ShmemPtrTestType = 21,
  PTestType = 22,
  GTestType = 23,
  WGGetTestType = 24,
  WGGetNBITestType = 25,
  WGPutTestType = 26,
  WGPutNBITestType = 27,
  WAVEGetTestType = 28,
  WAVEGetNBITestType = 29,
  WAVEPutTestType = 30,
  WAVEPutNBITestType = 31,
  TeamBroadcastTestType = 32,
  TeamReductionTestType = 33,
  TeamCtxGetTestType = 34,
  TeamCtxGetNBITestType = 35,
  TeamCtxPutTestType = 36,
  TeamCtxPutNBITestType = 37,
  TeamCtxInfraTestType = 38,
  PutNBIMRTestType = 39,
  AMO_SetTestType = 40,
  AMO_SwapTestType = 41,
  AMO_FetchAndTestType = 42,
  AMO_FetchOrTestType = 43,
  AMO_FetchXorTestType = 44,
  AMO_AndTestType = 45,
  AMO_OrTestType = 46,
  AMO_XorTestType = 47,
  PingAllTestType = 48,
  PutSignalTestType = 49,
  WGPutSignalTestType = 50,
  WAVEPutSignalTestType = 51,
  PutSignalNBITestType = 52,
  WGPutSignalNBITestType = 53,
  WAVEPutSignalNBITestType = 54,
  SignalFetchTestType = 55,
  WGSignalFetchTestType = 56,
  WAVESignalFetchTestType = 57,
  TeamWGBarrierTestType = 58,
  DefaultCTXGetTestType = 59,
  DefaultCTXGetNBITestType = 60,
  DefaultCTXPutTestType = 61,
  DefaultCTXPutNBITestType = 62,
  DefaultCTXPTestType = 63,
  DefaultCTXGTestType = 64,
  WAVEBarrierAllTestType = 65,
  WGBarrierAllTestType = 66,
  WAVESyncAllTestType = 67,
  WGSyncAllTestType = 68,
  TeamBarrierTestType = 69,
  TeamWAVEBarrierTestType = 70,
  WAVESyncTestType = 71,
  WGSyncTestType = 72,
};

enum OpType { PutType = 0, GetType = 1 };

typedef int ShmemContextType;

/******************************************************************************
 * TESTER INTERFACE
 *****************************************************************************/
class Tester {
 public:
  explicit Tester(TesterArguments args);
  virtual ~Tester();

  void execute();

  static std::vector<Tester *> create(TesterArguments args);

 protected:
  virtual void resetBuffers(uint64_t size) = 0;

  virtual void preLaunchKernel() {}

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) = 0;

  virtual void postLaunchKernel() {}

  virtual void verifyResults(uint64_t size) = 0;

  int num_msgs = 0;
  int num_timed_msgs = 0;
  int num_warps = 0;
  int bw_factor = 1;
  int device_id = 0;
  int wall_clk_rate = 0; //in kilohertz
  int wf_size = 0;

  TesterArguments args;

  TestType _type;
  ShmemContextType _shmem_context = 8;  // SHMEM_CTX_WP_PRIVATE

  hipStream_t stream;
  hipDeviceProp_t deviceProps;

  long long int *timer = nullptr;
  long long int *start_time = nullptr;
  long long int *end_time = nullptr;
  long long int min_start_time = 0;
  long long int max_end_time = 0;
  uint32_t num_timers = 0;

  bool *verification_error;

 private:
  bool _print_header = 1;
  void print(uint64_t size);

  void barrier();

  double gpuCyclesToMicroseconds(long long int cycles);

  double timerAvgInMicroseconds();

  bool peLaunchesKernel();

  hipEvent_t start_event;
  hipEvent_t stop_event;
};

#define CHECK_HIP(cmd)                                                        \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#endif /* _TESTER_HPP */
