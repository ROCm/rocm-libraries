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

#include "tester.hpp"

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <functional>
#include <iostream>
#include <rocshmem/rocshmem.hpp>
#include <vector>

#include "amo_bitwise_tester.hpp"
#include "amo_extended_tester.hpp"
#include "amo_standard_tester.hpp"
#include "default_ctx_primitive_tester.hpp"
#include "barrier_all_tester.hpp"
#include "empty_tester.hpp"
#include "ping_all_tester.hpp"
#include "ping_pong_tester.hpp"
#include "primitive_mr_tester.hpp"
#include "primitive_tester.hpp"
#include "random_access_tester.hpp"
#include "shmem_ptr_tester.hpp"
#include "signaling_operations_tester.hpp"
#include "sync_all_tester.hpp"
#include "sync_tester.hpp"
#include "team_alltoall_tester.hpp"
#include "team_barrier_tester.hpp"
#include "team_broadcast_tester.hpp"
#include "team_ctx_infra_tester.hpp"
#include "team_ctx_primitive_tester.hpp"
#include "team_fcollect_tester.hpp"
#include "team_reduction_tester.hpp"
#include "wavefront_primitives.hpp"
#include "workgroup_primitives.hpp"

Tester::Tester(TesterArguments args) : args(args) {
  _type = (TestType)args.algorithm;
  _shmem_context = args.shmem_context;
  CHECK_HIP(hipGetDevice(&device_id));
  CHECK_HIP(hipGetDeviceProperties(&deviceProps, device_id));
  wf_size = deviceProps.warpSize;
  num_warps = (args.wg_size - 1) / wf_size + 1;
  CHECK_HIP(hipStreamCreate(&stream));
  CHECK_HIP(hipEventCreate(&start_event));
  CHECK_HIP(hipEventCreate(&stop_event));
  CHECK_HIP(hipDeviceGetAttribute(&wall_clk_rate,
    hipDeviceAttributeWallClockRate, device_id));
  num_timers = args.num_wgs;
  switch (_type) {
    case WAVEGetTestType:
    case WAVEGetNBITestType:
    case WAVEPutTestType:
    case WAVEPutNBITestType:
      num_timers = args.num_wgs * num_warps;
      break;
    default:
      break;
  }
  CHECK_HIP(hipMalloc((void**)&timer, sizeof(long long int) * num_timers));
  CHECK_HIP(hipMalloc((void**)&start_time, sizeof(long long int) * num_timers));
  CHECK_HIP(hipMalloc((void**)&end_time, sizeof(long long int) * num_timers));
}

Tester::~Tester() {
  CHECK_HIP(hipFree(end_time));
  CHECK_HIP(hipFree(start_time));
  CHECK_HIP(hipFree(timer));
  CHECK_HIP(hipEventDestroy(stop_event));
  CHECK_HIP(hipEventDestroy(start_event));
  CHECK_HIP(hipStreamDestroy(stream));
}

std::vector<Tester*> Tester::create(TesterArguments args) {
  int rank = args.myid;
  std::vector<Tester*> testers;

  if (rank == 0) std::cout << "### Creating Test: ";

  TestType type = (TestType)args.algorithm;

  switch (type) {
    case InitTestType:
      if (rank == 0) std::cout << "Init ###" << std::endl;
      testers.push_back(new EmptyTester(args));
      return testers;
    case GetTestType:
      if (rank == 0) std::cout << "Blocking Gets ###" << std::endl;
      testers.push_back(new PrimitiveTester(args));
      return testers;
    case GetNBITestType:
      if (rank == 0) std::cout << "Non-Blocking Gets ###" << std::endl;
      testers.push_back(new PrimitiveTester(args));
      return testers;
    case PutTestType:
      if (rank == 0) std::cout << "Blocking Puts ###" << std::endl;
      testers.push_back(new PrimitiveTester(args));
      return testers;
    case PutNBITestType:
      if (rank == 0) std::cout << "Non-Blocking Puts ###" << std::endl;
      testers.push_back(new PrimitiveTester(args));
      return testers;
    case DefaultCTXGetTestType:
      if (rank == 0)
        std::cout << "Default context Blocking Gets ###" << std::endl;
      testers.push_back(new DefaultCTXPrimitiveTester(args));
      return testers;
    case DefaultCTXGetNBITestType:
      if (rank == 0)
        std::cout << "Default context Non-Blocking Gets ###" << std::endl;
      testers.push_back(new DefaultCTXPrimitiveTester(args));
      return testers;
    case DefaultCTXPutTestType:
      if (rank == 0)
        std::cout << "Default context Blocking Puts ###" << std::endl;
      testers.push_back(new DefaultCTXPrimitiveTester(args));
      return testers;
    case DefaultCTXPutNBITestType:
      if (rank == 0)
        std::cout << "Default context Non-Blocking Puts ###" << std::endl;
      testers.push_back(new DefaultCTXPrimitiveTester(args));
      return testers;
    case TeamCtxInfraTestType:
      if (rank == 0) std::cout << "Team Ctx Infra test ###" << std::endl;
      testers.push_back(new TeamCtxInfraTester(args));
      return testers;
    case TeamCtxGetTestType:
      if (rank == 0) std::cout << "Blocking Team Ctx Gets ###" << std::endl;
      testers.push_back(new TeamCtxPrimitiveTester(args));
      return testers;
    case TeamCtxGetNBITestType:
      if (rank == 0) std::cout << "Non-Blocking Team Ctx Gets ###" << std::endl;
      testers.push_back(new TeamCtxPrimitiveTester(args));
      return testers;
    case TeamCtxPutTestType:
      if (rank == 0) std::cout << "Blocking Team Ctx Puts ###" << std::endl;
      testers.push_back(new TeamCtxPrimitiveTester(args));
      return testers;
    case TeamCtxPutNBITestType:
      if (rank == 0) std::cout << "Non-Blocking Team Ctx Puts ###" << std::endl;
      testers.push_back(new TeamCtxPrimitiveTester(args));
      return testers;
    case PTestType:
      if (rank == 0) std::cout << "P Test ###" << std::endl;
      testers.push_back(new PrimitiveTester(args));
      return testers;
    case GTestType:
      if (rank == 0) std::cout << "G Test ###" << std::endl;
      testers.push_back(new PrimitiveTester(args));
      return testers;
    case TeamReductionTestType:
      if (rank == 0)
        std::cout << "All-to-All Team-based Reduction ###" << std::endl;
      testers.push_back(new TeamReductionTester<float, ROCSHMEM_SUM>(
          args,
          [](float& f1, float& f2) {
            f1 = 1;
            f2 = 1;
          },
          [](float v, float n_pes) {
            return (v == n_pes)
                       ? std::make_pair(true, "")
                       : std::make_pair(false, "Got " + std::to_string(v) +
                                                   ", Expect " +
                                                   std::to_string(n_pes));
          }));
      return testers;
    case TeamBroadcastTestType:
      if (rank == 0) {
        std::cout << "Team Broadcast Test ###" << std::endl;
      }
      testers.push_back(new TeamBroadcastTester<int64_t>(args));
      testers.push_back(new TeamBroadcastTester<int>(args));
      testers.push_back(new TeamBroadcastTester<long long>(args));
      testers.push_back(new TeamBroadcastTester<float>(args));
      testers.push_back(new TeamBroadcastTester<double>(args));
      testers.push_back(new TeamBroadcastTester<char>(args));
      testers.push_back(new TeamBroadcastTester<unsigned char>(args));
      return testers;
    case TeamAllToAllTestType:
      if (rank == 0) {
        std::cout << "Alltoall Test ###" << std::endl;
      }
      testers.push_back(new TeamAlltoallTester<int64_t>(args));
      testers.push_back(new TeamAlltoallTester<int>(args));
      testers.push_back(new TeamAlltoallTester<long long>(args));
      testers.push_back(new TeamAlltoallTester<float>(args));
      testers.push_back(new TeamAlltoallTester<double>(args));
      testers.push_back(new TeamAlltoallTester<char>(args));
      testers.push_back(new TeamAlltoallTester<unsigned char>(args));
      return testers;
    case TeamFCollectTestType:
      if (rank == 0) {
        std::cout << "Fcollect Test ###" << std::endl;
      }
      testers.push_back(new TeamFcollectTester<int64_t>(args));
      testers.push_back(new TeamFcollectTester<int>(args));
      testers.push_back(new TeamFcollectTester<long long>(args));
      testers.push_back(new TeamFcollectTester<float>(args));
      testers.push_back(new TeamFcollectTester<double>(args));
      testers.push_back(new TeamFcollectTester<char>(args));
      testers.push_back(new TeamFcollectTester<unsigned char>(args));
      return testers;
    case AMO_FAddTestType:
      if (rank == 0) std::cout << "AMO Fetch_Add ###" << std::endl;
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      testers.push_back(new AMOStandardTester<int>(args));
      return testers;
    case AMO_FIncTestType:
      if (rank == 0) std::cout << "AMO Fetch_Inc ###" << std::endl;
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      testers.push_back(new AMOStandardTester<int>(args));
      return testers;
    case AMO_FetchTestType:
      if (rank == 0) std::cout << "AMO Fetch ###" << std::endl;
      testers.push_back(new AMOExtendedTester<long long>(args));
      testers.push_back(new AMOExtendedTester<long>(args));
      testers.push_back(new AMOExtendedTester<int>(args));
      return testers;
    case AMO_FCswapTestType:
      if (rank == 0) std::cout << "AMO Fetch_CSWAP ###" << std::endl;
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      testers.push_back(new AMOStandardTester<int>(args));
      return testers;
    case AMO_AddTestType:
      if (rank == 0) std::cout << "AMO Add ###" << std::endl;
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      testers.push_back(new AMOStandardTester<int>(args));
      return testers;
    case AMO_SetTestType:
      if (rank == 0) std::cout << "AMO Set ###" << std::endl;
      testers.push_back(new AMOExtendedTester<long long>(args));
      testers.push_back(new AMOExtendedTester<long>(args));
      testers.push_back(new AMOExtendedTester<int>(args));
      return testers;
    case AMO_SwapTestType:
      if (rank == 0) std::cout << "AMO Swap ###" << std::endl;
      testers.push_back(new AMOExtendedTester<long long>(args));
      testers.push_back(new AMOExtendedTester<long>(args));
      testers.push_back(new AMOExtendedTester<int>(args));
      return testers;
    case AMO_FetchAndTestType:
      if (rank == 0) std::cout << "AMO Fetch And ###" << std::endl;
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      return testers;
    case AMO_AndTestType:
      if (rank == 0) std::cout << "AMO And ###" << std::endl;
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      return testers;
    case AMO_FetchOrTestType:
      if (rank == 0) std::cout << "AMO Fetch Or ###" << std::endl;
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      return testers;
    case AMO_OrTestType:
      if (rank == 0) std::cout << "AMO Or ###" << std::endl;
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      return testers;
    case AMO_FetchXorTestType:
      if (rank == 0) std::cout << "AMO Fetch Xor ###" << std::endl;
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      return testers;
    case AMO_XorTestType:
      if (rank == 0) std::cout << "AMO Xor ###" << std::endl;
      testers.push_back(new AMOBitwiseTester<unsigned long long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned long>(args));
      testers.push_back(new AMOBitwiseTester<unsigned int>(args));
      return testers;
    case AMO_IncTestType:
      if (rank == 0) std::cout << "AMO Inc ###" << std::endl;
      testers.push_back(new AMOStandardTester<long long>(args));
      testers.push_back(new AMOStandardTester<long>(args));
      testers.push_back(new AMOStandardTester<int>(args));
      return testers;
    case PingPongTestType:
      if (rank == 0) std::cout << "PingPong ###" << std::endl;
      testers.push_back(new PingPongTester(args));
      return testers;
    case PingAllTestType:
      if (rank == 0) std::cout << "PingAll ###" << std::endl;
      testers.push_back(new PingAllTester(args));
      return testers;
    case BarrierAllTestType:
      if (rank == 0) std::cout << "Barrier_All ###" << std::endl;
      testers.push_back(new BarrierAllTester(args));
      return testers;
    case WAVEBarrierAllTestType:
      if (rank == 0) std::cout << "WAVE Barrier_All ###" << std::endl;
      testers.push_back(new BarrierAllTester(args));
      return testers;
    case WGBarrierAllTestType:
      if (rank == 0) std::cout << "WG Barrier_All ###" << std::endl;
      testers.push_back(new BarrierAllTester(args));
      return testers;
    case TeamBarrierTestType:
      if (rank == 0) std::cout << "Team Barrier Test ###" << std::endl;
      testers.push_back(new TeamBarrierTester(args));
      return testers;
    case TeamWAVEBarrierTestType:
      if (rank == 0) std::cout << "Team WAVE Barrier Test ###" << std::endl;
      testers.push_back(new TeamBarrierTester(args));
      return testers;
    case TeamWGBarrierTestType:
      if (rank == 0) std::cout << "Team WG Barrier Test ###" << std::endl;
      testers.push_back(new TeamBarrierTester(args));
      return testers;
    case SyncAllTestType:
      if (rank == 0) std::cout << "SyncAll ###" << std::endl;
      testers.push_back(new SyncTester(args));
      return testers;
    case WAVESyncAllTestType:
      if (rank == 0) std::cout << "WAVE SyncAll ###" << std::endl;
      testers.push_back(new SyncTester(args));
      return testers;
    case WGSyncAllTestType:
      if (rank == 0) std::cout << "WG SyncAll ###" << std::endl;
      testers.push_back(new SyncTester(args));
      return testers;
    case SyncTestType:
      if (rank == 0) std::cout << "Sync ###" << std::endl;
      testers.push_back(new SyncTester(args));
      return testers;
    case WAVESyncTestType:
      if (rank == 0) std::cout << "WAVE Sync ###" << std::endl;
      testers.push_back(new SyncTester(args));
      return testers;
    case WGSyncTestType:
      if (rank == 0) std::cout << "WG Sync ###" << std::endl;
      testers.push_back(new SyncTester(args));
      return testers;
    case RandomAccessTestType:
      if (rank == 0) std::cout << "Random_Access ###" << std::endl;
      testers.push_back(new RandomAccessTester(args));
      return testers;
    case ShmemPtrTestType:
      if (rank == 0) std::cout << "Shmem_Ptr ###" << std::endl;
      testers.push_back(new ShmemPtrTester(args));
      return testers;
    case WGGetTestType:
      if (rank == 0)
        std::cout << "Blocking WG level Gets ###" << std::endl;
      testers.push_back(new WorkGroupPrimitiveTester(args));
      return testers;
    case WGGetNBITestType:
      if (rank == 0)
        std::cout << "Non-Blocking WG level Gets ###" << std::endl;
      testers.push_back(new WorkGroupPrimitiveTester(args));
      return testers;
    case WGPutTestType:
      if (rank == 0)
        std::cout << "Blocking WG level Puts ###" << std::endl;
      testers.push_back(new WorkGroupPrimitiveTester(args));
      return testers;
    case WGPutNBITestType:
      if (rank == 0)
        std::cout << "Non-Blocking WG level Puts ###" << std::endl;
      testers.push_back(new WorkGroupPrimitiveTester(args));
      return testers;
    case PutNBIMRTestType:
      if (rank == 0)
        std::cout << "Non-Blocking Put message rate ###" << std::endl;
      testers.push_back(new PrimitiveMRTester(args));
      return testers;
    case WAVEGetTestType:
      if (rank == 0)
        std::cout << "Blocking WAVE level Gets ###" << std::endl;
      testers.push_back(new WaveFrontPrimitiveTester(args));
      return testers;
    case WAVEGetNBITestType:
      if (rank == 0)
        std::cout << "Non-Blocking WAVE level Gets ###" << std::endl;
      testers.push_back(new WaveFrontPrimitiveTester(args));
      return testers;
    case WAVEPutTestType:
      if (rank == 0)
        std::cout << "Blocking WAVE level Puts ###" << std::endl;
      testers.push_back(new WaveFrontPrimitiveTester(args));
      return testers;
    case WAVEPutNBITestType:
      if (rank == 0)
        std::cout << "Non-Blocking WAVE level Puts ###" << std::endl;
      testers.push_back(new WaveFrontPrimitiveTester(args));
      return testers;
    case PutSignalTestType:
      if (rank == 0) std::cout << "Putmem Signal ###" << std::endl;
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      return testers;
    case WGPutSignalTestType:
      if (rank == 0) std::cout << "WG Putmem Signal ###" << std::endl;
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      return testers;
    case WAVEPutSignalTestType:
      if (rank == 0) std::cout << "Wave Putmem Signal ###" << std::endl;
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      return testers;
    case PutSignalNBITestType:
      if (rank == 0) std::cout << "Non-Blocking Putmem Signal ###" << std::endl;
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      return testers;
    case WGPutSignalNBITestType:
      if (rank == 0) std::cout << "Non-Blocking WG Putmem Signal ###" << std::endl;
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      return testers;
    case WAVEPutSignalNBITestType:
      if (rank == 0) std::cout << "Non-Blocking Wave Putmem Signal ###" << std::endl;
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_SET));
      testers.push_back(new SignalingOperationsTester(args, ROCSHMEM_SIGNAL_ADD));
      return testers;
    case SignalFetchTestType:
      if (rank == 0) std::cout << "Signal Fetch ###" << std::endl;
      testers.push_back(new SignalingOperationsTester(args));
      return testers;
    case WGSignalFetchTestType:
      if (rank == 0) std::cout << "WG Signal Fetch ###" << std::endl;
      testers.push_back(new SignalingOperationsTester(args));
      return testers;
    case WAVESignalFetchTestType:
      if (rank == 0) std::cout << "Wave Signal Fetch ###" << std::endl;
      testers.push_back(new SignalingOperationsTester(args));
      return testers;
    default:
      if (rank == 0) std::cout << "Empty Test ###" << std::endl;
      return testers;
  }
  return testers;
}

void Tester::execute() {
  if (_type == InitTestType) return;

  int num_loops = args.loop;

  /**
   * Some tests loop through data sizes in powers of 2 and report the
   * results for those ranges.
   */
  for (uint64_t size = args.min_msg_size; size <= args.max_msg_size;
       size <<= 1) {
    resetBuffers(size);

    /**
     * Restricts the number of iterations of really large messages.
     */
    if (size > args.large_message_size) num_loops = args.loop_large;

    barrier();

    preLaunchKernel();

    /**
     * This conditional launches the HIP kernel.
     *
     * Some tests may only launch a single kernel. These kernels will
     * be kicked off by the initiator (denoted by the args.myid check).
     *
     * Other tests will initiate of both sides and launch from both
     * rocshmem pes.
     */
    if (peLaunchesKernel()) {
      memset(timer, 0, sizeof(uint64_t) * args.num_wgs);

      const dim3 blockSize(args.wg_size, 1, 1);
      const dim3 gridSize(args.num_wgs, 1, 1);

      CHECK_HIP(hipEventRecord(start_event, stream));

      launchKernel(gridSize, blockSize, num_loops, size);

      CHECK_HIP(hipEventRecord(stop_event, stream));

      hipError_t err = hipStreamSynchronize(stream);
      if (err != hipSuccess) {
        printf("error = %d \n", err);
      }
    }

    barrier();

    postLaunchKernel();

    // data validation
    verifyResults(size);

    barrier();

    if (_type != TeamCtxInfraTestType) {
      print(size);
    }
  }
}

bool Tester::peLaunchesKernel() {
  bool is_launcher;

  /**
   * The PE assigned 0 is always active in these tests.
   */
  is_launcher = args.myid == 0;

  /**
   * Some test types are active on both sides.
   */
  is_launcher = is_launcher || (_type == TeamReductionTestType) ||
                (_type == TeamBroadcastTestType) || (_type == TeamCtxInfraTestType) ||
                (_type == TeamAllToAllTestType) || (_type == TeamFCollectTestType) ||
                (_type == PingPongTestType) || (_type == BarrierAllTestType) ||
                (_type == WAVEBarrierAllTestType) || (_type == WGBarrierAllTestType) ||
                (_type == SyncTestType) || (_type == WAVESyncTestType) ||
                (_type == WGSyncTestType) || (_type == SyncAllTestType) ||
                (_type == WAVESyncAllTestType) || (_type == WGSyncAllTestType) ||
                (_type == RandomAccessTestType) || (_type == PingAllTestType) ||
                (_type == TeamBarrierTestType) || (_type == TeamWAVEBarrierTestType) ||
                (_type == TeamWGBarrierTestType);

  return is_launcher;
}

void Tester::print(uint64_t size) {
  if (args.myid != 0) {
    return;
  }

  /**
   * Calculate total amount of data transfered
   */
  uint64_t total_size = size * num_timed_msgs;
  double timer_avg = timerAvgInMicroseconds();

  double time_us = gpuCyclesToMicroseconds(max_end_time - min_start_time);
  double time_s = time_us / 1e6;

  double latency_avg = time_us / num_timed_msgs;

  double avg_msg_rate = num_timed_msgs / time_s;

  double bandwidth_avg_gbs =
      static_cast<double>(total_size * bw_factor) / time_s / pow(2, 30);

  float total_kern_time_ms;
  CHECK_HIP(hipEventElapsedTime(&total_kern_time_ms, start_event, stop_event));
  float total_kern_time_s = total_kern_time_ms / 1000;

  int field_width = 20;
  int float_precision = 2;

  if (_print_header) {
    printf("%-*s%-*s%*s%*s%*s",
           15, "# Size (B)",
           15, "# of timed Msgs",
           field_width, "Latency (us)",
           field_width, "Bandwidth (GB/s)",
           field_width + 1, "Msg Rate (Msg/s)\n");
    _print_header = 0;
  }

  printf("%-*lu%-*d%*.*f%*.*f%*.*f\n",
         15, size,
         15, num_timed_msgs,
         field_width, float_precision, latency_avg,
         field_width, float_precision, bandwidth_avg_gbs,
         field_width, float_precision, avg_msg_rate);

  fflush(stdout);
}

void flush_hdp() {
  int hip_dev_id{};
  unsigned int* hdp_flush_ptr_{nullptr};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  CHECK_HIP(hipDeviceGetAttribute(reinterpret_cast<int*>(&hdp_flush_ptr_),
                        hipDeviceAttributeHdpMemFlushCntl, hip_dev_id));
  __atomic_store_n(hdp_flush_ptr_, 0x1, __ATOMIC_SEQ_CST);
}

void Tester::barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
  flush_hdp();
}

double Tester::gpuCyclesToMicroseconds(long long int cycles) {
  return static_cast<double>(cycles) /
         (static_cast<double>(wall_clk_rate) * 1e-3);
}

double Tester::timerAvgInMicroseconds() {
  double sum = 0;
  min_start_time = LLONG_MAX;
  max_end_time = 0;

  for (uint32_t i = 0; i < num_timers; i++) {
    timer[i] = end_time[i] - start_time[i];
    sum += gpuCyclesToMicroseconds(timer[i]);
    min_start_time = (start_time[i] < min_start_time)
                     ? start_time[i]
                     : min_start_time;
    max_end_time = (end_time[i] > max_end_time)
                     ? end_time[i]
                     : max_end_time;
  }

  return sum / num_timers;
}
