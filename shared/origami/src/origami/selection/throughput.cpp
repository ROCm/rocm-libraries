// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstddef>

#include "origami/core/problem.hpp"
#include "origami/hardware.hpp"
#include "origami/selection/throughput.hpp"

namespace origami {

double compute_perf_gflops(const hardware_t& hardware,
                           const problem_t& problem,
                           const double latency) {
  // Extract parameters from structured types
  size_t M = problem.size.m;
  size_t N = problem.size.n;
  size_t K = problem.size.k;

  // Compute total FLOPs
  double total_FLOPs = 2.0 * M * N * K;  // For GEMM, each multiply-add is 2 FLOPs
  // Compute total time in seconds
  double cycles_per_second = hardware.compute_clock_ghz * 1e9;  // 1 GHz = 1e9 cycles per second

  double total_time_seconds = latency / cycles_per_second;

  // Compute performance in FLOPS
  double FLOPS = total_FLOPs / total_time_seconds;
  // Convert to GFLOPS
  double GFLOPS = FLOPS / 1e9;  // 1 GFLOP = 1e9 FLOPs
  return GFLOPS;
}

}  // namespace origami
