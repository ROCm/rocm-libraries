// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Microbenchmarks for the issue-cost anchors that probe_cycle_budget.py needs:
//   1. matrix issue rate    -> cycles per WMMA per SIMD, and the TFLOPS ceiling
//   2. VALU issue rate      -> cycles per v_fmac_f32 per SIMD
//   3. transcendental rate  -> cycles per v_exp_f32 per SIMD
//
// Each kernel runs NACC independent accumulator chains so the measurement is
// issue-limited rather than dependency-limited, and each instruction class runs
// in its own dispatch so the driver can attribute counters per class.
//
// Cycles come from rocprofv3 GRBM_GUI_ACTIVE (see probe_roofline_peaks.py) rather
// than from wall time: parts that power-throttle swing 20-40% within seconds,
// which is larger than most of the effects worth measuring. Where s_memtime is
// unavailable (gfx11+), the counter is the only cycle source.
//
// The WMMA asm below is the gfx11/gfx12 wave32 form. On other families replace
// it with that family's matrix mnemonic (e.g. v_mfma_* on CDNA) and update
// FLOP_PER_MMA to match the instruction's shape.
//
// Build:
//   hipcc --offload-arch=gfx1151 -O3 -o probe_roofline_peaks probe_roofline_peaks.hip
//   probe_roofline_peaks <mma|valu|exp> <iters> <blocks> <threads>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

using v8f = float __attribute__((ext_vector_type(8)));
using v16h = _Float16 __attribute__((ext_vector_type(16)));

#define NACC 8

// FLOP per instruction: WMMA 16x16x16 = 2*16*16*16, VALU fma = 2 per lane
// (wave32), transcendental = 1 per lane.
#define FLOP_PER_MMA 8192.0
#define FLOP_PER_VALU 64.0
#define FLOP_PER_EXP 32.0

__global__ __launch_bounds__(256) void mma_peak(float *out, int iters) {
  v16h a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = (_Float16)(threadIdx.x * 0.01f + i);
    b[i] = (_Float16)(threadIdx.x * 0.02f + i);
  }
  v8f c[NACC];
  for (int k = 0; k < NACC; ++k)
    for (int i = 0; i < 8; ++i) c[k][i] = 0.0f;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < NACC; ++k)
      asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0"
                   : "+v"(c[k])
                   : "v"(a), "v"(b));
  }

  v8f s = c[0];
  for (int k = 1; k < NACC; ++k) s += c[k];
  float r = 0;
  for (int i = 0; i < 8; ++i) r += s[i];
  if (threadIdx.x == 0) out[blockIdx.x] = r;
}

__global__ __launch_bounds__(256) void valu_peak(float *out, int iters) {
  float c[NACC];
  for (int k = 0; k < NACC; ++k) c[k] = threadIdx.x + k;
  float m = 1.0001f, n = 0.5f;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < NACC; ++k)
      asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(c[k]) : "v"(m), "v"(n));
  }

  float r = 0;
  for (int k = 0; k < NACC; ++k) r += c[k];
  if (threadIdx.x == 0) out[blockIdx.x] = r;
}

__global__ __launch_bounds__(256) void exp_peak(float *out, int iters) {
  float c[NACC];
  for (int k = 0; k < NACC; ++k) c[k] = threadIdx.x * 1e-3f + k;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < NACC; ++k) asm volatile("v_exp_f32 %0, %0" : "+v"(c[k]));
  }

  float r = 0;
  for (int k = 0; k < NACC; ++k) r += c[k];
  if (threadIdx.x == 0) out[blockIdx.x] = r;
}

int main(int argc, char **argv) {
  const char *which = argc > 1 ? argv[1] : "mma";
  int iters = argc > 2 ? atoi(argv[2]) : 20000;
  int blocks = argc > 3 ? atoi(argv[3]) : 80;
  int thr = argc > 4 ? atoi(argv[4]) : 32;

  float *d_out;
  hipMalloc(&d_out, blocks * sizeof(float));
  auto launch = [&](int it) {
    if (!strcmp(which, "mma"))
      mma_peak<<<blocks, thr>>>(d_out, it);
    else if (!strcmp(which, "valu"))
      valu_peak<<<blocks, thr>>>(d_out, it);
    else
      exp_peak<<<blocks, thr>>>(d_out, it);
  };
  launch(64);  // warm up so module load is not inside the timed dispatch
  hipDeviceSynchronize();

  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  hipEventRecord(e0);
  launch(iters);
  hipEventRecord(e1);
  hipDeviceSynchronize();
  float ms = 0;
  hipEventElapsedTime(&ms, e0, e1);

  double flop_per_op = !strcmp(which, "mma")    ? FLOP_PER_MMA
                       : !strcmp(which, "valu") ? FLOP_PER_VALU
                                                : FLOP_PER_EXP;
  double waves = (double)blocks * (thr / 32);
  double ops = (double)iters * NACC;  // per wave
  double total_ops = ops * waves;
  printf(
      "%s blocks=%d thr=%d iters=%d  ops/wave=%.0f  total_ops=%.4g  "
      "wall=%.3f ms  %.2f TFLOP/s\n",
      which, blocks, thr, iters, ops, total_ops, ms, ms > 0 ? total_ops * flop_per_op / (ms * 1e-3) / 1e12 : 0.0);
  hipFree(d_out);
  return 0;
}
