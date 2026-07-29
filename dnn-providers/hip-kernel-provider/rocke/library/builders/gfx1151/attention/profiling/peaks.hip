// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Roofline anchors for gfx1151, measured rather than assumed:
//   1. WMMA f16 issue peak  -> FLOP/clk/CU and the implied TFLOPS ceiling
//   2. WMMA cost in cycles  -> cycles per v_wmma per SIMD
//   3. VALU peak            -> instructions/clk/CU (v_fma_f32 wave32)
//   4. transcendental peak  -> v_exp_f32 rate (the softmax inner op)
//
// Each kernel runs NACC independent chains so the measurement is issue-limited,
// not dependency-limited. gfx1151 has no s_memtime, so cycles come from
// rocprofv3 GRBM_GUI_ACTIVE (see peaks.py) -- wall time alone is unusable here
// because the part power-throttles. Kernel names are printed so the driver can
// match dispatches. Pass a kernel selector so each runs in its own dispatch.
//
// Build:
//   hipcc --offload-arch=gfx1151 -O3 -o peaks peaks.hip
//   peaks <wmma|valu|exp> <iters> <blocks> <threads>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>

using v8f = float __attribute__((ext_vector_type(8)));
using v16h = _Float16 __attribute__((ext_vector_type(16)));

#define NACC 8

// ---- 1/2: WMMA issue rate ---------------------------------------------------
__global__ __launch_bounds__(256) void wmma_peak(float *out, int iters) {
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

// ---- 3: VALU issue rate -----------------------------------------------------
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

// ---- 4: transcendental rate (softmax exp) -----------------------------------
__global__ __launch_bounds__(256) void exp_peak(float *out, int iters) {
  float c[NACC];
  for (int k = 0; k < NACC; ++k) c[k] = threadIdx.x * 1e-3f + k;

  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < NACC; ++k)
      asm volatile("v_exp_f32 %0, %0" : "+v"(c[k]));
  }

  float r = 0;
  for (int k = 0; k < NACC; ++k) r += c[k];
  if (threadIdx.x == 0) out[blockIdx.x] = r;
}

int main(int argc, char **argv) {
  const char *which = argc > 1 ? argv[1] : "wmma";
  int iters  = argc > 2 ? atoi(argv[2]) : 20000;
  int blocks = argc > 3 ? atoi(argv[3]) : 80;
  int thr    = argc > 4 ? atoi(argv[4]) : 32;

  float *d_out;
  hipMalloc(&d_out, blocks * sizeof(float));
  auto launch = [&](int it) {
    if (!strcmp(which, "wmma")) wmma_peak<<<blocks, thr>>>(d_out, it);
    else if (!strcmp(which, "valu")) valu_peak<<<blocks, thr>>>(d_out, it);
    else exp_peak<<<blocks, thr>>>(d_out, it);
  };
  launch(64);
  hipDeviceSynchronize();

  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0);
  launch(iters);
  hipEventRecord(e1);
  hipDeviceSynchronize();
  float ms = 0;
  hipEventElapsedTime(&ms, e0, e1);

  double flop_per_op = !strcmp(which, "wmma") ? 8192.0
                     : !strcmp(which, "valu") ? 64.0 : 32.0;
  double waves = (double)blocks * (thr / 32);
  double ops = (double)iters * NACC;          // per wave
  double total_ops = ops * waves;
  printf("%s blocks=%d thr=%d iters=%d  ops/wave=%.0f  total_ops=%.4g  "
         "wall=%.3f ms  %.2f TFLOP/s\n",
         which, blocks, thr, iters, ops, total_ops, ms,
         total_ops * flop_per_op / (ms * 1e-3) / 1e12);
  hipFree(d_out);
  return 0;
}
