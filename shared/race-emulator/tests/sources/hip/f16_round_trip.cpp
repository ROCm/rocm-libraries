// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compile for a target architecture with:
//   hipcc --cuda-device-only -S --offload-arch=<arch> -O2 f16_round_trip.cpp
//   -o ../<arch>/f16_round_trip.s
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Round-trip f32 -> f16 -> f32. Host provides and reads f32 data.
// Exercises v_cvt_f16_f32 then v_cvt_f32_f16 (opposite order to
// simple_adder_f16).
__global__ void f16_round_trip(float *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] = __half2float(__float2half(data[idx]));
  }
}
