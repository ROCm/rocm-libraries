// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Standalone C++ sanity benchmark for the warp-decode gate/up + down/reduce
// kernels. Runs both the FP8 activation variant
// (XScaleLayout = Block2D<1,128>) and the BF16 activation variant
// (XScaleLayout = PerTensor, scale = 1.0) for two realistic decode shapes,
// shared with the Python extension:
//
//   DeepSeek-V3-like: B=1..64, HIDDEN=7168, INTER=2048, TOPK=8, E=256
//   MiniMax-like    : B=1..64, HIDDEN=3072, INTER=1536, TOPK=8, E=256
//   Qwen3Next-like  : B=1..64, HIDDEN=2048, INTER=512,  TOPK=10, E=512
//
// Prints ms / TFLOP/s / GB/s per kernel per shape per batch.
//
// This binary does no token routing, sorting, or quantization; it just times
// the two CK kernels so we can cross-check the numbers the Python harness
// reports for the same input sizes.

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/warp_decode.hpp"
#include "ck_tile/ops/warp_decode/kernel/persistent_jtile_kernels_hip.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ck_tile;

namespace {

constexpr index_t kVectorDefault = 8;
constexpr index_t kVectorFP8     = 16;
constexpr index_t kVectorFP4     = 8; // packed FP4 fast path needs kVector % 8 == 0
constexpr index_t kWaveSize      = 64;
constexpr index_t kBlock_N = 128;
constexpr index_t kBlock_K = 128;
constexpr index_t kBlockXK = 128;

using XScaleLayoutFP8  = WarpDecodeScaleLayout::Block2D<1, kBlockXK>;
using XScaleLayoutBF16 = WarpDecodeScaleLayout::PerTensor;
using WScaleLayoutAll  = WarpDecodeScaleLayout::Block2D<kBlock_N, kBlock_K>;

using GateUpProblemFP8 = WarpDecodeGateUpProblem<fp8_t,
                                                 fp8_t,
                                                 float,
                                                 bf16_t,
                                                 float,
                                                 float,
                                                 XScaleLayoutFP8,
                                                 WScaleLayoutAll,
                                                 element_wise::Silu,
                                                 kVectorFP8>;
using GateUpKernelFP8 = WarpDecodeGateUpKernel<GateUpProblemFP8, WarpDecodePolicy>;
using GateUpProblemFP8Lds = WarpDecodeGateUpProblem<fp8_t,
                                                    fp8_t,
                                                    float,
                                                    bf16_t,
                                                    float,
                                                    float,
                                                    XScaleLayoutFP8,
                                                    WScaleLayoutAll,
                                                    element_wise::Silu,
                                                    kVectorFP8,
                                                    false,
                                                    false,
                                                    4>;
using GateUpLdsXKernelFP8 = WarpDecodeGateUpLdsXKernel<GateUpProblemFP8Lds, WarpDecodePolicy>;
using GateUpProblemFP8LdsN2 = WarpDecodeGateUpProblem<fp8_t,
                                                      fp8_t,
                                                      float,
                                                      bf16_t,
                                                      float,
                                                      float,
                                                      XScaleLayoutFP8,
                                                      WScaleLayoutAll,
                                                      element_wise::Silu,
                                                      kVectorFP8,
                                                      true,
                                                      false,
                                                      4,
                                                      2>;
using GateUpLdsXKernelFP8N2 = WarpDecodeGateUpLdsXKernel<GateUpProblemFP8LdsN2, WarpDecodePolicy>;
using GateUpProblemFP8Dot2 = WarpDecodeGateUpProblem<fp8_t,
                                                     fp8_t,
                                                     float,
                                                     bf16_t,
                                                     float,
                                                     float,
                                                     XScaleLayoutFP8,
                                                     WScaleLayoutAll,
                                                     element_wise::Silu,
                                                     kVectorFP8,
                                                     true>;
using GateUpKernelFP8Dot2 = WarpDecodeGateUpKernel<GateUpProblemFP8Dot2, WarpDecodePolicy>;
using GateUpProblemFP8Dot2N2 = WarpDecodeGateUpProblem<fp8_t,
                                                       fp8_t,
                                                       float,
                                                       bf16_t,
                                                       float,
                                                       float,
                                                       XScaleLayoutFP8,
                                                       WScaleLayoutAll,
                                                       element_wise::Silu,
                                                       kVectorFP8,
                                                       true,
                                                       false,
                                                       1,
                                                       2>;
using GateUpKernelFP8Dot2N2 = WarpDecodeGateUpKernel<GateUpProblemFP8Dot2N2, WarpDecodePolicy>;
using GateUpProblemFP8PkF32 = WarpDecodeGateUpProblem<fp8_t,
                                                      fp8_t,
                                                      float,
                                                      bf16_t,
                                                      float,
                                                      float,
                                                      XScaleLayoutFP8,
                                                      WScaleLayoutAll,
                                                      element_wise::Silu,
                                                      kVectorFP8,
                                                      false,
                                                      true>;
using GateUpKernelFP8PkF32 = WarpDecodeGateUpKernel<GateUpProblemFP8PkF32, WarpDecodePolicy>;

using GateUpProblemBF16 = WarpDecodeGateUpProblem<bf16_t,
                                                  fp8_t,
                                                  float,
                                                  bf16_t,
                                                  float,
                                                  float,
                                                  XScaleLayoutBF16,
                                                  WScaleLayoutAll,
                                                  element_wise::Silu,
                                                  kVectorDefault>;
using GateUpKernelBF16 = WarpDecodeGateUpKernel<GateUpProblemBF16, WarpDecodePolicy>;
using GateUpProblemBF16Lds = WarpDecodeGateUpProblem<bf16_t,
                                                     fp8_t,
                                                     float,
                                                     bf16_t,
                                                     float,
                                                     float,
                                                     XScaleLayoutBF16,
                                                     WScaleLayoutAll,
                                                     element_wise::Silu,
                                                     kVectorDefault,
                                                     false,
                                                     false,
                                                     4>;
using GateUpLdsXKernelBF16 = WarpDecodeGateUpLdsXKernel<GateUpProblemBF16Lds, WarpDecodePolicy>;
using GateUpProblemBF16LdsN2 = WarpDecodeGateUpProblem<bf16_t,
                                                       fp8_t,
                                                       float,
                                                       bf16_t,
                                                       float,
                                                       float,
                                                       XScaleLayoutBF16,
                                                       WScaleLayoutAll,
                                                       element_wise::Silu,
                                                       kVectorDefault,
                                                       true,
                                                       false,
                                                       4,
                                                       2>;
using GateUpLdsXKernelBF16N2 = WarpDecodeGateUpLdsXKernel<GateUpProblemBF16LdsN2, WarpDecodePolicy>;
using GateUpProblemBF16Dot2 = WarpDecodeGateUpProblem<bf16_t,
                                                      fp8_t,
                                                      float,
                                                      bf16_t,
                                                      float,
                                                      float,
                                                      XScaleLayoutBF16,
                                                      WScaleLayoutAll,
                                                      element_wise::Silu,
                                                      kVectorDefault,
                                                      true>;
using GateUpKernelBF16Dot2 = WarpDecodeGateUpKernel<GateUpProblemBF16Dot2, WarpDecodePolicy>;
using GateUpProblemBF16Dot2N2 = WarpDecodeGateUpProblem<bf16_t,
                                                        fp8_t,
                                                        float,
                                                        bf16_t,
                                                        float,
                                                        float,
                                                        XScaleLayoutBF16,
                                                        WScaleLayoutAll,
                                                        element_wise::Silu,
                                                        kVectorDefault,
                                                        true,
                                                        false,
                                                        1,
                                                        2>;
using GateUpKernelBF16Dot2N2 = WarpDecodeGateUpKernel<GateUpProblemBF16Dot2N2, WarpDecodePolicy>;
using GateUpProblemBF16PkF32 = WarpDecodeGateUpProblem<bf16_t,
                                                       fp8_t,
                                                       float,
                                                       bf16_t,
                                                       float,
                                                       float,
                                                       XScaleLayoutBF16,
                                                       WScaleLayoutAll,
                                                       element_wise::Silu,
                                                       kVectorDefault,
                                                       false,
                                                       true>;
using GateUpKernelBF16PkF32 = WarpDecodeGateUpKernel<GateUpProblemBF16PkF32, WarpDecodePolicy>;

using DownProblemDefault = WarpDecodeDownReduceProblem<bf16_t,
                                                       fp8_t,
                                                       float,
                                                       bf16_t,
                                                       float,
                                                       WScaleLayoutAll,
                                                       kVectorDefault>;
using DownKernelDefault = WarpDecodeDownReduceKernel<DownProblemDefault, WarpDecodePolicy>;
using DownProblemDefaultLds = WarpDecodeDownReduceProblem<bf16_t,
                                                          fp8_t,
                                                          float,
                                                          bf16_t,
                                                          float,
                                                          WScaleLayoutAll,
                                                          kVectorDefault,
                                                          true,
                                                          false,
                                                          4>;
using DownLdsInterKernelDefault =
    WarpDecodeDownReduceLdsInterKernel<DownProblemDefaultLds, WarpDecodePolicy>;
using DownProblemDefaultDot2 = WarpDecodeDownReduceProblem<bf16_t,
                                                           fp8_t,
                                                           float,
                                                           bf16_t,
                                                           float,
                                                           WScaleLayoutAll,
                                                           kVectorDefault,
                                                           true>;
using DownKernelDefaultDot2 = WarpDecodeDownReduceKernel<DownProblemDefaultDot2, WarpDecodePolicy>;
using DownProblemDefaultDot2H2 = WarpDecodeDownReduceProblem<bf16_t,
                                                             fp8_t,
                                                             float,
                                                             bf16_t,
                                                             float,
                                                             WScaleLayoutAll,
                                                             kVectorDefault,
                                                             true,
                                                             false,
                                                             1,
                                                             2>;
using DownKernelDefaultDot2H2 = WarpDecodeDownReduceKernel<DownProblemDefaultDot2H2, WarpDecodePolicy>;
using DownProblemDefaultPkF32 = WarpDecodeDownReduceProblem<bf16_t,
                                                            fp8_t,
                                                            float,
                                                            bf16_t,
                                                            float,
                                                            WScaleLayoutAll,
                                                            kVectorDefault,
                                                            false,
                                                            true>;
using DownKernelDefaultPkF32 = WarpDecodeDownReduceKernel<DownProblemDefaultPkF32, WarpDecodePolicy>;

using DownProblemFP8 = WarpDecodeDownReduceProblem<bf16_t,
                                                  fp8_t,
                                                  float,
                                                  bf16_t,
                                                  float,
                                                  WScaleLayoutAll,
                                                  kVectorFP8>;
using DownKernelFP8 = WarpDecodeDownReduceKernel<DownProblemFP8, WarpDecodePolicy>;
using DownProblemFP8Lds = WarpDecodeDownReduceProblem<bf16_t,
                                                      fp8_t,
                                                      float,
                                                      bf16_t,
                                                      float,
                                                      WScaleLayoutAll,
                                                      kVectorFP8,
                                                      true,
                                                      false,
                                                      4>;
using DownLdsInterKernelFP8 = WarpDecodeDownReduceLdsInterKernel<DownProblemFP8Lds, WarpDecodePolicy>;
using DownProblemFP8Dot2 = WarpDecodeDownReduceProblem<bf16_t,
                                                       fp8_t,
                                                       float,
                                                       bf16_t,
                                                       float,
                                                       WScaleLayoutAll,
                                                       kVectorFP8,
                                                       true>;
using DownKernelFP8Dot2 = WarpDecodeDownReduceKernel<DownProblemFP8Dot2, WarpDecodePolicy>;
using DownProblemFP8Dot2H2 = WarpDecodeDownReduceProblem<bf16_t,
                                                         fp8_t,
                                                         float,
                                                         bf16_t,
                                                         float,
                                                         WScaleLayoutAll,
                                                         kVectorFP8,
                                                         true,
                                                         false,
                                                         1,
                                                         2>;
using DownKernelFP8Dot2H2 = WarpDecodeDownReduceKernel<DownProblemFP8Dot2H2, WarpDecodePolicy>;
using DownProblemShort512Dot2 = WarpDecodeDownReduceProblem<bf16_t,
                                                            fp8_t,
                                                            float,
                                                            bf16_t,
                                                            float,
                                                            WScaleLayoutAll,
                                                            kVectorFP8,
                                                            true,
                                                            false,
                                                            1,
                                                            2,
                                                            32>;
using DownKernelShort512Dot2 = WarpDecodeDownReduceKernel<DownProblemShort512Dot2, WarpDecodePolicy>;
using DownProblemShort256Dot2 = WarpDecodeDownReduceProblem<bf16_t,
                                                            fp8_t,
                                                            float,
                                                            bf16_t,
                                                            float,
                                                            WScaleLayoutAll,
                                                            kVectorFP8,
                                                            true,
                                                            false,
                                                            1,
                                                            4,
                                                            16>;
using DownKernelShort256Dot2 = WarpDecodeDownReduceKernel<DownProblemShort256Dot2, WarpDecodePolicy>;
using DownProblemShort128Dot2 = WarpDecodeDownReduceProblem<bf16_t,
                                                            fp8_t,
                                                            float,
                                                            bf16_t,
                                                            float,
                                                            WScaleLayoutAll,
                                                            kVectorFP8,
                                                            true,
                                                            false,
                                                            1,
                                                            8,
                                                            8>;
using DownKernelShort128Dot2 = WarpDecodeDownReduceKernel<DownProblemShort128Dot2, WarpDecodePolicy>;
using DownProblemFP8PkF32 = WarpDecodeDownReduceProblem<bf16_t,
                                                        fp8_t,
                                                        float,
                                                        bf16_t,
                                                        float,
                                                        WScaleLayoutAll,
                                                        kVectorFP8,
                                                        false,
                                                        true>;
using DownKernelFP8PkF32 = WarpDecodeDownReduceKernel<DownProblemFP8PkF32, WarpDecodePolicy>;

// Packed MXFP4 (two FP4 per byte) weights: dot2 fast path with a per-tensor scale.
// FP4 halves the dominant weight read, so this is the primary BW win for `down`.
using DownProblemFP4 = WarpDecodeDownReduceProblem<bf16_t,
                                                   pk_fp4_t,
                                                   float,
                                                   bf16_t,
                                                   float,
                                                   WarpDecodeScaleLayout::PerTensor,
                                                   kVectorFP4,
                                                   true>;
using DownKernelFP4 = WarpDecodeDownReduceKernel<DownProblemFP4, WarpDecodePolicy>;

// Wide FP4: kVector=32 -> 16 packed bytes/lane = a single 128-bit weight load.
// Requires INTER % (64*32) == 0 (e.g. INTER=2048).
constexpr index_t kVectorFP4Wide = 32;
using DownProblemFP4Wide = WarpDecodeDownReduceProblem<bf16_t,
                                                       pk_fp4_t,
                                                       float,
                                                       bf16_t,
                                                       float,
                                                       WarpDecodeScaleLayout::PerTensor,
                                                       kVectorFP4Wide,
                                                       true>;
using DownKernelFP4Wide = WarpDecodeDownReduceKernel<DownProblemFP4Wide, WarpDecodePolicy>;

// FP4 H2: two hidden outputs per wave (FP4 analogue of down_h2_d2). Doubles the
// in-flight packed-FP4 weight loads per wave and reuses the BF16 activation, to
// close the 1-output path's HBM-rate gap (it only reaches ~half of HBM peak).
using DownProblemFP4H2 = WarpDecodeDownReduceProblem<bf16_t,
                                                     pk_fp4_t,
                                                     float,
                                                     bf16_t,
                                                     float,
                                                     WarpDecodeScaleLayout::PerTensor,
                                                     kVectorFP4,
                                                     true,
                                                     false,
                                                     1,
                                                     2>;
using DownKernelFP4H2 = WarpDecodeDownReduceKernel<DownProblemFP4H2, WarpDecodePolicy>;

struct Shape
{
    std::string name;
    index_t HIDDEN;
    index_t INTER;
    index_t TOPK;
    index_t E;
};

struct PerfResult
{
    double time_ms  = 0.0;
    double tflops   = 0.0;
    double gb_per_s = 0.0;
};

PerfResult make_perf(double time_ms, double flops, double bytes)
{
    PerfResult p;
    p.time_ms = time_ms;
    if(time_ms > 0.0)
    {
        p.tflops   = flops / (time_ms * 1.0e9);
        p.gb_per_s = bytes / (time_ms * 1.0e6);
    }
    return p;
}

template <typename T>
constexpr double element_bytes()
{
    return static_cast<double>(sizeof(T)) /
           static_cast<double>(numeric_traits<T>::PackedSize);
}

double gate_up_flops(index_t B, index_t HIDDEN, index_t INTER, index_t TOPK)
{
    constexpr double kActivationOps = 5.0;
    const double num_outputs =
        static_cast<double>(B) * static_cast<double>(TOPK) * static_cast<double>(INTER);
    return num_outputs * (4.0 * static_cast<double>(HIDDEN) + kActivationOps);
}

double gate_up_bytes(index_t B,
                     index_t HIDDEN,
                     index_t INTER,
                     index_t TOPK,
                     double x_elem_bytes,
                     double w_elem_bytes,
                     double inter_elem_bytes)
{
    const double num_outputs =
        static_cast<double>(B) * static_cast<double>(TOPK) * static_cast<double>(INTER);
    const double x_bytes       = num_outputs * static_cast<double>(HIDDEN) * x_elem_bytes;
    const double w_bytes       = num_outputs * static_cast<double>(HIDDEN) * w_elem_bytes * 2.0;
    const double router_bytes  = num_outputs * static_cast<double>(sizeof(int32_t));
    const double inter_bytes   = num_outputs * inter_elem_bytes;
    return x_bytes + w_bytes + router_bytes + inter_bytes;
}

double down_flops(index_t B, index_t HIDDEN, index_t INTER, index_t TOPK)
{
    const double num_outputs = static_cast<double>(B) * static_cast<double>(HIDDEN);
    const double work        = static_cast<double>(TOPK) * static_cast<double>(INTER) * 3.0;
    return num_outputs * work;
}

double down_bytes(index_t B,
                  index_t HIDDEN,
                  index_t INTER,
                  index_t TOPK,
                  double inter_elem_bytes,
                  double w_elem_bytes,
                  double y_elem_bytes)
{
    const double num_outputs = static_cast<double>(B) * static_cast<double>(HIDDEN);
    const double inter_b     = num_outputs * static_cast<double>(TOPK) *
                               static_cast<double>(INTER) * inter_elem_bytes;
    const double w_b = num_outputs * static_cast<double>(TOPK) *
                       static_cast<double>(INTER) * w_elem_bytes;
    const double rid_b = num_outputs * static_cast<double>(TOPK) *
                         static_cast<double>(sizeof(int32_t));
    const double rw_b = num_outputs * static_cast<double>(TOPK) *
                        static_cast<double>(sizeof(float));
    const double y_b = num_outputs * y_elem_bytes;
    return inter_b + w_b + rid_b + rw_b + y_b;
}

template <typename T>
void fill_random(HostTensor<T>& tensor, float lo, float hi, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for(index_t i = 0; i < static_cast<index_t>(tensor.get_element_space_size()); ++i)
    {
        tensor.mData[i] = type_convert<T>(dist(gen));
    }
}

// Packed FP4: mData is a flat byte array (two FP4 nibbles per byte). Fill raw
// random nibbles directly; get_element_space_size() already accounts for packing.
inline void fill_random_pk_fp4(HostTensor<pk_fp4_t>& tensor, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    for(index_t i = 0; i < static_cast<index_t>(tensor.get_element_space_size()); ++i)
    {
        const uint8_t lo = static_cast<uint8_t>(dist(gen));
        const uint8_t hi = static_cast<uint8_t>(dist(gen));
        tensor.mData[i]  = pk_fp4_t(static_cast<uint8_t>((hi << 4) | (lo & 0x0F)));
    }
}

template <typename T>
void fill_random_scale(std::vector<T>& buf, std::size_t count, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.5f, 2.0f);
    buf.resize(count);
    for(std::size_t i = 0; i < count; ++i)
    {
        buf[i] = type_convert<T>(dist(gen));
    }
}

void fill_router(HostTensor<int32_t>& router_ids,
                 HostTensor<float>& router_wts,
                 index_t B,
                 index_t TOPK,
                 index_t E,
                 unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> idist(0, E - 1);
    std::uniform_real_distribution<float>  wdist(0.1f, 1.0f);

    for(index_t b = 0; b < B; ++b)
    {
        float sum = 0.0f;
        for(index_t k = 0; k < TOPK; ++k)
        {
            router_ids(b, k) = idist(gen);
            const float w    = wdist(gen);
            router_wts(b, k) = w;
            sum += w;
        }
        for(index_t k = 0; k < TOPK; ++k)
        {
            router_wts(b, k) /= sum;
        }
    }
}

struct StageBufs
{
    // FP8 path host tensors
    HostTensor<fp8_t>   x_fp8;
    HostTensor<fp8_t>   w_gate_fp8;
    HostTensor<fp8_t>   w_up_fp8;
    HostTensor<fp8_t>   w_down_fp8;
    HostTensor<pk_fp4_t> w_down_fp4;   // packed MXFP4 down weights {E, HIDDEN, INTER} -> INTER/2 bytes/row
    HostTensor<bf16_t>  x_bf16;

    std::vector<float>  x_scale_fp8;       // Block2D<1,128>: [B, HIDDEN/128]
    std::vector<float>  w_gate_scale;      // Block2D<128,128>: [E*INTER/128, HIDDEN/128]
    std::vector<float>  w_up_scale;
    std::vector<float>  w_down_scale;      // [E*HIDDEN/128, INTER/128]

    HostTensor<int32_t> router_ids;
    HostTensor<float>   router_wts;

    HostTensor<bf16_t>  intermediate;
    HostTensor<bf16_t>  y;

    StageBufs(index_t B, index_t HIDDEN, index_t INTER, index_t TOPK, index_t E)
        : x_fp8({B, HIDDEN}),
          w_gate_fp8({E, INTER, HIDDEN}),
          w_up_fp8({E, INTER, HIDDEN}),
          w_down_fp8({E, HIDDEN, INTER}),
          w_down_fp4({E, HIDDEN, INTER}),
          x_bf16({B, HIDDEN}),
          router_ids({B, TOPK}),
          router_wts({B, TOPK}),
          intermediate({B, TOPK, INTER}),
          y({B, HIDDEN})
    {
        fill_random(x_fp8,      -0.5f, 0.5f,  42);
        fill_random(x_bf16,     -1.0f, 1.0f,  142);
        fill_random(w_gate_fp8, -0.25f, 0.25f, 43);
        fill_random(w_up_fp8,   -0.25f, 0.25f, 44);
        fill_random(w_down_fp8, -0.25f, 0.25f, 45);
        fill_random_pk_fp4(w_down_fp4, 45);

        fill_random_scale(x_scale_fp8,  static_cast<std::size_t>(B) * (HIDDEN / kBlockXK), 100);
        fill_random_scale(w_gate_scale,
                          static_cast<std::size_t>(E) * (INTER / kBlock_N) * (HIDDEN / kBlock_K),
                          200);
        fill_random_scale(w_up_scale,
                          static_cast<std::size_t>(E) * (INTER / kBlock_N) * (HIDDEN / kBlock_K),
                          201);
        fill_random_scale(w_down_scale,
                          static_cast<std::size_t>(E) * (HIDDEN / kBlock_N) * (INTER / kBlock_K),
                          202);

        fill_router(router_ids, router_wts, B, TOPK, E, 42);
    }
};

template <typename T>
void upload_host(const HostTensor<T>& h, DeviceMem& dev)
{
    dev.Realloc(h.get_element_space_size_in_bytes());
    dev.ToDevice(h.mData.data());
}

template <typename T>
void upload_vec(const std::vector<T>& v, DeviceMem& dev)
{
    dev.Realloc(v.size() * sizeof(T));
    dev.ToDevice(v.data());
}

void print_header()
{
    std::cout << std::left << std::setw(18) << "shape"
              << std::right << std::setw(5) << "B"
              << std::setw(20) << "kernel"
              << std::setw(10) << "ms"
              << std::setw(12) << "TFLOP/s"
              << std::setw(12) << "GB/s"
              << "\n";
}

void print_row(const std::string& shape,
               index_t            B,
               const std::string& kernel,
               const PerfResult&  p)
{
    std::cout << std::left << std::setw(18) << shape
              << std::right << std::setw(5) << B
              << std::setw(20) << kernel
              << std::setw(10) << std::fixed << std::setprecision(4) << p.time_ms
              << std::setw(12) << std::fixed << std::setprecision(2) << p.tflops
              << std::setw(12) << std::fixed << std::setprecision(1) << p.gb_per_s
              << "\n";
}

stream_config make_timing_cfg(int cold, int nrepeat)
{
    stream_config s;
    s.stream_id_   = nullptr;
    s.time_kernel_ = true;
    s.cold_niters_ = cold;
    s.nrepeat_     = nrepeat;
    s.is_gpu_timer_ = true;
    return s;
}

void bench_shape(const Shape& shape,
                 const std::vector<index_t>& batches,
                 int cold,
                 int nrepeat)
{
    for(const index_t B : batches)
    {
        const index_t HIDDEN = shape.HIDDEN;
        const index_t INTER  = shape.INTER;
        const index_t TOPK   = shape.TOPK;
        const index_t E      = shape.E;

        StageBufs bufs(B, HIDDEN, INTER, TOPK, E);

        DeviceMem x_fp8_dev, x_bf16_dev, w_gate_dev, w_up_dev, w_down_dev, w_down_fp4_dev;
        DeviceMem router_ids_dev, router_wts_dev;
        DeviceMem inter_dev(bufs.intermediate.get_element_space_size_in_bytes());
        DeviceMem y_dev(bufs.y.get_element_space_size_in_bytes());
        DeviceMem x_scale_fp8_dev, w_gate_scale_dev, w_up_scale_dev, w_down_scale_dev;

        upload_host(bufs.x_fp8,      x_fp8_dev);
        upload_host(bufs.x_bf16,     x_bf16_dev);
        upload_host(bufs.w_gate_fp8, w_gate_dev);
        upload_host(bufs.w_up_fp8,   w_up_dev);
        upload_host(bufs.w_down_fp8, w_down_dev);
        upload_host(bufs.w_down_fp4, w_down_fp4_dev);
        upload_host(bufs.router_ids, router_ids_dev);
        upload_host(bufs.router_wts, router_wts_dev);

        upload_vec(bufs.x_scale_fp8,  x_scale_fp8_dev);
        upload_vec(bufs.w_gate_scale, w_gate_scale_dev);
        upload_vec(bufs.w_up_scale,   w_up_scale_dev);
        upload_vec(bufs.w_down_scale, w_down_scale_dev);

        const auto cfg = make_timing_cfg(cold, nrepeat);

        // ---- Gate/up FP8 ----
        {
            GateUpKernelFP8::Kargs kargs{
                x_fp8_dev.GetDeviceBuffer(),
                x_scale_fp8_dev.GetDeviceBuffer(),
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpKernelFP8>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<fp8_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_up_fp8",  make_perf(ms, flops, bytes));
        }

        // ---- Gate/up FP8, BF16 dot2 accumulation ----
        {
            GateUpKernelFP8Dot2::Kargs kargs{
                x_fp8_dev.GetDeviceBuffer(),
                x_scale_fp8_dev.GetDeviceBuffer(),
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpKernelFP8Dot2>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<fp8_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_fp8_d2",  make_perf(ms, flops, bytes));
        }

        // ---- Gate/up FP8, BF16 dot2 accumulation, two rows per wave ----
        {
            GateUpKernelFP8Dot2N2::Kargs kargs{
                x_fp8_dev.GetDeviceBuffer(),
                x_scale_fp8_dev.GetDeviceBuffer(),
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpKernelFP8Dot2N2>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<fp8_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_fp8_n2",  make_perf(ms, flops, bytes));
        }

        // ---- Gate/up FP8, packed FP8->FP32 + packed FP32 FMA ----
        {
            GateUpKernelFP8PkF32::Kargs kargs{
                x_fp8_dev.GetDeviceBuffer(),
                x_scale_fp8_dev.GetDeviceBuffer(),
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpKernelFP8PkF32>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<fp8_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_fp8_pkf", make_perf(ms, flops, bytes));
        }

        // ---- Gate/up FP8, multi-warp LDS-staged X ----
        {
            GateUpLdsXKernelFP8::Kargs kargs{
                x_fp8_dev.GetDeviceBuffer(),
                x_scale_fp8_dev.GetDeviceBuffer(),
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpLdsXKernelFP8>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<fp8_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_fp8_lds", make_perf(ms, flops, bytes));
        }

        // ---- Gate/up FP8, multi-warp LDS-staged X plus two rows per wave ----
        {
            GateUpLdsXKernelFP8N2::Kargs kargs{
                x_fp8_dev.GetDeviceBuffer(),
                x_scale_fp8_dev.GetDeviceBuffer(),
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpLdsXKernelFP8N2>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<fp8_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_fp8_lds_n2", make_perf(ms, flops, bytes));
        }

        // ---- Gate/up BF16 ----
        {
            GateUpKernelBF16::Kargs kargs{
                x_bf16_dev.GetDeviceBuffer(),
                nullptr,
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpKernelBF16>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<bf16_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_up_bf16", make_perf(ms, flops, bytes));
        }

        // ---- Gate/up BF16, BF16 dot2 accumulation ----
        {
            GateUpKernelBF16Dot2::Kargs kargs{
                x_bf16_dev.GetDeviceBuffer(),
                nullptr,
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpKernelBF16Dot2>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<bf16_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_bf16_d2", make_perf(ms, flops, bytes));
        }

        // ---- Gate/up BF16, BF16 dot2 accumulation, two rows per wave ----
        {
            GateUpKernelBF16Dot2N2::Kargs kargs{
                x_bf16_dev.GetDeviceBuffer(),
                nullptr,
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpKernelBF16Dot2N2>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<bf16_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_bf16_n2", make_perf(ms, flops, bytes));
        }

        // ---- Gate/up BF16, packed FP8->FP32 + packed FP32 FMA ----
        {
            GateUpKernelBF16PkF32::Kargs kargs{
                x_bf16_dev.GetDeviceBuffer(),
                nullptr,
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpKernelBF16PkF32>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<bf16_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_bf16_pkf", make_perf(ms, flops, bytes));
        }

        // ---- Gate/up BF16, multi-warp LDS-staged X ----
        {
            GateUpLdsXKernelBF16::Kargs kargs{
                x_bf16_dev.GetDeviceBuffer(),
                nullptr,
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpLdsXKernelBF16>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<bf16_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_bf16_lds", make_perf(ms, flops, bytes));
        }

        // ---- Gate/up BF16, multi-warp LDS-staged X plus two rows per wave ----
        {
            GateUpLdsXKernelBF16N2::Kargs kargs{
                x_bf16_dev.GetDeviceBuffer(),
                nullptr,
                w_gate_dev.GetDeviceBuffer(),
                w_gate_scale_dev.GetDeviceBuffer(),
                w_up_dev.GetDeviceBuffer(),
                w_up_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                inter_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                HIDDEN, HIDDEN, HIDDEN, INTER};

            const float ms =
                launch_warp_decode_gate_up<GateUpLdsXKernelBF16N2>(kargs, cfg);
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                               element_bytes<bf16_t>(),
                                               element_bytes<fp8_t>(),
                                               element_bytes<bf16_t>());
            print_row(shape.name, B, "gate_bf16_lds_n2", make_perf(ms, flops, bytes));
        }

        const bool supports_regular_down = (INTER % (kWaveSize * kVectorDefault)) == 0;

        // ---- Down/reduce (shared) ----
        if(supports_regular_down)
        {
            const bool use_kvector_fp8 = (INTER % (kWaveSize * kVectorFP8)) == 0;
            float ms = 0.0f;
            if(use_kvector_fp8)
            {
                DownKernelFP8::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelFP8>(kargs, cfg);
            }
            else
            {
                DownKernelDefault::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelDefault>(kargs, cfg);
            }
            const double flops = down_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = down_bytes(B, HIDDEN, INTER, TOPK,
                                            element_bytes<bf16_t>(),
                                            element_bytes<fp8_t>(),
                                            element_bytes<bf16_t>());
            print_row(shape.name, B, "down_reduce",  make_perf(ms, flops, bytes));
        }

        // ---- Down/reduce, BF16 dot2 accumulation ----
        if(supports_regular_down)
        {
            const bool use_kvector_fp8 = (INTER % (kWaveSize * kVectorFP8)) == 0;
            float ms = 0.0f;
            if(use_kvector_fp8)
            {
                DownKernelFP8Dot2::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelFP8Dot2>(kargs, cfg);
            }
            else
            {
                DownKernelDefaultDot2::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelDefaultDot2>(kargs, cfg);
            }
            const double flops = down_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = down_bytes(B, HIDDEN, INTER, TOPK,
                                            element_bytes<bf16_t>(),
                                            element_bytes<fp8_t>(),
                                            element_bytes<bf16_t>());
            print_row(shape.name, B, "down_d2",  make_perf(ms, flops, bytes));
        }

        // ---- Down/reduce, BF16 dot2 accumulation, two hidden rows per wave ----
        if(supports_regular_down)
        {
            const bool use_kvector_fp8 = (INTER % (kWaveSize * kVectorFP8)) == 0;
            float ms = 0.0f;
            if(use_kvector_fp8)
            {
                DownKernelFP8Dot2H2::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelFP8Dot2H2>(kargs, cfg);
            }
            else
            {
                DownKernelDefaultDot2H2::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelDefaultDot2H2>(kargs, cfg);
            }
            const double flops = down_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = down_bytes(B, HIDDEN, INTER, TOPK,
                                            element_bytes<bf16_t>(),
                                            element_bytes<fp8_t>(),
                                            element_bytes<bf16_t>());
            print_row(shape.name, B, "down_h2_d2", make_perf(ms, flops, bytes));
        }

        // ---- Down/reduce, packed MXFP4 weights, pk_fma (halved weight read) ----
        if((INTER % (kWaveSize * kVectorFP4)) == 0)
        {
            DownKernelFP4::Kargs kargs{
                inter_dev.GetDeviceBuffer(),
                w_down_fp4_dev.GetDeviceBuffer(),
                w_down_scale_dev.GetDeviceBuffer(), // per-tensor scale reads element 0
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                y_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                INTER, INTER / 2, HIDDEN}; // stride_w_down is in packed bytes (INTER/2)
            const float ms = launch_warp_decode_down_reduce<DownKernelFP4>(kargs, cfg);
            const double flops = down_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = down_bytes(B, HIDDEN, INTER, TOPK,
                                            element_bytes<bf16_t>(),
                                            element_bytes<pk_fp4_t>(),
                                            element_bytes<bf16_t>());
            print_row(shape.name, B, "down_fp4", make_perf(ms, flops, bytes));
        }

        // ---- Down/reduce, packed MXFP4 weights, wide 128-bit loads (kVector=32) ----
        if((INTER % (kWaveSize * kVectorFP4Wide)) == 0)
        {
            DownKernelFP4Wide::Kargs kargs{
                inter_dev.GetDeviceBuffer(),
                w_down_fp4_dev.GetDeviceBuffer(),
                w_down_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                y_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                INTER, INTER / 2, HIDDEN};
            const float ms = launch_warp_decode_down_reduce<DownKernelFP4Wide>(kargs, cfg);
            const double flops = down_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = down_bytes(B, HIDDEN, INTER, TOPK,
                                            element_bytes<bf16_t>(),
                                            element_bytes<pk_fp4_t>(),
                                            element_bytes<bf16_t>());
            print_row(shape.name, B, "down_fp4_wide", make_perf(ms, flops, bytes));
        }

        // ---- Down/reduce, packed MXFP4 weights, two hidden rows per wave ----
        if((INTER % (kWaveSize * kVectorFP4)) == 0 && (HIDDEN % 2) == 0)
        {
            DownKernelFP4H2::Kargs kargs{
                inter_dev.GetDeviceBuffer(),
                w_down_fp4_dev.GetDeviceBuffer(),
                w_down_scale_dev.GetDeviceBuffer(),
                static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                y_dev.GetDeviceBuffer(),
                B, HIDDEN, INTER, TOPK, E,
                INTER, INTER / 2, HIDDEN};
            const float ms = launch_warp_decode_down_reduce<DownKernelFP4H2>(kargs, cfg);
            const double flops = down_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = down_bytes(B, HIDDEN, INTER, TOPK,
                                            element_bytes<bf16_t>(),
                                            element_bytes<pk_fp4_t>(),
                                            element_bytes<bf16_t>());
            print_row(shape.name, B, "down_fp4_h2", make_perf(ms, flops, bytes));
        }

        // ---- Down/reduce, short-INTER full-width dot2 subgroup path ----
        if(INTER == 512 || INTER == 256 || INTER == 128)
        {
            float ms = 0.0f;
            if(INTER == 512)
            {
                DownKernelShort512Dot2::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelShort512Dot2>(kargs, cfg);
            }
            else if(INTER == 256)
            {
                DownKernelShort256Dot2::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelShort256Dot2>(kargs, cfg);
            }
            else
            {
                DownKernelShort128Dot2::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelShort128Dot2>(kargs, cfg);
            }
            const double flops = down_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = down_bytes(B, HIDDEN, INTER, TOPK,
                                            element_bytes<bf16_t>(),
                                            element_bytes<fp8_t>(),
                                            element_bytes<bf16_t>());
            print_row(shape.name, B, "down_short_d2", make_perf(ms, flops, bytes));
        }

        // ---- Down/reduce, packed FP8->FP32 + packed FP32 FMA ----
        if(supports_regular_down)
        {
            const bool use_kvector_fp8 = (INTER % (kWaveSize * kVectorFP8)) == 0;
            float ms = 0.0f;
            if(use_kvector_fp8)
            {
                DownKernelFP8PkF32::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelFP8PkF32>(kargs, cfg);
            }
            else
            {
                DownKernelDefaultPkF32::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownKernelDefaultPkF32>(kargs, cfg);
            }
            const double flops = down_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = down_bytes(B, HIDDEN, INTER, TOPK,
                                            element_bytes<bf16_t>(),
                                            element_bytes<fp8_t>(),
                                            element_bytes<bf16_t>());
            print_row(shape.name, B, "down_pkf", make_perf(ms, flops, bytes));
        }

        // ---- Down/reduce, multi-warp LDS-staged intermediate ----
        if(supports_regular_down)
        {
            const bool use_kvector_fp8 = (INTER % (kWaveSize * kVectorFP8)) == 0;
            float ms = 0.0f;
            if(use_kvector_fp8)
            {
                DownLdsInterKernelFP8::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownLdsInterKernelFP8>(kargs, cfg);
            }
            else
            {
                DownLdsInterKernelDefault::Kargs kargs{
                    inter_dev.GetDeviceBuffer(),
                    w_down_dev.GetDeviceBuffer(),
                    w_down_scale_dev.GetDeviceBuffer(),
                    static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                    static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                    y_dev.GetDeviceBuffer(),
                    B, HIDDEN, INTER, TOPK, E,
                    INTER, INTER, HIDDEN};
                ms = launch_warp_decode_down_reduce<DownLdsInterKernelDefault>(kargs, cfg);
            }
            const double flops = down_flops(B, HIDDEN, INTER, TOPK);
            const double bytes = down_bytes(B, HIDDEN, INTER, TOPK,
                                            element_bytes<bf16_t>(),
                                            element_bytes<fp8_t>(),
                                            element_bytes<bf16_t>());
            print_row(shape.name, B, "down_lds",  make_perf(ms, flops, bytes));
        }

        // ---- Persistent JTile V2 (FP8 + BF16 X) and V3 J=16 (FP8 + BF16 X) ----
        //
        // These are the producer-owned single-stage warp-decode prototypes
        // exposed by the warp_decode_ext Python extension. They are linked in
        // here so the PMC sweep in docs/issues/warp_decode_profiling can
        // measure them side by side with the split-path kernels above.
        //
        // The persistent kernel internally:
        //   - zeroes y, then runs a resident-grid barrier;
        //   - assigns (b, topk, J_TILE) producer tiles to CTAs;
        //   - emits packed BF16x2 atomic adds into y for each H pair.
        //
        // Each timed iteration must start with control[0..3] = 0, so we wrap
        // hipMemsetAsync + hipLaunchKernelGGL into the launch_kernel callable.
        const bool persistent_supported = (HIDDEN % 2 == 0) && (INTER % 128 == 0) &&
                                          (HIDDEN % 128 == 0) && (TOPK > 0);

        auto bench_persistent_v2 = [&](const auto* x_ptr,
                                       const float* x_scale_ptr,
                                       const std::string& tag) {
            namespace wdp = ck_tile::warp_decode_persistent;
            using XDataType = std::remove_const_t<std::remove_pointer_t<decltype(x_ptr)>>;

            DeviceMem control_dev(4 * sizeof(int32_t));
            constexpr int block_threads =
                static_cast<int>(wdp::kPersistentJTileV2 * wdp::kWaveSize);
            const index_t j_tiles = (INTER + wdp::kPersistentJTileV2 - 1) /
                                    wdp::kPersistentJTileV2;
            const index_t total_tiles = static_cast<index_t>(B * TOPK) * j_tiles;
            const int blocks = wdp::persistent_block_count(total_tiles);
            const dim3 grid(blocks);
            const dim3 block(block_threads);

            auto launcher = [&](const ck_tile::stream_config& s) {
                HIP_CHECK_ERROR(hipMemsetAsync(control_dev.GetDeviceBuffer(),
                                               0,
                                               4 * sizeof(int32_t),
                                               s.stream_id_));
                hipLaunchKernelGGL((wdp::persistent_jtile_v2_atomic_kernel<
                                       XDataType,
                                       wdp::kPersistentJTileV2,
                                       wdp::kPersistentHPerThreadV2>),
                                   grid,
                                   block,
                                   0,
                                   s.stream_id_,
                                   x_ptr,
                                   x_scale_ptr,
                                   static_cast<const fp8_t*>(w_gate_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(w_gate_scale_dev.GetDeviceBuffer()),
                                   static_cast<const fp8_t*>(w_up_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(w_up_scale_dev.GetDeviceBuffer()),
                                   static_cast<const fp8_t*>(w_down_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(w_down_scale_dev.GetDeviceBuffer()),
                                   static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                                   static_cast<bf16_t*>(y_dev.GetDeviceBuffer()),
                                   static_cast<int32_t*>(control_dev.GetDeviceBuffer()),
                                   static_cast<index_t>(B),
                                   static_cast<index_t>(HIDDEN),
                                   static_cast<index_t>(INTER),
                                   static_cast<index_t>(TOPK),
                                   static_cast<index_t>(E));
            };

            const float ms = ck_tile::launch_kernel(cfg, launcher);
            // Persistent kernel touches the same bytes as gate/up + down per
            // routed expert, plus the in-kernel zero of y. Report a combined
            // byte count so the GB/s number matches the split path semantics.
            const double x_elem    = std::is_same_v<XDataType, fp8_t>
                                         ? element_bytes<fp8_t>()
                                         : element_bytes<bf16_t>();
            const double gate_b    = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                                   x_elem,
                                                   element_bytes<fp8_t>(),
                                                   element_bytes<bf16_t>());
            const double down_b    = down_bytes(B, HIDDEN, INTER, TOPK,
                                                element_bytes<bf16_t>(),
                                                element_bytes<fp8_t>(),
                                                element_bytes<bf16_t>());
            const double y_zero_b  = static_cast<double>(B) *
                                     static_cast<double>(HIDDEN) *
                                     element_bytes<bf16_t>();
            const double bytes     = gate_b + down_b + y_zero_b;
            const double flops     = gate_up_flops(B, HIDDEN, INTER, TOPK) +
                                     down_flops(B, HIDDEN, INTER, TOPK);
            print_row(shape.name, B, tag, make_perf(ms, flops, bytes));
        };

        auto bench_persistent_v3_j16 = [&](const auto* x_ptr,
                                           const float* x_scale_ptr,
                                           const std::string& tag) {
            namespace wdp = ck_tile::warp_decode_persistent;
            using XDataType = std::remove_const_t<std::remove_pointer_t<decltype(x_ptr)>>;

            DeviceMem control_dev(4 * sizeof(int32_t));
            constexpr int block_threads = static_cast<int>(
                (wdp::kPersistentJTileV3Large / wdp::kPersistentNPerWarpV3) *
                wdp::kWaveSize);
            const index_t j_tiles = (INTER + wdp::kPersistentJTileV3Large - 1) /
                                    wdp::kPersistentJTileV3Large;
            const index_t total_tiles = static_cast<index_t>(B * TOPK) * j_tiles;
            const int blocks = wdp::persistent_block_count(total_tiles);
            const dim3 grid(blocks);
            const dim3 block(block_threads);

            auto launcher = [&](const ck_tile::stream_config& s) {
                HIP_CHECK_ERROR(hipMemsetAsync(control_dev.GetDeviceBuffer(),
                                               0,
                                               4 * sizeof(int32_t),
                                               s.stream_id_));
                hipLaunchKernelGGL((wdp::persistent_jtile_v3_atomic_kernel<
                                       XDataType,
                                       wdp::kPersistentJTileV3Large,
                                       wdp::kPersistentNPerWarpV3,
                                       wdp::kPersistentHPerThreadV3>),
                                   grid,
                                   block,
                                   0,
                                   s.stream_id_,
                                   x_ptr,
                                   x_scale_ptr,
                                   static_cast<const fp8_t*>(w_gate_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(w_gate_scale_dev.GetDeviceBuffer()),
                                   static_cast<const fp8_t*>(w_up_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(w_up_scale_dev.GetDeviceBuffer()),
                                   static_cast<const fp8_t*>(w_down_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(w_down_scale_dev.GetDeviceBuffer()),
                                   static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                                   static_cast<bf16_t*>(y_dev.GetDeviceBuffer()),
                                   static_cast<int32_t*>(control_dev.GetDeviceBuffer()),
                                   static_cast<index_t>(B),
                                   static_cast<index_t>(HIDDEN),
                                   static_cast<index_t>(INTER),
                                   static_cast<index_t>(TOPK),
                                   static_cast<index_t>(E));
            };

            const float ms = ck_tile::launch_kernel(cfg, launcher);
            const double x_elem    = std::is_same_v<XDataType, fp8_t>
                                         ? element_bytes<fp8_t>()
                                         : element_bytes<bf16_t>();
            const double gate_b    = gate_up_bytes(B, HIDDEN, INTER, TOPK,
                                                   x_elem,
                                                   element_bytes<fp8_t>(),
                                                   element_bytes<bf16_t>());
            const double down_b    = down_bytes(B, HIDDEN, INTER, TOPK,
                                                element_bytes<bf16_t>(),
                                                element_bytes<fp8_t>(),
                                                element_bytes<bf16_t>());
            const double y_zero_b  = static_cast<double>(B) *
                                     static_cast<double>(HIDDEN) *
                                     element_bytes<bf16_t>();
            const double bytes     = gate_b + down_b + y_zero_b;
            const double flops     = gate_up_flops(B, HIDDEN, INTER, TOPK) +
                                     down_flops(B, HIDDEN, INTER, TOPK);
            print_row(shape.name, B, tag, make_perf(ms, flops, bytes));
        };

        auto bench_persistent_v4 = [&](const auto*        x_ptr,
                                       const float*       x_scale_ptr,
                                       const std::string& tag) {
            namespace wdp   = ck_tile::warp_decode_persistent;
            using XDataType = std::remove_const_t<std::remove_pointer_t<decltype(x_ptr)>>;

            DeviceMem     control_dev(4 * sizeof(int32_t));
            constexpr int block_threads = static_cast<int>(wdp::kPersistentV4BlockThreads);

            const index_t inter_pairs =
                INTER / wdp::kPersistentV4NPerWarpProducer;
            const index_t h_pairs =
                HIDDEN / wdp::kPersistentV4HPerWarpConsumer;
            const index_t producer_pairs =
                static_cast<index_t>(B * TOPK) * inter_pairs;
            const index_t consumer_pairs = static_cast<index_t>(B) * h_pairs;
            const index_t producer_groups =
                (producer_pairs + wdp::kPersistentV4WavesPerBlock - 1) /
                wdp::kPersistentV4WavesPerBlock;
            const index_t consumer_groups =
                (consumer_pairs + wdp::kPersistentV4WavesPerBlock - 1) /
                wdp::kPersistentV4WavesPerBlock;
            const index_t total_groups = std::max(producer_groups, consumer_groups);
            const int     blocks       = wdp::persistent_block_count(total_groups);
            const dim3    grid(blocks);
            const dim3    block(block_threads);

            // Reuse the inter_dev scratch buffer from this bench_shape (sized for
            // [B, TOPK, INTER] bf16, which matches what V4 expects).
            auto launcher = [&](const ck_tile::stream_config& s) {
                HIP_CHECK_ERROR(hipMemsetAsync(control_dev.GetDeviceBuffer(),
                                               0,
                                               4 * sizeof(int32_t),
                                               s.stream_id_));
                hipLaunchKernelGGL((wdp::persistent_v4_kernel<XDataType>),
                                   grid,
                                   block,
                                   0,
                                   s.stream_id_,
                                   x_ptr,
                                   x_scale_ptr,
                                   static_cast<const fp8_t*>(w_gate_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(w_gate_scale_dev.GetDeviceBuffer()),
                                   static_cast<const fp8_t*>(w_up_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(w_up_scale_dev.GetDeviceBuffer()),
                                   static_cast<const fp8_t*>(w_down_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(w_down_scale_dev.GetDeviceBuffer()),
                                   static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer()),
                                   static_cast<const float*>(router_wts_dev.GetDeviceBuffer()),
                                   static_cast<bf16_t*>(inter_dev.GetDeviceBuffer()),
                                   static_cast<bf16_t*>(y_dev.GetDeviceBuffer()),
                                   static_cast<int32_t*>(control_dev.GetDeviceBuffer()),
                                   static_cast<index_t>(B),
                                   static_cast<index_t>(HIDDEN),
                                   static_cast<index_t>(INTER),
                                   static_cast<index_t>(TOPK),
                                   static_cast<index_t>(E));
            };

            const float  ms     = ck_tile::launch_kernel(cfg, launcher);
            const double x_elem = std::is_same_v<XDataType, fp8_t>
                                      ? element_bytes<fp8_t>()
                                      : element_bytes<bf16_t>();
            const double gate_b = gate_up_bytes(B,
                                                HIDDEN,
                                                INTER,
                                                TOPK,
                                                x_elem,
                                                element_bytes<fp8_t>(),
                                                element_bytes<bf16_t>());
            const double down_b = down_bytes(B,
                                             HIDDEN,
                                             INTER,
                                             TOPK,
                                             element_bytes<bf16_t>(),
                                             element_bytes<fp8_t>(),
                                             element_bytes<bf16_t>());
            // V4 writes y exactly once (no atomics), so do not add a separate
            // y-zero traffic term the way V2/V3 do.
            const double bytes = gate_b + down_b;
            const double flops = gate_up_flops(B, HIDDEN, INTER, TOPK) +
                                 down_flops(B, HIDDEN, INTER, TOPK);
            print_row(shape.name, B, tag, make_perf(ms, flops, bytes));
        };

        if(persistent_supported)
        {
            bench_persistent_v2(
                static_cast<const fp8_t*>(x_fp8_dev.GetDeviceBuffer()),
                static_cast<const float*>(x_scale_fp8_dev.GetDeviceBuffer()),
                "wd_fp8_pj_v2");
            bench_persistent_v2(
                static_cast<const bf16_t*>(x_bf16_dev.GetDeviceBuffer()),
                static_cast<const float*>(nullptr),
                "wd_bf16_pj_v2");
            bench_persistent_v3_j16(
                static_cast<const fp8_t*>(x_fp8_dev.GetDeviceBuffer()),
                static_cast<const float*>(x_scale_fp8_dev.GetDeviceBuffer()),
                "wd_fp8_pj_v3_j16");
            bench_persistent_v3_j16(
                static_cast<const bf16_t*>(x_bf16_dev.GetDeviceBuffer()),
                static_cast<const float*>(nullptr),
                "wd_bf16_pj_v3_j16");
            bench_persistent_v4(
                static_cast<const fp8_t*>(x_fp8_dev.GetDeviceBuffer()),
                static_cast<const float*>(x_scale_fp8_dev.GetDeviceBuffer()),
                "wd_fp8_pv4");
            bench_persistent_v4(
                static_cast<const bf16_t*>(x_bf16_dev.GetDeviceBuffer()),
                static_cast<const float*>(nullptr),
                "wd_bf16_pv4");
        }
    }
}

int parse_int_env(const char* name, int fallback)
{
    if(const char* v = std::getenv(name))
    {
        try
        {
            return std::stoi(v);
        }
        catch(...)
        {
            return fallback;
        }
    }
    return fallback;
}

std::vector<std::string> split_csv_env(const char* name)
{
    std::vector<std::string> out;
    const char* v = std::getenv(name);
    if(!v || *v == '\0')
        return out;
    std::stringstream ss(v);
    std::string tok;
    while(std::getline(ss, tok, ','))
    {
        if(!tok.empty())
            out.push_back(tok);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    const int cold    = parse_int_env("CK_WARP_DECODE_BENCH_COLD", 5);
    const int nrepeat = parse_int_env("CK_WARP_DECODE_BENCH_ITERS", 30);

    const std::vector<Shape> all_shapes = {
        {"deepseek-v3", 7168, 2048, 8, 256},
        {"minimax",     3072, 1536, 8, 256},
        {"qwen3next",   2048,  512, 10, 512},
        {"qwen3next-tp2", 2048, 256, 10, 512},
        {"qwen3next-tp4", 2048, 128, 10, 512},
    };

    const std::vector<index_t> all_batches = {1, 2, 4, 8, 16, 32, 64};

    const auto shape_filter = split_csv_env("CK_WARP_DECODE_BENCH_SHAPES");
    const auto batch_filter = split_csv_env("CK_WARP_DECODE_BENCH_BATCHES");

    std::vector<Shape> shapes;
    if(shape_filter.empty())
    {
        shapes = all_shapes;
    }
    else
    {
        for(const auto& name : shape_filter)
        {
            for(const auto& s : all_shapes)
            {
                if(s.name == name)
                {
                    shapes.push_back(s);
                    break;
                }
            }
        }
    }

    std::vector<index_t> batches;
    if(batch_filter.empty())
    {
        batches = all_batches;
    }
    else
    {
        for(const auto& tok : batch_filter)
        {
            try
            {
                batches.push_back(static_cast<index_t>(std::stoi(tok)));
            }
            catch(...)
            {
            }
        }
    }

    std::cout << "bench_warp_decode: cold=" << cold << " nrepeat=" << nrepeat << "\n";
    print_header();
    for(const auto& shape : shapes)
    {
        bench_shape(shape, batches, cold, nrepeat);
    }

    return 0;
}
