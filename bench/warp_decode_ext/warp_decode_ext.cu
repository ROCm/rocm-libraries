// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Minimal PyTorch extension exposing the CKTile warp-decode kernels for the
// FP8 block-scale MoE benchmark. The unsuffixed entry points use the current
// default optimized dot2 path; _base/_pkf32/_lds variants are exposed for
// attribution benchmarks.
//
//   warp_decode_gate_up_fp8(x_fp8, x_scale, w_gate, w_gate_scale, w_up,
//                           w_up_scale, router_ids, intermediate_out)
//     XDataType   = fp8_t
//     XScaleLayout = Block2D<1, 128>  (per-1x128 activation scale)
//
//   warp_decode_gate_up_bf16(x_bf16, w_gate, w_gate_scale, w_up, w_up_scale,
//                            router_ids, intermediate_out)
//     XDataType   = bf16_t
//     XScaleLayout = PerTensor   (no x scale; pointer passed as nullptr)
//
//   warp_decode_down_reduce(intermediate, w_down, w_down_scale, router_ids,
//                           router_wts, y_out)
//     IntermediateDataType = bf16_t
//
// Shared config for all three: WDataType = fp8_t, WScaleLayout =
// Block2D<128, 128>, YDataType = bf16_t. FP8-heavy aligned paths use
// kVector = 16, while BF16 gate/up and non-aligned down/reduce keep kVector = 8.

#include <torch/extension.h>

#include <ATen/hip/HIPContext.h>

#include <stdexcept>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/warp_decode.hpp"

namespace {

using ck_tile::bf16_t;
using ck_tile::fp8_t;
using ck_tile::index_t;
using ck_tile::WarpDecodeScaleLayout;
using ck_tile::WarpDecodePolicy;
using ck_tile::WarpDecodeGateUpKernel;
using ck_tile::WarpDecodeGateUpLdsXKernel;
using ck_tile::WarpDecodeGateUpProblem;
using ck_tile::WarpDecodeDownReduceKernel;
using ck_tile::WarpDecodeDownReduceLdsInterKernel;
using ck_tile::WarpDecodeDownReduceProblem;

constexpr index_t kVectorDefault = 8;
constexpr index_t kVectorFP8     = 16;
constexpr index_t kWaveSize      = 64;

using XScaleLayoutFP8  = WarpDecodeScaleLayout::Block2D<1, 128>;
using XScaleLayoutBF16 = WarpDecodeScaleLayout::PerTensor;
using WScaleLayoutAll  = WarpDecodeScaleLayout::Block2D<128, 128>;

using GateUpProblemFP8Base = WarpDecodeGateUpProblem<fp8_t,
                                                     fp8_t,
                                                     float,
                                                     bf16_t,
                                                     float,
                                                     float,
                                                     XScaleLayoutFP8,
                                                     WScaleLayoutAll,
                                                     ck_tile::element_wise::Silu,
                                                     kVectorFP8,
                                                     false,
                                                     false>;
using GateUpProblemFP8Dot2 = WarpDecodeGateUpProblem<fp8_t,
                                                     fp8_t,
                                                     float,
                                                     bf16_t,
                                                     float,
                                                     float,
                                                     XScaleLayoutFP8,
                                                     WScaleLayoutAll,
                                                     ck_tile::element_wise::Silu,
                                                     kVectorFP8,
                                                     true,
                                                     false>;
using GateUpProblemFP8PkF32 = WarpDecodeGateUpProblem<fp8_t,
                                                      fp8_t,
                                                      float,
                                                      bf16_t,
                                                      float,
                                                      float,
                                                      XScaleLayoutFP8,
                                                      WScaleLayoutAll,
                                                      ck_tile::element_wise::Silu,
                                                      kVectorFP8,
                                                      false,
                                                      true>;
using GateUpKernelFP8Base  = WarpDecodeGateUpKernel<GateUpProblemFP8Base, WarpDecodePolicy>;
using GateUpKernelFP8Dot2  = WarpDecodeGateUpKernel<GateUpProblemFP8Dot2, WarpDecodePolicy>;
using GateUpKernelFP8PkF32 = WarpDecodeGateUpKernel<GateUpProblemFP8PkF32, WarpDecodePolicy>;
using GateUpKernelFP8Lds   = WarpDecodeGateUpLdsXKernel<GateUpProblemFP8Base, WarpDecodePolicy>;

using GateUpProblemBF16Base = WarpDecodeGateUpProblem<bf16_t,
                                                      fp8_t,
                                                      float,
                                                      bf16_t,
                                                      float,
                                                      float,
                                                      XScaleLayoutBF16,
                                                      WScaleLayoutAll,
                                                      ck_tile::element_wise::Silu,
                                                      kVectorDefault,
                                                      false,
                                                      false>;
using GateUpProblemBF16Dot2 = WarpDecodeGateUpProblem<bf16_t,
                                                      fp8_t,
                                                      float,
                                                      bf16_t,
                                                      float,
                                                      float,
                                                      XScaleLayoutBF16,
                                                      WScaleLayoutAll,
                                                      ck_tile::element_wise::Silu,
                                                      kVectorDefault,
                                                      true,
                                                      false>;
using GateUpProblemBF16PkF32 = WarpDecodeGateUpProblem<bf16_t,
                                                       fp8_t,
                                                       float,
                                                       bf16_t,
                                                       float,
                                                       float,
                                                       XScaleLayoutBF16,
                                                       WScaleLayoutAll,
                                                       ck_tile::element_wise::Silu,
                                                       kVectorDefault,
                                                       false,
                                                       true>;
using GateUpKernelBF16Base  = WarpDecodeGateUpKernel<GateUpProblemBF16Base, WarpDecodePolicy>;
using GateUpKernelBF16Dot2  = WarpDecodeGateUpKernel<GateUpProblemBF16Dot2, WarpDecodePolicy>;
using GateUpKernelBF16PkF32 = WarpDecodeGateUpKernel<GateUpProblemBF16PkF32, WarpDecodePolicy>;
using GateUpKernelBF16Lds   = WarpDecodeGateUpLdsXKernel<GateUpProblemBF16Base, WarpDecodePolicy>;

using DownProblemDefaultBase = WarpDecodeDownReduceProblem<bf16_t,
                                                           fp8_t,
                                                           float,
                                                           bf16_t,
                                                           float,
                                                           WScaleLayoutAll,
                                                           kVectorDefault,
                                                           false,
                                                           false>;
using DownProblemDefaultDot2 = WarpDecodeDownReduceProblem<bf16_t,
                                                           fp8_t,
                                                           float,
                                                           bf16_t,
                                                           float,
                                                           WScaleLayoutAll,
                                                           kVectorDefault,
                                                           true,
                                                           false>;
using DownProblemDefaultPkF32 = WarpDecodeDownReduceProblem<bf16_t,
                                                            fp8_t,
                                                            float,
                                                            bf16_t,
                                                            float,
                                                            WScaleLayoutAll,
                                                            kVectorDefault,
                                                            false,
                                                            true>;
using DownKernelDefaultBase  = WarpDecodeDownReduceKernel<DownProblemDefaultBase, WarpDecodePolicy>;
using DownKernelDefaultDot2  = WarpDecodeDownReduceKernel<DownProblemDefaultDot2, WarpDecodePolicy>;
using DownKernelDefaultPkF32 = WarpDecodeDownReduceKernel<DownProblemDefaultPkF32, WarpDecodePolicy>;
using DownKernelDefaultLds   = WarpDecodeDownReduceLdsInterKernel<DownProblemDefaultBase, WarpDecodePolicy>;

using DownProblemFP8Base = WarpDecodeDownReduceProblem<bf16_t,
                                                       fp8_t,
                                                       float,
                                                       bf16_t,
                                                       float,
                                                       WScaleLayoutAll,
                                                       kVectorFP8,
                                                       false,
                                                       false>;
using DownProblemFP8Dot2 = WarpDecodeDownReduceProblem<bf16_t,
                                                       fp8_t,
                                                       float,
                                                       bf16_t,
                                                       float,
                                                       WScaleLayoutAll,
                                                       kVectorFP8,
                                                       true,
                                                       false>;
using DownProblemFP8PkF32 = WarpDecodeDownReduceProblem<bf16_t,
                                                        fp8_t,
                                                        float,
                                                        bf16_t,
                                                        float,
                                                        WScaleLayoutAll,
                                                        kVectorFP8,
                                                        false,
                                                        true>;
using DownKernelFP8Base  = WarpDecodeDownReduceKernel<DownProblemFP8Base, WarpDecodePolicy>;
using DownKernelFP8Dot2  = WarpDecodeDownReduceKernel<DownProblemFP8Dot2, WarpDecodePolicy>;
using DownKernelFP8PkF32 = WarpDecodeDownReduceKernel<DownProblemFP8PkF32, WarpDecodePolicy>;
using DownKernelFP8Lds   = WarpDecodeDownReduceLdsInterKernel<DownProblemFP8Base, WarpDecodePolicy>;

hipStream_t current_hip_stream()
{
    return at::hip::getCurrentHIPStream().stream();
}

ck_tile::stream_config make_stream_cfg()
{
    ck_tile::stream_config s;
    s.stream_id_   = current_hip_stream();
    s.time_kernel_ = false;
    return s;
}

void check_tensor(const torch::Tensor& t,
                  const char* name,
                  torch::ScalarType expected_dtype,
                  std::initializer_list<int64_t> expected_shape)
{
    TORCH_CHECK(t.defined(), name, " is undefined");
    TORCH_CHECK(t.is_cuda(), name, " must be on GPU");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == expected_dtype,
                name, " has dtype ", t.scalar_type(), ", expected ", expected_dtype);
    if(expected_shape.size() != 0)
    {
        TORCH_CHECK(static_cast<size_t>(t.dim()) == expected_shape.size(),
                    name, " rank ", t.dim(), ", expected ", expected_shape.size());
        size_t i = 0;
        for(auto s : expected_shape)
        {
            TORCH_CHECK(t.size(i) == s,
                        name, " dim ", i, " is ", t.size(i), ", expected ", s);
            ++i;
        }
    }
}

template <typename DownKernel>
void launch_down_reduce_kernel(const torch::Tensor& intermediate,
                               const torch::Tensor& w_down,
                               const torch::Tensor& w_down_scale,
                               const torch::Tensor& router_ids,
                               const torch::Tensor& router_wts,
                               torch::Tensor& y,
                               index_t B,
                               index_t HIDDEN,
                               index_t INTER,
                               index_t TOP_K,
                               index_t E,
                               const ck_tile::stream_config& s)
{
    typename DownKernel::Kargs kargs{
        intermediate.data_ptr(),
        w_down.data_ptr(),
        w_down_scale.data_ptr(),
        static_cast<const int32_t*>(router_ids.data_ptr()),
        static_cast<const float*>(router_wts.data_ptr()),
        y.data_ptr(),
        B,
        HIDDEN,
        INTER,
        TOP_K,
        E,
        INTER,
        INTER,
        HIDDEN,
    };

    ck_tile::launch_warp_decode_down_reduce<DownKernel>(kargs, s);
}

template <typename GateKernel>
void launch_gate_up_fp8_kernel(const torch::Tensor& x,
                               const torch::Tensor& x_scale,
                               const torch::Tensor& w_gate,
                               const torch::Tensor& w_gate_scale,
                               const torch::Tensor& w_up,
                               const torch::Tensor& w_up_scale,
                               const torch::Tensor& router_ids,
                               torch::Tensor& intermediate,
                               index_t B,
                               index_t HIDDEN,
                               index_t INTER,
                               index_t TOP_K,
                               index_t E,
                               const ck_tile::stream_config& s)
{
    typename GateKernel::Kargs kargs{
        x.data_ptr(),
        x_scale.data_ptr(),
        w_gate.data_ptr(),
        w_gate_scale.data_ptr(),
        w_up.data_ptr(),
        w_up_scale.data_ptr(),
        static_cast<const int32_t*>(router_ids.data_ptr()),
        intermediate.data_ptr(),
        B,
        HIDDEN,
        INTER,
        TOP_K,
        E,
        HIDDEN,
        HIDDEN,
        HIDDEN,
        INTER,
    };

    ck_tile::launch_warp_decode_gate_up<GateKernel>(kargs, s);
}

template <typename GateKernel>
void launch_gate_up_bf16_kernel(const torch::Tensor& x,
                                const torch::Tensor& w_gate,
                                const torch::Tensor& w_gate_scale,
                                const torch::Tensor& w_up,
                                const torch::Tensor& w_up_scale,
                                const torch::Tensor& router_ids,
                                torch::Tensor& intermediate,
                                index_t B,
                                index_t HIDDEN,
                                index_t INTER,
                                index_t TOP_K,
                                index_t E,
                                const ck_tile::stream_config& s)
{
    typename GateKernel::Kargs kargs{
        x.data_ptr(),
        nullptr,  // p_x_scale: PerTensor with nullptr is interpreted as 1.0
        w_gate.data_ptr(),
        w_gate_scale.data_ptr(),
        w_up.data_ptr(),
        w_up_scale.data_ptr(),
        static_cast<const int32_t*>(router_ids.data_ptr()),
        intermediate.data_ptr(),
        B,
        HIDDEN,
        INTER,
        TOP_K,
        E,
        HIDDEN,
        HIDDEN,
        HIDDEN,
        INTER,
    };

    ck_tile::launch_warp_decode_gate_up<GateKernel>(kargs, s);
}

}  // namespace

// ---------------------------------------------------------------------------
// warp_decode_gate_up_fp8
// ---------------------------------------------------------------------------

template <typename GateKernel>
void warp_decode_gate_up_fp8_impl(const torch::Tensor& x,
                                  const torch::Tensor& x_scale,
                                  const torch::Tensor& w_gate,
                                  const torch::Tensor& w_gate_scale,
                                  const torch::Tensor& w_up,
                                  const torch::Tensor& w_up_scale,
                                  const torch::Tensor& router_ids,
                                  torch::Tensor& intermediate)
{
    const int64_t B      = x.size(0);
    const int64_t HIDDEN = x.size(1);

    TORCH_CHECK(w_gate.dim() == 3, "w_gate must be [E, INTER, HIDDEN]");
    const int64_t E     = w_gate.size(0);
    const int64_t INTER = w_gate.size(1);
    TORCH_CHECK(w_gate.size(2) == HIDDEN, "w_gate last dim must equal HIDDEN");

    TORCH_CHECK(router_ids.dim() == 2 && router_ids.size(0) == B,
                "router_ids must be [B, TOP_K]");
    const int64_t TOP_K = router_ids.size(1);

    check_tensor(x,            "x",            torch::kFloat8_e4m3fn, {B, HIDDEN});
    check_tensor(x_scale,      "x_scale",      torch::kFloat,         {B, HIDDEN / 128});
    check_tensor(w_gate,       "w_gate",       torch::kFloat8_e4m3fn, {E, INTER, HIDDEN});
    check_tensor(w_gate_scale, "w_gate_scale", torch::kFloat,         {E * INTER / 128, HIDDEN / 128});
    check_tensor(w_up,         "w_up",         torch::kFloat8_e4m3fn, {E, INTER, HIDDEN});
    check_tensor(w_up_scale,   "w_up_scale",   torch::kFloat,         {E * INTER / 128, HIDDEN / 128});
    check_tensor(router_ids,   "router_ids",   torch::kInt32,         {B, TOP_K});
    check_tensor(intermediate, "intermediate", torch::kBFloat16,      {B, TOP_K, INTER});

    auto s = make_stream_cfg();
    launch_gate_up_fp8_kernel<GateKernel>(x,
                                          x_scale,
                                          w_gate,
                                          w_gate_scale,
                                          w_up,
                                          w_up_scale,
                                          router_ids,
                                          intermediate,
                                          static_cast<index_t>(B),
                                          static_cast<index_t>(HIDDEN),
                                          static_cast<index_t>(INTER),
                                          static_cast<index_t>(TOP_K),
                                          static_cast<index_t>(E),
                                          s);
}

void warp_decode_gate_up_fp8_base(const torch::Tensor& x,
                                  const torch::Tensor& x_scale,
                                  const torch::Tensor& w_gate,
                                  const torch::Tensor& w_gate_scale,
                                  const torch::Tensor& w_up,
                                  const torch::Tensor& w_up_scale,
                                  const torch::Tensor& router_ids,
                                  torch::Tensor& intermediate)
{
    warp_decode_gate_up_fp8_impl<GateUpKernelFP8Base>(
        x, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, intermediate);
}

void warp_decode_gate_up_fp8(const torch::Tensor& x,
                             const torch::Tensor& x_scale,
                             const torch::Tensor& w_gate,
                             const torch::Tensor& w_gate_scale,
                             const torch::Tensor& w_up,
                             const torch::Tensor& w_up_scale,
                             const torch::Tensor& router_ids,
                             torch::Tensor& intermediate)
{
    warp_decode_gate_up_fp8_impl<GateUpKernelFP8Dot2>(
        x, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, intermediate);
}

void warp_decode_gate_up_fp8_pkf32(const torch::Tensor& x,
                                   const torch::Tensor& x_scale,
                                   const torch::Tensor& w_gate,
                                   const torch::Tensor& w_gate_scale,
                                   const torch::Tensor& w_up,
                                   const torch::Tensor& w_up_scale,
                                   const torch::Tensor& router_ids,
                                   torch::Tensor& intermediate)
{
    warp_decode_gate_up_fp8_impl<GateUpKernelFP8PkF32>(
        x, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, intermediate);
}

void warp_decode_gate_up_fp8_lds(const torch::Tensor& x,
                                 const torch::Tensor& x_scale,
                                 const torch::Tensor& w_gate,
                                 const torch::Tensor& w_gate_scale,
                                 const torch::Tensor& w_up,
                                 const torch::Tensor& w_up_scale,
                                 const torch::Tensor& router_ids,
                                 torch::Tensor& intermediate)
{
    warp_decode_gate_up_fp8_impl<GateUpKernelFP8Lds>(
        x, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, intermediate);
}

// ---------------------------------------------------------------------------
// warp_decode_gate_up_bf16
// ---------------------------------------------------------------------------

template <typename GateKernel>
void warp_decode_gate_up_bf16_impl(const torch::Tensor& x,
                                   const torch::Tensor& w_gate,
                                   const torch::Tensor& w_gate_scale,
                                   const torch::Tensor& w_up,
                                   const torch::Tensor& w_up_scale,
                                   const torch::Tensor& router_ids,
                                   torch::Tensor& intermediate)
{
    const int64_t B      = x.size(0);
    const int64_t HIDDEN = x.size(1);

    TORCH_CHECK(w_gate.dim() == 3, "w_gate must be [E, INTER, HIDDEN]");
    const int64_t E     = w_gate.size(0);
    const int64_t INTER = w_gate.size(1);
    TORCH_CHECK(w_gate.size(2) == HIDDEN, "w_gate last dim must equal HIDDEN");

    TORCH_CHECK(router_ids.dim() == 2 && router_ids.size(0) == B,
                "router_ids must be [B, TOP_K]");
    const int64_t TOP_K = router_ids.size(1);

    check_tensor(x,            "x",            torch::kBFloat16,      {B, HIDDEN});
    check_tensor(w_gate,       "w_gate",       torch::kFloat8_e4m3fn, {E, INTER, HIDDEN});
    check_tensor(w_gate_scale, "w_gate_scale", torch::kFloat,         {E * INTER / 128, HIDDEN / 128});
    check_tensor(w_up,         "w_up",         torch::kFloat8_e4m3fn, {E, INTER, HIDDEN});
    check_tensor(w_up_scale,   "w_up_scale",   torch::kFloat,         {E * INTER / 128, HIDDEN / 128});
    check_tensor(router_ids,   "router_ids",   torch::kInt32,         {B, TOP_K});
    check_tensor(intermediate, "intermediate", torch::kBFloat16,      {B, TOP_K, INTER});

    auto s = make_stream_cfg();
    launch_gate_up_bf16_kernel<GateKernel>(x,
                                           w_gate,
                                           w_gate_scale,
                                           w_up,
                                           w_up_scale,
                                           router_ids,
                                           intermediate,
                                           static_cast<index_t>(B),
                                           static_cast<index_t>(HIDDEN),
                                           static_cast<index_t>(INTER),
                                           static_cast<index_t>(TOP_K),
                                           static_cast<index_t>(E),
                                           s);
}

void warp_decode_gate_up_bf16_base(const torch::Tensor& x,
                                   const torch::Tensor& w_gate,
                                   const torch::Tensor& w_gate_scale,
                                   const torch::Tensor& w_up,
                                   const torch::Tensor& w_up_scale,
                                   const torch::Tensor& router_ids,
                                   torch::Tensor& intermediate)
{
    warp_decode_gate_up_bf16_impl<GateUpKernelBF16Base>(
        x, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, intermediate);
}

void warp_decode_gate_up_bf16(const torch::Tensor& x,
                              const torch::Tensor& w_gate,
                              const torch::Tensor& w_gate_scale,
                              const torch::Tensor& w_up,
                              const torch::Tensor& w_up_scale,
                              const torch::Tensor& router_ids,
                              torch::Tensor& intermediate)
{
    warp_decode_gate_up_bf16_impl<GateUpKernelBF16Dot2>(
        x, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, intermediate);
}

void warp_decode_gate_up_bf16_pkf32(const torch::Tensor& x,
                                    const torch::Tensor& w_gate,
                                    const torch::Tensor& w_gate_scale,
                                    const torch::Tensor& w_up,
                                    const torch::Tensor& w_up_scale,
                                    const torch::Tensor& router_ids,
                                    torch::Tensor& intermediate)
{
    warp_decode_gate_up_bf16_impl<GateUpKernelBF16PkF32>(
        x, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, intermediate);
}

void warp_decode_gate_up_bf16_lds(const torch::Tensor& x,
                                  const torch::Tensor& w_gate,
                                  const torch::Tensor& w_gate_scale,
                                  const torch::Tensor& w_up,
                                  const torch::Tensor& w_up_scale,
                                  const torch::Tensor& router_ids,
                                  torch::Tensor& intermediate)
{
    warp_decode_gate_up_bf16_impl<GateUpKernelBF16Lds>(
        x, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, intermediate);
}

// ---------------------------------------------------------------------------
// warp_decode_down_reduce
// ---------------------------------------------------------------------------

template <typename DownKernelDefault, typename DownKernelFP8>
void warp_decode_down_reduce_impl(const torch::Tensor& intermediate,
                                  const torch::Tensor& w_down,
                                  const torch::Tensor& w_down_scale,
                                  const torch::Tensor& router_ids,
                                  const torch::Tensor& router_wts,
                                  torch::Tensor& y)
{
    TORCH_CHECK(intermediate.dim() == 3, "intermediate must be [B, TOP_K, INTER]");
    const int64_t B     = intermediate.size(0);
    const int64_t TOP_K = intermediate.size(1);
    const int64_t INTER = intermediate.size(2);

    TORCH_CHECK(w_down.dim() == 3, "w_down must be [E, HIDDEN, INTER]");
    const int64_t E      = w_down.size(0);
    const int64_t HIDDEN = w_down.size(1);
    TORCH_CHECK(w_down.size(2) == INTER, "w_down last dim must equal INTER");

    check_tensor(intermediate, "intermediate", torch::kBFloat16,      {B, TOP_K, INTER});
    check_tensor(w_down,       "w_down",       torch::kFloat8_e4m3fn, {E, HIDDEN, INTER});
    check_tensor(w_down_scale, "w_down_scale", torch::kFloat,         {E * HIDDEN / 128, INTER / 128});
    check_tensor(router_ids,   "router_ids",   torch::kInt32,         {B, TOP_K});
    check_tensor(router_wts,   "router_wts",   torch::kFloat,         {B, TOP_K});
    check_tensor(y,            "y",            torch::kBFloat16,      {B, HIDDEN});

    auto s = make_stream_cfg();
    const bool use_kvector_fp8 = (INTER % (kWaveSize * kVectorFP8)) == 0;
    if(use_kvector_fp8)
    {
        launch_down_reduce_kernel<DownKernelFP8>(intermediate,
                                                 w_down,
                                                 w_down_scale,
                                                 router_ids,
                                                 router_wts,
                                                 y,
                                                 static_cast<index_t>(B),
                                                 static_cast<index_t>(HIDDEN),
                                                 static_cast<index_t>(INTER),
                                                 static_cast<index_t>(TOP_K),
                                                 static_cast<index_t>(E),
                                                 s);
    }
    else
    {
        launch_down_reduce_kernel<DownKernelDefault>(intermediate,
                                                     w_down,
                                                     w_down_scale,
                                                     router_ids,
                                                     router_wts,
                                                     y,
                                                     static_cast<index_t>(B),
                                                     static_cast<index_t>(HIDDEN),
                                                     static_cast<index_t>(INTER),
                                                     static_cast<index_t>(TOP_K),
                                                     static_cast<index_t>(E),
                                                     s);
    }
}

void warp_decode_down_reduce_base(const torch::Tensor& intermediate,
                                  const torch::Tensor& w_down,
                                  const torch::Tensor& w_down_scale,
                                  const torch::Tensor& router_ids,
                                  const torch::Tensor& router_wts,
                                  torch::Tensor& y)
{
    warp_decode_down_reduce_impl<DownKernelDefaultBase, DownKernelFP8Base>(
        intermediate, w_down, w_down_scale, router_ids, router_wts, y);
}

void warp_decode_down_reduce(const torch::Tensor& intermediate,
                             const torch::Tensor& w_down,
                             const torch::Tensor& w_down_scale,
                             const torch::Tensor& router_ids,
                             const torch::Tensor& router_wts,
                             torch::Tensor& y)
{
    warp_decode_down_reduce_impl<DownKernelDefaultPkF32, DownKernelFP8PkF32>(
        intermediate, w_down, w_down_scale, router_ids, router_wts, y);
}

void warp_decode_down_reduce_dot2(const torch::Tensor& intermediate,
                                  const torch::Tensor& w_down,
                                  const torch::Tensor& w_down_scale,
                                  const torch::Tensor& router_ids,
                                  const torch::Tensor& router_wts,
                                  torch::Tensor& y)
{
    warp_decode_down_reduce_impl<DownKernelDefaultDot2, DownKernelFP8Dot2>(
        intermediate, w_down, w_down_scale, router_ids, router_wts, y);
}

void warp_decode_down_reduce_pkf32(const torch::Tensor& intermediate,
                                   const torch::Tensor& w_down,
                                   const torch::Tensor& w_down_scale,
                                   const torch::Tensor& router_ids,
                                   const torch::Tensor& router_wts,
                                   torch::Tensor& y)
{
    warp_decode_down_reduce_impl<DownKernelDefaultPkF32, DownKernelFP8PkF32>(
        intermediate, w_down, w_down_scale, router_ids, router_wts, y);
}

void warp_decode_down_reduce_lds(const torch::Tensor& intermediate,
                                 const torch::Tensor& w_down,
                                 const torch::Tensor& w_down_scale,
                                 const torch::Tensor& router_ids,
                                 const torch::Tensor& router_wts,
                                 torch::Tensor& y)
{
    warp_decode_down_reduce_impl<DownKernelDefaultLds, DownKernelFP8Lds>(
        intermediate, w_down, w_down_scale, router_ids, router_wts, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "CKTile warp-decode kernels for FP8 block-scale MoE decode benchmarks";
    m.def("warp_decode_gate_up_fp8",
          &warp_decode_gate_up_fp8,
          "Warp-decode gate+up dot2 default (fp8 activation, Block2D<1,128> x scale, "
          "Block2D<128,128> w scale, bf16 intermediate)",
          py::arg("x"),
          py::arg("x_scale"),
          py::arg("w_gate"),
          py::arg("w_gate_scale"),
          py::arg("w_up"),
          py::arg("w_up_scale"),
          py::arg("router_ids"),
          py::arg("intermediate"));
    m.def("warp_decode_gate_up_fp8_base",
          &warp_decode_gate_up_fp8_base,
          "Warp-decode gate+up baseline scalar-conversion variant (fp8 activation)",
          py::arg("x"),
          py::arg("x_scale"),
          py::arg("w_gate"),
          py::arg("w_gate_scale"),
          py::arg("w_up"),
          py::arg("w_up_scale"),
          py::arg("router_ids"),
          py::arg("intermediate"));
    m.def("warp_decode_gate_up_fp8_pkf32",
          &warp_decode_gate_up_fp8_pkf32,
          "Warp-decode gate+up packed-FP32 FMA variant (fp8 activation)",
          py::arg("x"),
          py::arg("x_scale"),
          py::arg("w_gate"),
          py::arg("w_gate_scale"),
          py::arg("w_up"),
          py::arg("w_up_scale"),
          py::arg("router_ids"),
          py::arg("intermediate"));
    m.def("warp_decode_gate_up_fp8_lds",
          &warp_decode_gate_up_fp8_lds,
          "Warp-decode gate+up dot2 plus multi-warp LDS-staged X variant (fp8 activation)",
          py::arg("x"),
          py::arg("x_scale"),
          py::arg("w_gate"),
          py::arg("w_gate_scale"),
          py::arg("w_up"),
          py::arg("w_up_scale"),
          py::arg("router_ids"),
          py::arg("intermediate"));
    m.def("warp_decode_gate_up_bf16",
          &warp_decode_gate_up_bf16,
          "Warp-decode gate+up dot2 default (bf16 activation, no x scale, Block2D<128,128> "
          "w scale, bf16 intermediate)",
          py::arg("x"),
          py::arg("w_gate"),
          py::arg("w_gate_scale"),
          py::arg("w_up"),
          py::arg("w_up_scale"),
          py::arg("router_ids"),
          py::arg("intermediate"));
    m.def("warp_decode_gate_up_bf16_base",
          &warp_decode_gate_up_bf16_base,
          "Warp-decode gate+up baseline scalar-conversion variant (bf16 activation)",
          py::arg("x"),
          py::arg("w_gate"),
          py::arg("w_gate_scale"),
          py::arg("w_up"),
          py::arg("w_up_scale"),
          py::arg("router_ids"),
          py::arg("intermediate"));
    m.def("warp_decode_gate_up_bf16_pkf32",
          &warp_decode_gate_up_bf16_pkf32,
          "Warp-decode gate+up packed-FP32 FMA variant (bf16 activation)",
          py::arg("x"),
          py::arg("w_gate"),
          py::arg("w_gate_scale"),
          py::arg("w_up"),
          py::arg("w_up_scale"),
          py::arg("router_ids"),
          py::arg("intermediate"));
    m.def("warp_decode_gate_up_bf16_lds",
          &warp_decode_gate_up_bf16_lds,
          "Warp-decode gate+up dot2 plus multi-warp LDS-staged X variant (bf16 activation)",
          py::arg("x"),
          py::arg("w_gate"),
          py::arg("w_gate_scale"),
          py::arg("w_up"),
          py::arg("w_up_scale"),
          py::arg("router_ids"),
          py::arg("intermediate"));
    m.def("warp_decode_down_reduce",
          &warp_decode_down_reduce,
          "Warp-decode down projection accurate packed-FP32 default with in-register weighted reduction "
          "(bf16 intermediate, fp8 weight, Block2D<128,128> w scale, bf16 out)",
          py::arg("intermediate"),
          py::arg("w_down"),
          py::arg("w_down_scale"),
          py::arg("router_ids"),
          py::arg("router_wts"),
          py::arg("y"));
    m.def("warp_decode_down_reduce_dot2",
          &warp_decode_down_reduce_dot2,
          "Warp-decode down projection BF16 dot2 attribution variant",
          py::arg("intermediate"),
          py::arg("w_down"),
          py::arg("w_down_scale"),
          py::arg("router_ids"),
          py::arg("router_wts"),
          py::arg("y"));
    m.def("warp_decode_down_reduce_base",
          &warp_decode_down_reduce_base,
          "Warp-decode down projection baseline scalar-conversion variant",
          py::arg("intermediate"),
          py::arg("w_down"),
          py::arg("w_down_scale"),
          py::arg("router_ids"),
          py::arg("router_wts"),
          py::arg("y"));
    m.def("warp_decode_down_reduce_pkf32",
          &warp_decode_down_reduce_pkf32,
          "Warp-decode down projection packed-FP32 FMA variant",
          py::arg("intermediate"),
          py::arg("w_down"),
          py::arg("w_down_scale"),
          py::arg("router_ids"),
          py::arg("router_wts"),
          py::arg("y"));
    m.def("warp_decode_down_reduce_lds",
          &warp_decode_down_reduce_lds,
          "Warp-decode down projection dot2 plus multi-warp LDS-staged intermediate variant",
          py::arg("intermediate"),
          py::arg("w_down"),
          py::arg("w_down_scale"),
          py::arg("router_ids"),
          py::arg("router_wts"),
          py::arg("y"));
}
