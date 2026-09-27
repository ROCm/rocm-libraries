// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/epilogue_type.hpp"
#include "ck/utility/type_convert.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// dW[k,c] = sum_n sum_s dY[n,k,s] * x[n,c,s]. Each CTA owns a
// [split,image,k,c] FP32 partial; the second kernel reduces those partials
// before the one and only BF16 conversion. Neither kernel clears an output or
// uses atomics. A and B are views of the original packed NCHW tensors.
using NchwPointwiseBf16WmmaGemm =
    DeviceBatchedGemm_Wmma_CShuffleV3<tensor_layout::gemm::RowMajor,
                                      tensor_layout::gemm::ColumnMajor,
                                      tensor_layout::gemm::RowMajor,
                                      bhalf_t,
                                      bhalf_t,
                                      float,
                                      float,
                                      float,
                                      element_wise::PassThrough,
                                      element_wise::PassThrough,
                                      element_wise::PassThrough,
                                      GemmSpecialization::MNKPadding,
                                      64,
                                      32,
                                      64,
                                      64,
                                      8,
                                      8,
                                      16,
                                      16,
                                      2,
                                      2,
                                      Sequence<4, 16, 1>,
                                      Sequence<1, 0, 2>,
                                      Sequence<1, 0, 2>,
                                      2,
                                      8,
                                      8,
                                      false,
                                      Sequence<4, 16, 1>,
                                      Sequence<1, 0, 2>,
                                      Sequence<1, 0, 2>,
                                      2,
                                      8,
                                      8,
                                      false,
                                      1,
                                      1,
                                      Sequence<1, 16, 1, 4>,
                                      8,
                                      BlockGemmPipelineScheduler::Intrawave,
                                      BlockGemmPipelineVersion::v1,
                                      bhalf_t,
                                      bhalf_t>;
using NchwPointwiseBf16WmmaGridwise = NchwPointwiseBf16WmmaGemm::GridwiseGemm;

// The public batched GEMM invoker clears C and uses AtomicAdd for KBatch>1.
// Here each (split,image) has a distinct FP32 C matrix instead.
static __global__ void
kernel_nchw_pointwise_bf16_wmma_partials(NchwPointwiseBf16WmmaGridwise::Argument karg,
                                         index_t batch)
{
#if defined(__gfx12__)
    using Gridwise               = NchwPointwiseBf16WmmaGridwise;
    constexpr auto epilogue_type = Gridwise::IsBWaveTransferApplicable && Gridwise::UseDirectStore
                                       ? EpilogueType::DirectStore
                                       : EpilogueType::CShuffle;
    using Epilogue               = get_epilogue_t<epilogue_type, Gridwise>;
    constexpr index_t nbytes     = Gridwise::GetSharedMemoryNumberOfByte<Epilogue>();
    __shared__ char shared[nbytes];

    const long_index_t image = blockIdx.y;
    const long_index_t split = blockIdx.z;
    // SplitKBatchOffset shortens CTA-local K but keeps KRead and the input
    // offsets based on the full spatial extent.
    const auto offset = Gridwise::SplitKBatchOffset(karg, blockIdx.z);
    Gridwise::AsGridPointer a_shift;
    Gridwise::BsGridPointer b_shift;
    a_shift(Number<0>{}) = karg.p_as_grid[Number<0>{}] + image * karg.M * karg.StrideAs[0] +
                           offset.a_k_split_offset[0];
    b_shift(Number<0>{}) = karg.p_bs_grid[Number<0>{}] + image * karg.N * karg.StrideBs[0] +
                           offset.b_k_split_offset[0];
    float* partial = karg.p_e_grid + (split * batch + image) * karg.M * karg.N;
    Epilogue epilogue_args{};
    Gridwise::Run<true, InMemoryDataOperationEnum::Set, TailNumber::Full>(a_shift,
                                                                          b_shift,
                                                                          karg.p_ds_grid,
                                                                          partial,
                                                                          shared,
                                                                          karg,
                                                                          karg.a_element_op,
                                                                          karg.b_element_op,
                                                                          karg.cde_element_op,
                                                                          epilogue_args);
#else
    (void)karg;
    (void)batch;
#endif
}

// 16 weight lanes x 16 split lanes: only the first eight splits are active.
// Each lane sums images in order, then the eight splits form a fixed FP32 tree.
static __global__ void kernel_nchw_pointwise_bf16_wmma_finalize(const float* partial,
                                                                bhalf_t* weight,
                                                                index_t batch,
                                                                index_t weights)
{
    constexpr index_t Splits          = 8;
    constexpr index_t WeightsPerBlock = 16;
    __shared__ float reduction[Splits][WeightsPerBlock];
    const index_t lane  = threadIdx.x;
    const index_t split = lane / WeightsPerBlock;
    const index_t w     = blockIdx.x * WeightsPerBlock + lane % WeightsPerBlock;
    if(split < Splits)
    {
        float sum = 0;
        if(w < weights)
            for(index_t image = 0; image < batch; ++image)
                sum += partial[(static_cast<long_index_t>(split) * batch + image) * weights + w];
        reduction[split][lane % WeightsPerBlock] = sum;
    }
    __syncthreads();
    for(index_t stride = Splits / 2; stride > 0; stride /= 2)
    {
        if(split < stride)
            reduction[split][lane % WeightsPerBlock] +=
                reduction[split + stride][lane % WeightsPerBlock];
        __syncthreads();
    }
    if(split == 0 && w < weights)
        weight[w] = type_convert<bhalf_t>(reduction[0][lane % WeightsPerBlock]);
}

struct DeviceGroupedConvBwdWeightNchwPointwiseBf16Wmma final
    : DeviceGroupedConvBwdWeight<2,
                                 tensor_layout::convolution::NGCHW,
                                 tensor_layout::convolution::GKCYX,
                                 tensor_layout::convolution::NGKHW,
                                 bhalf_t,
                                 bhalf_t,
                                 bhalf_t,
                                 element_wise::PassThrough,
                                 element_wise::PassThrough,
                                 element_wise::PassThrough>
{
    static constexpr index_t Splits = 8;

    struct Argument : BaseArgument, ArgumentSplitK
    {
        const bhalf_t* in      = nullptr;
        bhalf_t* wei           = nullptr;
        const bhalf_t* out     = nullptr;
        bool valid             = false;
        index_t batch          = 0;
        index_t cin            = 0;
        index_t kout           = 0;
        index_t spatial        = 0;
        index_t grid_x         = 0;
        index_t weights        = 0;
        size_t in_bytes        = 0;
        size_t out_bytes       = 0;
        size_t wei_bytes       = 0;
        size_t workspace_bytes = 0;
    };

    static bool
    CheckedMultiply(long_index_t x, long_index_t y, long_index_t limit, long_index_t& result)
    {
        if(x <= 0 || y <= 0 || x > limit / y)
            return false;
        result = x * y;
        return true;
    }

    // Difference comparison cannot overflow even if a caller passes an address
    // near the end of uintptr_t. Sizes are already checked against PTRDIFF_MAX.
    static bool Disjoint(const void* lhs, size_t lhs_bytes, const void* rhs, size_t rhs_bytes)
    {
        const auto a = reinterpret_cast<std::uintptr_t>(lhs);
        const auto b = reinterpret_cast<std::uintptr_t>(rhs);
        return a <= b ? b - a >= lhs_bytes : a - b >= rhs_bytes;
    }

    static bool PointersSupported(const Argument& a, bool require_real)
    {
        if(!a.in || !a.wei || !a.out)
            return !require_real && !a.in && !a.wei && !a.out;
        if(reinterpret_cast<std::uintptr_t>(a.in) % (8 * sizeof(bhalf_t)) != 0 ||
           reinterpret_cast<std::uintptr_t>(a.out) % (8 * sizeof(bhalf_t)) != 0 ||
           reinterpret_cast<std::uintptr_t>(a.wei) % alignof(bhalf_t) != 0 ||
           !Disjoint(a.in, a.in_bytes, a.out, a.out_bytes) ||
           !Disjoint(a.in, a.in_bytes, a.wei, a.wei_bytes) ||
           !Disjoint(a.out, a.out_bytes, a.wei, a.wei_bytes))
            return false;
        if(a.p_workspace_)
        {
            if(reinterpret_cast<std::uintptr_t>(a.p_workspace_) % 256 != 0 ||
               !Disjoint(a.p_workspace_, a.workspace_bytes, a.in, a.in_bytes) ||
               !Disjoint(a.p_workspace_, a.workspace_bytes, a.out, a.out_bytes) ||
               !Disjoint(a.p_workspace_, a.workspace_bytes, a.wei, a.wei_bytes))
                return false;
        }
        return true;
    }

    template <typename Index>
    static bool Validate(const std::array<Index, 5>& il,
                         const std::array<Index, 5>& is,
                         const std::array<Index, 5>& wl,
                         const std::array<Index, 5>& ws,
                         const std::array<Index, 5>& ol,
                         const std::array<Index, 5>& os,
                         const std::array<Index, 2>& fs,
                         const std::array<Index, 2>& fd,
                         const std::array<Index, 2>& lp,
                         const std::array<Index, 2>& rp,
                         index_t split,
                         Argument& a)
    {
        if(!is_gfx125_supported() || (split != -1 && split != 0 && split != 1))
            return false;
        constexpr long_index_t MaxIndex = std::numeric_limits<index_t>::max();
        constexpr long_index_t MaxBytes = std::numeric_limits<std::ptrdiff_t>::max();
        constexpr long_index_t MaxBf16  = MaxBytes / sizeof(bhalf_t);
        constexpr long_index_t MaxFloat = MaxBytes / sizeof(float);
        for(index_t d = 0; d < 5; ++d)
        {
            if(il[d] <= 0 || il[d] > MaxIndex || wl[d] <= 0 || wl[d] > MaxIndex || ol[d] <= 0 ||
               ol[d] > MaxIndex)
                return false;
        }
        const long_index_t batch = il[1];
        const long_index_t cin   = il[2];
        const long_index_t kout  = ol[2];
        if(il[0] != 1 || ol[0] != 1 || wl[0] != 1 || ol[1] != batch || batch > 42 ||
           wl[1] != kout || wl[2] != cin || wl[3] != 1 || wl[4] != 1 || cin < 8 || cin > 24 ||
           cin % 8 != 0 || kout < 16 || kout > 128 || kout % 16 != 0 || il[3] != ol[3] ||
           il[4] != ol[4] || fs[0] != 1 || fs[1] != 1 || fd[0] != 1 || fd[1] != 1 || lp[0] != 0 ||
           lp[1] != 0 || rp[0] != 0 || rp[1] != 0)
            return false;

        long_index_t spatial, in_image, out_image, in_count, out_count;
        long_index_t weights, partials, grid_x, ctas;
        if(!CheckedMultiply(il[3], il[4], MaxIndex, spatial) || spatial < 1024 || spatial > 76800 ||
           spatial % (Splits * 64) != 0 || !CheckedMultiply(cin, spatial, MaxIndex, in_image) ||
           !CheckedMultiply(kout, spatial, MaxIndex, out_image) ||
           !CheckedMultiply(batch, in_image, MaxIndex, in_count) ||
           !CheckedMultiply(batch, out_image, MaxIndex, out_count) ||
           !CheckedMultiply(kout, cin, MaxIndex, weights) ||
           !CheckedMultiply(Splits * batch, weights, MaxIndex, partials) ||
           !CheckedMultiply((kout + 31) / 32, (cin + 63) / 64, MaxIndex, grid_x) ||
           !CheckedMultiply(grid_x, batch * Splits, MaxIndex, ctas))
            return false;
        if(is[0] != in_count || is[1] != in_image || is[2] != spatial || is[3] != il[4] ||
           is[4] != 1 || os[0] != out_count || os[1] != out_image || os[2] != spatial ||
           os[3] != ol[4] || os[4] != 1 || ws[0] != weights || ws[1] != cin || ws[2] != 1 ||
           ws[3] != 1 || ws[4] != 1 || in_count > MaxBf16 || out_count > MaxBf16 ||
           weights > MaxBf16 || partials > MaxFloat ||
           grid_x > std::numeric_limits<std::uint32_t>::max() ||
           batch > std::numeric_limits<std::uint32_t>::max() ||
           ctas > std::numeric_limits<std::uint32_t>::max())
            return false;

        // Gridwise uses the original S for pointer offsets, but KRead=S/8
        // for each independently owned GEMM. Check the exact selected tail.
        NchwPointwiseBf16WmmaGridwise::Argument karg(
            std::array<const void*, 1>{nullptr},
            std::array<const void*, 1>{nullptr},
            std::array<const void*, 0>{},
            nullptr,
            static_cast<index_t>(kout),
            static_cast<index_t>(cin),
            static_cast<index_t>(spatial),
            std::array<index_t, 1>{static_cast<index_t>(spatial)},
            std::array<index_t, 1>{static_cast<index_t>(spatial)},
            std::array<index_t, 0>{},
            static_cast<index_t>(cin),
            Splits,
            element_wise::PassThrough{},
            element_wise::PassThrough{},
            element_wise::PassThrough{},
            true);
        if(karg.KRead != spatial / Splits || !NchwPointwiseBf16WmmaGridwise::CheckValidity(karg) ||
           !NchwPointwiseBf16WmmaGridwise::CalculateHasMainKBlockLoop(karg.KRead) ||
           NchwPointwiseBf16WmmaGridwise::CalculateKBlockLoopTailNum(karg.KRead) !=
               TailNumber::Full)
            return false;

        const auto bytes = static_cast<size_t>(partials) * sizeof(float);
        if(bytes > static_cast<size_t>(MaxBytes - 255))
            return false;
        a.batch           = static_cast<index_t>(batch);
        a.cin             = static_cast<index_t>(cin);
        a.kout            = static_cast<index_t>(kout);
        a.spatial         = static_cast<index_t>(spatial);
        a.grid_x          = static_cast<index_t>(grid_x);
        a.weights         = static_cast<index_t>(weights);
        a.in_bytes        = static_cast<size_t>(in_count) * sizeof(bhalf_t);
        a.out_bytes       = static_cast<size_t>(out_count) * sizeof(bhalf_t);
        a.wei_bytes       = static_cast<size_t>(weights) * sizeof(bhalf_t);
        a.workspace_bytes = (bytes + 255) & ~size_t{255};
        return true;
    }

    template <typename Index>
    static std::unique_ptr<BaseArgument> MakeArgument(const void* in,
                                                      void* wei,
                                                      const void* out,
                                                      const std::array<Index, 5>& il,
                                                      const std::array<Index, 5>& is,
                                                      const std::array<Index, 5>& wl,
                                                      const std::array<Index, 5>& ws,
                                                      const std::array<Index, 5>& ol,
                                                      const std::array<Index, 5>& os,
                                                      const std::array<Index, 2>& fs,
                                                      const std::array<Index, 2>& fd,
                                                      const std::array<Index, 2>& lp,
                                                      const std::array<Index, 2>& rp,
                                                      index_t split)
    {
        auto a      = std::make_unique<Argument>();
        a->in       = static_cast<const bhalf_t*>(in);
        a->wei      = static_cast<bhalf_t*>(wei);
        a->out      = static_cast<const bhalf_t*>(out);
        a->k_batch_ = 1; // Public split -1/0/1 maps to eight private disjoint P slices.
        a->valid    = Validate(il, is, wl, ws, ol, os, fs, fd, lp, rp, split, *a);
        return a;
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* in,
                                                      void* wei,
                                                      const void* out,
                                                      const std::array<index_t, 5>& il,
                                                      const std::array<index_t, 5>& is,
                                                      const std::array<index_t, 5>& wl,
                                                      const std::array<index_t, 5>& ws,
                                                      const std::array<index_t, 5>& ol,
                                                      const std::array<index_t, 5>& os,
                                                      const std::array<index_t, 2>& fs,
                                                      const std::array<index_t, 2>& fd,
                                                      const std::array<index_t, 2>& lp,
                                                      const std::array<index_t, 2>& rp,
                                                      element_wise::PassThrough,
                                                      element_wise::PassThrough,
                                                      element_wise::PassThrough,
                                                      index_t split) override
    {
        return MakeArgument(in, wei, out, il, is, wl, ws, ol, os, fs, fd, lp, rp, split);
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* in,
                                                      void* wei,
                                                      const void* out,
                                                      const std::array<long_index_t, 5>& il,
                                                      const std::array<long_index_t, 5>& is,
                                                      const std::array<long_index_t, 5>& wl,
                                                      const std::array<long_index_t, 5>& ws,
                                                      const std::array<long_index_t, 5>& ol,
                                                      const std::array<long_index_t, 5>& os,
                                                      const std::array<long_index_t, 2>& fs,
                                                      const std::array<long_index_t, 2>& fd,
                                                      const std::array<long_index_t, 2>& lp,
                                                      const std::array<long_index_t, 2>& rp,
                                                      element_wise::PassThrough,
                                                      element_wise::PassThrough,
                                                      element_wise::PassThrough,
                                                      index_t split) override
    {
        return MakeArgument(in, wei, out, il, is, wl, ws, ol, os, fs, fd, lp, rp, split);
    }

    struct Invoker : BaseInvoker
    {
        float Run(const BaseArgument* base, const StreamConfig& stream = StreamConfig{}) override
        {
            const auto* a = dynamic_cast<const Argument*>(base);
            if(!a || !a->valid || !PointersSupported(*a, true) || !a->p_workspace_)
                throw std::runtime_error(
                    "Unsupported native NCHW BF16 pointwise WRW argument or scratch");

            NchwPointwiseBf16WmmaGridwise::Argument karg(std::array<const void*, 1>{a->out},
                                                         std::array<const void*, 1>{a->in},
                                                         std::array<const void*, 0>{},
                                                         static_cast<float*>(a->p_workspace_),
                                                         a->kout,
                                                         a->cin,
                                                         a->spatial,
                                                         std::array<index_t, 1>{a->spatial},
                                                         std::array<index_t, 1>{a->spatial},
                                                         std::array<index_t, 0>{},
                                                         a->cin,
                                                         Splits,
                                                         element_wise::PassThrough{},
                                                         element_wise::PassThrough{},
                                                         element_wise::PassThrough{},
                                                         true);
            const auto stage1_ms =
                launch_and_time_kernel(stream,
                                       kernel_nchw_pointwise_bf16_wmma_partials,
                                       dim3(static_cast<std::uint32_t>(a->grid_x),
                                            static_cast<std::uint32_t>(a->batch),
                                            Splits),
                                       dim3(64),
                                       0,
                                       karg,
                                       a->batch);
            const auto stage2_ms =
                launch_and_time_kernel(stream,
                                       kernel_nchw_pointwise_bf16_wmma_finalize,
                                       dim3(static_cast<std::uint32_t>((a->weights + 15) / 16)),
                                       dim3(256),
                                       0,
                                       static_cast<const float*>(a->p_workspace_),
                                       a->wei,
                                       a->batch,
                                       a->weights);
            return stage1_ms + stage2_ms;
        }
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    bool IsSupportedArgument(const BaseArgument* base) override
    {
        const auto* a = dynamic_cast<const Argument*>(base);
        return a && a->valid && PointersSupported(*a, false);
    }

    size_t GetWorkSpaceSize(const BaseArgument* base) const override
    {
        const auto* a = dynamic_cast<const Argument*>(base);
        return a && a->valid ? a->workspace_bytes : 0;
    }

    void SetWorkSpacePointer(BaseArgument* base,
                             void* workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto* a = dynamic_cast<Argument*>(base);
        if(!a)
            throw std::runtime_error("Invalid native NCHW BF16 pointwise WRW argument");
        a->p_workspace_ = workspace;
    }

    std::string GetTypeString() const override
    {
        return "DeviceGroupedConvBwdWeightNchwPointwiseBf16Wmma<8,32,64,64>";
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
