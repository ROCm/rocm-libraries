// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_contraction/utils/tensor_descriptor_utils.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"

namespace ck_tile {

/// @brief E viewed as [M0, M1, N0, N1], with m = m0 * M1 + m1 and n = n0 * N1 + n1. Each
///        sub-dimension has its own stride; the stride of N1 is 1.
/// @par   Example: M0 = T, M1 = B, N0 = H, N1 = D with strides (D, T * D, B * T * D) store the
///        GEMM result C[t * B + b, h * D + d] to Y[h, b, t, d].
struct GemmEPermuteDesc
{
    index_t M0, M1, N0, N1;
    index_t stride_M0, stride_M1, stride_N0;
};

struct GemmEPermuteHostArgs : public GemmHostArgs
{
    // stride_E = N: the base size checks then see the dense [M, N] footprint.
    CK_TILE_HOST GemmEPermuteHostArgs(const void* a_ptr_,
                                      const void* b_ptr_,
                                      void* e_ptr_,
                                      index_t k_batch_,
                                      index_t K_,
                                      index_t stride_A_,
                                      index_t stride_B_,
                                      const GemmEPermuteDesc& e_permute_)
        : GemmHostArgs(a_ptr_,
                       b_ptr_,
                       e_ptr_,
                       k_batch_,
                       e_permute_.M0 * e_permute_.M1,
                       e_permute_.N0 * e_permute_.N1,
                       K_,
                       stride_A_,
                       stride_B_,
                       e_permute_.N0 * e_permute_.N1),
          e_permute(e_permute_)
    {
    }

    GemmEPermuteDesc e_permute;
};

/// @brief The [M, N] view of the permuted E, built on the host so that the merge's magic numbers
///        are computed once rather than per workgroup.
template <index_t VecC>
CK_TILE_HOST auto make_gemm_e_permute_descriptor(const GemmEPermuteDesc& p)
{
    const auto mn = TensorDescriptorUtils<0, 2, 2, 1, 1, 1, VecC>::Make_E_GridDescriptor_M_N(
        {p.M0, p.M1, p.N0, p.N1}, {p.stride_M0, p.stride_M1, p.stride_N0, 1});
    // A row past M would fold back into E through the merge (m = M -> m0 = M0, m1 = 0), so M is
    // masked. N needs no mask: MakeCBlockWindows pads an N tail, or IsSupportedArgument rejects it.
    return transform_tensor_descriptor(mn,
                                       make_tuple(make_right_pad_transform(p.M0 * p.M1, index_t{0}),
                                                  make_pass_through_transform(p.N0 * p.N1)),
                                       make_tuple(sequence<0>{}, sequence<1>{}),
                                       make_tuple(sequence<0>{}, sequence<1>{}));
}

/// @brief GEMM whose E tile is stored through the [M0, M1, N0, N1] view of GemmEPermuteDesc: a
///        GEMM and a permute of its output in one kernel, with no dense C in memory.
/// @note  v1: k_batch == 1, no D tensors, row-major E, no TDM epilogue, E span below 2 GB, the
///        rows a masked M (N) tail folds into past M0 (N0) included.
template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GemmEPermuteKernel
    : public UniversalGemmKernel<
          TilePartitioner_,
          GemmPipeline_,
          EpiloguePipeline_,
          GemmEPermuteKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>>
{
    using Base =
        UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_, GemmEPermuteKernel>;
    using typename Base::EDataType;

    static constexpr index_t VecC = Base::EpiloguePipeline::GetVectorSizeC();
    using EDesc = decltype(make_gemm_e_permute_descriptor<VecC>(GemmEPermuteDesc{}));

    struct KernelArgs : Base::KernelArgs
    {
        GemmEPermuteDesc e_permute; // read by IsSupportedArgument only
        EDesc e_desc;
    };

    static_assert(std::is_same_v<typename Base::CLayout, tensor_layout::gemm::RowMajor>);
    static_assert(Base::NumATensor == 1 && Base::NumBTensor == 1 && Base::NumDTensor == 0);
    static_assert(!is_tdm_epilogue_v<typename Base::EpiloguePipeline>,
                  "a TDM store needs an affine 2D descriptor");
    static_assert(!Base::has_tile_partitioner_output_offset);
    static_assert(!Base::UseLargeTensorGlobalLoad());

    CK_TILE_HOST static KernelArgs MakeKernelArgs(const GemmEPermuteHostArgs& h)
    {
        // The merge's magic numbers need lengths >= 1; IsSupportedArgument rejects the others.
        const auto& p         = h.e_permute;
        const bool positive   = std::min({p.M0, p.M1, p.N0, p.N1}) >= 1;
        const auto desc_input = positive ? p : GemmEPermuteDesc{1, 1, 1, 1, 1, 1, 1};
        return KernelArgs{
            GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>::MakeKernelArgs(h),
            p,
            make_gemm_e_permute_descriptor<VecC>(desc_input)};
    }

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        const auto reject = [](const char* reason) {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("GemmEPermute: ", reason);
            }
            return false;
        };
        const auto& p = kargs.e_permute;
        if(kargs.k_batch != 1)
        {
            return reject("split-K (k_batch != 1) is not supported");
        }
        if(std::min({p.M0, p.M1, p.N0, p.N1}) < 1 ||
           long_index_t{p.M0} * p.M1 != long_index_t{kargs.M} ||
           long_index_t{p.N0} * p.N1 != long_index_t{kargs.N})
        {
            return reject("needs M0, M1, N0, N1 >= 1 with M0 * M1 == M and N0 * N1 == N");
        }
        // A vector store must not straddle an n0 boundary.
        if(p.N1 % VecC != 0)
        {
            return reject("N1 must be a multiple of the C vector size");
        }
        // No two (m, n) may share an address: sorted by stride, each dimension of length > 1 must
        // start past the extent of the previous one.
        std::array<std::array<long_index_t, 2>, 4> dims{
            {{p.stride_M0, p.M0}, {p.stride_M1, p.M1}, {p.stride_N0, p.N0}, {1, p.N1}}};
        const auto end =
            std::partition(dims.begin(), dims.end(), [](const auto& d) { return d[1] > 1; });
        std::sort(dims.begin(), end);
        long_index_t span = 1;
        for(auto d = dims.begin(); d != end; ++d)
        {
            if((*d)[0] < 1 || (d != dims.begin() && (*d)[0] < (*(d - 1))[0] * (*(d - 1))[1]))
            {
                return reject("strides must be positive and must not make E overlap itself");
            }
            span += ((*d)[1] - 1) * (*d)[0];
        }
        // A tile past M (or past N under kPadN) is masked, but its rows still fold to m0 >= M0
        // (n0 >= N0); `tail` counts the sub-indices they reach past the last valid one. The masked
        // buffer store is dropped by adding 2^31 to its byte offset, which only works while that
        // offset is itself in [0, 2^31), so the folded rows count as span.
        const auto tail =
            [](long_index_t len, long_index_t per_block, index_t inner, index_t outer) {
                return (integer_least_multiple(len, per_block) - 1) / inner - (outer - 1);
            };
        using Partitioner          = typename Base::TilePartitioner;
        const long_index_t m0_tail = tail(kargs.M, Partitioner::MPerBlock, p.M1, p.M0);
        const long_index_t n0_tail =
            Base::GemmPipeline::kPadN ? tail(kargs.N, Partitioner::NPerBlock, p.N1, p.N0) : 0;
        if((m0_tail > 0 && p.stride_M0 < 0) || (n0_tail > 0 && p.stride_N0 < 0))
        {
            return reject("the stride of a dimension that masked tail rows fold into must be >= 0");
        }
        // 32-bit buffer offsets and index_t descriptor math.
        if(span + m0_tail * p.stride_M0 + n0_tail * p.stride_N0 >=
           (long_index_t{1} << 31) / long_index_t{sizeof(EDataType)})
        {
            return reject("the E span, masked tail rows included, must be below 2 GB");
        }
        return Base::IsSupportedArgument(kargs);
    }

    CK_TILE_HOST static auto GetName() -> const std::string
    {
        return Base::GetName() + "_e_permute";
    }

    /// Hides the base overload that RunGemm calls through SelfType.
    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto MakeCBlockWindows(EDataType* e_ptr,
                                                 const KernelArgs& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        return Base::template MakeCBlockWindows<DstInMemOp>(e_ptr, i_m, i_n, kargs.e_desc);
    }
};

} // namespace ck_tile
