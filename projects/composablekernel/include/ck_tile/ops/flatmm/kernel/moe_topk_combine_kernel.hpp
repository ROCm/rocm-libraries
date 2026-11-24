// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Combines the per-slot rows of an MoE down projection:
//
//   out[t][n] = sum_{s < topk} weight[t][s] * p[t * topk + s][n]
//
// p is the output of MoeFlatmmKernel with MoeFlatmmKind::kFFN_gemm2_per_slot, one row per
// (token, slot); weight is the router's top-k weights in token order, [num_tokens, topk]. The sum
// is taken in fp32 in slot order, so the result is deterministic. Each thread produces
// kVectorSize consecutive outputs of one token.
template <typename PDataType_,
          typename ODataType_,
          index_t kBlockSize_  = 256,
          index_t kVectorSize_ = 4>
struct MoeTopkCombineKernel
{
    using PDataType = remove_cvref_t<PDataType_>;
    using ODataType = remove_cvref_t<ODataType_>;

    static constexpr index_t kBlockSize  = kBlockSize_;
    static constexpr index_t kVectorSize = kVectorSize_;

    static_assert(kVectorSize * sizeof(PDataType) % sizeof(uint32_t) == 0 &&
                      kVectorSize * sizeof(ODataType) % sizeof(uint32_t) == 0,
                  "a vector must be a whole number of dwords");

    struct Kargs
    {
        const void* p_ptr;            // [num_tokens * topk, stride_p]
        const float* topk_weight_ptr; // [num_tokens, topk]
        void* o_ptr;                  // [num_tokens, stride_o]
        index_t num_tokens;
        index_t topk;
        index_t n;
        index_t stride_p;
        index_t stride_o;
    };

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* p_ptr,
                                                  const float* topk_weight_ptr,
                                                  void* o_ptr,
                                                  index_t num_tokens,
                                                  index_t topk,
                                                  index_t n,
                                                  index_t stride_p,
                                                  index_t stride_o)
    {
        return Kargs{p_ptr, topk_weight_ptr, o_ptr, num_tokens, topk, n, stride_p, stride_o};
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static auto GridSize(const Kargs& kargs)
    {
        const long_index_t vectors = long_index_t(kargs.num_tokens) * (kargs.n / kVectorSize);
        return dim3(static_cast<uint32_t>((vectors + kBlockSize - 1) / kBlockSize));
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Kargs& kargs)
    {
        return kargs.topk > 0 && kargs.n % kVectorSize == 0 && kargs.stride_p % kVectorSize == 0 &&
               kargs.stride_o % kVectorSize == 0 && kargs.stride_p >= kargs.n &&
               kargs.stride_o >= kargs.n;
    }

    [[nodiscard]] CK_TILE_HOST static std::string GetName()
    {
        return "moe_topk_combine_" + std::to_string(kBlockSize) + "x" + std::to_string(kVectorSize);
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using PRaw = ext_vector_t<uint32_t, kVectorSize * sizeof(PDataType) / sizeof(uint32_t)>;
        using ORaw = ext_vector_t<uint32_t, kVectorSize * sizeof(ODataType) / sizeof(uint32_t)>;

        const index_t vectors_per_row = kargs.n / kVectorSize;
        const long_index_t i          = long_index_t(blockIdx.x) * kBlockSize + threadIdx.x;
        if(i >= long_index_t(kargs.num_tokens) * vectors_per_row)
            return;
        const index_t t = static_cast<index_t>(i / vectors_per_row);
        const index_t n = static_cast<index_t>(i % vectors_per_row) * kVectorSize;

        const auto* p = static_cast<const PDataType*>(kargs.p_ptr);
        array<float, kVectorSize> acc;
        static_for<0, kVectorSize, 1>{}([&](auto j) { acc(j) = 0.f; });
        for(index_t s = 0; s < kargs.topk; ++s)
        {
            const float w = kargs.topk_weight_ptr[long_index_t(t) * kargs.topk + s];
            // Rows and n are multiples of kVectorSize elements: one aligned vector load.
            PRaw raw;
            __builtin_memcpy(
                &raw,
                __builtin_assume_aligned(
                    p + (long_index_t(t) * kargs.topk + s) * kargs.stride_p + n, sizeof(PRaw)),
                sizeof(PRaw));
            const auto v = bit_cast<array<PDataType, kVectorSize>>(raw);
            static_for<0, kVectorSize, 1>{}(
                [&](auto j) { acc(j) += w * type_convert<float>(v[j]); });
        }

        array<ODataType, kVectorSize> out;
        static_for<0, kVectorSize, 1>{}([&](auto j) { out(j) = type_convert<ODataType>(acc[j]); });
        const ORaw packed = bit_cast<ORaw>(out);
        __builtin_memcpy(__builtin_assume_aligned(static_cast<ODataType*>(kargs.o_ptr) +
                                                      long_index_t(t) * kargs.stride_o + n,
                                                  sizeof(ORaw)),
                         &packed,
                         sizeof(ORaw));
    }
};

} // namespace ck_tile
