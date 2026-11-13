// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "engines/kernel_ingestor_engine/descriptors/conv/igemm/kernels/ConvIgemmArgs.h"

/**
 * @file ConvTf32Split.hpp
 * @brief The host half of the three-bf16 channel split (K1Split3Args, ConvIgemmCore.h's
 *        ConvTf32Split): the TF32 form of an engine that runs an fp32 graph as its bf16 problem
 *        with the reduction channels tripled.
 *
 * For a reduction a x b over channels, a' = [a hi | a hi' | a lo] and b' = [b hi | b lo | b hi']
 * along the (per-group) reduction channels give a' . b' = a hi b hi + a hi' b lo + a lo b hi'
 * (lo x lo dropped, about 2^-16 of each product), with hi' the high part with infinities zeroed
 * so an infinity reaches the result through hi x hi alone (inf x b = inf, NaN only where b is 0,
 * as in fp32). The engine then runs a bf16 config on the problem with 3x the reduction channels
 * and writes its output in fp32.
 *
 * TF32 is a permission (conv_igemm::tf32Permitted, HIP_KERNEL_PROVIDER_CONV_TF32): a config of
 * math 1 is admitted only for an fp32 graph under it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_tf32
{

/// The split kernel's entry point (in every engine source, under CONV_TF32_SPLIT3).
inline constexpr std::string_view SPLIT_ENTRY_POINT = "ConvTf32Split";
/// The compile definition that adds it.
inline constexpr std::string_view SPLIT_DEFINE = "CONV_TF32_SPLIT3";

/// Block patterns (2 bits a block): the operand whose hi meets the other's lo, and the other.
inline constexpr int PATTERN_A = 0 | (2 << 2) | (1 << 4); ///< [hi | hi' | lo]
inline constexpr int PATTERN_B = 0 | (1 << 2) | (2 << 4); ///< [hi | lo | hi']

/// A packed fp32 tensor viewed as [outer][groups][cg][inner], split to [outer][groups][3 cg][inner].
struct SplitTensor
{
    int64_t outer = 0;
    int64_t groups = 1;
    int64_t cg = 0;
    int64_t inner = 0;
    int pattern = PATTERN_A;

    int64_t elements() const
    {
        return outer * groups * cg * inner;
    }
    /// The split tensor's bytes (3x the elements, bf16), from a 256-byte boundary.
    size_t splitBytes() const
    {
        return (static_cast<size_t>(elements()) * 3 * 2 + 255) / 256 * 256;
    }
};

/// Both operands' layout in a workspace: a at 0, b after it, the engine's own workspace after
/// both (at `bytes`).
struct SplitPlan
{
    SplitTensor a;
    SplitTensor b;
    size_t bOffset = 0;
    size_t bytes = 0;
    unsigned int grid = 0;
};

inline SplitPlan splitPlanFor(const SplitTensor& a, const SplitTensor& b)
{
    SplitPlan plan;
    plan.a = a;
    plan.b = b;
    plan.bOffset = a.splitBytes();
    plan.bytes = plan.bOffset + b.splitBytes();
    const int64_t most = std::max(a.elements(), b.elements());
    plan.grid = static_cast<unsigned int>(
        std::min<int64_t>(std::max<int64_t>((most + 255) / 256, 1), int64_t{1} << 16));
    return plan;
}

/// The kernel arguments of @p plan for sources @p a and @p b and a workspace at @p workspace.
inline K1Split3Args
    splitArgsFor(const SplitPlan& plan, const void* a, const void* b, void* workspace)
{
    auto* base = static_cast<char*>(workspace);
    const auto tensor = [](const SplitTensor& t, const void* src, void* dst) {
        K1Split3Tensor k{};
        k.src = static_cast<const float*>(src);
        k.dst = static_cast<unsigned short*>(dst);
        k.outer = t.outer;
        k.inner = t.inner;
        k.groups = static_cast<int>(t.groups);
        k.cg = static_cast<int>(t.cg);
        k.pattern = t.pattern;
        return k;
    };
    K1Split3Args args{};
    args.t[0] = tensor(plan.a, a, base);
    args.t[1] = tensor(plan.b, b, base + plan.bOffset);
    return args;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_tf32

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
