// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/igemm/kernels/ConvIgemmArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_igemm
{

/// A strided copy into or out of the workspace: the operand as the convolution reads it, where
/// the graph's own layout differs. The device pointers are filled at launch; the graph-side
/// pointer is advanced by `graphOffset` elements (a reflected filter starts at its far end).
struct Staging
{
    bool needed = false;
    /// Packed channels-first activations move by a tiled transpose; anything else by the copy.
    bool transpose = false;
    ConvIgemmTransposeArgs transposeArgs{};
    ConvIgemmCopyArgs copy{};
    int64_t graphOffset = 0;
    size_t workspaceOffset = 0;
    size_t bytes = 0;
};

/// A forward or backward-data convolution as the implicit-GEMM kernels run it.
struct IgemmProblem
{
    conv_naive::ConvNaiveProblem base;
    conv_naive::ConvDirection direction = conv_naive::ConvDirection::FORWARD;
    /// The kernels' argument block; pointers are filled at launch.
    K1Args args{};
    int64_t xUid = 0;
    int64_t wUid = 0;
    int64_t yUid = 0;
    int elementBytes = 0;
    /// x and y channels-last (N D H W C), the filter K T R S C/g (forward) or transposed to
    /// [g][T R S][C/g][K/g] (backward data, always staged).
    Staging x;
    Staging w;
    Staging y;
    size_t workspaceBytes = 0;
    /// Rows of the GEMM per launch column: forward m; backward data the largest stride phase.
    long long rows = 0;
    long long phases = 1;
    /// Bytes one block's rows can span, from its first image: decides the 32-bit buffer loads.
    long long blockSpanBytes = 0;
};

/**
 * @brief The convolution @p context's graph describes in @p direction, or nullopt when the
 *        implicit-GEMM kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: every extent, padding, stride, dilation and
 * channel count within 32 bits, and grid dimensions within the launch limits for the smallest
 * tile. Any operand layout is served: those that are not what the kernels read are staged
 * through the workspace.
 */
std::optional<IgemmProblem>
    parseIgemmProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                      conv_naive::ConvDirection direction);

/// One matrix-core atom (or "fma") of the catalog: the kernel template's name, the waves'
/// width, and whether the device runs it.
struct AtomInfo
{
    std::string_view name;
    std::string_view kernelType; ///< K1Atom* type name; empty for the FMA kernel
    int wave = 0; ///< 64 MFMA, 32 WMMA, 0 FMA (block from `threads`)
    bool tf32 = false; ///< Rounds fp32 operands (xf32): served only with TF32 permission
};

/// The atom @p name names, or nullopt for an unknown one.
std::optional<AtomInfo> atomNamed(std::string_view name);

/// The element type @p atom multiplies: "f16", "bf16", "f32", "any" (FMA), or empty if unknown.
std::string_view atomElement(std::string_view atom);

/// "f32", "f16" or "bf16" for the convolution element types, empty otherwise.
std::string_view dataElement(hipdnn_flatbuffers_sdk::data_objects::DataType dataType);

/// Whether the device @p gcnArchName (suffix intact) runs @p atom.
bool atomRunsOn(std::string_view atom, std::string_view gcnArchName);

/// TF32 permission: HIP_KERNEL_PROVIDER_CONV_TF32 set to a true value. Temporary, until
/// hipDNN can express the permission on a graph.
bool tf32Permitted();

/// The launch grid of @p problem for a BM x BN tile, or nullopt past the launch limits.
struct Grid
{
    unsigned int x = 0, y = 0, z = 0;
};
std::optional<Grid> gridFor(const IgemmProblem& problem, int64_t bm, int64_t bn);

/// Rows of shift range a halo config stages beyond its row tile (the kernels' K1_HALO_CAP).
constexpr long long HALO_CAP = 64;

/// Whether the halo A loop serves @p problem with a @p bm row tile: unit stride, output spatial
/// dims equal to the input's (a "same" convolution), a pixel-shift range over the taps within
/// HALO_CAP rows, and the staged rows' byte offsets within 32 bits.
bool haloFits(const IgemmProblem& problem, int64_t bm);

/// Whether @p gcnArchName (suffix intact) deals workgroups over several XCDs, so the
/// XCD-contiguous block order changes anything (gfx942, gfx950).
bool hasXcds(std::string_view gcnArchName);

/// Split-K for a config: how many contiguous ranges of the reduction run as separate blocks,
/// and where their fp32 partials sit in the workspace (after the staged operands).
struct SplitPlan
{
    int splits = 1;
    long long outElems = 0;
    size_t partialOffset = 0;
    size_t workspaceBytes = 0; ///< staged operands and partials
};

/// The analytic default: split only when the tiles leave the device under-filled (fewer than
/// two blocks per compute unit), keeping at least two k-iterations per split, at most 16
/// splits, and the partials under 1 GiB.
SplitPlan splitPlanFor(
    const IgemmProblem& problem, int64_t bm, int64_t bn, int64_t bk, int64_t computeUnits);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_igemm
