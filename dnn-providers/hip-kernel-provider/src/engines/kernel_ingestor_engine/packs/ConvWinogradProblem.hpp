// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/winograd/kernels/ConvWinogradArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvWinogradProblem.hpp
 * @brief The graph-to-kernel half of the Winograd convolution engines (CONV_WINOGRAD_FWD,
 *        CONV_WINOGRAD_BWD, CONV_WINOGRAD_WRW): which graphs they accept, how a catalog config
 *        runs one, and the workspace that takes. Kept apart from the dispatch code so the rules
 *        can be tested without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_winograd
{

/// How an activation pair (x and y) or a filter is laid out: packed channels-first (NCHW, or
/// KCRS for the filter), packed channels-last (NHWC, KRSC), or anything else.
enum class Layout
{
    NCHW,
    NHWC,
    OTHER,
};

/// A 2-D, 3x3, stride-1 or stride-2, dilation-1 convolution in the kernels' forward form (see
/// ConvWinogradArgs.h; stride 2 as its phase convolution, `mode` 1 or 2), or (backward weights,
/// stride 1) in the F(3,r) form of `wrw`. Pointers and tile counts are filled per config and
/// launch.
struct WinogradProblem
{
    conv_naive::ConvNaiveProblem base;
    conv_naive::ConvDirection direction = conv_naive::ConvDirection::FORWARD;
    K5Args args{};
    /// Backward weights: the shape fields (channels over all groups, extents, padding, flip,
    /// dw strides); tiles, splits and pointers per config and launch.
    K5WrwArgs wrw{};
    int64_t xUid = 0;
    int64_t wUid = 0;
    int64_t yUid = 0;
    int elementBytes = 0;
    /// x and y together; OTHER unless both are packed the same way.
    Layout activations = Layout::OTHER;
    /// The filter as the graph holds it.
    Layout filter = Layout::OTHER;
};

/**
 * @brief The convolution @p context's graph describes in @p direction, or nullopt when the
 *        Winograd kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: two spatial axes, a 3x3 filter, stride 1 or 2
 * (the same on both axes; backward weights stride 1 only), dilation 1, and every extent,
 * channel count and padding within 32 bits. Any padding,
 * asymmetric included (the output extent the graph declares is the one computed), both
 * convolution modes, groups, fp32 / fp16 / bf16, and any operand layout (staged when the
 * variant does not read it).
 */
std::optional<WinogradProblem>
    parseWinogradProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                         conv_naive::ConvDirection direction);

/**
 * @brief Whether the engines admit bf16 graphs: only with HIP_KERNEL_PROVIDER_CONV_WINOGRAD_BF16
 *        set to a true value.
 *
 * The matrix-core bf16 operands round the transformed input and filter tiles to bf16, an error
 * (about a third of bf16's unit roundoff of each tile's sum of |products|) the convolution's own
 * precision model does not allow, and MIOpen has no bf16 Winograd to match. Declined until a plan
 * can carry a numerical note; the bf16 instances and code stay and serve with the variable set.
 */
bool bf16Permitted();

/**
 * @brief Whether the F(4,3) configs (fp32 multipass) are admitted: only with
 *        HIP_KERNEL_PROVIDER_CONV_WINOGRAD_F43 set to a true value.
 *
 * F(4,3)'s transforms (points 0, +-1, +-2) carry entries up to 8 and 1/24: about 27 fp32 ulps of
 * each tile's sum of |products| on a 3x3 c128 layer, against about 1 for F(2,3) and the 5.7
 * coefficient MIOpen's fp32 Winograd needs. Declined until a plan can carry a numerical note;
 * the instances stay and serve with the variable set.
 */
bool f43Permitted();

enum class Variant
{
    MULTIPASS,
    FUSED,
    WMMA,
    WREG, ///< fused WMMA with register-built operands (gfx11 / gfx12)
    SREG, ///< fused VALU with register operands (wave64 gfx9)
    COOP, ///< fused WMMA, the block transforms each input chunk once into LDS (gfx11 / gfx12)
    WRW, ///< backward weights F(3,r): x / dy transforms, a split GEMM per position, dW transform
};

/// A catalog config's compile-time choices, read from its metadata.
struct Config
{
    Variant variant = Variant::MULTIPASS;
    int m = 2; ///< output tile: F(m,3), or F(3,2) (m = 3) on stride 2's phase sub-filters
    Layout layout = Layout::NCHW; ///< what the kernels read; the fused variants read NCHW
    std::string arch; ///< "any", or the WMMA family: "gfx11", "gfx12"
    bool llcPlan = false; ///< multipass: slice the batch to fit the last-level cache
    int gemmBm = 0, gemmBn = 0, gemmBk = 0; ///< multipass GEMM tile
    int bk = 0, bt = 0, ccv = 0, tm = 0, tn = 0; ///< fused VALU block
    int waves = 0; ///< fused WMMA waves per workgroup (register / cooperative: derived, checked)
    int bins = 0; ///< register WMMA: bins of a transform row per wave (4 or 2)
    bool f16Math = false; ///< register / cooperative, fp16: the input transform in packed fp16 math
    /// 1: prefetch further (cooperative: the next item's first chunk behind the epilogue; fused
    /// VALU: two input chunks ahead).
    int pf = 0;
    // Backward weights (variant WRW) reads m = r of F(3,r) (2 or 4), arch = the GEMM's atom
    // family ("any": FMA on fp32 operands; "gfx9": MFMA 16x16; "gfx950": MFMA 16x16x32, 16-bit;
    // "gfx11" / "gfx12": WMMA, 16-bit), the block gemmBm (out channels) x gemmBn (in channels)
    // x gemmBk (tiles), and tm x tn: the FMA thread tile, or a wave's atoms along out and in
    // channels (waves: their count, 0 for FMA).
};

/// The config @p kernel describes, or nullopt for metadata the pack does not know.
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Whether the device @p gcnArchName (suffix intact) runs @p config's instructions.
bool configRunsOn(const Config& config, std::string_view gcnArchName);

/// A strided copy of one operand into (read) or out of (written) the workspace.
struct Staging
{
    bool needed = false;
    /// Backward weights: a packed NHWC operand transposed to NCHW (copy.dims: N, H x W, C).
    bool transpose = false;
    unsigned int transposeBlocks[3] = {1, 1, 1};
    K5CopyArgs copy{};
    size_t workspaceOffset = 0;
    size_t bytes = 0;
};

/// How a config runs a problem: what it stages, the batch slices, and the workspace.
struct Plan
{
    /// The problem with this config's tile counts for one full slice.
    K5Args args{};
    Staging in; ///< forward-form input (x, or dy)
    Staging out; ///< forward-form output (y, or dx)
    Staging filter;
    size_t uOffset = 0, uBytes = 0;
    size_t vOffset = 0, vBytes = 0;
    size_t mOffset = 0, mBytes = 0;
    int sliceImages = 1; ///< images per launch
    int slices = 1;
    size_t workspaceBytes = 0;
    /// Launch grids for one slice: transforms and GEMM (multipass), or the persistent kernel.
    unsigned int filterBlocks[3] = {1, 1, 1};
    unsigned int inputBlocks[3] = {1, 1, 1};
    unsigned int gemmBlocks[3] = {1, 1, 1};
    unsigned int outputBlocks[3] = {1, 1, 1};
    unsigned int mainBlocks = 1;
    unsigned int mainThreads = 256;
    /// Backward weights: the transformed dy (U), the transformed x (V), the fp32 partials, and
    /// the GEMM's operand bytes, atom code (CONV_WINO_WRW_ATOM) and threads.
    K5WrwArgs wrw{}; ///< the problem with this config's tiles, ldt and splits for one full slice
    size_t partOffset = 0, partBytes = 0;
    int operandBytes = 0;
    int wrwAtom = 0;
    /// CONV_WINO_WRW_OPT: 0 fp32 operands, 1 the I/O type (fp16 graphs' F(3,2)).
    int wrwOpt = 0;
    unsigned int gemmThreads = 256;
};

/// Workspace limit the batch slicing keeps the multipass buffers under (the engine's own
/// default; HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB overrides).
size_t workspaceLimit();

/// The last-level (MALL) cache of an RDNA device, or 0 where the LLC-fit plan does not apply.
/// HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES overrides it on any device (tests).
size_t llcBytes(std::string_view gcnArchName);

/// The plan @p config runs @p problem with on @p device, or nullopt when the config does not
/// serve it (layout, launch limits, buffer limits, an LLC plan that would not slice, or a
/// persistent config on a device that reports no units).
std::optional<Plan> planFor(const WinogradProblem& problem,
                            const Config& config,
                            const hipdnn_plugin_sdk::ingestor::DeviceProperties& device);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_winograd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
