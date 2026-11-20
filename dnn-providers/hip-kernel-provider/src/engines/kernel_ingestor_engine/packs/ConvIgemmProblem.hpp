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
    /// Group fusion: the filter expanded into block-diagonal fused groups (ConvIgemmExpand).
    bool expand = false;
    ConvIgemmExpandArgs expandArgs{};
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
    /// One image has more than 2^31 - 1 pixels (input or output): only the large-image configs
    /// (64-bit positions within an image, K1_LARGE) serve it, and only it.
    bool largeImage = false;
    /// The graph's filter is packed K T R S C/g and not reflected: backward data can read it
    /// directly (the direct-filter configs) instead of staging the transposed copy.
    bool filterKtrsc = false;
    /// The graph's filter is packed K T R S C/g or K C/g T R S and not reflected: group fusion
    /// can expand it. Element (k, c, tap) at k * wSrcK + c * wSrcC + tap * wSrcTap.
    bool filterExpandable = false;
    long long wSrcK = 0, wSrcC = 0, wSrcTap = 0;
    /// Col2im backward data (withCol2imView): the fp32 GEMM columns ConvIgemmCol2im sums into
    /// dx, after the staged operands in the workspace; 0 bytes otherwise.
    size_t col2imOffset = 0;
    size_t col2imBytes = 0;
};

/// @p problem without the filter staging, for a direct-filter config: the remaining staged
/// operands keep their order in the workspace.
IgemmProblem withoutFilterStaging(const IgemmProblem& problem);

/// @p problem with @p fuse consecutive groups run as one (group fusion): the filter expanded into
/// block-diagonal fused groups of fuse x C/g input channels in the workspace (packed K T R S C/g,
/// which forward and the direct-filter backward-data configs read), or nullopt where the groups or
/// the filter do not allow it.
std::optional<IgemmProblem> withFusedGroups(const IgemmProblem& problem, int64_t fuse);

/**
 * @brief @p problem as a patch config runs it, or nullopt where it is not a patch convolution.
 *
 * One group, and along width (forward) or every axis (backward data) a filter extent equal to
 * the stride, no padding, unit dilation and an input extent the stride divides; at least one
 * axis folded (a filter extent above one).
 *
 * Forward reads the channels-last input as W / S pixels of S C channels each and the K T R S C/g
 * filter as K T R 1 (S C): the same bytes, so each filter row's taps become one contiguous run
 * of the reduction and the vector loads span whole runs. The kernel is unchanged.
 *
 * Backward data runs dx[(n, od, oh, ow)][(t, r, s, c)] = sum_k dy[(n, od, oh, ow)][k]
 * w[k][(t, r, s, c)] as one GEMM over dy's positions (a 1x1, unit-stride problem of T R S C
 * columns) with the patch scatter in the epilogue (K1_PATCH): every dx element is written once,
 * with no stride phases. The filter is read as given (direct-filter configs) or staged as
 * [t][r][s][c][k] (reflected for true convolution).
 */
std::optional<IgemmProblem> withPatchView(const IgemmProblem& problem);

/**
 * @brief Backward data @p problem as the col2im configs run it, or nullopt where they do not.
 *
 * One GEMM over dy's positions with the filter's T R S C as columns (as withPatchView's
 * backward data), for any filter, stride, padding and dilation: its fp32 columns go to the
 * workspace and ConvIgemmCol2im adds, for each dx element, those of the dy positions it reaches,
 * tap by tap. Served for one group and a packed, unreflected K T R S C filter (read in place),
 * where the GEMM has fewer multiply-adds than the GEMMs over dx's rows (dy positions times taps
 * below dx positions times the taps a stride phase reaches: "valid" or partly padded layers,
 * whose dx rows mostly multiply padding), for convolutions that are not patch convolutions
 * (withPatchView serves those with no columns in between), and within 1 GiB of columns.
 */
std::optional<IgemmProblem> withCol2imView(const IgemmProblem& problem);

/// Whether @p problem's output (forward y, backward data dx) is packed channels-first, so a
/// channels-first-output config writes it in place of the staging transpose.
bool outputChannelsFirst(const IgemmProblem& problem);

/// Removes the output staging for a channels-first epilogue, and also input staging when
/// @p removeInput selects the bounded pointwise loader. Recomputes the remaining workspace.
IgemmProblem withoutActivationStaging(const IgemmProblem& problem, bool removeInput);

/**
 * @brief The convolution @p context's graph describes in @p direction, or nullopt when the
 *        implicit-GEMM kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: every extent, padding, stride, dilation and
 * channel count within 32 bits, and grid dimensions within the launch limits for the smallest
 * tile. Any operand layout is served: those that are not what the kernels read are staged
 * through the workspace. With @p fusion EPILOGUE a forward graph may carry a pointwise epilogue
 * (base.epilogue), which the kernels apply before their single rounding.
 */
std::optional<IgemmProblem>
    parseIgemmProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                      conv_naive::ConvDirection direction,
                      conv_naive::Fusion fusion = conv_naive::Fusion::NONE);

/// One matrix-core atom (or "fma") of the catalog: the kernel template's name, the waves'
/// width, and whether the device runs it.
struct AtomInfo
{
    std::string_view name;
    std::string_view kernelType; ///< K1Atom* type name; empty for the FMA kernel
    int wave = 0; ///< 64 MFMA, 32 WMMA, 0 FMA (block from `threads`)
    bool tf32 = false; ///< Rounds fp32 operands (xf32): a TF32 config
    /// 16-bit fragments of 8, 16 or 32 bytes, which the swizzled-tile (swz) configs read in
    /// 16-byte chunks: the k-step those reads span (8-byte fragments two atom k-steps), which
    /// the config's k-step must be a multiple of; 0: no swizzled tiles.
    int swizzleK = 0;
};

/// The atom @p name names, or nullopt for an unknown one.
std::optional<AtomInfo> atomNamed(std::string_view name);

/// The element type @p atom multiplies: "f16", "bf16", "f32", "any" (FMA), or empty if unknown.
std::string_view atomElement(std::string_view atom);

/// "f32", "f16" or "bf16" for the convolution element types, empty otherwise.
std::string_view dataElement(hipdnn_flatbuffers_sdk::data_objects::DataType dataType);

/// Whether the device @p gcnArchName (suffix intact) runs @p atom.
bool atomRunsOn(std::string_view atom, std::string_view gcnArchName);

/// The launch grid of @p problem for a BM x BN tile, or nullopt past the launch limits.
struct Grid
{
    unsigned int x = 0, y = 0, z = 0;
};
std::optional<Grid> gridFor(const IgemmProblem& problem, int64_t bm, int64_t bn);

/// Row tiles one launch of @p threads-thread blocks takes: all @p rowTiles, or as many as one
/// dispatch addresses (2^32 - 1 work items along x); the rest run as further launches.
long long launchTilesFor(long long rowTiles, long long threads);

/// The waves per SIMD the register allocator targets for a config on @p gcnArchName, or 0 for
/// the compiler's own choice. The swizzled 128x128x64 WMMA tile with 4x2 waves gets 4 on the
/// gfx11 chips with the 1.5x VGPR file (gfx1100, gfx1101, gfx1151): aiming for 8 there, HIPRTC
/// spills it (256 VGPRs, 168 B scratch on gfx1101) and it runs 1.6x slower.
int occupancyTarget(std::string_view atom,
                    int64_t bm,
                    int64_t bn,
                    int64_t bk,
                    int64_t wm,
                    int64_t wn,
                    int64_t swz,
                    std::string_view gcnArchName);

/// Rows of shift range a halo config stages beyond its row tile (the kernels' K1_HALO_CAP).
constexpr long long HALO_CAP = 64;

/// Whether the halo A loop serves @p problem with a @p bm row tile: unit stride, output spatial
/// dims equal to the input's (a "same" convolution), a pixel-shift range over the taps within
/// HALO_CAP rows, and the staged rows' byte offsets within 32 bits.
bool haloFits(const IgemmProblem& problem, int64_t bm);

/// Whether @p problem's gathered activation (forward x, backward data dy) is packed
/// channels-first, so a channels-first-input config (cfx) reads it in place of the staging
/// transpose.
bool inputChannelsFirst(const IgemmProblem& problem);

/// Whether @p problem's output is packed channels-first: staged by the transpose
/// (outputChannelsFirst), or packed channels-last with every spatial extent 1, where the two
/// layouts coincide.
bool outputPackedChannelsFirst(const IgemmProblem& problem);

/// The same for @p problem's gathered input (forward x, backward data dy).
bool inputPackedChannelsFirst(const IgemmProblem& problem);

/// @p problem without the input staging, for a channels-first-input config: the remaining staged
/// operands keep their order in the workspace.
IgemmProblem withoutInputStaging(const IgemmProblem& problem);

/**
 * @brief @p problem as a 1-D channels-first line, for the channels-first-input (cfx) configs.
 *
 * A pointwise convolution (every filter extent 1, unit stride, no padding, output extents equal to
 * the input's) whose input and output are packed channels-first is the 1-D problem over each
 * channel's D x H x W pixels as one run of L = D H W: the same bytes, read and written in place.
 * The view has depth and height 1 and width L. A problem that is already 1-D, or that is not such
 * a pointwise convolution, is returned unchanged (cfxFits then decides as before).
 */
IgemmProblem withChannelsFirstLineView(const IgemmProblem& problem);

/// The tile fields the channels-first-input fit depends on.
struct CfxTile
{
    int64_t bm = 0, bn = 0, bk = 0, wm = 1, wn = 1;
    int64_t halo = 0; ///< 0: K1TilesCf (FMA kernel); h: K1HaloCf staging h pixels a row
    int64_t dim = 16; ///< the atom's tile width (the epilogue strips)
    long long ldsBytes = 0; ///< the device's LDS per workgroup
};

/**
 * @brief Whether a channels-first-input config (cfx) serves @p problem.
 *
 * 1-D problems only (depth and height extents and filter 1). Along the width, in every gather the
 * blocks run (forward, or each backward-data stride phase), a halo config's block reaches at most
 * the pixels its rows stage, h x bm plus HALO_CAP and an aligning vector, and its LDS fits @p
 * tile's ldsBytes; and the pixels a block reads stay within 32-bit byte offsets of its first image.
 */
bool cfxFits(const IgemmProblem& problem, const CfxTile& tile);

/// Whether @p gcnArchName (suffix intact) deals workgroups over several XCDs, so the
/// XCD-contiguous block order changes anything (gfx942, gfx950).
bool hasXcds(std::string_view gcnArchName);

/// Whether a tile's block order serves the device: XCD order only where there are XCDs, and the
/// 256 x 256 tile there only in XCD order (its plain-order copy is for single-die devices).
bool blockOrderServes(int64_t bm, int64_t bn, int64_t xcd, std::string_view gcnArchName);

/// Split-K for a config: how many contiguous ranges of the reduction run as separate blocks,
/// and where their fp32 partials sit in the workspace (after the staged operands).
struct SplitPlan
{
    int splits = 1;
    long long outElems = 0;
    size_t partialOffset = 0;
    size_t workspaceBytes = 0; ///< staged operands and partials
    /// The tail split (register-staged flex tile): the row tiles before this one run unsplit in
    /// a launch of their own and only the rest split; 0 splits every row tile.
    long long splitRowTile = 0;
    /// Elements of one split's partials: the output's past row tile splitRowTile.
    long long partialElems = 0;
};

/// The split policy's largest explicit split count (split_policy N, 2 <= N <= this).
inline constexpr int64_t MAX_SPLIT_POLICY = 64;

/**
 * @brief The split-K plan of a config on @p problem.
 *
 * @p splitPolicy (the config's `split_policy` knob):
 * - 0, the analytic default: split only when the tiles leave the device under-filled, at most
 *   16 splits, the partials under 1 GiB. Under-filled is fewer blocks than the units hold: on
 *   gfx9 @p residentBlocks a CU (8-wave blocks' occupancy, residentBlocksOf) up to 2, 2 when 0,
 *   keeping at least two k-iterations per split; one block per WGP on legacy gfx10+
 *   (device_units), keeping at least 64; one block per CU on gfx1250, keeping eight.
 * - 1: no split.
 * - N from 2 to MAX_SPLIT_POLICY: exactly N splits where the unsplit grid under-fills the
 *   device, each split keeps a k-iteration and the partials and the grid fit; otherwise no split
 *   (the kernel match declines such a config, and any whose plan equals the default's, so
 *   autotune times only distinct plans, and a filled grid's engine workspace carries no
 *   partials it never runs).
 *
 * A device that reports no units gets no split (the kernel match declines it). @p pack: 1 the
 * config's k-steps span whole taps (tap packing), so a k-step covers bk / (reduction channels)
 * taps; 2 the reduction runs flat (taps x channels in k-steps of bk). @p noSplit (patch configs):
 * one launch whatever the fill; their GEMMs are microsecond-scale, so the reduce launch and the
 * partials' round trip cost more than an under-filled grid. @p flex (flex tiles, one block a
 * CU; 1 the LDS-DMA form, 2 the register-staged form): the default splits the row tiles past
 * the whole rounds or all of them, whichever the launch rounds make shortest at the form's split
 * cost (flexTailSplit, flexSplitCost). Policies 1 and N apply unchanged.
 */
SplitPlan splitPlanFor(const IgemmProblem& problem,
                       int64_t bm,
                       int64_t bn,
                       int64_t bk,
                       const hipdnn_plugin_sdk::ingestor::DeviceProperties& device,
                       int64_t pack = 0,
                       bool noSplit = false,
                       int64_t splitPolicy = 0,
                       int64_t residentBlocks = 0,
                       int64_t flex = 0);

/**
 * @brief Blocks of a matrix-core config of at least 8 waves (512 threads on gfx9) a CU holds at
 *        once, from its threads, its LDS and its accumulators; 0 for smaller blocks, the FMA
 *        kernel (wave 0) or a device whose dispatch units are not gfx9 CUs (the split plan's arch
 *        default, 2 a CU on gfx9, applies).
 *
 * Waves: 32 a gfx9 CU (8 a SIMD). LDS: (bm + bn) rows of bk padded by a 16-byte vector, by the
 * DMA stages or the k-stages, within the CU's 64 KiB (160 KiB on gfx950). Registers: bm x bn / threads
 * fp32 accumulators a lane plus 64, within 512 a lane.
 */
int64_t residentBlocksOf(int64_t bm,
                         int64_t bn,
                         int64_t bk,
                         int64_t threads,
                         int64_t wave,
                         int64_t stages,
                         int64_t dma,
                         int elementBytes,
                         std::string_view gcnArchName);

/// The flex tile's split: the first row tile that splits (0: all) and the
/// split count (1: none).
struct FlexTailSplit
{
    long long splitRowTile = 0;
    long long splits = 1;
};

/// The flex tile's split of @p rowTiles row tiles of @p rowWidth tiles each, @p kSteps k-steps a
/// tile, one block a unit: the row tiles that fill whole rounds of @p units run unsplit and the
/// rest split S ways, or all of them split S ways, for the plan with the fewest k-steps along the
/// launch rounds, a split block paying @p splitCost k-steps more. Rounds are
/// ceil(blocks / units) of the longest block. @p tailAlone false (a fused epilogue, which the
/// reduce applies over the whole output) leaves only the all-split plans.
FlexTailSplit flexTailSplit(long long rowTiles,
                            long long rowWidth,
                            long long kSteps,
                            long long units,
                            bool tailAlone,
                            long long splitCost);

/// The flex tile's fixed cost of one more split, in its k-steps (the partials' round trip and
/// the reduce launch), by form: 1, the LDS-DMA form, runs a k-step in about half the time of 2,
/// the register-staged form, so the same round trip costs it more k-steps.
constexpr long long flexSplitCost(int64_t form)
{
    return form == 1 ? 20 : 14;
}

/// Whether tap packing serves @p problem with a @p bk k-step: the reduction channels per tap
/// are fewer than bk and divide it.
bool packFits(const IgemmProblem& problem, int64_t bk);

/// Whether the flat reduction (pack 2) serves @p problem with a @p bk k-step: whole taps of
/// reduction channels would pad (bk does not divide them), so running the taps end to end saves
/// k-steps. The channel vector dividing the reduction channels is the kernel match's own check.
bool flatFits(const IgemmProblem& problem, int64_t bk);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_igemm
