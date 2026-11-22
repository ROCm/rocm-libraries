// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/direct/kernels/ConvDirectArgs.h"
#include "engines/kernel_ingestor_engine/descriptors/conv/igemm/kernels/ConvIgemmArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvDirectProblem.hpp
 * @brief The graph-to-kernel half of the stride-1 halo-tile direct convolution engines
 *        (CONV_DIRECT_FWD, CONV_DIRECT_BWD): which graphs they accept, how a catalog config runs
 *        one, and the workspace that takes. Kept apart from the dispatch code so the rules can be
 *        tested without a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_direct
{

/// The filter extents (depth, height, width) the kernels are instantiated for. A 2-D filter has
/// depth 1; 1-D graphs are not served.
inline constexpr std::array<std::array<int, 3>, 7> FILTERS{{
    {1, 3, 3},
    {1, 5, 5},
    {1, 7, 7},
    {1, 1, 7},
    {1, 7, 1},
    {3, 3, 3},
    {5, 5, 5},
}};

/// The filter widths (1 x S) the line configs serve on one-row images: 1-D graphs and 2-D graphs
/// of one row (audio convolutions).
inline constexpr std::array<int, 4> LINE_FILTERS{{3, 5, 7, 11}};

/// Output planes per compute unit a z-ring grid needs at least (planFor): a range of L planes
/// stages L + KD - 1 input planes and runs KD - 1 partial edge steps at each end.
inline constexpr long long ZRING_MIN_PLANES = 6;

/// A strided copy (or, for packed channels-first activations, a tiled transpose) of one
/// forward-form activation into (in) or out of (out) the workspace's packed channels-last copy.
struct Staging
{
    bool needed = false;
    bool transpose = false;
    ConvIgemmTransposeArgs transposeArgs{};
    ConvIgemmCopyArgs copy{};
    size_t workspaceOffset = 0;
    size_t bytes = 0;
};

/// A forward or backward-data convolution in the kernels' forward form: backward data is the
/// forward convolution of dy with the filter rotated and its channel roles swapped (both done by
/// the weight pre-pass), padded by (F - 1) d - p at dilation d.
struct DirectProblem
{
    conv_naive::ConvNaiveProblem base;
    conv_naive::ConvDirection direction = conv_naive::ConvDirection::FORWARD;
    /// The forward form; pointers, tile and split are filled per config and launch.
    K2Args args{};
    int64_t xUid = 0;
    int64_t wUid = 0;
    int64_t yUid = 0;
    int elementBytes = 0;
    /// The graph filter's element strides in forward roles (k, c, t, r, s), and whether it is
    /// reflected (true convolution).
    std::array<long long, 5> wStrides = {};
    int flip = 0;
    /// The dilation of every axis the filter spans (1 when undilated); a 3-D problem is undilated.
    int dilation = 1;
    /// A one-row image (1-D, or 2-D of one row) under a 1 x S filter: the line configs' shape.
    bool oneRow = false;
    /// The graph's spatial rank (1 to 3); 1-D graphs run as one-row 2-D ones on line configs.
    int spatialRank = 0;
    /// The forward-form input (x, or dy) and output (y, or dx), staged when not channels-last.
    Staging in;
    Staging out;
    /// Bytes of the staged activations (the start of every config's own workspace).
    size_t stagingBytes = 0;
};

/**
 * @brief The convolution @p context's graph describes in @p direction, or nullopt when the
 *        direct kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: two or three spatial axes (or one, as a one-row
 * 2-D image), groups 1, stride 1 on every axis, one dilation of the axes the filter spans (1 in
 * 3-D), a filter in FILTERS (or a 1 x S filter of LINE_FILTERS on a one-row image), every
 * extent, channel count and padding
 * within 32 bits, and one image of the forward-form input within 2^31 bytes (the kernels' 32-bit
 * buffer offsets). Any padding, asymmetric included (the output extent the graph declares is the
 * one computed), both convolution modes, fp32 / fp16 / bf16, and any operand layout (staged to
 * channels-last when it is not). With @p fusion EPILOGUE a forward graph may carry a pointwise
 * epilogue (base.epilogue), which the kernels apply before their single rounding.
 */
std::optional<DirectProblem>
    parseDirectProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                       conv_naive::ConvDirection direction,
                       conv_naive::Fusion fusion = conv_naive::Fusion::NONE);

/// A catalog config's compile-time choices, read from its metadata.
struct Config
{
    bool fma = false;
    std::string atom; ///< conv_igemm atom name ("fma" for the FMA kernel)
    std::string_view kernelType; ///< K1Atom* type of the atom; empty for FMA
    int wave = 0; ///< 64 MFMA, 32 WMMA, 0 FMA
    int ka = 1; ///< the atom's reduction step (weight format granule); 1 for FMA
    bool math = false; ///< three-bf16 (fp32 graph, bf16 operands): a TF32 config
    int kd = 0, r = 0, s = 0;
    int bm = 0, bn = 0, bc = 0, wm = 0, wn = 0, vec = 0;
    int ring = 1, bl = 0, split = 0;
    int tm = 0, tn = 0, nt = 0, threads = 0, pipe = 0;
    /// The filter-row reuse kernel (kind "rows", ConvDirectRows.h): waves WP x WQ x WK, each PW
    /// output rows x 16 columns x 16 FN channels; ping-pong (pp) and staged epilogue (epi).
    bool rows = false;
    int pw = 0, fn = 0, wp = 0, wq = 0, wk = 0, pp = 0, epi = 0;
    /// Rows batch unfold (2-D): an MFMA's 16 columns are 16 / un columns of
    /// each of un consecutive images; 1 is the plain tile.
    int un = 0;
    /// Rows on packed channels-first activations in place (2-D, 16-bit, gfx950): the halo is
    /// fetched as it lies and transposed in LDS, the output stored channels-first; no staging.
    int cf = 0;
    /// Halo tile with register-path weights (BL 0), channels-last: 1 prefetches the weight
    /// fragments across channel chunks, skips ring sets by scalar branches and pads 16-bit halo
    /// rows by 8 bytes (CONV_DIRECT_XPF); 0 elsewhere.
    int xpf = 0;
    /// Run tiles (halo tile, ring 0, register-path weights, channels-last): a block owns BM
    /// consecutive pixels of a 16-column strip of the output, its rows over the planes, instead of
    /// a th x tw tile, so widths the 2-D tiles fit badly (130, 66, 34) pad no pixel slots. The
    /// value is the stage's pixels (CONV_DIRECT_RUN); 0 elsewhere.
    int run = 0;
    /// Filter dilation (2-D rows configs; 1 elsewhere): the config serves exactly this dilation of
    /// the axes its filter spans.
    int dil = 1;
    /// Line tiles (2-D rows, 1 x S filters): one-row images, a tile wp pw segments of 16 wq
    /// columns of the row staged as one halo line.
    int line = 0;
    /// Rows: persistent blocks resident per unit (1 to 3); each holds its own halo stages, so the
    /// blocks' memory phases and barriers overlap the others' MMAs.
    int bpu = 1;
    /// The z-ring tile (kind "zring", ring 2; ConvDirectZring.h): 3-D, 16-bit, 16 x 16 MFMA atoms.
    /// A tile is bm / 16 rows x 16 columns x bn channels, and one workgroup per compute unit walks
    /// a balanced range of (tile, output plane) units along depth with the KD live output planes
    /// in its accumulators; each input plane is staged once per range.
    bool zring = false;
    std::string archs; ///< ship list: comma-separated arch groups, or "any"

    /// Output pixels per block (the planner's tile budget) and output channels per block.
    int blockPixels() const
    {
        if(fma)
        {
            return (threads / nt) * tm;
        }
        return rows ? wp * pw * wq * 16 : bm;
    }
    int blockColumns() const
    {
        if(fma)
        {
            return nt * tn;
        }
        return rows ? wk * fn * 16 : bn;
    }
    int blockThreads() const
    {
        if(fma)
        {
            return threads;
        }
        return rows ? wp * wq * wk * wave : wm * wn * wave;
    }
};

/// The config @p kernel describes, or nullopt for metadata the pack does not know.
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Whether @p archs (a config's ship list) names the device @p gcnArchName (suffix intact):
/// "any", the exact architecture, or its family ("gfx11", "gfx12").
bool shipsTo(std::string_view archs, std::string_view gcnArchName);

/// LDS per workgroup on @p gcnArchName: 160 KiB on CDNA4 (gfx950), 64 KiB on every other family.
int64_t ldsLimitOf(std::string_view gcnArchName);

/// LDS a rows config's kernel declares: two halo stages of (wp pw + (r - 1) dil) x
/// (16 wq + (s - 1) dil) pixels x bc channels of the atom's operand (ConvDirectRows.h; the staged
/// epilogue reuses a stage); a channels-first config one such stage plus one raw stage of
/// (wp pw + (r - 1) dil) x bc channel rows of 8-column runs. 0 for a config of another kind.
int64_t rowsLdsBytes(const Config& config);

/// LDS a z-ring config's kernel declares on @p gcnArchName: two stages of (bm / 16 + r - 1) x
/// (16 + s - 1) positions x bc channels, each 16-byte channel chunk's positions padded to the
/// device's conflict-free period for 128-bit reads (256 bytes on gfx950, 128 elsewhere), plus a
/// 16-byte slot each (ConvDirectZring.h). 0 for a config of another kind.
int64_t zringLdsBytes(const Config& config, std::string_view gcnArchName);

/// Whether the device runs @p config: its atom exists there, its ship list names it, and (rows,
/// z-ring) the stages of its resident blocks fit the device's LDS.
bool configRunsOn(const Config& config, std::string_view gcnArchName);

/// How a config runs a problem: the tile, the split, the launch grid and the workspace.
struct Plan
{
    K2Args args{}; ///< the problem with this config's tile and split; pointers filled at launch
    std::array<unsigned int, 3> grid = {1, 1, 1};
    int cpad = 0; ///< reduction channels of the formatted weights (x3 for three-bf16)
    long long weightElems = 0;
    long long outElems = 0;
    size_t weightOffset = 0, weightBytes = 0;
    size_t splitInputOffset = 0, splitInputBytes = 0; ///< three-bf16 input
    size_t partialOffset = 0, partialBytes = 0; ///< split-C fp32 partials
    size_t workspaceBytes = 0;
};

/**
 * @brief The plan @p config runs @p problem with on a device of @p computeUnits, or nullopt when
 *        the config cannot (its filter or channel vector, the LDS halo bound, launch or 32-bit
 *        buffer limits) or the device reports no units (@p computeUnits < 1).
 *
 * @p computeUnits counts the units workgroups are dispatched onto (device_units::dispatchUnits:
 * CUs on gfx9, WGPs on gfx10+); the per-unit block targets below are in that unit.
 *
 * A rows config has a fixed tile (wp pw rows x 16 wq columns) and launches at most one block per
 * compute unit, each walking its share of the tiles (rowsLdsBytes: up to ~153 KiB on gfx950).
 *
 * A z-ring config has a fixed tile (bm / 16 rows x 16 columns x bn channels) and launches one
 * block per compute unit over the (tile, output plane) units; it serves 16-bit 3-D problems
 * whose reduction channels are whole stages (bc) and whose output channels are a multiple of 4,
 * and only grids of at least ZRING_MIN_PLANES output planes per unit, where the ring's reuse pays
 * for its pencil edges.
 *
 * The tile is th x tw output pixels (tw a multiple of the FMA run length) of ds planes, chosen to
 * minimize padded pixel slots plus halo loads within the config's halo bound; depth is split
 * until there are 4 (ring) or 8 blocks per compute unit. A split-C config splits the channel
 * chunks when the grid has fewer than 2 blocks per compute unit, keeping 2 chunks per split and
 * at most 16 splits.
 */
std::optional<Plan>
    planFor(const DirectProblem& problem, const Config& config, int64_t computeUnits, int xcds = 0);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_direct

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
