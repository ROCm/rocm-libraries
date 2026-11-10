// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>

#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/wrw/kernels/ConvWrwArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvWrwProblem.hpp
 * @brief The host half of the matrix-core backward-weights engine (CONV_WRW, K3): the graphs it
 *        serves, the config a kernel's metadata declares, and the launch of a config on a
 *        problem (the reduction split, the fp32 partials' workspace and the reduce). Kept apart
 *        from the dispatch code so every gate can be tested without a device.
 *
 * The kernel reads x and dy in place, both packed channels-first (N C D H W) or both
 * channels-last (N D H W C), and writes fp32 partials [split][g][tap][k][c] into the workspace;
 * a second kernel adds the splits in a fixed order and writes dw, packed K C T R S or K T R S C,
 * in its element type (deterministic, no atomics). At one split with dw's channels contiguous
 * (K T R S C, or one tap) a config compiled with the direct store (dws 1) writes dw itself: no
 * workspace, no reduce. The LDS a config takes is the kernels' own
 * arithmetic (ConvWrwGeom.h, which the kernels assert their tiles against).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_wrw
{

/// The atoms a config may name, in atom_id order: K1's matrix-core atoms (no TF32 form), then
/// the FMA row ring's "fma".
inline constexpr std::array<std::string_view, 17> ATOMS{"mfma16_f16",
                                                        "mfma16_bf16",
                                                        "mfma16_bf16_908",
                                                        "mfma16_f32",
                                                        "mfma32_f16",
                                                        "mfma32_bf16",
                                                        "mfma32_bf16_908",
                                                        "mfma32_f32",
                                                        "mfma16x32_f16",
                                                        "mfma16x32_bf16",
                                                        "mfma32x16_f16",
                                                        "mfma32x16_bf16",
                                                        "wmma11_f16",
                                                        "wmma11_bf16",
                                                        "wmma12_f16",
                                                        "wmma12_bf16",
                                                        "fma"};

/// The atom_id of @p atom, or -1 for an unknown one.
int64_t atomIdOf(std::string_view atom);

/// Whether a matrix-core atom of ATOMS multiplies @p element ("f32", "f16", "bf16") on
/// @p gcnArchName (suffix intact). Where none does (gfx906, gfx900), the FMA loops serve that
/// element type too.
bool runsMatrixAtomFor(std::string_view element, std::string_view gcnArchName);

/// A matrix-core atom's shape: the fragment's edge (DIM), the positions one step reduces (KA)
/// and the positions a lane holds (KPL).
struct AtomShape
{
    int64_t dim = 0;
    int64_t ka = 0;
    int64_t kpl = 0;
};

/// The shape of @p atom; zero for an unknown one and for "fma".
AtomShape atomShapeOf(std::string_view atom);

/// The kernel's loops (CONV_K3_LOOP).
inline constexpr int64_t LOOP_ROWS = 0; ///< output rows, any stride
inline constexpr int64_t LOOP_RING = 1; ///< row ring (height strides > 1 as phases)
inline constexpr int64_t LOOP_RING_DMA = 2; ///< row ring with LDS DMA (gfx950)
inline constexpr int64_t LOOP_RING_FMA = 3; ///< row ring on FMA (exact fp32)
inline constexpr int64_t LOOP_ROWS_FMA = 4; ///< output rows on FMA (exact fp32, any stride)
inline constexpr int64_t LOOP_DIRECT = 5; ///< direct (gfx950, 16-bit, persistent grid)
inline constexpr int64_t LOOP_ROWS_SMALLC = 6; ///< output rows, (channel, tap) N axis (stems)

/// The activation layouts (CONV_K3_LAYOUT): the graph's x and dy are channels-first (0) or
/// channels-last (1: transposed into LDS; 2: kept channels-last, read with LDS transpose reads).
inline constexpr int64_t LAYOUT_CHANNELS_FIRST = 0;
inline constexpr int64_t LAYOUT_CHANNELS_LAST = 1;
inline constexpr int64_t LAYOUT_CHANNELS_LAST_TR = 2;

/// One kernel config: the KMD fields minus direction and dtype.
struct WrwConfig
{
    std::string atom;
    int64_t atomId = -1;
    int64_t loop = -1;
    int64_t layout = -1;
    /// Tile: output channels, input channels, positions per reduction step.
    int64_t bk = 0;
    int64_t bc = 0;
    int64_t bq = 0;
    /// Filter rows per block (loops 0 and 4); the filter height (loops 1 to 3).
    int64_t fr = 0;
    /// The filter width, its stride and its dilation.
    int64_t s = 0;
    int64_t sw = 0;
    int64_t dw = 0;
    /// Waves over output and input channels (loops 0 to 2; 0 for loop 3).
    int64_t wk = 0;
    int64_t wc = 0;
    /// Output and input channels per thread (loop 3; 0 for loops 0 to 2).
    int64_t tk = 0;
    int64_t tc = 0;
    /// Elements per 16-byte global load.
    int64_t vec = 0;
    /// 1: XCD-contiguous block order (gfx942, gfx950).
    int64_t xcd = 0;
    /// 1: row-aligned units (loop 1, channels-first, width stride 1); 2: dy units at any element
    /// (loop 1, channels-last transposed into LDS, output channels not whole vectors).
    int64_t align = 0;
    /// LDS stages of loop 2 (2 to 4), ring slots of loop 5 (2 to 5); 0 for the other loops.
    int64_t ns = 0;
    /// Loop 5: items a workgroup runs side by side (waves_q: 1, 2 or 4; wk wc wq = 8); else 0.
    int64_t wq = 0;
    /// 1: the epilogue also writes dw itself, used at one split into K T R S C or one tap
    /// (CONV_K3_DWS); 0: fp32 partials only, as compiled before the direct store. 0 for loop 5.
    int64_t dws = 0;
    /// The split planner's target blocks per dispatch unit (1, 2, 4 or 8; 1 for loop 5, whose
    /// persistent grid has one workgroup per unit; 8 for small FMA blocks): host-side only.
    int64_t bpu = 0;
};

/// A backward-weights convolution the kernels serve.
struct WrwProblem
{
    conv_naive::ConvNaiveProblem base;
    int64_t xUid = 0;
    int64_t dyUid = 0;
    int64_t dwUid = 0;
    int elementBytes = 0;
    /// x and dy are both packed channels-first / both packed channels-last (both hold when the
    /// channel axes are 1).
    bool channelsFirst = false;
    bool channelsLast = false;
    /// dw is packed K T R S C (else K C T R S).
    bool filterChannelsLast = false;
    /// Filter taps (ft fr fs).
    int64_t taps = 0;
};

/**
 * @brief The backward-weights convolution @p context's graph describes, or nullopt when the
 *        kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates (both convolution modes), plus: fp32, fp16 or bf16;
 * x and dy both packed channels-first or both packed channels-last; dw packed K C T R S or
 * K T R S C; the batch within 32 bits; per spatial axis the reach of the kernel's 32-bit
 * position arithmetic ((output extent + 512) x stride + filter x dilation + padding) within 32
 * bits, which bounds every extent, padding, stride and dilation; the taps within 32 bits; one
 * image of x and one of dy within 2^31 bytes (32-bit buffer offsets).
 */
std::optional<WrwProblem> parseWrwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// LDS bytes a config's ConvWrw kernel declares: k3LdsBytes (ConvWrwGeom.h), the arithmetic the
/// kernels assert their tiles against.
int64_t ldsBytesOf(const WrwConfig& config, int elementBytes);

/// The most load passes (loads per thread and step, each tracked by a bit of a 32-bit mask) of
/// a config's dy and x tiles; @p wave is the atom's wave width (ignored for loop 3).
int64_t loadPassesOf(const WrwConfig& config, int elementBytes, int64_t wave);

/// The LDS a workgroup may use on @p gcnArchName: 160 KiB on gfx950, else 64 KiB.
int64_t ldsLimitOf(std::string_view gcnArchName);

/// The launch of a config on a problem: both kernels' grids and argument blocks, and the
/// workspace of the fp32 partials.
struct WrwLaunch
{
    unsigned int gridX = 0;
    unsigned int gridY = 0;
    unsigned int gridZ = 0;
    unsigned int block = 0;
    unsigned int reduceGrid = 0;
    unsigned int reduceBlock = 256;
    /// K1's atom template (empty for the FMA ring).
    std::string_view kernelType;
    /// Pointers are filled at launch.
    K3Args args{};
    K3ReduceArgs reduce{};
    size_t workspaceBytes = 0;
    /// The kernel writes dw itself (args.dwStore): no workspace, no reduce launch.
    bool dwStore = false;
};

/// The workspace the splits' fp32 partials may take at most; splits are reduced to fit.
inline constexpr int64_t MAX_WORKSPACE_BYTES = int64_t{1} << 30;

/// The splits the planner wants for a ConvWrw grid of @p blocksXY blocks per split, aiming at
/// @p blocksPerUnit blocks per dispatch unit of a device of @p dispatchUnits, for @p steps
/// reduction steps and @p partialElems partials per split: ceil(units x blocksPerUnit /
/// blocksXY), at most steps / 4, 65535, and as many as keep the workspace within
/// MAX_WORKSPACE_BYTES; at least 1. Then, of that count down to half of it (64 counts at most),
/// the one whose grid best fills its last round of dispatchUnits blocks (the most on a tie).
int64_t splitsWanted(int64_t blocksXY,
                     int64_t blocksPerUnit,
                     int64_t steps,
                     int64_t partialElems,
                     int64_t dispatchUnits);

/// The direct loop's work (loop 5): units are (channel tile, item), an item one image plane's
/// 32-column block over a segment of its rows; a persistent grid of about one workgroup per
/// dispatch unit takes them in pieces of `per` items.
struct DirectPlan
{
    int64_t segs = 1; ///< row segments per image plane
    int64_t segRows = 0; ///< rows per segment
    int64_t walk = 0; ///< 0 cells (tile fastest, whole splits), 1 equal runs of units
    int64_t items = 0; ///< items per tile
    int64_t per = 0; ///< items per piece (walk 0) / units per workgroup (walk 1)
    int64_t splits = 0; ///< partials slots
    int64_t grid = 0; ///< workgroups
};

/**
 * @brief The direct loop's plan for @p tiles channel tiles over @p planes image planes of
 *        @p rows input rows and @p colBlocks 32-column blocks, a filter of @p filterRows rows,
 *        partials of @p partialBytes per split, on @p units dispatch units, @p tileMacs
 *        multiply-adds per tile and input row of a block (bc bk s fr x 32).
 *
 * Walk 0 (cells) when the tiles leave each unit a whole split (2 tiles <= units): splits =
 * min(max(1, units / tiles), items), per = ceil(items / splits). Else walk 1: per = ceil(tiles
 * items / units) units of work per workgroup, splits the most workgroups any tile spans. The
 * row segments (1 to 64, at least 4 rows each) minimise the modelled time: the busiest
 * workgroup's rows (each segment adds fr + 1) at tileMacs per row plus the partials' traffic;
 * ties keep fewer segments.
 */
DirectPlan directPlanFor(int64_t tiles,
                         int64_t planes,
                         int64_t rows,
                         int64_t colBlocks,
                         int64_t filterRows,
                         int64_t partialBytes,
                         int64_t units,
                         int64_t tileMacs);

/**
 * @brief The launch of @p config on @p problem on @p device, or nullopt when the config does
 *        not compute it correctly or cannot launch it there.
 *
 * Reads and gates every field of @p config: the atom (known, its atom_id, running on the
 * device, taking the graph's element type; "fma" exactly for loops 3 and 4, on fp32 graphs, or on
 * 16-bit graphs where runsMatrixAtomFor finds no atom for the type); the loop
 * (0 any stride; 1 to 3 height stride 1 with fr the filter height, or 1 at height stride sh > 1
 * and unit height dilation with fr = ceil(filter height / sh), one grid row per height phase;
 * 2 gfx950 and width stride 1;
 * 3 width stride 1, rows of x and dy whole vectors); the layout, one the graph's x and dy are
 * packed in (channels-last: channels per group whole vectors; 2: gfx950, 16-bit, an atom whose
 * lanes hold 8 positions, not loop 3; loop 2 takes 0 or 2); the tile against the atom (bk, bc
 * whole fragments per wave, bq whole steps; wk, wc >= 1 and tk = tc = 0) or the FMA thread tile
 * (wk = wc = 0; tk, tc multiples of 4 dividing bk, bc; bk, bc multiples of 4) and the vector (16
 * bytes; channels-first: bq whole vectors, vec whole width-stride phases); s, sw, dw equal to
 * the filter's width, its stride and dilation; fr >= 1; xcd 0, or 1 on gfx942 / gfx950 (loops
 * 0 to 2); align 0, or 1 for loop 1, channels-first, sw 1 and widths of x and dy whole vectors,
 * or 2 for loop 1, layout 1 and output channels per group not whole vectors (which the other
 * configs require);
 * ns 2 to 4 for loop 2, else 0; dws 0 or 1 (0 for loop 5); the block within 1024 threads and
 * its load passes within 32; the LDS within the device's; images of at least one vector (channels-first); the grids within
 * the launch limits (x, and the reduce, 2^31 - 1; y and z 65535); bpu 1, 2, 4 or 8. The splits follow
 * splitsWanted at bpu blocks per unit, so a device that reports no dispatch units is declined.
 *
 * Loop 5 (direct): gfx950, layout 2, a 16 x 16 x 32 atom, bq 32, wq 1, 2 or 4 with wk wc wq =
 * 8 (512 threads), ns 2 to 5 with (ns - 2) loads per row within 63, unit width stride, unit
 * height dilation (or a one-row filter), bc and bk powers of two from 32 to 512 whose rows have a
 * conflict-free rotation (ConvWrwGeom.h), bpu 1; its grid follows directPlanFor. Every other
 * loop takes wq 0.
 *
 * Loop 6 (small-C rows): loop 0 on a matrix atom with the N axis the (input channel, filter-width
 * tap) pairs: channels-first, at most 16 input channels per group, bc at most 16, wc 1, bk whole
 * fragments per wave, bq whole atom steps.
 */
std::optional<WrwLaunch> wrwLaunchFor(const WrwProblem& problem,
                                      const WrwConfig& config,
                                      const hipdnn_plugin_sdk::ingestor::DeviceProperties& device);

/**
 * @brief The preprocessor definitions ConvWrw.cpp is compiled with for @p config on
 *        @p dataType (CONV_K3_T, CONV_K3_ATOM = @p kernelType unless empty (the FMA ring), and
 *        one CONV_K3_<FIELD> per integer field), in a fixed order. Both entry points compile
 *        from the same definitions.
 *
 * @throws std::invalid_argument for an element type the kernels do not take.
 */
std::vector<std::pair<std::string, std::string>>
    compileDefinesFor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                      const WrwConfig& config,
                      std::string_view kernelType);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_wrw
