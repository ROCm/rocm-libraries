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
 * channels-last (N D H W C), or (the generic loop) through any strides, and writes fp32 partials [split][g][tap][k][c] into the workspace;
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
inline constexpr int64_t LOOP_ROWS_PACKED = 7; ///< output rows, several narrow rows a step
inline constexpr int64_t LOOP_GENERIC = 8; ///< generic taps: runtime filter, strides (coverage)

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
    /// 1: block-based buffers (CONV_K3_PB), for images past the 32-bit buffer range and admitted
    /// only there: the channels-first K3Tiles loops (0, 1, 6), whose x and dy buffers start at
    /// the block's first channel (its channel planes within the range), and the generic loop,
    /// whose buffers start each step at the first element the block can read (its step windows
    /// within the range; any layout and strides). 0 elsewhere.
    int64_t pb = 0;
    /// The split planner's target blocks per dispatch unit (1, 2, 4 or 8; 1 for loop 5, whose
    /// persistent grid has one workgroup per unit; 8 for small FMA blocks): host-side only.
    int64_t bpu = 0;
    /// Loop 8 only (the "spec" metadata string): the specialized geometry classes the catalog
    /// serves on this config's archs, which the generic config declines. ';'-separated keys
    /// "<atom_id><kind><layout>.<s>.<sw>.<dw>.<fr>", kind R (loop 0, any filter height), C
    /// (loop 6, at most 16 input channels per group), P (loop 1, the filter height fr, height
    /// stride 1 or phases), U (P with align 2: channels-last output channels in any count), G
    /// (loop 2, gfx950, fr at height stride 1) or D (loop 5, gfx950, as G with unit width
    /// stride and unit height dilation or one filter row); fr 0 for R and C. F, S, Q (atom fma
    /// alone, for the generic FMA form): the FMA output rows of vectors (channels-first rows of
    /// whole vectors), of scalar units, and the FMA ring (fr at height stride 1). Lower case r, p, c:
    /// their block-based buffer forms, which serve large images alone (channels-first, each
    /// channel plane within the range); the upper-case kinds serve the other images.
    std::string spec;
    /// 0: the graph's element type. 1: three-bf16 for an fp32 graph under TF32 permission
    /// (tf32Permitted): a bf16 config run on x and dy split into 3N bf16 images by
    /// ConvWrwSplit (K3SplitArgs), dw written in fp32 by the reduce.
    int64_t math = 0;
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
    /// One image of x or of dy spans past the 32-bit buffer range: only block-based buffer
    /// configs (pb 1) serve it.
    bool largeImage = false;
    /// x and dy have unit channel strides, every other stride a whole number of 16-byte vectors
    /// and channels per group whole vectors (packed or not): the generic loop's channels-last
    /// configs (16-byte channel loads) serve exactly these, its channels-first ones the rest.
    bool vectorChannels = false;
    /// Elements one image of x / of dy spans through its strides (the packed image when packed).
    int64_t xSpan = 0;
    int64_t ySpan = 0;
};

/// Whether specialized key @p key (WrwConfig::spec) serves @p problem on @p gcnArchName: its
/// atom runs there and takes the graph's element type (G and D, and layout 2, on gfx950 only),
/// the graph's x and dy are packed in its layout (channels-last: output channels per group
/// whole 16-byte vectors except for U, input channels except for C) and within the 32-bit buffer
/// range, its width class is the filter's (width, stride, dilation) and its filter-height rule
/// holds. nullopt for a malformed key.
std::optional<bool>
    specServes(const WrwProblem& problem, std::string_view key, std::string_view gcnArchName);

/**
 * @brief The backward-weights convolution @p context's graph describes, or nullopt when the
 *        kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates (both convolution modes), plus: fp32, fp16 or bf16;
 * x and dy both packed channels-first or both packed channels-last, or (for the generic loop
 * only, which reads them through their strides) any strides; dw packed K C T R S or
 * K T R S C; the batch within 32 bits; per spatial axis the reach of the kernel's 32-bit
 * position arithmetic ((output extent + 512) x stride + filter x dilation + padding) within 32
 * bits, which bounds every extent, padding, stride and dilation; the taps within 32 bits. An
 * image of x or dy spanning past 2^31 bytes is a large image (WrwProblem::largeImage), for the
 * block-based buffer configs alone.
 */
std::optional<WrwProblem> parseWrwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// LDS bytes a config's ConvWrw kernel declares: k3LdsBytes (ConvWrwGeom.h), the arithmetic the
/// kernels assert their tiles against.
int64_t ldsBytesOf(const WrwConfig& config, int elementBytes);

/// The most load passes (loads per thread and step, each tracked by a bit of a 32-bit mask) of
/// a config's dy and x tiles; @p wave is the atom's wave width (ignored for loop 3). Loop 7's
/// element units (K3PackTiles), tracked by no mask; loop 8's loads a thread holds (elements, or
/// 16-byte channel vectors at each position of a 32-bit LDS word).
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
    /// Three-bf16 (math 1): ConvWrwSplit runs first and writes the split operands to the
    /// workspace's first splitBytes (xs, then ys at splitYOffset); the partials follow them.
    bool split = false;
    unsigned int splitGrid = 0;
    size_t splitYOffset = 0;
    size_t splitBytes = 0;
    K3SplitArgs splitArgs{};
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
 * tap) pairs: at most 16 input channels per group, bc at most 16, wc 1, bk whole fragments per
 * wave, bq whole atom steps; channels-first, or channels-last (layout 1: any input channel count,
 * output channels per group whole vectors, bq whole vectors and vec whole width-stride phases,
 * as the channels-first tiles it stages into).
 *
 * Loop 7 (packed rows): loop 0 on a matrix atom, channels-first, with a step's bq positions
 * holding several output rows of one plane, each in a virtual row of wv = wo + (s - 1) dw / sw
 * columns: only for narrow outputs, 2 wv at most bq; the filter width whole (s the filter
 * width), bq dividing the block's threads, align 0. Its units are elements (no pass masks): at
 * most 64 passes.
 *
 * Loop 8 (generic taps, the coverage tier): a matrix atom, or "fma" (the FMA form: tk x tc
 * thread tiles, wk wc 0, only where no matrix atom takes the element, not for fewer than 8
 * input channels a group or width strides of 4 or more); the filter, strides and dilations
 * runtime (s, sw 0), fr the taps a block (>= 1); dw 0 (per-tap gathers) or the extra positions
 * of a shared width window (layout 1, a multiple of 8, unit width stride, (min(fr, filter
 * width) - 1) x width dilation within dw; fr width taps of one filter row a block, so
 * ft fr ceil(fs / fr) grid rows a group); layout 1 exactly for problems with
 * vectorChannels, layout 0 for the rest (any strides, read element by element); bq dividing the
 * block's threads, bk and bc whole channel rows a pass; align, ns and wq 0; pb 1 (large images)
 * with each step's windows within the range: x 1 + (min(bc, cg) - 1) xs1 + the filter's depth
 * and height reach (within the image) x xs2, xs3 + ((bq - 1) sw + (fs - 1) dw + dw) columns
 * (within the image) x xs4, dy 1 + (min(bk, kg) - 1) ys1 + (bq - 1) ys4; at most 64
 * loads a thread in flight; not for filters of at most 9 taps dilated 6 or more on a filter
 * axis (ASPP-class, where the BLAS path wins). Declined
 * wherever a key of its spec serves the problem (specServes): a specialized config's class.
 * math 0, or 1 (three-bf16): an fp32 graph, TF32 permitted (tf32Permitted), a bf16 atom, dws 0,
 * 3N within 32 bits; the launch is the config's on the bf16 problem of 3N images, with
 * ConvWrwSplit ahead of it and the split operands ahead of the partials in the workspace.
 */
std::optional<WrwLaunch> wrwLaunchFor(const WrwProblem& problem,
                                      const WrwConfig& config,
                                      const hipdnn_plugin_sdk::ingestor::DeviceProperties& device);

/**
 * @brief Whether three-bf16 (math 1, x and dy as 3N bf16 images) is admitted for @p problem on
 *        @p gcnArchName: the classes where it measured faster than exact fp32 K3
 *        (~/old-asic-hipdnn/tf32/PLAN.md). The 3N images triple the operand traffic, so it pays
 *        only where the bf16 kernel is about 3x the fp32 one or the exact path is weak.
 *
 * gfx942, gfx950: unit stride, at least 9 filter taps, at least 128 input and 64 output channels
 * per group. gfx11, gfx12 (exact fp32 on FMA): at most 4 input channels per group (stems), or at
 * least 256 with at least 9 taps. Elsewhere (gfx90a, whose bf16 rate is only 4x fp32 MFMA): never.
 */
bool threeBf16Pays(const WrwProblem& problem, std::string_view gcnArchName);

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
