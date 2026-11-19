// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvWrwProblem.hpp"

// The kernels' LDS arithmetic, shared verbatim with them.
#include "engines/kernel_ingestor_engine/descriptors/conv/wrw/kernels/ConvWrwGeom.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/DeviceUnits.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_wrw
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
constexpr int64_t UINT_LIMIT = std::numeric_limits<unsigned int>::max();
constexpr int64_t MAX_GRID_YZ = 65535;
constexpr int64_t MAX_BLOCK = 1024;
/// The kernel tracks each thread's load passes in a 32-bit mask.
constexpr int64_t MAX_PASSES = 32;
/// Packed rows' element units carry no pass mask; their registers bound them.
constexpr int64_t MAX_PACK_PASSES = 64;
/// Positions past an output row a block's chunk may reach (bq at most 512).
constexpr int64_t CHUNK_MARGIN = 512;
/// The generic loop declines filters of at most this many taps dilated at least this far.
constexpr int64_t GENERIC_SPARSE_MAX_TAPS = 9;
constexpr int64_t GENERIC_SPARSE_MIN_DILATION = 6;
/// The generic loop declines grouped layers whose groups have fewer input and fewer output
/// channels than this (depthwise, ResNeXt-class narrow groups).
constexpr int64_t GENERIC_MIN_GROUP_CHANNELS = 16;
/// The generic loop declines fewer input channels a group (stems), and output rows narrower
/// than this many positions.
constexpr int64_t GENERIC_MIN_CHANNELS = 8;
constexpr int64_t GENERIC_MIN_OUTPUT_WIDTH = 4;
/// The generic FMA form declines this width stride and past.
constexpr int64_t GENERIC_FMA_MAX_STRIDE = 4;
/// The reduce: 256 threads, 64 four-channel quads of dw per block.
constexpr int64_t REDUCE_QUADS_PER_BLOCK = 64;
/// The tile reduce's widest channel tile, its narrowest when shrunk for blocks, its most output
/// channels a tile, and the blocks per dispatch unit it aims for.
constexpr int64_t REDUCE_TILE_MAX_CHANNELS = 128;
constexpr int64_t REDUCE_TILE_MIN_CHANNELS = 16;
constexpr int64_t REDUCE_TILE_MAX_K = 4;
constexpr int64_t REDUCE_TILE_BLOCKS_PER_UNIT = 2;

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

/// True when @p strides are packed for @p extents with the axes in @p order (slowest first);
/// extent-1 axes may carry any stride.
// NOLINTNEXTLINE(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
bool packedInOrder(const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
                   const std::array<long long, CONV_NAIVE_TENSOR_DIMS>& extents,
                   const std::array<size_t, CONV_NAIVE_TENSOR_DIMS>& order)
{
    long long expected = 1;
    for(size_t i = CONV_NAIVE_TENSOR_DIMS; i-- > 0;)
    {
        const size_t axis = order[i];
        if(extents[axis] != 1 && strides[axis] != expected)
        {
            return false;
        }
        expected *= extents[axis];
    }
    return true;
}

/// The elements one image (n fixed) of an operand of @p extents spans through @p strides (non-
/// negative, as the base parse admits them): 1 + the sum over the other axes of (extent - 1) x
/// stride. nullopt when the sum overflows.
// NOLINTNEXTLINE(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
std::optional<int64_t> imageSpan(const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
                                 const std::array<long long, CONV_NAIVE_TENSOR_DIMS>& extents)
{
    int64_t span = 1;
    for(size_t axis = 1; axis < CONV_NAIVE_TENSOR_DIMS; ++axis)
    {
        // An axis of extent 1, or of stride 0, reaches nothing (checkedMultiply takes positive
        // factors only).
        if(extents[axis] == 1 || strides[axis] == 0)
        {
            continue;
        }
        const auto reach = pack_features::checkedMultiply(extents[axis] - 1, strides[axis]);
        if(!reach.has_value() || *reach > std::numeric_limits<int64_t>::max() - span)
        {
            return std::nullopt;
        }
        span += *reach;
    }
    return span;
}

constexpr std::array<size_t, CONV_NAIVE_TENSOR_DIMS> CHANNELS_FIRST_ORDER{0, 1, 2, 3, 4};
constexpr std::array<size_t, CONV_NAIVE_TENSOR_DIMS> CHANNELS_LAST_ORDER{0, 2, 3, 4, 1};

/// Packed element strides (n, c, d, h, w) of an operand of @p extents in a layout.
void packedStrides(
    const std::array<long long, CONV_NAIVE_TENSOR_DIMS>& extents,
    bool channelsLast,
    // NOLINTNEXTLINE(modernize-avoid-c-arrays) -- mirrors the kernel argument block's C arrays
    long long (&strides)[5])
{
    const auto& order = channelsLast ? CHANNELS_LAST_ORDER : CHANNELS_FIRST_ORDER;
    long long running = 1;
    for(size_t i = CONV_NAIVE_TENSOR_DIMS; i-- > 0;)
    {
        strides[order[i]] = running;
        running *= extents[order[i]];
    }
}

/// The atoms' shapes (DIM, KA, KPL), in ATOMS order.
constexpr std::array<AtomShape, ATOMS.size()> SHAPES{{
    {16, 16, 4}, // mfma16_f16: v_mfma_f32_16x16x16_f16
    {16, 16, 4}, // mfma16_bf16: v_mfma_f32_16x16x16_bf16 (1k)
    {16, 8, 2}, // mfma16_bf16_908: v_mfma_f32_16x16x8_bf16
    {16, 4, 1}, // mfma16_f32: v_mfma_f32_16x16x4_f32
    {32, 8, 4}, // mfma32_f16
    {32, 8, 4}, // mfma32_bf16
    {32, 4, 2}, // mfma32_bf16_908
    {32, 2, 1}, // mfma32_f32
    {16, 32, 8}, // mfma16x32_f16 (gfx950)
    {16, 32, 8}, // mfma16x32_bf16 (gfx950)
    {32, 16, 8}, // mfma32x16_f16 (gfx950)
    {32, 16, 8}, // mfma32x16_bf16 (gfx950)
    {16, 16, 16}, // wmma11_f16
    {16, 16, 16}, // wmma11_bf16
    {16, 16, 8}, // wmma12_f16
    {16, 16, 8}, // wmma12_bf16
    {0, 0, 0}, // fma: no matrix core
    {16, 32, 16}, // wmma125_f16: native gfx1250 16x16x32
    {16, 32, 16}, // wmma125_bf16: native gfx1250 16x16x32
    {16, 4, 2}, // wmma125_f32: native gfx1250 16x16x4
}};

} // namespace

std::optional<WrwProblem> parseWrwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context)
{
    auto base = conv_naive::parseConvNaiveProblem(context, ConvDirection::BACKWARD_WEIGHTS);
    if(!base.has_value())
    {
        return std::nullopt;
    }
    int elementBytes = 0;
    switch(base->dataType)
    {
    case data_objects::DataType::FLOAT:
        elementBytes = 4;
        break;
    case data_objects::DataType::HALF:
    case data_objects::DataType::BFLOAT16:
        elementBytes = 2;
        break;
    default:
        return std::nullopt;
    }
    const auto& a = base->args;
    const long long cg = a.c / a.groups;
    // x and dy: both packed channels-first or both packed channels-last (every loop), or any
    // strides (the buffer generic loop 8 and native scalar gather loop 11).
    const std::array<long long, CONV_NAIVE_TENSOR_DIMS> xExtents{
        a.n, a.c, a.inSpatial[0], a.inSpatial[1], a.inSpatial[2]};
    const std::array<long long, CONV_NAIVE_TENSOR_DIMS> yExtents{
        a.n, a.k, a.outSpatial[0], a.outSpatial[1], a.outSpatial[2]};
    const std::array<long long, CONV_NAIVE_TENSOR_DIMS> wExtents{
        a.k, cg, a.filter[0], a.filter[1], a.filter[2]};
    const bool channelsFirst = packedInOrder(a.xStrides, xExtents, CHANNELS_FIRST_ORDER)
                               && packedInOrder(a.yStrides, yExtents, CHANNELS_FIRST_ORDER);
    const bool channelsLast = packedInOrder(a.xStrides, xExtents, CHANNELS_LAST_ORDER)
                              && packedInOrder(a.yStrides, yExtents, CHANNELS_LAST_ORDER);
    // Otherwise (the generic loop only) any positive strides: an image spans its strides' reach.
    const auto xSpan = imageSpan(a.xStrides, xExtents);
    const auto ySpan = imageSpan(a.yStrides, yExtents);
    if(!xSpan.has_value() || !ySpan.has_value())
    {
        return std::nullopt;
    }
    // dw: packed K C T R S or K T R S C (the reduce writes either through its strides).
    const bool filterChannelsFirst = packedInOrder(a.wStrides, wExtents, CHANNELS_FIRST_ORDER);
    const bool filterChannelsLast = packedInOrder(a.wStrides, wExtents, CHANNELS_LAST_ORDER);
    if(!filterChannelsFirst && !filterChannelsLast)
    {
        return std::nullopt;
    }
    // The argument block holds these extents in 32 bits.
    if(a.n > INT_LIMIT || a.c > INT_LIMIT || a.k > INT_LIMIT || a.groups > INT_LIMIT
       || a.inSpatial[0] > INT_LIMIT || a.inSpatial[1] > INT_LIMIT || a.inSpatial[2] > INT_LIMIT)
    {
        return std::nullopt;
    }
    // Per axis, the positions the kernel forms in 32 bits: an output position (with a chunk's
    // margin) times the stride, plus the filter's reach and the padding. Bounds every extent,
    // padding, stride and dilation (all non-negative after the base parse).
    for(int axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
    {
        const auto outReach
            = pack_features::checkedMultiply(a.outSpatial[axis] + CHUNK_MARGIN, a.stride[axis]);
        const auto filterReach = pack_features::checkedMultiply(a.filter[axis], a.dilation[axis]);
        if(!outReach.has_value() || !filterReach.has_value() || *outReach > INT_LIMIT - *filterReach
           || *outReach + *filterReach > INT_LIMIT - a.prePadding[axis])
        {
            return std::nullopt;
        }
    }
    const auto taps = pack_features::checkedProduct({a.filter[0], a.filter[1], a.filter[2]});
    if(!taps.has_value() || *taps > INT_LIMIT)
    {
        return std::nullopt;
    }
    // One image of x and of dy is one 32-bit-addressed buffer; or (channels-first, block-based
    // buffers) each channel plane is, and a block's buffers span its channel planes.
    const auto xImage
        = pack_features::checkedProduct({a.c, a.inSpatial[0], a.inSpatial[1], a.inSpatial[2]});
    const auto yImage
        = pack_features::checkedProduct({a.k, a.outSpatial[0], a.outSpatial[1], a.outSpatial[2]});
    const auto xBytes = pack_features::checkedProduct({a.n, xImage.value_or(0), elementBytes});
    const auto yBytes = pack_features::checkedProduct({a.n, yImage.value_or(0), elementBytes});
    const auto wBytes = pack_features::checkedProduct({a.k, cg, *taps, elementBytes});
    if(!xImage.has_value() || !yImage.has_value() || !xBytes.has_value() || !yBytes.has_value()
       || !wBytes.has_value())
    {
        return std::nullopt;
    }
    // An image whose span passes the 32-bit buffer range: only the block-based buffer configs
    // (pb 1) serve it, each with its own window gate.
    const bool largeImage = *xSpan > INT_LIMIT / elementBytes || *ySpan > INT_LIMIT / elementBytes;
    // Unit channel strides, every other stride (of an axis past 1) whole 16-byte vectors, and
    // channels per group whole vectors: the generic loop's channel-vector loads.
    const long long v = 16 / elementBytes;
    const long long kg = a.k / a.groups;
    bool vectorChannels = a.xStrides[1] == 1 && a.yStrides[1] == 1 && cg % v == 0 && kg % v == 0;
    for(size_t axis = 0; axis < CONV_NAIVE_TENSOR_DIMS; ++axis)
    {
        if(axis != 1
           && ((xExtents[axis] != 1 && a.xStrides[axis] % v != 0)
               || (yExtents[axis] != 1 && a.yStrides[axis] % v != 0)))
        {
            vectorChannels = false;
        }
    }
    WrwProblem problem;
    problem.elementBytes = elementBytes;
    problem.channelsFirst = channelsFirst;
    problem.channelsLast = channelsLast;
    problem.filterChannelsLast = !filterChannelsFirst;
    problem.taps = *taps;
    problem.largeImage = largeImage;
    problem.vectorChannels = vectorChannels;
    problem.xSpan = *xSpan;
    problem.ySpan = *ySpan;
    problem.xUid = base->operands.x;
    problem.dyUid = base->operands.y;
    problem.dwUid = base->operands.w;
    problem.base = *base;
    return problem;
}

bool runsMatrixAtomFor(std::string_view element, std::string_view gcnArchName)
{
    return std::any_of(ATOMS.begin(), ATOMS.end(), [&](std::string_view atom) {
        return atom != "fma" && conv_igemm::atomElement(atom) == element
               && conv_igemm::atomRunsOn(atom, gcnArchName);
    });
}

int64_t atomIdOf(std::string_view atom)
{
    const auto it = std::find(ATOMS.begin(), ATOMS.end(), atom);
    return it == ATOMS.end() ? -1 : static_cast<int64_t>(it - ATOMS.begin());
}

AtomShape atomShapeOf(std::string_view atom)
{
    const auto id = atomIdOf(atom);
    return id < 0 ? AtomShape{} : SHAPES[static_cast<size_t>(id)];
}

int64_t ldsBytesOf(const WrwConfig& config, int elementBytes)
{
    // Every field as int, as the kernels instantiate it; a field outside int declines first.
    for(const int64_t value : {config.loop,
                               config.layout,
                               config.bk,
                               config.bc,
                               config.bq,
                               config.fr,
                               config.s,
                               config.sw,
                               config.dw,
                               config.wk,
                               config.wc,
                               config.tk,
                               config.tc,
                               config.vec,
                               config.align,
                               config.ns,
                               config.wq})
    {
        if(value < 0 || value > INT_LIMIT)
        {
            return std::numeric_limits<int64_t>::max();
        }
    }
    if(config.loop == LOOP_ACCURATE)
    {
        if((config.bk != 8 && config.bk != 16) || (config.bc != 8 && config.bc != 16)
           || (config.bq != 32 && config.bq != 64) || elementBytes != 4)
        {
            return std::numeric_limits<int64_t>::max();
        }
        return (config.bk + config.bc) * (config.bq + 1) * static_cast<int64_t>(sizeof(float));
    }
    const auto i = [](int64_t value) { return static_cast<int>(value); };
    // The matrix-core ring's channels-last pitch on CDNA follows its fragment reads' bytes.
    const bool cdna = config.atom.rfind("mfma", 0) == 0;
    const int fragmentBytes
        = cdna ? static_cast<int>(atomShapeOf(config.atom).kpl) * elementBytes : 0;
    return k3LdsBytes(i(config.loop),
                      i(config.layout),
                      elementBytes,
                      i(config.bk),
                      i(config.bc),
                      i(config.bq),
                      i(config.fr),
                      i(config.s),
                      i(config.sw),
                      i(config.dw),
                      i(config.wk),
                      i(config.wc),
                      i(config.tk),
                      i(config.tc),
                      i(config.vec),
                      i(config.align),
                      i(config.ns),
                      i(config.wq),
                      fragmentBytes);
}

int64_t loadPassesOf(const WrwConfig& config, int elementBytes, int64_t wave)
{
    // Scalar gathers have no register load-pass bitmask or outstanding inline-asm loads.
    if(config.loop == LOOP_NATIVE_GENERIC || config.loop == LOOP_TAP_TILED
       || config.loop == LOOP_SMALL_K || config.loop == LOOP_ACCURATE)
    {
        return 0;
    }
    const int64_t vec = config.vec;
    if(config.loop == LOOP_GENERIC)
    {
        // k3WrwGeneric's loads a thread holds: channels-first, a position's bk and fr bc channel
        // rows over threads / bq rows a pass; channels-last, units of vec channels at kw
        // positions (one 32-bit LDS word), kw loads each.
        // (The FMA form: tk x tc channels a thread.)
        const int64_t threads = config.tk > 0 && config.tc > 0
                                    ? (config.bk / config.tk) * (config.bc / config.tc)
                                    : config.wk * config.wc * wave;
        if(threads < 1 || config.bq < 1 || vec < 1 || threads % config.bq != 0)
        {
            return std::numeric_limits<int64_t>::max();
        }
        if(config.layout == LAYOUT_CHANNELS_FIRST)
        {
            const int64_t rows = threads / config.bq;
            return divUp(config.bk, rows) + divUp(config.fr * config.bc, rows);
        }
        const int64_t kw = 4 / elementBytes;
        const int64_t words = divUp(config.bq, kw);
        // dw > 0: one window of bq + dw positions for every tap.
        const int64_t xUnits = config.dw > 0
                                   ? divUp(config.bc, vec) * divUp(config.bq + config.dw, kw)
                                   : config.fr * divUp(config.bc, vec) * words;
        return (divUp(divUp(config.bk, vec) * words, threads) + divUp(xUnits, threads)) * kw;
    }
    if(config.loop == LOOP_DIRECT)
    {
        // The x and dy rounds of a row over an item's 8 / wq waves (k3dLoads).
        if((config.wq != 1 && config.wq != 2 && config.wq != 4) || config.bc < 1 || config.bc > 512
           || config.bk < 1 || config.bk > 512 || config.s < 1 || config.s > 64 || config.dw < 1
           || config.dw > 64)
        {
            return std::numeric_limits<int64_t>::max();
        }
        return k3dLoads(static_cast<int>(config.bc),
                        static_cast<int>(config.bk),
                        static_cast<int>(config.wq),
                        static_cast<int>(config.s),
                        static_cast<int>(config.dw));
    }
    if(config.loop == LOOP_ROWS_FMA)
    {
        // K3FmaRowTiles: as the FMA ring's, over fr windows of the strided window rows.
        const int64_t threads = (config.bk / config.tk) * (config.bc / config.tc);
        const int64_t win = (config.bq - 1) * config.sw + (config.s - 1) * config.dw + 1;
        const int64_t nbv = divUp(win, vec) + (vec > 1 ? 1 : 0);
        const bool cf = config.layout == LAYOUT_CHANNELS_FIRST;
        const int64_t aUnits
            = cf ? (config.bk / 4) * (config.bq / vec) : config.bq * (config.bk / vec);
        const int64_t bUnits
            = cf ? config.fr * (config.bc / 4) * nbv : config.fr * nbv * vec * (config.bc / vec);
        return std::max(divUp(aUnits, threads), divUp(bUnits, threads));
    }
    if(config.loop == LOOP_RING_FMA)
    {
        // K3FmaTiles: channels-first units are 4 channels by a vector of positions; channels-last
        // units a position's vector of channels; over the dy chunk and the aligned x window.
        const int64_t threads = (config.bk / config.tk) * (config.bc / config.tc);
        const int64_t win = config.bq + (config.s - 1) * config.dw;
        const int64_t nbv = divUp(win, vec) + 1;
        const bool cf = config.layout == LAYOUT_CHANNELS_FIRST;
        const int64_t aUnits
            = cf ? (config.bk / 4) * (config.bq / vec) : config.bq * (config.bk / vec);
        const int64_t bUnits = cf ? (config.bc / 4) * nbv : nbv * vec * (config.bc / vec);
        return std::max(divUp(aUnits, threads), divUp(bUnits, threads));
    }
    const int64_t waves = config.wk * config.wc;
    if(config.loop == LOOP_RING_DMA)
    {
        // K3DmaGeom / K3DmaTrGeom: 64-unit groups of 16 bytes over the waves.
        const int64_t win = config.bq + (config.s - 1) * config.dw;
        int64_t aGroups = 0;
        int64_t bGroups = 0;
        if(config.layout == LAYOUT_CHANNELS_LAST_TR)
        {
            aGroups = divUp(config.bq * (k3TrPitch(static_cast<int>(config.bk)) / 8), 64);
            bGroups = divUp(win * (k3TrPitch(static_cast<int>(config.bc)) / 8), 64);
        }
        else
        {
            const int64_t v = 16 / elementBytes;
            aGroups = divUp(config.bk * (config.bq / v + 1), 64);
            bGroups = divUp(config.bc * (divUp(win + 2, v) | 1), 64);
        }
        return std::max(divUp(aGroups, waves), divUp(bGroups, waves));
    }
    const int64_t threads = waves * wave;
    if(config.loop == LOOP_ROWS_PACKED)
    {
        // K3PackTiles: dy units one element of the bk x bq tile; x units one element of the
        // channels-first tiles' window (fr rows, bc channels, nbv vectors of positions).
        const int64_t win = (config.bq - 1) * config.sw + (config.s - 1) * config.dw + 1;
        const int64_t span = divUp(win, vec) * vec;
        return std::max(divUp(config.bk * config.bq, threads),
                        divUp(config.fr * config.bc * span, threads));
    }
    if(config.loop == LOOP_ROWS_SMALLC && config.layout == LAYOUT_CHANNELS_LAST)
    {
        // K3SmallClTiles: dy units KV positions by a vector of output channels; x units one
        // element of the channels-first tiles' window (fr rows, sw phases of ph positions, bc
        // channels).
        const int64_t kv = 4 / elementBytes;
        const int64_t win = (config.bq - 1) * config.sw + (config.s - 1) * config.dw + 1;
        const int64_t ph = divUp(win, vec) * vec / config.sw;
        const int64_t aUnits = (config.bq / kv) * (config.bk / vec);
        const int64_t bUnits = config.fr * config.sw * ph * config.bc;
        return std::max(divUp(aUnits, threads), divUp(bUnits, threads));
    }
    // K3Tiles: channels-first units are a vector of positions of one channel row; channels-last
    // ones KV positions (1 for layout 2) by a vector of channels.
    const bool cf = config.layout == LAYOUT_CHANNELS_FIRST;
    const int64_t kv = config.layout == LAYOUT_CHANNELS_LAST ? 4 / elementBytes : 1;
    const int64_t rows
        = config.loop == LOOP_ROWS || config.loop == LOOP_ROWS_SMALLC ? config.fr : 1;
    const int64_t win = (config.bq - 1) * config.sw + (config.s - 1) * config.dw + 1;
    const int64_t nbv = divUp(win + (config.align != 0 ? vec - 1 : 0), vec);
    const int64_t phRaw = cf ? nbv * vec / config.sw : divUp(win, config.sw);
    const int64_t ph = divUp(phRaw, kv) * kv;
    const int64_t aUnits
        = cf ? config.bk * (config.bq / vec) : (config.bq / kv) * (config.bk / vec);
    const int64_t bUnits
        = cf ? rows * config.bc * nbv : rows * config.sw * (ph / kv) * (config.bc / vec);
    return std::max(divUp(aUnits, threads), divUp(bUnits, threads));
}

std::optional<bool>
    specServes(const WrwProblem& problem, std::string_view key, std::string_view gcnArchName)
{
    // "<atom_id><kind><layout>.<s>.<sw>.<dw>.<fr>"
    size_t at = 0;
    const auto readInt = [&](int64_t& value) {
        const auto [end, ec] = std::from_chars(key.data() + at, key.data() + key.size(), value);
        if(ec != std::errc() || end == key.data() + at)
        {
            return false;
        }
        at = static_cast<size_t>(end - key.data());
        return true;
    };
    const auto readDotInt = [&](int64_t& value) {
        if(at >= key.size() || key[at] != '.')
        {
            return false;
        }
        ++at;
        return readInt(value);
    };
    int64_t atomId = -1;
    int64_t layout = -1;
    int64_t s = 0;
    int64_t sw = 0;
    int64_t dw = 0;
    int64_t fr = -1;
    // The legacy generic tier's spec namespace ends at FMA; appended native atoms use their
    // own qualified runtime paths, not specialized-class keys.
    constexpr int64_t SPEC_FMA_ATOM_ID = 16;
    static_assert(ATOMS[SPEC_FMA_ATOM_ID] == "fma");
    if(!readInt(atomId) || atomId < 0 || atomId > SPEC_FMA_ATOM_ID || at >= key.size())
    {
        return std::nullopt;
    }
    // Lower case r, p, c: the block-based buffer (pb) forms of R, P, C. F, S, Q: the FMA loops
    // (the "fma" atom alone): output rows of vectors, of scalar units, the FMA ring.
    const char letter = key[at++];
    const bool blockBuffers = letter == 'r' || letter == 'p' || letter == 'c';
    const char kind = blockBuffers ? static_cast<char>(letter - 'a' + 'A') : letter;
    const bool fmaKind = kind == 'F' || kind == 'S' || kind == 'Q';
    const bool ring = kind == 'P' || kind == 'U' || kind == 'G' || kind == 'D' || kind == 'Q';
    const bool fmaAtom = atomId == SPEC_FMA_ATOM_ID;
    if((kind != 'R' && kind != 'C' && kind != 'F' && kind != 'S' && !ring) || fmaKind != fmaAtom
       || !readInt(layout) || layout < 0 || layout > 2 || !readDotInt(s) || !readDotInt(sw)
       || !readDotInt(dw) || !readDotInt(fr) || at != key.size() || s < 1 || sw < 1 || dw < 1
       || (ring ? fr < 1 : fr != 0))
    {
        return std::nullopt;
    }
    const std::string_view atom = ATOMS[static_cast<size_t>(atomId)];
    const auto& a = problem.base.args;
    const bool gfx950 = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName) == "gfx950";
    const auto element = conv_igemm::dataElement(problem.base.dataType);
    // The FMA loops take fp32 everywhere and 16-bit only where no matrix atom takes it.
    if(!conv_igemm::atomRunsOn(atom, gcnArchName)
       || (fmaAtom ? element != "f32" && runsMatrixAtomFor(element, gcnArchName)
                   : conv_igemm::atomElement(atom) != element)
       || ((kind == 'G' || kind == 'D' || layout == LAYOUT_CHANNELS_LAST_TR) && !gfx950))
    {
        return false;
    }
    const int64_t vec = 16 / problem.elementBytes;
    const int64_t kg = a.k / a.groups;
    const int64_t cg = a.c / a.groups;
    if((layout == LAYOUT_CHANNELS_FIRST ? !problem.channelsFirst : !problem.channelsLast)
       || (layout != LAYOUT_CHANNELS_FIRST && kind != 'S'
           && ((kind != 'U' && kg % vec != 0) || (kind != 'C' && cg % vec != 0)))
       || problem.largeImage != blockBuffers || s != a.filter[2] || sw != a.stride[2]
       || dw != a.dilation[2])
    {
        return false;
    }
    // The FMA loops' vectors: channels-first rows of x and dy whole vectors.
    if((kind == 'F' || kind == 'Q') && layout == LAYOUT_CHANNELS_FIRST
       && (a.inSpatial[2] % vec != 0 || a.outSpatial[2] % vec != 0))
    {
        return false;
    }
    // Block-based buffers: channels-first, each channel plane of x and dy within the range.
    if(blockBuffers
       && (layout != LAYOUT_CHANNELS_FIRST
           || a.inSpatial[0] * a.inSpatial[1] * a.inSpatial[2] > INT_LIMIT / problem.elementBytes
           || a.outSpatial[0] * a.outSpatial[1] * a.outSpatial[2]
                  > INT_LIMIT / problem.elementBytes))
    {
        return false;
    }
    const bool wholeHeight = a.stride[1] == 1 && fr == a.filter[1];
    switch(kind)
    {
    case 'C':
        return cg <= 16;
    case 'P':
    case 'U':
        // Or height phases (unit height dilation): ceil(filter height / sh) rows each.
        return wholeHeight || (a.dilation[1] == 1 && fr == divUp(a.filter[1], a.stride[1]));
    case 'G':
    case 'Q':
        return wholeHeight;
    case 'D':
        return wholeHeight && (a.dilation[1] == 1 || a.filter[1] == 1);
    default: // 'R'
        return true;
    }
}

int64_t ldsLimitOf(std::string_view gcnArchName)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    if(arch == "gfx1250")
    {
        return int64_t{320} * 1024;
    }
    return arch == "gfx950" ? int64_t{160} * 1024 : int64_t{64} * 1024;
}

int64_t splitsWanted(int64_t blocksXY,
                     int64_t blocksPerUnit,
                     int64_t steps,
                     int64_t partialElems,
                     int64_t dispatchUnits,
                     int64_t partialElementBytes)
{
    int64_t wanted
        = divUp(std::max<int64_t>(dispatchUnits, 1) * std::max<int64_t>(blocksPerUnit, 1),
                std::max<int64_t>(blocksXY, 1));
    wanted = std::min({wanted, std::max<int64_t>(1, steps / 4), MAX_GRID_YZ});
    const int64_t maxSplits = MAX_WORKSPACE_BYTES / std::max<int64_t>(partialElementBytes, 1)
                              / std::max<int64_t>(partialElems, 1);
    wanted = std::max<int64_t>(1, std::min(wanted, maxSplits));
    // Whole rounds of blocks over the units: of wanted / 2 .. wanted splits, the one whose
    // grid leaves the least of its last round idle (the most splits on a tie). 10 splits of 64
    // blocks on 304 units run 3 rounds for 2.1 rounds of work; 9 run 2 for 1.9.
    const int64_t units = std::max<int64_t>(dispatchUnits, 1);
    const int64_t blocks = std::max<int64_t>(blocksXY, 1);
    int64_t best = wanted;
    double bestUse = -1;
    for(int64_t s = wanted; s >= std::max({int64_t{1}, wanted / 2, wanted - 64}); --s)
    {
        const double work = static_cast<double>(blocks) * static_cast<double>(s);
        const double rounds = std::ceil(work / static_cast<double>(units));
        const double use = work / (rounds * static_cast<double>(units));
        if(use > bestUse + 1e-9)
        {
            best = s;
            bestUse = use;
        }
    }
    return best;
}

ReduceTile reduceTileFor(int64_t groups, int64_t kg, int64_t cg, int64_t taps, int64_t units)
{
    ReduceTile t;
    // Whole quads a tile, all taps within the LDS.
    const int64_t ldsChannels = K3_REDUCE_TILE_FLOATS / std::max<int64_t>(taps, 1) / 4 * 4;
    int64_t tileC = std::min({divUp(cg, 4) * 4, REDUCE_TILE_MAX_CHANNELS, ldsChannels});
    if(tileC < 4 || groups < 1 || kg < 1)
    {
        return t;
    }
    const int64_t wanted = REDUCE_TILE_BLOCKS_PER_UNIT * std::max<int64_t>(units, 1);
    const auto blocksAt
        = [&](int64_t k, int64_t c) { return groups * divUp(kg, k) * divUp(cg, c); };
    // Narrower tiles (shorter runs) until the blocks fill the units, then as many output
    // channels a tile as keep them filled.
    while(tileC > REDUCE_TILE_MIN_CHANNELS && blocksAt(1, tileC) < wanted)
    {
        tileC = std::max(REDUCE_TILE_MIN_CHANNELS, tileC / 2 / 4 * 4);
    }
    int64_t tileK = 1;
    for(int64_t k = REDUCE_TILE_MAX_K; k > 1; k /= 2)
    {
        if(k * tileC * taps <= K3_REDUCE_TILE_FLOATS && blocksAt(k, tileC) >= wanted)
        {
            tileK = k;
            break;
        }
    }
    tileK = std::min(tileK, kg);
    t.tileK = tileK;
    t.tileC = tileC;
    t.tilesK = divUp(kg, tileK);
    t.tilesC = divUp(cg, tileC);
    t.blocks = blocksAt(tileK, tileC);
    return t;
}

RingTrack ringTrackOf(int64_t ho, int64_t hi, int64_t ph, int64_t dh, int64_t filterRows)
{
    const RingTrack uniform{divUp(ho, dh) + filterRows - 1, dh};
    // Classes past the output height have no dy row (lengths of three values).
    if(dh > ho)
    {
        return uniform;
    }
    const auto lenOf = [&](int64_t rho) -> int64_t {
        const int64_t rows = rho < ho ? divUp(ho - rho, dh) : 0;
        const int64_t reach = hi + ph - rho;
        const int64_t xSteps = reach > 0 ? divUp(reach, dh) : 0;
        return rows > 0 ? std::min(rows + filterRows - 1, xSteps) : 0;
    };
    RingTrack track{lenOf(0), 0};
    while(track.longClasses < dh && lenOf(track.longClasses) == track.len)
    {
        ++track.longClasses;
    }
    for(int64_t rho = track.longClasses; rho < dh; ++rho)
    {
        if(lenOf(rho) != track.len - 1)
        {
            return uniform;
        }
    }
    if(track.len < 1 || (track.longClasses < dh && track.len < 2) || track.len > uniform.len)
    {
        return uniform;
    }
    return track;
}

DirectPlan directPlanFor(int64_t tiles,
                         int64_t planes,
                         int64_t rows,
                         int64_t colBlocks,
                         int64_t filterRows,
                         int64_t partialBytes,
                         int64_t units,
                         int64_t tileMacs)
{
    tiles = std::max<int64_t>(tiles, 1);
    units = std::max<int64_t>(units, 1);
    // A block row at ~6.25 TFLOP/s per unit (gfx950), partials at ~4 TB/s written and read.
    const double rowUs = 2.0 * static_cast<double>(tileMacs) / 6.25e6;
    const double splitUs = 2.0 * static_cast<double>(partialBytes) / 4e6;
    const int64_t walk = tiles * 2 <= units ? 0 : 1;
    DirectPlan best;
    double bestCost = 0;
    for(int64_t segs = 1; segs <= 64; ++segs)
    {
        if(segs > 1 && divUp(rows, segs) < 4)
        {
            break;
        }
        DirectPlan p;
        p.segs = segs;
        p.segRows = divUp(rows, segs);
        p.walk = walk;
        p.items = planes * segs * colBlocks;
        int64_t busiest = 0;
        if(walk == 0)
        {
            const int64_t wanted = std::min(std::max<int64_t>(1, units / tiles), p.items);
            p.per = divUp(p.items, wanted);
            p.splits = divUp(p.items, p.per);
            p.grid = std::min(units, tiles * p.splits);
            busiest = divUp(tiles * p.splits, units) * p.per;
        }
        else
        {
            p.per = divUp(tiles * p.items, units);
            p.splits = 1;
            // A tile's pieces depend on where its first unit falls in a run, which repeats
            // with period per.
            for(int64_t t = 0; t < std::min(tiles, p.per); ++t)
            {
                const int64_t first = t * p.items / p.per;
                const int64_t last = (t * p.items + p.items - 1) / p.per;
                p.splits = std::max(p.splits, last - first + 1);
            }
            p.grid = divUp(tiles * p.items, p.per);
            busiest = p.per;
        }
        const double cost
            = static_cast<double>(busiest) * static_cast<double>(p.segRows + filterRows + 1) * rowUs
              + static_cast<double>(p.splits) * splitUs;
        if(segs == 1 || cost < bestCost)
        {
            best = p;
            bestCost = cost;
        }
    }
    return best;
}

std::optional<WrwProblem> widthFoldOf(const WrwProblem& problem)
{
    const auto& a = problem.base.args;
    const int64_t s = a.stride[2];
    const int64_t width = a.inSpatial[2];
    const int64_t filterW = a.filter[2];
    const int64_t pad = a.prePadding[2];
    const bool staged = problem.channelsFirst;
    if(problem.noFold || problem.folded || s < 2 || filterW % s != 0 || a.dilation[2] != 1
       || a.groups != 1 || pad < 0 || problem.largeImage
       || !(problem.channelsFirst || problem.channelsLast) || (!staged && width % s != 0))
    {
        return std::nullopt;
    }
    // Staged: shifted by the padding's phase, exactly S / s taps. In place: ceil-padded.
    const int64_t shift = staged ? pad % s : 0;
    const int64_t reach = staged ? divUp(width + shift, s) : width / s;
    const int64_t pre = staged ? (pad - shift) / s : divUp(pad, s);
    const int64_t fw = staged ? filterW / s : divUp(filterW - pad, s) + pre;
    const auto channels = pack_features::checkedMultiply(a.c, s);
    // The folded output width is dy's: reach + pre + post - fw + 1 with a padding past the end
    // of at least 0 (every dy position's window starts inside the folded row's reach).
    const int64_t post = a.outSpatial[2] - reach - pre + fw - 1;
    // The staged rows are padded to whole 16-byte vectors (ConvWrwFoldX's stores); the extra
    // positions are zero and no dy position reads them.
    const int64_t vector = 16 / problem.elementBytes;
    const int64_t u = staged ? divUp(reach, vector) * vector : reach;
    if(!channels.has_value() || *channels > INT_LIMIT || reach < 1 || fw < 1 || post < 0
       || u > INT_LIMIT)
    {
        return std::nullopt;
    }
    WrwProblem view = problem;
    auto& v = view.base.args;
    v.c = *channels;
    v.inSpatial[2] = u;
    v.filter[2] = fw;
    v.stride[2] = 1;
    v.prePadding[2] = pre;
    v.dilation[2] = 1;
    const int64_t planes = a.inSpatial[0] * a.inSpatial[1];
    if(problem.channelsFirst)
    {
        // The staged x': packed N (C s) D H U.
        v.xStrides[4] = 1;
        v.xStrides[3] = u;
        v.xStrides[2] = a.inSpatial[1] * u;
        v.xStrides[1] = planes * u;
        v.xStrides[0] = *channels * planes * u;
    }
    else
    {
        // Channels-last x in place: position u s + j of channel c is channel j C + c of
        // position u, so only the width stride changes (s C elements a folded position).
        v.xStrides[1] = 1;
        v.xStrides[4] = *channels;
    }
    // The folded filter (K T R fw by C s); its partials are reduced into dw by the fold reduce.
    view.taps = a.filter[0] * a.filter[1] * fw;
    view.channelsFirst = problem.channelsFirst;
    view.channelsLast = problem.channelsLast;
    view.folded = true;
    view.foldShift = shift;
    return view;
}

std::optional<WrwLaunch> wrwLaunchFor(const WrwProblem& problem,
                                      const WrwConfig& config,
                                      const hipdnn_plugin_sdk::ingestor::DeviceProperties& device)
{
    if(config.math != 0)
    {
        // Three-bf16: the bf16 config on 3N images of split x and dy (x hi dy hi + x hi dy lo +
        // x lo dy hi), partials and reduce as ever (the reduce writes fp32 dw). The
        // CONV_WRW_TF32 engine's configs, fp32 graphs only; the direct dw store would write bf16, so dws 0. The split
        // copies x and dy whole, images outermost, so both must be packed (channels-first or
        // channels-last); not the generic tier (loop 8) or the block-based buffer forms (pb).
        if(config.math != 1 || problem.base.dataType != data_objects::DataType::FLOAT
           || conv_igemm::atomElement(config.atom) != "bf16" || config.dws != 0
           || config.loop >= LOOP_GENERIC || config.pb != 0
           || !(problem.channelsFirst || problem.channelsLast)
           || problem.base.args.n * 3 > INT_LIMIT || !threeBf16Pays(problem, device.gcnArchName))
        {
            return std::nullopt;
        }
        WrwProblem split = problem;
        split.noFold = true;
        split.base.dataType = data_objects::DataType::BFLOAT16;
        split.elementBytes = 2;
        split.base.args.n = problem.base.args.n * 3;
        WrwConfig plain = config;
        plain.math = 0;
        auto launch = wrwLaunchFor(split, plain, device);
        if(!launch.has_value())
        {
            return std::nullopt;
        }
        const auto& a = problem.base.args;
        const auto xElems = pack_features::checkedProduct(
            {a.n, a.c, a.inSpatial[0], a.inSpatial[1], a.inSpatial[2]});
        const auto yElems = pack_features::checkedProduct(
            {a.n, a.k, a.outSpatial[0], a.outSpatial[1], a.outSpatial[2]});
        if(!xElems.has_value() || !yElems.has_value())
        {
            return std::nullopt;
        }
        // xs (3 x xElems bf16), ys (3 x yElems bf16), each from a 256-byte boundary.
        const auto align = [](int64_t bytes) { return (bytes + 255) / 256 * 256; };
        const int64_t xsBytes = align(*xElems * 3 * 2);
        const int64_t splitBytes = xsBytes + align(*yElems * 3 * 2);
        launch->split = true;
        launch->splitYOffset = static_cast<size_t>(xsBytes);
        launch->splitBytes = static_cast<size_t>(splitBytes);
        launch->splitArgs.xElems = *xElems;
        launch->splitArgs.yElems = *yElems;
        launch->splitGrid = static_cast<unsigned int>(
            std::min<int64_t>(divUp(std::max(*xElems, *yElems), 256), int64_t{1} << 16));
        launch->workspaceBytes += static_cast<size_t>(splitBytes);
        return launch;
    }

    // The width fold: a specialized config (compile-time geometry) of unit width stride serves a
    // strided width whose stride divides the filter width as the folded stride-1 problem
    // (widthFoldOf).
    if(config.loop < LOOP_GENERIC && config.sw == 1 && problem.base.args.stride[2] > 1)
    {
        const auto folded = widthFoldOf(problem);
        if(!folded.has_value())
        {
            return std::nullopt;
        }
        auto launch = wrwLaunchFor(*folded, config, device);
        const int64_t units = device_units::dispatchUnits(device);
        if(!launch.has_value() || launch->dwStore || units < 1)
        {
            return std::nullopt;
        }
        const auto& a = problem.base.args;
        const auto& f = folded->base.args;
        const int64_t s = a.stride[2];
        auto& r = launch->reduce;
        const int64_t quads = r.elems / r.cg * divUp(int64_t{r.cg}, int64_t{4});
        const int64_t perBlock = deepReduceQuads(quads, units);
        if(divUp(quads, perBlock) > INT_LIMIT)
        {
            return std::nullopt;
        }
        r.tile = 4;
        r.quads = static_cast<int>(perBlock);
        launch->reduceGrid = static_cast<unsigned int>(divUp(quads, perBlock));
        launch->reduceBlock = static_cast<unsigned int>(4 * perBlock);
        auto& rf = launch->reduceFold;
        rf.r = r;
        // dw packed K C T R S or K T R S C, the graph's taps.
        const int64_t taps = problem.taps;
        rf.wk = a.c * taps;
        rf.wc = problem.filterChannelsLast ? 1 : taps;
        rf.wtap = problem.filterChannelsLast ? a.c : 1;
        rf.c = static_cast<int>(a.c);
        rf.s = static_cast<int>(s);
        rf.fw = static_cast<int>(f.filter[2]);
        rf.pre = static_cast<int>(f.prePadding[2]);
        rf.pad = static_cast<int>(a.prePadding[2] - folded->foldShift);
        rf.filterW = static_cast<int>(a.filter[2]);
        rf.outer = static_cast<int>(a.filter[0] * a.filter[1]);
        rf.flip = a.flipFilter != 0 ? 1 : 0;
        launch->fold = true;
        if(problem.channelsFirst)
        {
            const auto rows
                = pack_features::checkedProduct({a.n, a.c, a.inSpatial[0], a.inSpatial[1]});
            if(!rows.has_value())
            {
                return std::nullopt;
            }
            // The staged x' (ceil((L + shift) / s) positions a row by s phases), by tiles of
            // K3_FOLD_TILE x elements of one row.
            const int64_t tilePositions
                = K3_FOLD_TILE / s / (16 / problem.elementBytes) * (16 / problem.elementBytes);
            const auto blocks
                = pack_features::checkedMultiply(*rows, divUp(f.inSpatial[2], tilePositions));
            const auto stagedBytes = pack_features::checkedProduct(
                {*rows, f.inSpatial[2], s, int64_t{problem.elementBytes}});
            if(tilePositions < 1 || !blocks.has_value() || *blocks > INT_LIMIT
               || !stagedBytes.has_value())
            {
                return std::nullopt;
            }
            launch->foldStage = true;
            launch->foldOffset = (launch->workspaceBytes + 255) / 256 * 256;
            launch->workspaceBytes = launch->foldOffset + static_cast<size_t>(*stagedBytes);
            auto& fa = launch->foldArgs;
            fa.rows = *rows;
            fa.planes = a.inSpatial[0] * a.inSpatial[1];
            fa.width = a.inSpatial[2];
            fa.c = static_cast<int>(a.c);
            fa.s = static_cast<int>(s);
            fa.u = static_cast<int>(f.inSpatial[2]);
            fa.shift = static_cast<int>(folded->foldShift);
            launch->foldGrid = static_cast<unsigned int>(*blocks);
        }
        return launch;
    }

    // A 1x1 filter at unit stride and no padding over packed channels-first images reads every
    // image plane (depth x height x width) as one contiguous run in x and dy alike: the
    // specialized loops take it as one row of that many positions, so their position chunks
    // run across row ends (resnext's 16-wide rows filled half of each 32-position chunk).
    if(config.loop < LOOP_GENERIC && problem.channelsFirst)
    {
        const auto& p = problem.base.args;
        bool pointwise = p.inSpatial[0] * p.inSpatial[1] > 1;
        for(int axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
        {
            pointwise = pointwise && p.filter[axis] == 1 && p.stride[axis] == 1
                        && p.prePadding[axis] == 0 && p.inSpatial[axis] == p.outSpatial[axis];
        }
        const auto plane
            = pack_features::checkedProduct({p.inSpatial[0], p.inSpatial[1], p.inSpatial[2]});
        if(pointwise && plane.has_value() && *plane <= INT_LIMIT / 4)
        {
            WrwProblem flat = problem;
            auto& f = flat.base.args;
            for(int axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
            {
                f.inSpatial[axis] = f.outSpatial[axis] = axis == 2 ? *plane : 1;
                f.dilation[axis] = 1;
            }
            // Packed channels-first strides of the one-row planes (n, c, d, h, w).
            f.xStrides[2] = f.xStrides[3] = *plane;
            f.yStrides[2] = f.yStrides[3] = *plane;
            return wrwLaunchFor(flat, config, device);
        }
    }

    // The atom: known, its id, on this device, taking the graph's element type. "fma" is the
    // FMA ring's, which takes fp32 only (16-bit runs on the matrix cores everywhere).
    const auto atomId = atomIdOf(config.atom);
    const auto info = conv_igemm::atomNamed(config.atom);
    if(atomId < 0 || atomId != config.atomId || !info.has_value())
    {
        return std::nullopt;
    }
    const bool accurate = config.loop == LOOP_ACCURATE;
    const bool smallk = config.loop == LOOP_SMALL_K;
    const bool nativeGeneric = config.loop == LOOP_NATIVE_GENERIC;
    const bool tapTiled = config.loop == LOOP_TAP_TILED;
    const bool runtime = nativeGeneric || tapTiled || smallk || accurate;
    const bool generic = config.loop == LOOP_GENERIC;
    // The generic loop's FMA form: the "fma" atom (thread tiles tk x tc).
    const bool genericFma = generic && info->wave == 0;
    const bool fma = config.loop == LOOP_RING_FMA || config.loop == LOOP_ROWS_FMA || genericFma
                     || smallk || accurate;
    const bool direct = config.loop == LOOP_DIRECT;
    const bool smallc = config.loop == LOOP_ROWS_SMALLC;
    const bool packed = config.loop == LOOP_ROWS_PACKED;
    const bool rows
        = config.loop == LOOP_ROWS || config.loop == LOOP_ROWS_FMA || smallc || packed || generic;
    if(fma != (info->wave == 0))
    {
        return std::nullopt;
    }
    const std::string_view arch = device.gcnArchName;
    const auto element = conv_igemm::dataElement(problem.base.dataType);
    // The FMA loops take fp32 everywhere, and 16-bit (accumulating in fp32) only where no
    // matrix atom takes it (gfx906, gfx900): elsewhere they would only add slow candidates.
    // The generic FMA form only where no matrix atom takes the element (fp32 on RDNA): the
    // matrix generic configs serve the rest.
    if(!conv_igemm::atomRunsOn(config.atom, arch)
       || (fma ? !smallk && (element != "f32" || genericFma) && runsMatrixAtomFor(element, arch)
               : conv_igemm::atomElement(config.atom) != element))
    {
        return std::nullopt;
    }
    const auto shape = atomShapeOf(config.atom);
    const auto& a = problem.base.args;
    const int eb = problem.elementBytes;
    const int64_t kg = a.k / a.groups;
    const int64_t cg = a.c / a.groups;
    const bool gfx950 = hipdnn_plugin_sdk::stripArchFeatures(arch) == "gfx950";
    // Flattened positions (flat 1): every output position, and the whole x and dy spans.
    const int64_t flatPositions = a.n * a.outSpatial[0] * a.outSpatial[1] * a.outSpatial[2];
    const int64_t xTensor = (a.n - 1) * a.xStrides[0] + problem.xSpan;
    const int64_t yTensor = (a.n - 1) * a.yStrides[0] + problem.ySpan;
    const bool gfx1250 = hipdnn_plugin_sdk::stripArchFeatures(arch) == "gfx1250";

    // Loop and layout.
    if(config.loop < LOOP_ROWS || config.loop > LOOP_TAP_TILED)
    {
        return std::nullopt;
    }
    const bool cf = config.layout == LAYOUT_CHANNELS_FIRST;
    const bool tr = config.layout == LAYOUT_CHANNELS_LAST_TR;
    const bool cl = config.layout == LAYOUT_CHANNELS_LAST || tr;
    if(generic || (nativeGeneric && !problem.channelsFirst && !problem.channelsLast))
    {
        // The buffer generic uses channel vectors where they load whole, scalar elements
        // elsewhere. Native scalar gathers use that same layout choice for strided operands.
        if(tr || cl != problem.vectorChannels)
        {
            return std::nullopt;
        }
    }
    else if(!(cf && problem.channelsFirst) && !(cl && problem.channelsLast))
    {
        return std::nullopt;
    }
    // Transpose reads: gfx950 and an atom whose lanes hold 8 positions (16-bit atoms only).
    if(tr && (!gfx950 || shape.kpl != 8))
    {
        return std::nullopt;
    }
    // Runtime matrix configs use native gfx1250 atoms and scalar gathers. Keep their tile
    // arithmetic and register footprint bounded.
    if(nativeGeneric || tapTiled)
    {
        if(!gfx1250 || config.atom.compare(0, 8, "wmma125_") != 0 || tr || config.vec != 1
           || config.fr != 0 || config.s != 0 || config.sw != 0 || config.dw != 0 || config.bk > 128
           || config.bc > 128 || config.bq > 256 || config.wk > 8 || config.wc > 8
           || config.xcd != 0 || config.align != 0 || config.ns != 0
           || (config.dws != 1 && config.dws != 2))
        {
            return std::nullopt;
        }
    }
    if(smallk
       && (!gfx1250 || config.layout != LAYOUT_CHANNELS_LAST || kg > 4 || config.bk != 4
           || config.tk != 4 || config.tc != 1
           || (config.bc != 64 && config.bc != 128 && config.bc != 256)
           || (config.bq != 64 && config.bq != 128 && config.bq != 256) || config.vec != 1
           || config.wk != 0 || config.wc != 0 || config.fr != 0 || config.s != 0 || config.sw != 0
           || config.dw != 0 || config.xcd != 0 || config.align != 0 || config.ns != 0
           || config.wq != 0 || config.dws != 1))
    {
        return std::nullopt;
    }
    // A separate complete FP32-output kernel: no matrix atom, vector assumptions, direct
    // epilogue, or specialized-filter knobs. Every thread owns one (k,c) FP64 accumulator.
    // The FP64 path is qualified only on gfx1250; FMA's broad atom gate is not its contract.
    if(accurate
       && (!gfx1250 || element != "f32" || eb != 4 || tr || (config.bk != 8 && config.bk != 16)
           || (config.bc != 8 && config.bc != 16) || (config.bq != 32 && config.bq != 64)
           || config.tk != 1 || config.tc != 1 || config.vec != 1 || config.wk != 0
           || config.wc != 0 || config.fr != 0 || config.s != 0 || config.sw != 0 || config.dw != 0
           || config.xcd != 0 || config.align != 0 || config.ns != 0 || config.wq != 0
           || config.dws != 0))
    {
        return std::nullopt;
    }
    // The vector and the tile: 16 bytes, or (FMA output rows) one element.
    if((config.vec * eb != 16 && ((config.loop != LOOP_ROWS_FMA && !runtime) || config.vec != 1))
       || config.bk < 1 || config.bc < 1 || config.bq < 1)
    {
        return std::nullopt;
    }
    int64_t threads = 0;
    if(accurate)
    {
        threads = config.bk * config.bc;
    }
    else if(smallk)
    {
        threads = config.bc;
    }
    else if(fma)
    {
        // Thread tiles of whole channel quads dividing the tile (so bk and bc are quads too).
        if(config.wk != 0 || config.wc != 0 || config.tk < 4 || config.tc < 4 || config.tk % 4 != 0
           || config.tc % 4 != 0 || config.bk % config.tk != 0 || config.bc % config.tc != 0)
        {
            return std::nullopt;
        }
        threads = (config.bk / config.tk) * (config.bc / config.tc);
    }
    else
    {
        if(config.wk < 1 || config.wc < 1 || config.tk != 0 || config.tc != 0)
        {
            return std::nullopt;
        }
        // Small-C rows: the N axis is bc x s (channel, tap) pairs, the waves split k only.
        if(config.bk % (config.wk * shape.dim) != 0
           || (smallc ? config.wc != 1 : config.bc % (config.wc * shape.dim) != 0)
           || config.bq % shape.ka != 0)
        {
            return std::nullopt;
        }
        threads = config.wk * config.wc * info->wave;
    }
    if(tapTiled)
    {
        const int64_t qlanes = std::min(config.bq, threads);
        if(threads % config.bk != 0 || threads % config.bc != 0 || threads % qlanes != 0
           || config.bq % qlanes != 0)
        {
            return std::nullopt;
        }
    }
    // waves_q: loop 5's items side by side, the 8 waves split wk x wc x wq. The matrix-core
    // ring's steps a stage (2 or 4): only for output rows within one step's positions, which a
    // step of fewer positions then fills (a wider row takes the one-step ring). 0 elsewhere.
    const bool ringStages = config.loop == LOOP_RING && (config.wq == 2 || config.wq == 4)
                            && a.outSpatial[2] <= config.bq;
    if(direct ? ((config.wq != 1 && config.wq != 2 && config.wq != 4)
                 || config.wk * config.wc * config.wq != 8)
              : (config.wq != 0 && !ringStages))
    {
        return std::nullopt;
    }
    // The direct dw store: compiled in or not; loop 5 writes partials only.
    if(config.dws < 0 || config.dws > 2 || (direct && config.dws != 0)
       || (config.dws == 2
           && (!tapTiled || !cf || problem.filterChannelsLast || config.atom != "wmma125_f16"
               || config.bk != 64 || config.bc != 64 || config.bq != 32 || config.wk != 2
               || config.wc != 2)))
    {
        return std::nullopt;
    }
    // Block-based buffers: exactly the graphs whose image is past the 32-bit buffer range (the
    // parse admits those channels-first only), on the K3Tiles loops, with a block's bk dy and
    // bc x channel planes within it.
    if((config.pb != 0 && config.pb != 1) || (runtime && config.pb != 0))
    {
        return std::nullopt;
    }
    if(!runtime && problem.largeImage != (config.pb == 1))
    {
        return std::nullopt;
    }
    // The generic loop's per-step windows (pb 1): from the step's first reachable element, the
    // block's channels, the filter's reach in depth and height, and its chunk's columns.
    int64_t xWindow = 0;
    int64_t yWindow = 0;
    if(config.pb == 1 && generic)
    {
        const auto& xs = a.xStrides;
        const auto& ys = a.yStrides;
        const int64_t columns = (config.bq - 1) * a.stride[2] + (a.filter[2] - 1) * a.dilation[2]
                                + std::max<int64_t>(config.dw, 0);
        xWindow = 1 + (std::min<int64_t>(config.bc, cg) - 1) * xs[1]
                  + std::min<int64_t>((a.filter[0] - 1) * a.dilation[0], a.inSpatial[0] - 1) * xs[2]
                  + std::min<int64_t>((a.filter[1] - 1) * a.dilation[1], a.inSpatial[1] - 1) * xs[3]
                  + std::min<int64_t>(columns, a.inSpatial[2] - 1) * xs[4];
        yWindow = 1 + (std::min<int64_t>(config.bk, kg) - 1) * ys[1]
                  + std::min<int64_t>(config.bq - 1, a.outSpatial[2] - 1) * ys[4];
        if(xWindow > INT_LIMIT / eb || yWindow > INT_LIMIT / eb)
        {
            return std::nullopt;
        }
    }
    else if(config.pb == 1)
    {
        // The K3Tiles loops: channels-first, a block's channel planes within the range.
        const int64_t xPlane = a.inSpatial[0] * a.inSpatial[1] * a.inSpatial[2];
        const int64_t yPlane = a.outSpatial[0] * a.outSpatial[1] * a.outSpatial[2];
        if((config.loop != LOOP_ROWS && config.loop != LOOP_RING && config.loop != LOOP_ROWS_SMALLC)
           || !problem.channelsFirst || config.bc * xPlane > INT_LIMIT / eb
           || config.bk * yPlane > INT_LIMIT / eb)
        {
            return std::nullopt;
        }
    }
    if(direct)
    {
        threads = int64_t{8} * info->wave;
    }
    if(generic)
    {
        // Runtime geometry (s, sw, dw 0), fr taps a block; a position per thread (bq dividing
        // the block) and whole channel rows a pass. (Channel vectors and 32-bit position words
        // follow from whole fragments and atom steps.)
        // (A negative dw is declined with the LDS: ldsBytesOf takes no negative field.)
        if(config.s != 0 || config.sw != 0 || config.fr < 1 || threads % config.bq != 0
           || config.bk % (threads / config.bq) != 0 || config.bc % (threads / config.bq) != 0)
        {
            return std::nullopt;
        }
        // dw > 0, the shared width window of bq + dw positions: channel vectors, whole 16-byte
        // units of extra positions, unit width stride, and the block's width taps within it.
        if(config.dw > 0
           && (!cl || config.dw % 8 != 0 || a.stride[2] != 1
               || (std::min<int64_t>(config.fr, a.filter[2]) - 1) * a.dilation[2] > config.dw))
        {
            return std::nullopt;
        }
        // Widely dilated small filters (ASPP-class: at most 9 taps, dilation 6 or more on a
        // filter axis): each tap gathers a far-apart window. Measured against the BLAS path on
        // deeplab 3x3 d12-d36 and SSD's fp32 3x3 d6 (best generic config against best BLAS):
        // gfx1101 0.9-12x (fp32 FMA form 1.3-2.2x) and gfx1201 16-bit 0.8-3.3x (geomean 1.5x)
        // serve them; gfx1201's FMA form (0.4-0.7x), gfx942 (geomean 0.64x) and gfx90a (0.73x)
        // leave them to BLAS, as does every architecture not measured.
        int64_t dilation = 1;
        for(int axis = 0; axis < CONV_NAIVE_SPATIAL_DIMS; ++axis)
        {
            if(a.filter[axis] > 1)
            {
                dilation = std::max<int64_t>(dilation, a.dilation[axis]);
            }
        }
        const auto baseArch = hipdnn_plugin_sdk::stripArchFeatures(arch);
        const bool servesSparse
            = baseArch.rfind("gfx11", 0) == 0 || (baseArch.rfind("gfx120", 0) == 0 && !genericFma);
        if(problem.taps <= GENERIC_SPARSE_MAX_TAPS && dilation >= GENERIC_SPARSE_MIN_DILATION
           && !servesSparse)
        {
            return std::nullopt;
        }
        // Narrow groups (depthwise, ResNeXt-class): a group's GEMM fills a sliver of the block's
        // channel tile and every group rereads each tap's gather. On gfx1101 it ran 14-20x
        // behind CONV_GROUP_WRW at 4 channels a group and 30-5000x behind the vector-ALU engines
        // on depthwise layers (CONV_DEPTHWISE_VALU_WRW, CONV_GROUP_VALU_WRW serve both).
        if(a.groups > 1 && cg < GENERIC_MIN_GROUP_CHANNELS && kg < GENERIC_MIN_GROUP_CHANNELS)
        {
            return std::nullopt;
        }
        // Stems (few input channels a group: the channel tile mostly empty) and output rows of a
        // few positions (a step's chunk mostly empty) go to BLAS. Measured against it: ViT,
        // ConvNeXt and R(2+1)D stems (3 channels) 2.5-30x behind on gfx1101, 4.2-7.9x on
        // gfx1201 (FMA form); Inception's 5x5 over a 5x5 map (one position a row) 5-25x behind
        // on gfx1101. The flattened-position form (flat 1) fills its steps across rows, so it
        // keeps narrow rows.
        if(cg < GENERIC_MIN_CHANNELS
           || (config.flat == 0 && a.outSpatial[2] < GENERIC_MIN_OUTPUT_WIDTH))
        {
            return std::nullopt;
        }
        // The FMA form also leaves patchify-class width strides (no reuse between a chunk's
        // taps) to BLAS, which won on gfx1201 (AlexNet 11x11 s4, wav2vec k10 s5).
        if(genericFma && a.stride[2] >= GENERIC_FMA_MAX_STRIDE)
        {
            return std::nullopt;
        }
    }
    // Every other loop: the filter width, its stride and dilation are compile-time.
    else if(!runtime
            && (config.s != a.filter[2] || config.sw != a.stride[2] || config.dw != a.dilation[2]
                || config.fr < 1))
    {
        return std::nullopt;
    }
    // (The generic loop's channel vectors are the problem's vectorChannels, gated above.)
    if(!generic && !runtime && cf)
    {
        if(config.bq % config.vec != 0 || (!fma && config.vec % config.sw != 0))
        {
            return std::nullopt;
        }
    }
    else if(!generic && !runtime
            && ((kg % config.vec != 0 && config.align != 2) || (cg % config.vec != 0 && !smallc)))
    {
        return std::nullopt;
    }
    // Small-C rows: at most 16 input channels per group (stems), a tile of at most 16 of them.
    // Channels-last operands are staged into the channels-first tiles (x one element at a time,
    // so any channel count): their geometry's whole-vector chunk and phases hold too.
    if(smallc
       && (tr || cg > 16 || config.bc > 16
           || (!cf && (config.bq % config.vec != 0 || config.vec % config.sw != 0))))
    {
        return std::nullopt;
    }
    // Packed rows: channels-first, a dy position per thread (bq dividing the block), and narrow
    // outputs only: two virtual rows of wv = wo + (s - 1) dw / sw columns within bq (so at least
    // two rows a step).
    int64_t packWidth = 0;
    int64_t packRows = 0;
    if(packed)
    {
        packWidth = a.outSpatial[2] + (config.s - 1) * config.dw / config.sw;
        packRows = config.bq / packWidth;
        if(!cf || threads % config.bq != 0 || 2 * packWidth > config.bq)
        {
            return std::nullopt;
        }
    }
    // The row rings: height stride 1, the whole filter height per block; or (the matrix ring,
    // loop 1) height stride sh > 1 at unit height dilation, phases of ceil(fr / sh) rows.
    const bool phased = config.loop == LOOP_RING && a.stride[1] > 1 && a.dilation[1] == 1
                        && config.fr == divUp(a.filter[1], a.stride[1]);
    if(!runtime && !rows && !phased && (a.stride[1] != 1 || config.fr != a.filter[1]))
    {
        return std::nullopt;
    }
    if(config.loop == LOOP_RING_DMA)
    {
        if(!gfx950 || config.layout == LAYOUT_CHANNELS_LAST || config.sw != 1 || config.ns < 2
           || config.ns > 4)
        {
            return std::nullopt;
        }
    }
    else if(direct)
    {
        // Rows kept channels-last (so gfx950 and an atom of 8 positions per lane), the 16 x 16 x
        // 32 atom (the other such atom steps 16 positions), 32 output columns per item, unit
        // width stride and height dilation (a one-row filter has none), ns ring slots whose
        // loads in flight fit vmcnt, rows of a conflict-free rotation.
        const int64_t loads = loadPassesOf(config, eb, info->wave);
        if(!tr || shape.ka != 32 || config.bq != 32 || config.sw != 1
           || (a.dilation[1] != 1 && a.filter[1] != 1) || config.ns < 2 || config.ns > 5
           || loads > 63 || (config.ns - 2) * loads > 63 || config.bpu != 1
           || k3dRotation(static_cast<int>(config.bc),
                          static_cast<int>((config.s - 1) * config.dw + 1))
                  < 0
           || k3dRotation(static_cast<int>(config.bk), 1) < 0)
        {
            return std::nullopt;
        }
    }
    else if(config.ns != 0)
    {
        return std::nullopt;
    }
    // The FMA ring: unit width stride. Both FMA loops: channels-first rows of whole vectors
    // (units wholly in or out of a row).
    if(config.loop == LOOP_RING_FMA && config.sw != 1)
    {
        return std::nullopt;
    }
    // Both FMA loops' tiles are whole vectors of positions and (channels-last units; the
    // kernels assert it for either layout) of channels: 16-bit vectors of 8 take tiles of 8.
    if(fma
       && (config.bq % config.vec != 0 || config.bk % config.vec != 0
           || config.bc % config.vec != 0))
    {
        return std::nullopt;
    }
    if(fma && !generic && !runtime && cf
       && (a.inSpatial[2] % config.vec != 0 || a.outSpatial[2] % config.vec != 0))
    {
        return std::nullopt;
    }
    // Align 2: the ring's channels-last dy units at any element, for output channel counts that
    // are not whole vectors (only there: whole vectors take the aligned configs).
    if(config.align == 2)
    {
        if(config.loop != LOOP_RING || config.layout != LAYOUT_CHANNELS_LAST
           || kg % config.vec == 0)
        {
            return std::nullopt;
        }
    }
    else if(config.align != 0)
    {
        if(config.align != 1 || config.loop != LOOP_RING || !cf || config.sw != 1
           || a.inSpatial[2] % config.vec != 0 || a.outSpatial[2] % config.vec != 0)
        {
            return std::nullopt;
        }
    }
    if(config.xcd != 0 && (config.xcd != 1 || fma || !conv_igemm::hasXcds(arch)))
    {
        return std::nullopt;
    }
    if(config.bpu != 1 && config.bpu != 2 && config.bpu != 4 && config.bpu != 8)
    {
        return std::nullopt;
    }
    // The block, its load passes and its LDS.
    if(threads > MAX_BLOCK
       || loadPassesOf(config, eb, info->wave) > (packed || generic ? MAX_PACK_PASSES : MAX_PASSES)
       || ldsBytesOf(config, eb) > static_cast<int64_t>(
              gfx1250 && device.sharedMemPerBlock > 0
                  ? std::min(static_cast<size_t>(ldsLimitOf(arch)), device.sharedMemPerBlock)
                  : static_cast<size_t>(ldsLimitOf(arch))))
    {
        return std::nullopt;
    }
    // Channels-first runs are loaded whole from within an image. The generic loop's buffers
    // span the images' strides.
    // (Block-based generic buffers: the per-step windows.)
    int64_t xImage = a.c * a.inSpatial[0] * a.inSpatial[1] * a.inSpatial[2];
    int64_t yImage = a.k * a.outSpatial[0] * a.outSpatial[1] * a.outSpatial[2];
    if(generic && config.pb == 1)
    {
        xImage = xWindow;
        yImage = yWindow;
    }
    else if(generic && config.flat != 0)
    {
        xImage = xTensor;
        yImage = yTensor;
    }
    else if(generic)
    {
        xImage = problem.xSpan;
        yImage = problem.ySpan;
    }
    if(!generic && !runtime && cf && (xImage < config.vec || yImage < config.vec))
    {
        return std::nullopt;
    }
    // Flattened positions (flat 1): the generic loop without a shared window or block-based
    // buffers, where an output row is shorter than a step (row chunks would run part empty);
    // x and dy each one buffer, and the positions within 32 bits.
    if(config.flat != 0
       && (config.flat != 1 || !generic || config.dw != 0 || config.pb != 0
           || a.outSpatial[2] >= config.bq || flatPositions > INT_LIMIT - config.bq
           || xTensor > INT_LIMIT / eb || yTensor > INT_LIMIT / eb))
    {
        return std::nullopt;
    }
    // The two tiers: the generic loop only where no specialized config's class serves (flat
    // configs serve beside them: the specialized loops step over row chunks).
    if(generic && config.flat == 0)
    {
        size_t from = 0;
        while(from < config.spec.size())
        {
            const size_t to = std::min(config.spec.find(';', from), config.spec.size());
            const auto served
                = specServes(problem, std::string_view(config.spec).substr(from, to - from), arch);
            if(!served.has_value() || *served)
            {
                return std::nullopt;
            }
            from = to + 1;
        }
    }
    const int64_t units = device_units::dispatchUnits(device);
    if(units < 1)
    {
        return std::nullopt;
    }

    // Grid and reduction steps.
    const auto partialElems = pack_features::checkedProduct({a.groups, problem.taps, kg, cg});
    int64_t gridX = 0;
    int64_t gridY = 1;
    int64_t gridZ = 1;
    int64_t chunks = 0;
    int64_t trackLen = 0;
    int64_t trackLong = 0;
    int64_t splits = 0;
    int64_t stepsPerSplit = 0;
    std::optional<int64_t> steps;
    DirectPlan plan;
    if(runtime)
    {
        const auto columns = pack_features::checkedProduct({problem.taps, cg});
        if(!columns.has_value())
        {
            return std::nullopt;
        }
        const auto tiles
            = tapTiled
                  ? pack_features::checkedProduct(
                        {a.groups, divUp(*columns, config.bc), divUp(kg, config.bk)})
                  : pack_features::checkedProduct({a.groups,
                                                   problem.taps,
                                                   divUp(cg, config.bc),
                                                   smallk ? int64_t{1} : divUp(kg, config.bk)});
        const auto positions = pack_features::checkedProduct(
            {a.n, a.outSpatial[0], a.outSpatial[1], a.outSpatial[2]});
        if(!tiles.has_value() || !positions.has_value())
        {
            return std::nullopt;
        }
        gridX = *tiles;
        steps = divUp(*positions, config.bq);
    }
    else if(direct)
    {
        // Loop 5: a persistent grid over (tile, item) units; steps are a tile's items and
        // stepsPerSplit the items (walk 0) or units (walk 1) of a piece.
        chunks = divUp(a.outSpatial[2], 32);
        const auto tiles = pack_features::checkedProduct(
            {a.groups, a.filter[0], divUp(cg, config.bc), divUp(kg, config.bk)});
        if(!tiles.has_value() || !partialElems.has_value())
        {
            return std::nullopt;
        }
        plan = directPlanFor(*tiles,
                             a.n * a.outSpatial[0],
                             a.inSpatial[1],
                             chunks,
                             a.filter[1],
                             *partialElems * int64_t{sizeof(float)},
                             units,
                             config.bc * config.bk * config.s * config.fr * 32);
        const auto unitsOfWork = pack_features::checkedProduct({*tiles, plan.items});
        if(!unitsOfWork.has_value() || plan.items > INT_LIMIT)
        {
            return std::nullopt;
        }
        gridX = plan.grid;
        splits = plan.splits;
        steps = plan.items;
        stepsPerSplit = plan.per;
    }
    else
    {
        gridX = divUp(kg, config.bk) * divUp(cg, config.bc);
        const int64_t frows = a.filter[0] * a.filter[1];
        // A phased ring runs one grid row per height phase; its tracks walk the dy rows.
        int64_t rowsPerGroup = a.filter[0] * (phased ? a.stride[1] : 1);
        if(generic && config.dw > 0)
        {
            rowsPerGroup = a.filter[0] * a.filter[1] * divUp(a.filter[2], config.fr);
        }
        else if(generic)
        {
            rowsPerGroup = divUp(problem.taps, config.fr);
        }
        else if(rows)
        {
            rowsPerGroup = divUp(frows, config.fr);
        }
        gridY = a.groups * rowsPerGroup;
        // Packed rows: a plane's chunks are its groups of packRows rows.
        chunks = packed ? divUp(a.outSpatial[1], packRows) : divUp(a.outSpatial[2], config.bq);
        // A phased ring's one class keeps every step; a stride-1 ring's tracks end at their
        // last x row in the image (ringTrackOf).
        const RingTrack track = phased ? RingTrack{a.outSpatial[1] + config.fr - 1, a.dilation[1]}
                                       : ringTrackOf(a.outSpatial[1],
                                                     a.inSpatial[1],
                                                     a.prePadding[1],
                                                     a.dilation[1],
                                                     a.filter[1]);
        trackLen = track.len;
        trackLong = track.longClasses;
        // Steps per (image, output depth) plane: chunks tracks of each class.
        const auto planeSteps = pack_features::checkedProduct(
            {chunks,
             trackLong * trackLen
                 + (a.dilation[1] - trackLong) * std::max<int64_t>(trackLen - 1, 0)});
        if(packed)
        {
            steps = pack_features::checkedProduct({a.n, a.outSpatial[0], chunks});
        }
        else if(generic && config.flat != 0)
        {
            chunks = 1;
            steps = divUp(flatPositions, config.bq);
        }
        else if(rows)
        {
            steps = pack_features::checkedProduct({a.n, a.outSpatial[0], a.outSpatial[1], chunks});
        }
        else if(planeSteps.has_value())
        {
            steps = pack_features::checkedProduct({a.n, a.outSpatial[0], *planeSteps});
        }
        else
        {
            steps = std::nullopt;
        }
    }
    if(gridX < 1 || gridX > INT_LIMIT || gridY < 1 || gridY > MAX_GRID_YZ || !steps.has_value()
       || *steps < 1 || !partialElems.has_value() || trackLen > INT_LIMIT || chunks > INT_LIMIT)
    {
        return std::nullopt;
    }
    if(!direct)
    {
        const int64_t wanted = splitsWanted(gridX * gridY,
                                            config.bpu,
                                            *steps,
                                            *partialElems,
                                            units,
                                            accurate ? sizeof(double) : sizeof(float));
        stepsPerSplit = divUp(*steps, wanted);
        splits = divUp(*steps, stepsPerSplit);
        gridZ = splits;
    }
    // Channels-first filters also have contiguous taps when the channel axis has extent one.
    // Direct stores do not allocate partials or launch the reducer.
    const bool dwStore
        = splits == 1 && !problem.folded
          && (config.dws == 2
              || (config.dws == 1 && (problem.filterChannelsLast || problem.taps == 1 || cg == 1)));
    const auto workspace = pack_features::checkedProduct(
        {splits, *partialElems, accurate ? int64_t{sizeof(double)} : int64_t{sizeof(float)}});
    const int64_t quads = a.groups * problem.taps * kg * divUp(cg, 4);
    // K C T R S filters of several taps at few splits: tile blocks, which write each output
    // channel's run of channels by every tap contiguously (reduceTileFor). With more splits, K T
    // R S C or one tap (channels already contiguous), the quad blocks.
    const ReduceTile reduceTile = accurate || dwStore || problem.filterChannelsLast
                                          || problem.taps < 2 || splits > K3_REDUCE_TILE_MAX_SPLITS
                                      ? ReduceTile{}
                                      : reduceTileFor(a.groups, kg, cg, problem.taps, units);
    // gfx950 at one split: the row reduce (one output channel's run of 64 channels a block)
    // measured 3-4 us ahead of the tile reduce in the call (StyleGAN2 c512 8^2 n2, R(2+1)D c512).
    const bool reduceRows = reduceTile.blocks > 0 && splits == 1
                            && problem.taps <= K3_REDUCE_ROW_MAX_TAPS
                            && hipdnn_plugin_sdk::stripArchFeatures(device.gcnArchName) == "gfx950";
    int64_t reduceGrid = divUp(quads, REDUCE_QUADS_PER_BLOCK);
    if(accurate)
    {
        reduceGrid = divUp(*partialElems, 256);
    }
    else if(reduceTile.blocks > 0)
    {
        reduceGrid = reduceTile.blocks;
    }
    if(reduceRows)
    {
        reduceGrid = a.groups * kg * divUp(cg, int64_t{K3_REDUCE_ROW_CHANNELS});
    }
    if(splits > MAX_GRID_YZ
       || (!dwStore
           && (!workspace.has_value() || ((direct || gfx1250) && *workspace > MAX_WORKSPACE_BYTES)
               || reduceGrid > INT_LIMIT))
       || (config.xcd != 0 && gridX * gridY * gridZ > UINT_LIMIT))
    {
        return std::nullopt;
    }

    WrwLaunch launch;
    launch.gridX = static_cast<unsigned int>(gridX);
    launch.gridY = static_cast<unsigned int>(gridY);
    launch.gridZ = static_cast<unsigned int>(gridZ);
    launch.block = static_cast<unsigned int>(threads);
    launch.reduceGrid = static_cast<unsigned int>(reduceGrid);
    launch.kernelType = info->kernelType;
    launch.dwStore = dwStore;
    launch.accurate = accurate;
    launch.workspaceBytes = dwStore ? 0 : static_cast<size_t>(*workspace);
    auto& k3 = launch.args;
    k3.n = static_cast<int>(a.n);
    k3.c = static_cast<int>(a.c);
    k3.k = static_cast<int>(a.k);
    k3.groups = static_cast<int>(a.groups);
    k3.di = static_cast<int>(a.inSpatial[0]);
    k3.hi = static_cast<int>(a.inSpatial[1]);
    k3.wi = static_cast<int>(a.inSpatial[2]);
    k3.dout = static_cast<int>(a.outSpatial[0]);
    k3.ho = static_cast<int>(a.outSpatial[1]);
    k3.wo = static_cast<int>(a.outSpatial[2]);
    k3.ft = static_cast<int>(a.filter[0]);
    k3.fr = static_cast<int>(a.filter[1]);
    k3.fs = static_cast<int>(a.filter[2]);
    k3.sd = static_cast<int>(a.stride[0]);
    k3.sh = static_cast<int>(a.stride[1]);
    k3.sw = static_cast<int>(a.stride[2]);
    k3.pd = static_cast<int>(a.prePadding[0]);
    k3.ph = static_cast<int>(a.prePadding[1]);
    k3.pw = static_cast<int>(a.prePadding[2]);
    k3.dd = static_cast<int>(a.dilation[0]);
    k3.dh = static_cast<int>(a.dilation[1]);
    k3.dw = static_cast<int>(a.dilation[2]);
    k3.chunks = static_cast<int>(chunks);
    k3.splits = static_cast<int>(splits);
    k3.steps = *steps;
    k3.stepsPerSplit = stepsPerSplit;
    k3.trackLen = rows || direct || runtime ? 0 : static_cast<int>(trackLen);
    k3.trackLong = rows || direct || runtime ? 0 : static_cast<int>(trackLong);
    k3.segs = direct ? static_cast<int>(plan.segs) : static_cast<int>(packWidth);
    k3.segRows = direct ? static_cast<int>(plan.segRows) : static_cast<int>(packRows);
    k3.walk = direct ? static_cast<int>(plan.walk) : 0;
    if(generic || nativeGeneric)
    {
        // The graph's own strides (packed or not).
        std::copy(std::begin(a.xStrides), std::end(a.xStrides), std::begin(k3.xs));
        std::copy(std::begin(a.yStrides), std::end(a.yStrides), std::begin(k3.ys));
    }
    else
    {
        const bool channelsLast = !cf;
        packedStrides(
            {a.n, a.c, a.inSpatial[0], a.inSpatial[1], a.inSpatial[2]}, channelsLast, k3.xs);
        packedStrides(
            {a.n, a.k, a.outSpatial[0], a.outSpatial[1], a.outSpatial[2]}, channelsLast, k3.ys);
    }
    k3.xImage = xImage;
    k3.yImage = yImage;
    k3.partialElems = *partialElems;
    k3.dwStore = dwStore ? 1 : 0;
    k3.flip = a.flipFilter != 0 ? 1 : 0;
    k3.dwk = cg * problem.taps;
    k3.dwtap = problem.filterChannelsLast ? cg : 1;

    auto& r = launch.reduce;
    r.elems = a.k * cg * problem.taps;
    r.partialElems = *partialElems;
    r.splits = static_cast<int>(splits);
    r.groups = static_cast<int>(a.groups);
    r.kg = static_cast<int>(kg);
    r.cg = static_cast<int>(cg);
    r.taps = static_cast<int>(problem.taps);
    r.flip = a.flipFilter != 0 ? 1 : 0;
    // dw packed K C T R S (taps contiguous per channel) or K T R S C (channels contiguous).
    r.wk = cg * problem.taps;
    r.wc = problem.filterChannelsLast ? 1 : problem.taps;
    r.wtap = problem.filterChannelsLast ? cg : 1;
    r.tile = 0;
    if(reduceRows)
    {
        r.tile = 2;
    }
    else if(reduceTile.blocks > 0)
    {
        r.tile = 1;
    }
    r.tileK = static_cast<int>(reduceTile.tileK);
    r.tileC = static_cast<int>(reduceTile.tileC);
    r.tilesK = static_cast<int>(reduceTile.tilesK);
    r.tilesC = static_cast<int>(reduceTile.tilesC);
    // Many splits into few quads: the deep reduce (the same sum, more loads in flight, smaller
    // blocks), where the quad reduce's grid leaves most units idle and waits on load latency.
    if(!dwStore && !accurate && r.tile == 0 && splits >= K3_REDUCE_DEEP_MIN_SPLITS
       && K3_REDUCE_DEEP_UNIT_SHARE * divUp(quads, REDUCE_QUADS_PER_BLOCK) <= units)
    {
        const int64_t perBlock = deepReduceQuads(quads, units);
        if(divUp(quads, perBlock) <= INT_LIMIT)
        {
            r.tile = 3;
            r.quads = static_cast<int>(perBlock);
            launch.reduceGrid = static_cast<unsigned int>(divUp(quads, perBlock));
            launch.reduceBlock = static_cast<unsigned int>(4 * perBlock);
        }
    }
    return launch;
}

int64_t deepReduceQuads(int64_t quads, int64_t dispatchUnits)
{
    // The largest block (64 quads) that still gives every unit two blocks, down to 16.
    int64_t perBlock = K3_REDUCE_DEEP_MAX_QUADS;
    while(perBlock > 16 && divUp(quads, perBlock) < 2 * std::max<int64_t>(dispatchUnits, 1))
    {
        perBlock /= 2;
    }
    return perBlock;
}

bool threeBf16Pays(const WrwProblem& problem, std::string_view gcnArchName)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    const auto& a = problem.base.args;
    const int64_t cg = a.c / a.groups;
    const int64_t kg = a.k / a.groups;
    const int64_t taps = problem.taps;
    if(arch == "gfx942" || arch == "gfx950")
    {
        const bool unitStride = a.stride[0] == 1 && a.stride[1] == 1 && a.stride[2] == 1;
        return unitStride && taps >= 9 && cg >= 128 && kg >= 64;
    }
    if(arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0)
    {
        return cg <= 4 || (cg >= 256 && taps >= 9);
    }
    return false;
}

std::vector<std::pair<std::string, std::string>> compileDefinesFor(data_objects::DataType dataType,
                                                                   const WrwConfig& config,
                                                                   std::string_view kernelType)
{
    std::string element;
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        element = "float";
        break;
    case data_objects::DataType::HALF:
        element = "_Float16";
        break;
    case data_objects::DataType::BFLOAT16:
        element = "unsigned short"; // bf16 bits
        break;
    default:
        throw std::invalid_argument("conv_wrw: unsupported element type");
    }
    // Three-bf16: the bf16 kernel, ConvWrwSplit (CONV_K3_MATH), and fp32 dw from the reduces
    // (CONV_K3_TO). Defined only then, so no other config's code changes.
    const bool split = config.math == 1 && dataType == data_objects::DataType::FLOAT;
    std::vector<std::pair<std::string, std::string>> defines{
        {"CONV_K3_T", split ? std::string("unsigned short") : element}};
    if(split)
    {
        defines.emplace_back("CONV_K3_TO", "float");
        defines.emplace_back("CONV_K3_MATH", "1");
    }
    if(!kernelType.empty())
    {
        defines.emplace_back("CONV_K3_ATOM", std::string(kernelType));
    }
    const std::array<std::pair<const char*, int64_t>, 19> fields{{{"LOOP", config.loop},
                                                                  {"LAYOUT", config.layout},
                                                                  {"BK", config.bk},
                                                                  {"BC", config.bc},
                                                                  {"BQ", config.bq},
                                                                  {"FR", config.fr},
                                                                  {"S", config.s},
                                                                  {"SW", config.sw},
                                                                  {"DW", config.dw},
                                                                  {"WK", config.wk},
                                                                  {"WC", config.wc},
                                                                  {"TK", config.tk},
                                                                  {"TC", config.tc},
                                                                  {"VEC", config.vec},
                                                                  {"XCD", config.xcd},
                                                                  {"ALIGN", config.align},
                                                                  {"NS", config.ns},
                                                                  {"WQ", config.wq},
                                                                  {"DWS", config.dws}}};
    for(const auto& [name, value] : fields)
    {
        defines.emplace_back(std::string("CONV_K3_") + name, std::to_string(value));
    }
    // Defined only when set, so every other config compiles as before.
    if(config.pb != 0)
    {
        defines.emplace_back("CONV_K3_PB", std::to_string(config.pb));
    }
    if(config.flat != 0)
    {
        defines.emplace_back("CONV_K3_FLAT", std::to_string(config.flat));
    }
    return defines;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_wrw

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
