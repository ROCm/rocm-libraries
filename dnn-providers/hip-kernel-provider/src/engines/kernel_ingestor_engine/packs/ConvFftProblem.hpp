// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/fft/kernels/ConvFftArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvFftProblem.hpp
 * @brief The graph and config half of the FFT convolution engines (CONV_FFT_FWD, CONV_FFT_BWD,
 *        CONV_FFT_WRW): which graphs they take, what a catalog config declares, and how a config
 *        runs a problem (tiles, chunks, workspace, grids). Kept apart from the dispatch code so
 *        the rules can be tested without a device.
 *
 * The engines are overlap-save tiled FFT convolutions (ConvFftCore.h): fp32 transforms and
 * products whatever the tensors' element type, so they carry the FFT numerical note: the error
 * of an output is relative to the magnitude of the whole transform (about log2 of the tile size
 * units of fp32 rounding of the largest terms), not to the output itself.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_fft
{

/// A convolution the engines serve: one conv node, 1-D or 2-D, fp32 / fp16 / bf16 tensors of one
/// element type, compute type unset or fp32, any layout (element strides; the written tensor's
/// elements distinct), any groups, unit stride, any dilation and padding, both conv modes, every
/// extent and channel count within 32 bits.
struct FftProblem
{
    conv_naive::ConvNaiveProblem base;
    int n = 0;
    int groups = 0;
    int cg = 0; ///< input channels a group
    int kg = 0; ///< output channels a group
    int h = 0, w = 0; ///< x extents (1-D: h = 1)
    int ho = 0, wo = 0; ///< y extents
    int r = 0, s = 0; ///< filter extents
    int padH = 0, padW = 0; ///< pre-padding
    int dilH = 1, dilW = 1;
    int flip = 0; ///< convolution mode: the filter is reflected
    long long xStr[4] = {}; ///< n, c, h, w element strides
    long long wStr[4] = {}; ///< k, c, r, s
    long long yStr[4] = {}; ///< n, k, h, w

    int spanH() const
    {
        return (r - 1) * dilH + 1;
    }
    int spanW() const
    {
        return (s - 1) * dilW + 1;
    }
};

/// The problem @p context's graph describes in @p direction, or nullopt.
std::optional<FftProblem> parseFftProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                                          conv_naive::ConvDirection direction);

/// A catalog config, from its metadata: the transform size th x tw (th 1 for a 1-D transform),
/// tiles a transform block (pb) and its threads (tb), and the product tile's output channels (gn).
struct Config
{
    int th = 0;
    int tw = 0;
    int pb = 0;
    int tb = 0;
    int gn = 0;
    int tm = 4; ///< tiles a product thread (4, or 8 with gn 64)

    int bins() const
    {
        return th * (tw / 2 + 1);
    }
};

/// The config @p kernel describes, or nullopt for one the kernels cannot compile: th 1 or a power
/// of two 8..64, tw a power of two 8..64, pb >= 1 (even for th 1) with pb x bins x 8 bytes of LDS
/// within 64 KiB, tb a multiple of 64 up to 256 and at least one thread a row pair, gn 16, 32 or
/// 64, tm 4 (or 8 with gn 64).
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Upper bound of the workspace a plan asks for; tiles are processed in chunks below it.
inline constexpr size_t WORKSPACE_CAP_BYTES = size_t{1} << 30;

/// WORKSPACE_CAP_BYTES, or HIP_KERNEL_PROVIDER_CONV_FFT_WORKSPACE_CAP (bytes, at least 4096) when
/// set: tests force several chunks with it.
size_t workspaceCapBytes();

/// The admission floor: a config serves a problem only when its frequency-domain product does at
/// most 1 / ADMIT_RATIO of the direct convolution's multiply-accumulates (counted in real
/// multiply-accumulates, 4 per complex one), and the problem has at least ADMIT_MIN_MACS of them.
/// FFT pays transforms and a frequency-domain workspace on top of its product, so it only wins
/// where the filter is large against the tile. Calibrated on gfx950 (engine bench of every
/// config, every engine): fp32 dense wins from 5x5 (1.8x) and 1 x 7 (1.3x) up.
inline constexpr double ADMIT_RATIO = 4.0;

/// The floor for fp16 / bf16 tensors: the product stays fp32 (no reduced-precision product)
/// while the direct engines run 16-bit matrix cores at several times the fp32 rate, so FFT
/// needs a far larger advantage: on gfx950 it ties at 11x11 (advantage ~25) and loses at 7x7
/// (~7-10).
inline constexpr double ADMIT_RATIO_16BIT = 16.0;
inline constexpr int64_t ADMIT_MIN_MACS = int64_t{1} << 26;

/// False when HIP_KERNEL_PROVIDER_CONV_FFT_FLOOR=0: every config whose transform fits the filter
/// is admitted (tests, and measuring where the floor belongs).
bool admissionFloorEnabled();

/// The direct convolution's multiply-accumulates over the product's (real) ones for @p config.
/// Depthwise problems (one input and one output channel a group) have no product over channels
/// to amortize the transforms, so their count adds the forward and inverse transforms of a tile
/// (2.5 th tw log2(th tw) multiply-accumulate equivalents each) per output tile.
double fftAdvantage(const FftProblem& problem, const Config& config);

/// The admission floor of the depthwise path (same meaning as ADMIT_RATIO, transforms counted):
/// depthwise direct kernels are bound by memory and latency rather than arithmetic, so parity
/// in arithmetic pays: on gfx950 a 13x13 depthwise layer (advantage 1.1) runs 2x faster than
/// the direct engines forward and 24x in backward weights; 7x7 (0.45) is left to them.
inline constexpr double ADMIT_RATIO_DEPTHWISE = 1.0;

/// The admission floor that applies to @p problem: ADMIT_RATIO_DEPTHWISE for the depthwise path,
/// else ADMIT_RATIO for fp32 tensors and ADMIT_RATIO_16BIT for fp16 / bf16.
double admissionRatio(const FftProblem& problem);

/// True for the depthwise path: one input and one output channel a group.
inline bool isDepthwise(const FftProblem& problem)
{
    return problem.cg == 1 && problem.kg == 1;
}

/// How a config runs a problem.
struct Plan
{
    conv_naive::ConvDirection direction = conv_naive::ConvDirection::FORWARD;
    Config config;
    ConvFftArgs args{}; ///< the fields every launch shares; per-launch fields set at launch
    int bins = 0;
    /// The depthwise path: fwd / bwd as filter transform + one fused transform-multiply-inverse
    /// launch; wrw as per-partition sums, an ordered fold and the inverse.
    bool depthwise = false;
    int tiles = 0; ///< P
    int chunk = 0; ///< tiles a chunk
    int chunks = 0;
    // workspace layout (bytes from the start)
    size_t fixedOffset = 0; ///< filter spectra (fwd / bwd) or weight-gradient spectra (wrw)
    size_t fixedBytes = 0;
    size_t aOffset = 0; ///< data spectra of a chunk
    size_t aBytes = 0;
    size_t bOffset = 0; ///< product spectra (fwd / bwd) or dy spectra (wrw) of a chunk
    size_t bBytes = 0;
    size_t workspaceBytes = 0;
    // launch geometry (grid.x; every launch is one-dimensional)
    long long filterGrid = 0; ///< fwd / bwd filter transform; wrw weight-gradient inverse
    long long productGridPerChunk = 0;
    int transformThreads = 0;

    /// Grid of a data transform of @p tilesInChunk tiles over @p channels channels a group.
    long long dataGrid(int tilesInChunk, int channels) const;
};

/**
 * @brief The plan @p config runs @p problem with in @p direction on a device of @p computeUnits
 *        dispatch units (0: unknown; sizes the depthwise wrw partitions), or nullopt when the
 *        config does not serve it: the dilated filter does not fit the transform (span > th or tw, or a 1-D
 *        transform for a filter taller than one row), the config is below the admission floor,
 *        the filter (or weight-gradient) spectra alone exceed the workspace cap, or a launch
 *        exceeds a 32-bit grid or index.
 */
std::optional<Plan> planFor(const FftProblem& problem,
                            conv_naive::ConvDirection direction,
                            const Config& config,
                            int64_t computeUnits = 0);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_fft

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
