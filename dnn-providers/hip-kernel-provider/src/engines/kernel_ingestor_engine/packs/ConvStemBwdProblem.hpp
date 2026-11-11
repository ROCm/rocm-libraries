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

#include "engines/kernel_ingestor_engine/descriptors/conv/stem_bwd/kernels/ConvStemBwdArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvStemBwdProblem.hpp
 * @brief The host half of the stem backward-data engine (CONV_STEM_BWD): the graphs it serves,
 *        the config a kernel's metadata declares, and the launch of a config on a problem (the
 *        filter transform, its workspace, the main kernel's grid). Kept apart from the dispatch
 *        code so every gate can be tested without a device.
 *
 * dx of a convolution with at most 4 input channels (network stems: 7x7 stride 2, 3x3 stride
 * 2, ...) as one GEMM per block of "super-pixels", each the stride-phase group of sh x sw dx
 * pixels: M the super-pixels, N the input channels times the phases (filling the matrix core's
 * columns), the reduction the output channels times the window offsets u x v (the filter rows /
 * columns of one phase). ConvStemBwdTransform first rearranges the filter into that N x
 * (u v k) operand in the workspace; the main kernel then reads dy in place (packed N K H W or
 * N H W K) and writes dx through its strides. No split, no atomics: deterministic.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_stem_bwd
{

/// The atoms a config may name, in atom_id order: K1's matrix-core atoms (no TF32 form), then
/// "fma" (exact fp32 on the vector ALUs, for devices without an fp32 matrix core).
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
/// @p gcnArchName (suffix intact). Where none does (gfx906, gfx900), "fma" takes that type too.
bool runsMatrixAtomFor(std::string_view element, std::string_view gcnArchName);

/// An atom's fragment edge (DIM: the super-columns of a fragment and the GEMM columns of one)
/// and the output channels one step reduces (KA). The FMA loop's: 16 and 4.
struct AtomShape
{
    int64_t dim = 0;
    int64_t ka = 0;
};

/// The shape of @p atom; zero for an unknown one.
AtomShape atomShapeOf(std::string_view atom);

/// The dy layouts (CONV_SB_LAYOUT).
inline constexpr int64_t LAYOUT_CHANNELS_FIRST = 0;
inline constexpr int64_t LAYOUT_CHANNELS_LAST = 1;

/// The most input channels the engine serves (stems: RGB, RGBA, grey).
inline constexpr int64_t MAX_CHANNELS = 4;

/// The most filter depth taps a 3-D config folds into the reduction.
inline constexpr int64_t MAX_DEPTH_TAPS = 8;

/// The super-pixel edge of space-to-depth configs (stride 1, StemBwdConfig::sd).
inline constexpr int64_t SPACE_TO_DEPTH = 2;

/// The most GEMM columns (input channels x stride phases) a config may compute.
inline constexpr int64_t MAX_COLUMNS = 64;

/// One kernel config: the KMD fields minus direction and dtype.
struct StemBwdConfig
{
    std::string atom;
    int64_t atomId = -1;
    int64_t layout = -1;
    /// Channels-first dy loads: 1 element, or 16 bytes of aligned columns. Channels-last: 16
    /// bytes of output channels.
    int64_t vec = 0;
    /// Window offsets: filter rows / columns per stride phase (the problem's, exactly).
    int64_t u = 0;
    int64_t v = 0;
    /// GEMM columns computed: at least the input channels times the stride phases.
    int64_t nb = 0;
    /// Output channels per chunk.
    int64_t kc = 0;
    /// Super-rows per block (one wave each on a matrix atom) and fragments of super-columns.
    int64_t th = 0;
    int64_t fm = 0;
    /// The filter's height and width strides.
    int64_t sh = 0;
    int64_t sw = 0;
    /// 3-D: the filter's depth taps (unit depth stride and dilation; the dx planes in the grid,
    /// the taps in the reduction). 0: 2-D (a trivial depth axis).
    int64_t ft = 0;
    /// Stride 1: SPACE_TO_DEPTH, super-pixels of sd x sd dx pixels on a space-to-depth view of
    /// dy (the phases fill the GEMM columns, the sub-pixels join the reduction). 0: off.
    int64_t sd = 0;
};

/// A backward-data convolution the kernels serve.
struct StemBwdProblem
{
    conv_naive::ConvNaiveProblem base;
    int64_t dxUid = 0;
    int64_t wUid = 0;
    int64_t dyUid = 0;
    int elementBytes = 0;
    /// dy is packed channels-first / channels-last (both hold when K is 1).
    bool dyChannelsFirst = false;
    bool dyChannelsLast = false;
    /// The window: offsets (oh, ow) and extents (u, v) the problem needs (windowOf).
    int64_t oh = 0;
    int64_t ow = 0;
    int64_t u = 0;
    int64_t v = 0;
    /// The filter's depth taps when the depth axis is not trivial, else 0 (StemBwdConfig::ft).
    int64_t depthTaps = 0;
    /// Height and width stride 1: the space-to-depth window (spaceToDepthWindowOf), else zero.
    int64_t sdOh = 0;
    int64_t sdOw = 0;
    int64_t sdU = 0;
    int64_t sdV = 0;
    int64_t sdShiftH = 0;
    int64_t sdShiftW = 0;
};

/// The window of a filter of @p filter taps at @p stride and @p padding along one axis: super-
/// pixel i reads dy positions i + offset - t for t < extent. offset = floor((stride - 1 +
/// padding) / stride), extent = offset - ceil((padding - filter + 1) / stride) + 1.
struct Window
{
    int64_t offset = 0;
    int64_t extent = 0;
};
Window windowOf(int64_t filter, int64_t stride, int64_t padding);

/// The window of a stride-1 axis on its space-to-depth view: super-pixel i (dx 2 i + p, p < 2)
/// reads dy super-positions i + offset - t for t < extent, super-position y holding dy 2 y +
/// shift + e (e < 2). The dy span 2 i + lo .. 2 i + padding + 1, lo = padding - filter + 1:
/// shift = lo mod 2, offset = floor((padding + 1 - shift) / 2), extent = offset - floor(lo / 2)
/// + 1.
struct SpaceToDepthWindow
{
    int64_t offset = 0;
    int64_t extent = 0;
    int64_t shift = 0;
};
SpaceToDepthWindow spaceToDepthWindowOf(int64_t filter, int64_t padding);

/**
 * @brief The stem backward-data convolution @p context's graph describes, or nullopt when the
 *        kernels do not serve it.
 *
 * Everything parseConvNaiveProblem gates (both convolution modes), plus: fp32, fp16 or bf16;
 * one group; at most MAX_CHANNELS input channels; a trivial depth axis (dx, dy and the filter
 * of depth 1, no depth padding; the depth stride and dilation then change nothing) or one of
 * unit stride and dilation, 1 to MAX_DEPTH_TAPS filter taps and padding at most 64; unit height
 * and width dilation; height and width strides, filter extents and padding at most 64 (the
 * window and the GEMM columns stay small; configs fix the strides and window); dy packed
 * channels-first or channels-last; dx packed either way (written through its strides); the
 * filter packed K C R S or K R S C; the batch times dx's depth within 65535 (the grid's z); every extent and
 * the output channels within 32 bits; one image of dy within 2^31 bytes (32-bit buffer
 * offsets).
 */
std::optional<StemBwdProblem>
    parseStemBwdProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// LDS bytes a config's kernel declares (sbLdsBytes, the arithmetic the kernel asserts).
int64_t ldsBytesOf(const StemBwdConfig& config, int elementBytes);

/// The LDS a workgroup may use on @p gcnArchName: 160 KiB on gfx950, else 64 KiB.
int64_t ldsLimitOf(std::string_view gcnArchName);

/// The launch of a config on a problem: both kernels' grids and argument blocks, and the
/// workspace of the transformed filter.
struct StemBwdLaunch
{
    unsigned int gridX = 0;
    unsigned int gridY = 0;
    unsigned int gridZ = 0;
    unsigned int block = 0;
    unsigned int transformGrid = 0;
    unsigned int transformBlock = 256;
    /// K1's atom template (empty for FMA).
    std::string_view kernelType;
    /// Pointers are filled at launch.
    ConvStemBwdArgs args{};
    ConvStemBwdTransformArgs transform{};
    size_t workspaceBytes = 0;
};

/// The transform's most blocks (a grid-stride loop covers the rest).
inline constexpr int64_t MAX_TRANSFORM_BLOCKS = 1024;

/**
 * @brief The launch of @p config on @p problem on @p device, or nullopt when the config does
 *        not compute it correctly or cannot launch it there.
 *
 * Reads and gates every field of @p config: the atom (known, its atom_id, running on the
 * device, taking the graph's element type; "fma" on fp32 graphs, or 16-bit ones where
 * runsMatrixAtomFor finds no atom for the type); the layout, one dy is
 * packed in; vec (channels-last: 16 bytes, the output channels whole vectors; channels-first:
 * 1, or 16 bytes when dy rows are whole vectors); u and v the problem's window extents; sh and
 * sw the problem's strides; ft the problem's depth taps (0 on a trivial depth axis); sd 0, or
 * SPACE_TO_DEPTH on a stride-1 problem of trivial depth (u and v then its space-to-depth
 * window's, channels-first vec 1, the phases sd x sd); nb at least the input channels times the phases, at most
 * MAX_COLUMNS, whole fragments on a matrix atom; kc whole atom steps and 16-byte units; th and
 * fm >= 1 with the block within 1024 threads; the LDS within the device's; the grid within the
 * launch limits (x 2^31 - 1; y and z 65535); the transformed filter within 2^31 bytes (32-bit buffer
 * offsets).
 */
std::optional<StemBwdLaunch>
    stemBwdLaunchFor(const StemBwdProblem& problem,
                     const StemBwdConfig& config,
                     const hipdnn_plugin_sdk::ingestor::DeviceProperties& device);

/**
 * @brief The preprocessor definitions ConvStemBwd.cpp is compiled with for @p config on
 *        @p dataType (CONV_SB_T, CONV_SB_ATOM = @p kernelType unless empty (FMA), and one
 *        CONV_SB_<FIELD> per integer field), in a fixed order. Both entry points compile from
 *        the same definitions.
 *
 * @throws std::invalid_argument for an element type the kernels do not take.
 */
std::vector<std::pair<std::string, std::string>>
    compileDefinesFor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                      const StemBwdConfig& config,
                      std::string_view kernelType);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_stem_bwd
