// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/igemm_nchw/kernels/ConvIgemmNchwArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvTf32Split.hpp"

/**
 * @file ConvIgemmNchwProblem.hpp
 * @brief The host half of the channels-first backward-data implicit-GEMM engine
 *        (CONV_IGEMM_NCHW_BWD, K7): the graphs it serves, the config a kernel's metadata
 *        declares, and the launch of a config on a problem. Kept apart from the dispatch code so
 *        every gate can be tested without a device.
 *
 * The kernel reads dy, the K C T R S filter and writes dx in place, so it serves backward data
 * whose operands are all packed channels-first, with unit dilation along depth and height: each
 * dx row reduces over every (dy row, filter row) pair that reaches it (none: the row is zero),
 * and the convolution runs along the width. That is the shape of ConvTranspose1d (as 1-D, or 2-D
 * with height 1), of per-row 1-D convolutions, and of strided NCHW/NCDHW backward data: strided
 * 1x1, patchify (filter equal to stride), 2x2x2 and 3x3x3 stride 2.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_igemm_nchw
{

/// The atoms a config may name, in atom_id order: the FMA kernel, then K1's matrix-core atoms.
inline constexpr std::array<std::string_view, 11> ATOMS{"fma",
                                                        "mfma16_f16",
                                                        "mfma16_bf16",
                                                        "mfma16_f32",
                                                        "mfma32_f16",
                                                        "mfma32_bf16",
                                                        "mfma32_f32",
                                                        "wmma11_f16",
                                                        "wmma11_bf16",
                                                        "wmma12_f16",
                                                        "wmma12_bf16"};

/// One kernel config: the KMD fields minus direction and dtype. Fields the kernel kind does not
/// use are 0 (wm, wn, wp for FMA; tq, tc for matrix cores); a config that sets one serves
/// nothing.
struct NchwConfig
{
    std::string atom;
    int64_t atomId = -1;
    /// Tile: phase indices (each covering every stride phase), channels, k-step.
    int64_t bmq = 0;
    int64_t bn = 0;
    int64_t bk = 0;
    /// Matrix core: waves along the phase indices, the channels and the stride phases (wp
    /// groups of stride / wp phases).
    int64_t wm = 0;
    int64_t wn = 0;
    int64_t wp = 0;
    /// FMA: phase indices and channels per thread.
    int64_t tq = 0;
    int64_t tc = 0;
    /// Elements per global load.
    int64_t vec = 0;
    /// The convolution's stride along the width, and the most taps a stride phase holds.
    int64_t stride = 0;
    int64_t taps = 0;
    /// LDS tiles in turn: FMA 1, or 2 (one barrier per k-step); matrix core 1.
    int64_t stages = 0;
    /// 0, or the stride phases computed over the flattened dy plane (the rest zero): the
    /// problem must be flat (NchwProblem::flat) with at most this many filter taps.
    int64_t flat = 0;
    /// The dy window's elements past the tile, whole vectors: at least the farthest tap's reach
    /// ((taps - 1) dilation, plus one at strides past 1). FMA: the vector (a register window,
    /// unit dilation), or wider at stride 1.
    int64_t halo = 0;
    /// 0: the graph's element type. 1: three-bf16 for an fp32 graph under TF32 permission
    /// (conv_igemm::tf32Permitted): a bf16 matrix-core config on dy and w split along K into 3K
    /// channels (ConvTf32Split.hpp), dx written in fp32.
    int64_t math = 0;
};

/// A backward-data convolution the kernel serves.
struct NchwProblem
{
    conv_naive::ConvNaiveProblem base;
    /// The kernel's argument block; pointers are filled at launch.
    K7Args args{};
    int64_t dxUid = 0;
    int64_t wUid = 0;
    int64_t dyUid = 0;
    int elementBytes = 0;
    /// The stride along the width.
    int64_t stride = 0;
    /// The dy plane can be read flattened: no padding on any axis, every filter extent at most its
    /// stride, dx exactly the strides times dy on every axis.
    bool flat = false;
};

/// The launch of a config on a problem.
struct NchwLaunch
{
    unsigned int gridX = 0;
    unsigned int gridY = 0;
    unsigned int gridZ = 0;
    unsigned int block = 0;
    /// K1's atom template (empty for the FMA kernel).
    std::string_view kernelType;
    /// Three-bf16 (math 1): the split pre-pass and the kernel's arguments for the 3K problem.
    bool split = false;
    conv_tf32::SplitPlan plan;
    K7Args args{};
};

/**
 * @brief The backward-data convolution @p context's graph describes, or nullopt when the kernel
 *        does not serve it.
 *
 * Everything parseConvNaiveProblem gates, plus: fp32, fp16 or bf16; one group;
 * cross-correlation; dy, dx and the filter packed channels-first (N C D H W, K C T R S); along
 * depth and height non-negative padding and a dilation only at stride 1;
 * dilation along the width only at stride 1; every extent, stride and padding within 32 bits, one image of
 * dy and the whole filter within 32-bit byte offsets, images times dx rows within the grid's
 * third axis.
 */
std::optional<NchwProblem>
    parseNchwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// The atom_id of @p atom, or -1 for an unknown one.
int64_t atomIdOf(std::string_view atom);

/// LDS bytes a config's kernel declares (the kernels' own arithmetic), for @p elementBytes (the
/// operands'; a math-1 config writes its output tile in fp32).
int64_t ldsBytesOf(const NchwConfig& config, int elementBytes);

/**
 * @brief The bf16 problem a three-bf16 config runs for fp32 @p problem: K tripled (dy and w
 *        split along K, ConvTf32Split.hpp), or nullopt when its 3K offsets leave 32 bits.
 */
std::optional<NchwProblem> threeBf16ProblemOf(const NchwProblem& problem);

/// Whether three-bf16 is admitted for @p problem on @p gcnArchName (where it pays).
bool threeBf16Pays(const NchwProblem& problem, std::string_view gcnArchName);

/**
 * @brief The launch of @p config on @p problem, or nullopt when the config does not compute it
 *        correctly or cannot launch it.
 *
 * Reads and gates every field of @p config: the atom and its atom_id; the kind's own fields set
 * and the other kind's 0; the tile against the atom's step (matrix core) or the thread tile
 * (FMA); the vector against the element size, the tile and the filter run; the stride equal to
 * the problem's and the taps per phase exactly ceil(S / stride); the halo covering the farthest
 * tap's reach (FMA: one vector at unit dilation, wider at stride 1); flat against the problem; the LDS within 64 KiB;
 * the grid within the launch limits. The device and dtype checks are the kernel
 * match's (atomRunsOn, the dtype string).
 *
 * math 0, or 1 (three-bf16): an fp32 problem, TF32 permitted, a bf16 matrix-core atom (the
 * kernel match also requires threeBf16Pays for the device); the config is then gated on
 * threeBf16ProblemOf(problem), and the launch carries the split pre-pass (plan) and the 3K
 * problem's arguments.
 */
std::optional<NchwLaunch> nchwLaunchFor(const NchwProblem& problem, const NchwConfig& config);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_igemm_nchw
