// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/group/kernels/ConvGroupArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvTf32Split.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvWrwProblem.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_group
{

/// One launch of the K4 kernel and its filter expansion: forward and unit-stride backward data
/// run one; strided backward data one per stride phase that has outputs.
struct GroupPhase
{
    K4Args args{}; ///< pointers are filled at launch
    K4ExpandArgs expand{}; ///< pointers are filled at launch
    size_t expandOffset = 0; ///< this phase's expanded filter in the workspace
};

/// A small-group convolution as the K4 kernels run it: the activations staged channels-last
/// exactly as the implicit-GEMM engine stages them (its parse, its stagings), each phase's
/// expanded filter in the workspace after them.
struct GroupProblem
{
    conv_igemm::IgemmProblem igemm;
    std::vector<GroupPhase> phases;
    int filterT = 1, filterR = 0, filterS = 0; ///< the graph's filter extent
    int stride = 1; ///< the graph's row and column stride
    /// The kernel's compile-time filter extent and gather stride: forward the graph's;
    /// backward data the taps a stride phase reaches (ceil(F / stride)) at unit stride.
    /// Wide groups (16 < cg <= 64): the 16-channel blocks a group spans; 0 for small groups.
    int nbi = 0;
    int kernelT = 1, kernelR = 0, kernelS = 0, kernelStride = 1;
    size_t workspaceBytes = 0; ///< staged activations and the expanded filters
};

/// The convolution @p context's graph describes in @p direction, or nullopt when the K4 kernels
/// do not serve it. Everything parseIgemmProblem gates, plus: groups of cg channels each way with
/// 2 <= cg <= 16, cg dividing 16 and channels a multiple of 16 (not depthwise: those are the
/// depthwise engines'), or 16 < cg <= 64 with cg a multiple of 8 (wide groups), unit dilation, equal row and column strides (backward data: row, column and
/// depth strides 1 or 2), and grids within the launch limits for the smallest tile. 2-D or 3-D.
std::optional<GroupProblem>
    parseGroupProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                      conv_naive::ConvDirection direction);

/// One launch of the K4 kernel: its arguments (pointers filled at launch) and the phase whose
/// expanded filter it reads (phase-fused launches read that phase's and the next three).
struct GroupLaunch
{
    K4Args args{};
    size_t phase = 0;
};

/// The launches @p problem runs: one per phase, or (@p phased: backward data at row and column
/// stride 2, the phase-fused kernel) one per depth phase, each running its four row and column
/// phases with their paddings and filter stride in the arguments. nullopt where the phased form
/// does not apply (not stride 2 on rows and columns, or a depth phase without all four).
std::optional<std::vector<GroupLaunch>> groupLaunches(const GroupProblem& problem, bool phased);

/// LDS bytes of the phase-fused kernel: the halo (one row and column wider than a phase's) and
/// the 2 th x 2 tw output tile of ob output blocks (0: cb, the small-group kernel's).
long long groupPhasedLdsBytes(
    int elementBytes, int r, int s, int th, int tw, int cb, bool cf, int ob = 0);

/// @p problem with its activations read and written in place by the channels-first (cf) kernels:
/// nullopt unless x and y are both packed channels-first (both staged by transposes).
std::optional<GroupProblem> channelsFirstInPlace(const GroupProblem& problem);

/// Workgroups of a th x tw output tile over cb channel blocks for one launch: images x output
/// planes x row tiles x column tiles x channel tiles, or nullopt past the launch limit.
std::optional<unsigned int> phaseBlocks(const K4Args& args, int th, int tw, int cb);

/// Workgroups of a wide-group launch: images x output planes x row tiles x column tiles x groups
/// x (nbi / nbo) output-block tiles, or nullopt past the launch limit.
std::optional<unsigned int> phaseBlocksWide(const K4Args& args, int th, int tw, int nbi, int nbo);

/// Whether @p blocks workgroups of @p threads fit one dispatch (at most 2^32 - 1 work items
/// along x).
bool fitsOneDispatch(long long blocks, long long threads);

/// Elements of one phase's expanded filter: block-diagonal per 16-channel block (small groups)
/// or dense per group, padded to nbi blocks each way (wide groups).
size_t expandedElements(const GroupProblem& problem);

/// LDS bytes of the input halo a config stages for a th x tw output tile (the kernel's static
/// array).
long long groupLdsBytes(int elementBytes, int r, int s, int stride, int th, int tw, int cb);

/// A small-group backward-weights convolution as the K4 wrw kernel runs it (CONV_GROUP_WRW): x
/// and dy read in place, both packed channels-first or both packed channels-last, dw written by
/// the reduce through its strides.
struct GroupWrwProblem
{
    conv_wrw::WrwProblem wrw;
    int cg = 0; ///< channels a group (x, dy and dw alike)
    /// Wide groups (16 < cg <= 32): the 16-channel blocks a group spans; 0 for small groups.
    int nbi = 0;
    int filterT = 1, filterR = 0, filterS = 0;
    int stride = 1; ///< the row and column stride
};

/// The backward-weights convolution @p context's graph describes, or nullopt when the K4 wrw
/// kernel does not serve it. Everything parseWrwProblem gates (x and dy both packed
/// channels-first or both packed channels-last, dw packed K C T R S or K T R S C, 32-bit
/// extents), plus: groups of cg channels each way with 2 <= cg <= 16, cg dividing 16 and the
/// channels a multiple of 16 (not depthwise), or 16 < cg <= 32 with cg a multiple of 8 (wide
/// groups, nbi = 2 blocks), unit dilation, equal row and column strides.
std::optional<GroupWrwProblem>
    parseGroupWrwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// LDS bytes of the wrw kernel's tile: the x halo of a th x tw dy tile at @p stride, the dy tile
/// and one zero pixel, each pixel 16 cb channels plus a 16-byte pad; wide groups (@p nbi > 0) x
/// pixels of 16 nbi channels and dy pixels of 16 nbo (@p cb's role), and (below 4 blocks a group)
/// the waves' fp32 sums, whichever is larger.
long long groupWrwLdsBytes(
    int elementBytes, int r, int s, int stride, int th, int tw, int cb, int nbi = 0);

/// One wrw launch: the kernel's grid and arguments, the reduce's, and the partials' workspace.
struct GroupWrwLaunch
{
    K4WrwArgs args{}; ///< pointers filled at launch
    K4WrwReduceArgs reduce{}; ///< pointers filled at launch
    unsigned int grid = 0;
    unsigned int reduceGrid = 0;
    size_t workspaceBytes = 0;
};

/// The launch of a th x tw tile of @p blocks 16-channel blocks (cb; wide groups: nbo dy blocks of
/// a group) on @p problem, its splits sized for @p bpu workgroups a
/// dispatch unit of @p device (conv_wrw::splitsWanted, at least 4 tiles a split, the partials
/// within 1 GiB), or nullopt past the launch limits or on a device that reports no units.
std::optional<GroupWrwLaunch>
    groupWrwLaunch(const GroupWrwProblem& problem,
                   int th,
                   int tw,
                   int blocks,
                   int bpu,
                   const hipdnn_plugin_sdk::ingestor::DeviceProperties& device);

/// Three-bf16 (TF32 permission, math 1 configs): the split copies a launch reads and where they
/// sit in the workspace (after everything else the dispatch keeps there).
struct GroupSplit
{
    conv_tf32::SplitPlan plan;
    size_t offset = 0; ///< the split copies' first byte in the workspace
    size_t workspaceBytes = 0; ///< the whole workspace, the split copies included
};

/// Forward and backward data, wide groups: the kernel's input (x forward, dy backward, as staged
/// or in place) split whole into parts hi | hi' | lo, and every phase's fp32 expanded filter (one
/// span) into hi | lo | hi'. The bf16 wide kernel runs P 3: each depth tap multiplies the three
/// part pairs into the same fp32 accumulators.
GroupSplit groupSplitOf(const GroupProblem& problem);

/// Whether the three-bf16 form of @p problem is admitted on @p gcnArchName: wide groups (small
/// groups and backward weights measured slower than exact) on the matrix-core architectures it
/// is built for (gfx90a, gfx942, gfx950, gfx11, gfx12).
bool threeBf16Pays(const GroupProblem& problem, std::string_view gcnArchName);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_group
