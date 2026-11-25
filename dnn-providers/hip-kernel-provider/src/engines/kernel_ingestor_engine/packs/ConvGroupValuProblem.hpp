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

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/descriptors/conv/group_valu/kernels/ConvGroupValuArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"

/**
 * @file ConvGroupValuProblem.hpp
 * @brief The graph and config half of CONV_GROUP_VALU_FWD / CONV_GROUP_VALU_BWD (grouped
 *        convolution on vector ALU, ConvGroupValu.cpp): which graphs they take, what a catalog
 *        config declares and how a config runs a problem (kernel roles, lanes a row, load and
 *        store widths, row blocks, the prepared filter's workspace).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_group_valu
{

/// A forward or backward-data convolution the engines serve: one conv node, 1-D, 2-D or 3-D,
/// fp32 / fp16 / bf16 tensors of one element type, compute type unset or fp32, any groups, both
/// conv modes, extents and channel counts within 32 bits, any stride. x and y
/// both channels-first with the width packed, or both channels-last with the channels packed.
struct GvProblem
{
    conv_naive::ConvNaiveProblem base;
    conv_naive::ConvDirection direction = conv_naive::ConvDirection::FORWARD;
    int n = 0;
    int groups = 0;
    int cpg = 0; ///< x channels a group
    int kpg = 0; ///< y channels a group
    std::array<int, 3> in{}; ///< x extents (d, h, w; leading 1s below 3-D)
    std::array<int, 3> out{}; ///< y extents
    std::array<int, 3> filter{}; ///< t, r, s
    std::array<int, 3> pad{}; ///< pre-padding
    std::array<int, 3> stride{};
    std::array<int, 3> dilation{};
    int flip = 0;
    int elementBytes = 0;
    bool channelsFirst = false;
    bool channelsLast = false;
    std::array<long long, 5> xStr{}; ///< n, c, d, h, w element strides
    std::array<long long, 5> yStr{};
    std::array<long long, 5> wStr{}; ///< k, c, t, r, s
    long long xBytes = 0;
    long long yBytes = 0;
    int xAlign = 0; ///< the declared byte alignment of x's pointer, at most 16
    int yAlign = 0;
};

/// The problem @p context's graph describes in @p direction, or nullopt.
std::optional<GvProblem> parseGvProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                                        conv_naive::ConvDirection direction);

/// A catalog config: its direction (`FWD` or `BWD`), layout (`nchw` / `nhwc`), the output
/// channels a wave (kb), input channels a step (cb), output columns a lane (p), the filter class
/// (fd x fr x fs taps, width stride sw, width dilation dw, the graph's width pre-padding pw) and
/// the waves a workgroup (nw).
struct Config
{
    std::string direction;
    std::string layout;
    int kb = 0;
    int cb = 0;
    int p = 0;
    int fd = 0;
    int fr = 0;
    int fs = 0;
    int sw = 0;
    int dw = 0;
    int pw = 0;
    int nw = 0;
};

/// Accumulators a lane holds (kb x p) at most.
inline constexpr int MAX_ACCUMULATORS = 64;

/// The config @p kernel describes, or nullopt: direction FWD or BWD; layout nchw or nhwc; kb 1,
/// 2, 4, 8 or 16; cb 1, 2 or 4; p 1 to 16; kb x p at most MAX_ACCUMULATORS; fd, fr, fs 1 to 7; sw 1
/// or 2 (BWD: p a multiple of sw); dw 1 to 4; pw at least 0; nw 1, 2, 4 or 8.
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// How a config runs a problem on a device.
struct Plan
{
    Config config;
    ConvGroupValuArgs args{}; ///< every field but the pointers
    int wave = 0;
    int kernelPw = 0; ///< the kernel's width padding (backward: span - 1 - pad)
    int vbx = 0; ///< bytes a load of the input operand
    int vby = 0; ///< bytes a store of the output operand
    bool ragged = false;
    bool colmask = false;
    long long prepareGrid = 0;
    size_t workspaceBytes = 0; ///< the prepared filter
};

/// Lanes of a wave on @p gcnArchName when the device does not report it (64 on gfx9, 32 after).
int planWave(std::string_view gcnArchName, int reportedWarpSize);

/// The plan @p config runs @p problem with on a device of @p wave lanes and @p computeUnits
/// dispatch units (0: unknown), or nullopt when it does not serve it.
std::optional<Plan>
    planFor(const GvProblem& problem, const Config& config, int wave, int64_t computeUnits);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_group_valu

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
