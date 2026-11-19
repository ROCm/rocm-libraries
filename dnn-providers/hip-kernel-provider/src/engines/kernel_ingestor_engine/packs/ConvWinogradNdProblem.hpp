// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "engines/kernel_ingestor_engine/descriptors/conv/winograd_nd/kernels/ConvWinogradNdArgs.h"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"
#include <array>
#include <cstddef>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <optional>
#include <string>
#include <string_view>
namespace hip_kernel_provider::kernel_ingestor_engine::conv_winograd_nd
{
// FLOAT forward rank-two dilated stride-one 3x3 residue-class convolution.
struct Config
{
    int atom = 5, bm = 64, bn = 64, bk = 16, chunkTiles = 4096;
    std::string arch = "gfx1250";
};
struct Problem
{
    conv_naive::ConvNaiveProblem base;
    KNdArgs args{};
};
struct Plan
{
    KNdArgs args{};
    long long totalTiles = 0;
    int bins = 0;
    size_t uOffset = 0, vOffset = 0, mOffset = 0, workspaceBytes = 0;
    unsigned int filterBlocks = 1, inputBlocks = 1, outputBlocks = 1;
    std::array<unsigned int, 3> gemmBlocks = {1, 1, 1};
};
std::optional<Problem> parseProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                                    conv_naive::ConvDirection direction);
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& k);
bool configRunsOn(const Config& c, std::string_view arch);
// limit includes fixed U and the two spatially chunked buffers, aligned to 256 bytes.
std::optional<Plan> planFor(const Problem& p, const Config& c, size_t limit = 256ULL * 1024 * 1024);
}
#endif
