// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvRdnaFmaIgemm.hpp"

#include <array>
#include <limits>
#include <string>

#include "engines/kernel_ingestor_engine/IngestorLaunchRestore.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::fma_igemm
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;

namespace
{

// Must match kernels/ConvFwdFmaIgemm.cpp.
constexpr int64_t TILE_M = 128;
// Fewer output channels per group leave most of the (at least 32-wide) channel tile idle;
// depthwise convolutions belong to a depthwise kernel.
constexpr int64_t MIN_OUT_CHANNELS_PER_GROUP = 16;
constexpr int64_t MAX_GRID_X_BLOCKS = 0x7fffffff / BLOCK_SIZE;
constexpr int64_t MAX_GRID_Y = 0xffff;

constexpr std::string_view FAMILY_FIELD = "kernel_family";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view LAYOUT_FIELD = "layout";
constexpr std::string_view TILE_N_FIELD = "tile_n";

constexpr std::string_view N_VALUE = "n";
constexpr std::string_view C_VALUE = "c";
constexpr std::string_view K_VALUE = "k";
constexpr std::string_view GROUPS_VALUE = "groups";
// Per spatial axis (depth, height, width).
constexpr std::array<std::string_view, 3> IN_VALUES{"in_d", "in_h", "in_w"};
constexpr std::array<std::string_view, 3> FILTER_VALUES{"filter_d", "filter_h", "filter_w"};
constexpr std::array<std::string_view, 3> OUT_VALUES{"out_d", "out_h", "out_w"};
constexpr std::array<std::string_view, 3> PAD_VALUES{"pad_d", "pad_h", "pad_w"};
constexpr std::array<std::string_view, 3> STRIDE_VALUES{"stride_d", "stride_h", "stride_w"};
constexpr std::array<std::string_view, 3> DILATION_VALUES{"dilation_d", "dilation_h", "dilation_w"};

/// The kernel's CONV_DTYPE for a dtype.
int dtypeCode(data_objects::DataType dtype)
{
    if(dtype == data_objects::DataType::FLOAT)
    {
        return 0;
    }
    return dtype == data_objects::DataType::HALF ? 1 : 2;
}

} // namespace

int64_t tileN(const ConvProblem& problem)
{
    const int64_t kPerGroup = problem.outChannelsPerGroup();
    if(kPerGroup <= 32)
    {
        return 32;
    }
    return kPerGroup <= 64 ? 64 : 128;
}

int64_t gridXBlocks(const ConvProblem& problem)
{
    const int64_t m = int64_t{problem.n} * problem.out[0] * problem.out[1] * problem.out[2];
    return (m + TILE_M - 1) / TILE_M;
}

int64_t gridY(const ConvProblem& problem)
{
    const int64_t kPerGroup = problem.outChannelsPerGroup();
    const int64_t tile = tileN(problem);
    return problem.groups * ((kPerGroup + tile - 1) / tile);
}

bool serves(const ConvProblem& problem)
{
    if(problem.groups > 1 && problem.outChannelsPerGroup() < MIN_OUT_CHANNELS_PER_GROUP)
    {
        return false;
    }
    return gridXBlocks(problem) <= MAX_GRID_X_BLOCKS && gridY(problem) <= MAX_GRID_Y;
}

bool isVariantOf(const KernelDefinition& kernel, const ConvProblem& problem)
{
    return kernel.getStringMetadata(std::string(FAMILY_FIELD)) == FAMILY && serves(problem)
           && kernel.getStringMetadata(std::string(DTYPE_FIELD))
                  == data_objects::EnumNameDataType(problem.dtype)
           && kernel.getStringMetadata(std::string(LAYOUT_FIELD)) == layoutName(problem.layout)
           && kernel.getIntMetadata(std::string(TILE_N_FIELD)) == tileN(problem);
}

const std::vector<KernelArgument>& kernelSignature(bool hasBias)
{
    static const KernelArgument s_buffer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_scalar{"by_value", static_cast<uint32_t>(sizeof(int)), 0, ""};
    const auto make = [](size_t buffers) {
        std::vector<KernelArgument> signature(buffers, s_buffer);
        // N, C, Di, Hi, Wi, K, Z, Y, X, Do, Ho, Wo, sd, sh, sw, pd, ph, pw, dd, dh, dw, groups.
        signature.insert(signature.end(), 22, s_scalar);
        return signature;
    };
    static const std::vector<KernelArgument> s_plain = make(3);
    static const std::vector<KernelArgument> s_bias = make(4);
    return hasBias ? s_bias : s_plain;
}

void addCompileOptions(compilation::KernelCompileOptions& options,
                       const ConvProblem& problem,
                       bool hasBias)
{
    const int channelsLast = problem.layout == Layout::CHANNELS_LAST ? 1 : 0;
    options.add("CONV_DTYPE", dtypeCode(problem.dtype));
    options.add("CONV_CHANNELS_LAST", channelsLast);
    // Channels-first weights are read in place; channels-last ones are K x taps x C already.
    options.add("CONV_WEI_KCZYX", 1 - channelsLast);
    options.add("TILE_N", static_cast<int>(tileN(problem)));
    if(hasBias)
    {
        options.add("CONV_HAS_BIAS", 1);
    }
}

void launch(const compilation::IRunnableKernel& kernel,
            hipStream_t stream,
            const void* x,
            const void* w,
            void* y,
            const void* bias,
            bool hasBias,
            const ConvProblem& p)
{
    // A 2-D problem carries depth 1, depth stride and dilation 1 and depth padding 0.
    const auto run = [&](auto&&... buffers) {
        kernel.launch(stream,
                      buffers...,
                      p.n,
                      p.c,
                      p.in[0],
                      p.in[1],
                      p.in[2],
                      p.k,
                      p.filter[0],
                      p.filter[1],
                      p.filter[2],
                      p.out[0],
                      p.out[1],
                      p.out[2],
                      p.stride[0],
                      p.stride[1],
                      p.stride[2],
                      p.pad[0],
                      p.pad[1],
                      p.pad[2],
                      p.dilation[0],
                      p.dilation[1],
                      p.dilation[2],
                      p.groups);
    };
    if(hasBias)
    {
        run(x, w, y, bias);
    }
    else
    {
        run(x, w, y);
    }
}

void addProblemLaunchValues(MetadataValues& values, const ConvProblem& problem)
{
    values[std::string(N_VALUE)] = int64_t{problem.n};
    values[std::string(C_VALUE)] = int64_t{problem.c};
    values[std::string(K_VALUE)] = int64_t{problem.k};
    values[std::string(GROUPS_VALUE)] = int64_t{problem.groups};
    for(size_t axis = 0; axis < 3; ++axis)
    {
        values[std::string(IN_VALUES[axis])] = int64_t{problem.in[axis]};
        values[std::string(FILTER_VALUES[axis])] = int64_t{problem.filter[axis]};
        values[std::string(OUT_VALUES[axis])] = int64_t{problem.out[axis]};
        values[std::string(PAD_VALUES[axis])] = int64_t{problem.pad[axis]};
        values[std::string(STRIDE_VALUES[axis])] = int64_t{problem.stride[axis]};
        values[std::string(DILATION_VALUES[axis])] = int64_t{problem.dilation[axis]};
    }
}

std::vector<std::string_view> problemLaunchValueNames()
{
    std::vector<std::string_view> names{N_VALUE, C_VALUE, K_VALUE, GROUPS_VALUE};
    for(const auto* axisNames :
        {&IN_VALUES, &FILTER_VALUES, &OUT_VALUES, &PAD_VALUES, &STRIDE_VALUES, &DILATION_VALUES})
    {
        names.insert(names.end(), axisNames->begin(), axisNames->end());
    }
    return names;
}

ConvProblem readProblemLaunchValues(const SavedLaunchInputs& inputs)
{
    constexpr int64_t INT_MAX_VALUE = std::numeric_limits<int>::max();
    const auto positive = [&inputs](std::string_view name) {
        return static_cast<int>(requireLaunchInt(inputs, name, 1, INT_MAX_VALUE));
    };

    ConvProblem problem;
    problem.n = positive(N_VALUE);
    problem.c = positive(C_VALUE);
    problem.k = positive(K_VALUE);
    problem.groups = positive(GROUPS_VALUE);
    for(size_t axis = 0; axis < 3; ++axis)
    {
        problem.in[axis] = positive(IN_VALUES[axis]);
        problem.filter[axis] = positive(FILTER_VALUES[axis]);
        problem.out[axis] = positive(OUT_VALUES[axis]);
        problem.pad[axis]
            = static_cast<int>(requireLaunchInt(inputs, PAD_VALUES[axis], 0, INT_MAX_VALUE));
        problem.stride[axis] = positive(STRIDE_VALUES[axis]);
        problem.dilation[axis] = positive(DILATION_VALUES[axis]);
    }
    problem.spatialRank = problem.in[0] == 1 && problem.filter[0] == 1 ? 2 : 3;

    if(problem.c % problem.groups != 0 || problem.k % problem.groups != 0
       || gridXBlocks(problem) > MAX_GRID_X_BLOCKS || gridY(problem) > MAX_GRID_Y)
    {
        serialization::refuseIngestorPlan(
            serialization::IngestorPlanRefusal::INCOMPATIBLE,
            "launch values of dispatch '" + inputs.dispatchSymbol
                + "' give groups that do not divide 'c' and 'k', or a grid beyond the launch "
                  "limits");
    }
    return problem;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_rdna::fma_igemm

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
