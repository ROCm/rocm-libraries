// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogueTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file ConvEpilogueTiming.hpp
 * @brief Opt-in timing of a conv pack's configs with and without a fused bias + relu epilogue,
 *        config for config, on shapes listed in a CSV
 *        (name,dtype,n,c,k,groups,layout,input,filter,pad,stride,dilation; extents AxB[xC];
 *        dtype float|half|bfloat16). Not a correctness test: inputs are a constant, outputs
 *        unchecked. Every admitted config is timed bare; the fastest few (CONV_EPI_TIMING_TOP,
 *        default 4) are then timed bare and fused, interleaved, and one
 *        "FUSBENCH|shape|config|bare_ms|fused_ms|fused/bare" line is printed for each. The chain
 *        is bias then CONV_EPI_TIMING_ACT: relu (default), silu, gelu, gelu_tanh, sigmoid, tanh,
 *        elu or softplus.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_epilogue_timing
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_epilogue_test::FusedGraphSpec;
using conv_naive_test::ConvShape;

inline std::vector<int64_t> extentsOf(const std::string& text)
{
    std::vector<int64_t> extents;
    std::stringstream stream(text);
    std::string part;
    while(std::getline(stream, part, 'x'))
    {
        extents.push_back(std::stoll(part));
    }
    return extents;
}

inline std::vector<std::pair<std::string, ConvShape>> shapesOf(const std::string& path)
{
    std::vector<std::pair<std::string, ConvShape>> shapes;
    std::ifstream file(path);
    std::string line;
    std::getline(file, line); // header
    while(std::getline(file, line))
    {
        std::vector<std::string> f;
        std::stringstream stream(line);
        std::string cell;
        while(std::getline(stream, cell, ','))
        {
            f.push_back(cell);
        }
        if(f.size() != 12)
        {
            continue;
        }
        ConvShape shape;
        shape.dataType = data_objects::DataType::FLOAT;
        if(f[1] == "half")
        {
            shape.dataType = data_objects::DataType::HALF;
        }
        else if(f[1] == "bfloat16")
        {
            shape.dataType = data_objects::DataType::BFLOAT16;
        }
        shape.n = std::stoll(f[2]);
        shape.c = std::stoll(f[3]);
        shape.k = std::stoll(f[4]);
        shape.groups = std::stoll(f[5]);
        shape.channelsLast = f[6] == "nhwc";
        shape.input = extentsOf(f[7]);
        shape.filter = extentsOf(f[8]);
        shape.prePadding = extentsOf(f[9]);
        shape.postPadding = shape.prePadding;
        shape.stride = extentsOf(f[10]);
        shape.dilation = extentsOf(f[11]);
        shapes.emplace_back(f[0], shape);
    }
    return shapes;
}

/// The activation step CONV_EPI_TIMING_ACT names (relu when unset).
inline conv_epilogue_test::Step activationStep()
{
    const char* env = std::getenv("CONV_EPI_TIMING_ACT");
    const std::string name = env != nullptr ? env : "relu";
    using data_objects::PointwiseMode;
    if(name == "silu")
    {
        return conv_epilogue_test::unary(PointwiseMode::SWISH_FWD);
    }
    if(name == "gelu")
    {
        return conv_epilogue_test::unary(PointwiseMode::GELU_FWD);
    }
    if(name == "gelu_tanh")
    {
        return conv_epilogue_test::unary(PointwiseMode::GELU_APPROX_TANH_FWD);
    }
    if(name == "sigmoid")
    {
        return conv_epilogue_test::unary(PointwiseMode::SIGMOID_FWD);
    }
    if(name == "tanh")
    {
        return conv_epilogue_test::unary(PointwiseMode::TANH_FWD);
    }
    if(name == "elu")
    {
        return conv_epilogue_test::unary(PointwiseMode::ELU_FWD);
    }
    if(name == "softplus")
    {
        return conv_epilogue_test::unary(PointwiseMode::SOFTPLUS_FWD);
    }
    return conv_epilogue_test::relu();
}

inline size_t bytesOf(const conv_naive_test::TensorSpec& tensor)
{
    int64_t elements = 1;
    for(const auto extent : tensor.dims)
    {
        elements *= extent;
    }
    return static_cast<size_t>(elements)
           * (tensor.dataType == data_objects::DataType::FLOAT ? 4 : 2);
}

/// One graph's prepared launch of one config.
struct Prepared
{
    std::unique_ptr<hipdnn_plugin_sdk::ingestor::PreparedDispatch> dispatch;
    void* workspace = nullptr;
};

inline void
    timeFusedAgainstBare(const PackSymbols& pack,
                         const std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>& kernels,
                         const char* env)
{
    const char* path = std::getenv(env);
    if(path == nullptr || *path == '\0')
    {
        GTEST_SKIP() << "set " << env << " to a shape CSV to time";
    }
    const char* topEnv = std::getenv("CONV_EPI_TIMING_TOP");
    const size_t top = topEnv != nullptr ? static_cast<size_t>(std::atoi(topEnv)) : 4;
    const char* onlyEnv = std::getenv("CONV_EPI_TIMING_ONLY");
    const std::string only = onlyEnv != nullptr ? onlyEnv : "";
    const auto& handler = dispatchHandler(pack);
    const Handle handle;
    const auto stream = handle.getStream();
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    ASSERT_EQ(hipEventCreate(&start), hipSuccess);
    ASSERT_EQ(hipEventCreate(&stop), hipSuccess);

    for(const auto& [name, shape] : shapesOf(path))
    {
        const auto bareSpec = conv_naive_test::specFor(shape, conv_naive::ConvDirection::FORWARD);
        const FusedGraphSpec fusedSpec = conv_epilogue_test::fusedSpecFor(
            shape, {conv_epilogue_test::bias(shape.dataType), activationStep()});
        const GraphFixture bare(conv_naive_test::buildGraph(bareSpec), currentDeviceProperties());
        const GraphFixture fused(conv_epilogue_test::buildFusedGraph(fusedSpec),
                                 currentDeviceProperties());
        const auto bareBound = matchesGraph(pack, bare.context());
        const auto fusedBound = matchesGraph(pack, fused.context());
        if(!bareBound.has_value() || !fusedBound.has_value())
        {
            std::cout << "FUSBENCH|" << name << "|refused|" << bareBound.has_value() << "|"
                      << fusedBound.has_value() << '\n';
            continue;
        }
        void* x = nullptr;
        void* w = nullptr;
        void* y = nullptr;
        void* b = nullptr;
        ASSERT_EQ(hipMalloc(&x, bytesOf(bareSpec.x)), hipSuccess);
        ASSERT_EQ(hipMalloc(&w, bytesOf(bareSpec.w)), hipSuccess);
        ASSERT_EQ(hipMalloc(&y, bytesOf(bareSpec.y)), hipSuccess);
        ASSERT_EQ(hipMalloc(&b, static_cast<size_t>(shape.k) * 4), hipSuccess);
        ASSERT_EQ(hipMemset(x, 0x3c, bytesOf(bareSpec.x)), hipSuccess);
        ASSERT_EQ(hipMemset(w, 0x3c, bytesOf(bareSpec.w)), hipSuccess);
        ASSERT_EQ(hipMemset(b, 0x3c, static_cast<size_t>(shape.k) * 4), hipSuccess);
        const std::array<hipdnnPluginDeviceBuffer_t, 3> bareBuffers{
            hipdnnPluginDeviceBuffer_t{conv_naive_test::X_UID, x},
            hipdnnPluginDeviceBuffer_t{conv_naive_test::W_UID, w},
            hipdnnPluginDeviceBuffer_t{conv_naive_test::Y_UID, y}};
        const std::array<hipdnnPluginDeviceBuffer_t, 4> fusedBuffers{
            hipdnnPluginDeviceBuffer_t{conv_naive_test::X_UID, x},
            hipdnnPluginDeviceBuffer_t{conv_naive_test::W_UID, w},
            hipdnnPluginDeviceBuffer_t{conv_epilogue_test::outputUid(2), y},
            hipdnnPluginDeviceBuffer_t{conv_epilogue_test::OPERAND_UID, b}};

        const auto prepare = [&](const GraphFixture& fixture,
                                 const auto& bound,
                                 const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel) {
            Prepared p;
            p.dispatch = handler.prepare(fixture.context(), bound, kernel);
            const size_t bytes = handler.workspaceBytes(fixture.context(), bound, kernel);
            if(bytes > 0 && hipMalloc(&p.workspace, bytes) != hipSuccess)
            {
                p.dispatch.reset();
            }
            return p;
        };
        const auto timeOnce
            = [&](const Prepared& p, const hipdnnPluginDeviceBuffer_t* buffers, uint32_t count) {
                  EXPECT_EQ(hipEventRecord(start, stream), hipSuccess);
                  handler.launch(handle, *p.dispatch, buffers, count, p.workspace);
                  EXPECT_EQ(hipEventRecord(stop, stream), hipSuccess);
                  EXPECT_EQ(hipEventSynchronize(stop), hipSuccess);
                  float ms = 0.0f;
                  EXPECT_EQ(hipEventElapsedTime(&ms, start, stop), hipSuccess);
                  return ms;
              };
        const auto release = [](Prepared& p) {
            if(p.workspace != nullptr)
            {
                EXPECT_EQ(hipFree(p.workspace), hipSuccess);
                p.workspace = nullptr;
            }
        };

        // Phase 1: every admitted config bare, the minimum of 3.
        std::vector<std::pair<float, size_t>> bareTimes;
        for(size_t i = 0; i < kernels.size(); ++i)
        {
            const auto& kernel = kernels[i];
            if(!only.empty() && kernel.name.find(only) == std::string::npos)
            {
                continue;
            }
            if(!matchesKernel(pack, bare.context(), kernel, *bareBound))
            {
                continue;
            }
            auto p = prepare(bare, *bareBound, kernel);
            if(!p.dispatch)
            {
                continue;
            }
            float ms = timeOnce(p, bareBuffers.data(), 3);
            for(int r = 0; r < 3; ++r)
            {
                ms = std::min(ms, timeOnce(p, bareBuffers.data(), 3));
            }
            bareTimes.emplace_back(ms, i);
            release(p);
        }
        std::sort(bareTimes.begin(), bareTimes.end());

        // Phase 2: the fastest bare configs, bare and fused interleaved, medians of 21.
        for(size_t t = 0; t < std::min(top, bareTimes.size()); ++t)
        {
            const auto& kernel = kernels[bareTimes[t].second];
            if(!matchesKernel(pack, fused.context(), kernel, *fusedBound))
            {
                std::cout << "FUSBENCH|" << name << "|" << kernel.name << "|fused-declined\n";
                continue;
            }
            auto pb = prepare(bare, *bareBound, kernel);
            auto pf = prepare(fused, *fusedBound, kernel);
            if(!pb.dispatch || !pf.dispatch)
            {
                continue;
            }
            std::vector<float> tb;
            std::vector<float> tf;
            for(int r = 0; r < 3; ++r)
            {
                timeOnce(pb, bareBuffers.data(), 3);
                timeOnce(pf, fusedBuffers.data(), 4);
            }
            for(int r = 0; r < 21; ++r)
            {
                tb.push_back(timeOnce(pb, bareBuffers.data(), 3));
                tf.push_back(timeOnce(pf, fusedBuffers.data(), 4));
            }
            std::sort(tb.begin(), tb.end());
            std::sort(tf.begin(), tf.end());
            std::cout << "FUSBENCH|" << name << "|" << kernel.name << "|" << tb[10] << "|" << tf[10]
                      << "|" << tf[10] / tb[10] << '\n';
            release(pb);
            release(pf);
        }
        EXPECT_EQ(hipFree(x), hipSuccess);
        EXPECT_EQ(hipFree(w), hipSuccess);
        EXPECT_EQ(hipFree(y), hipSuccess);
        EXPECT_EQ(hipFree(b), hipSuccess);
    }
    EXPECT_EQ(hipEventDestroy(start), hipSuccess);
    EXPECT_EQ(hipEventDestroy(stop), hipSuccess);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_epilogue_timing

#endif
