// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file ConvTf32Timing.hpp
 * @brief Opt-in timing of a conv pack's configs by math form (exact fp32 against TF32-class), on
 *        shapes listed in a CSV (name,n,c,k,groups,layout,input,filter,pad,stride,dilation;
 *        extents as AxB[xC]). Not a correctness test: inputs are a constant, outputs unchecked.
 *        Prints one "TF32BENCH|shape|form|best_ms|config|configs" line per shape and form.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_tf32_timing
{

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

/// The shapes of @p path, fp32.
inline std::vector<std::pair<std::string, conv_naive_test::ConvShape>>
    shapesOf(const std::string& path)
{
    std::vector<std::pair<std::string, conv_naive_test::ConvShape>> shapes;
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
        if(f.size() != 11)
        {
            continue;
        }
        conv_naive_test::ConvShape shape;
        shape.n = std::stoll(f[1]);
        shape.c = std::stoll(f[2]);
        shape.k = std::stoll(f[3]);
        shape.groups = std::stoll(f[4]);
        shape.channelsLast = f[5] == "nhwc";
        shape.input = extentsOf(f[6]);
        shape.filter = extentsOf(f[7]);
        shape.prePadding = extentsOf(f[8]);
        shape.postPadding = shape.prePadding;
        shape.stride = extentsOf(f[9]);
        shape.dilation = extentsOf(f[10]);
        shape.dataType = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
        shapes.emplace_back(f[0], shape);
    }
    return shapes;
}

inline int64_t bytesOf(const conv_naive_test::TensorSpec& tensor)
{
    int64_t elements = 1;
    for(const auto extent : tensor.dims)
    {
        elements *= extent;
    }
    return elements * 4;
}

/// Times every config @p kernels holds that the pack admits for each shape of the CSV named by
/// @p env (skips when unset), grouped by @p formOf(kernel); prints the best per form, each
/// shape's name followed by @p suffix (a direction, where one CSV is timed in several).
inline void timeByForm(
    const PackSymbols& pack,
    conv_naive::ConvDirection direction,
    const std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>& kernels,
    const char* env,
    const std::function<std::string(const hipdnn_plugin_sdk::ingestor::KernelDefinition&)>& formOf,
    const std::string& suffix = "")
{
    const char* path = std::getenv(env);
    if(path == nullptr || *path == '\0')
    {
        GTEST_SKIP() << "set " << env << " to a shape CSV to time";
    }
    const auto& handler = dispatchHandler(pack);
    for(const auto& [shapeName, shape] : shapesOf(path))
    {
        const std::string name = shapeName + suffix;
        const auto spec = conv_naive_test::specFor(shape, direction);
        const GraphFixture fixture(conv_naive_test::buildGraph(spec), currentDeviceProperties());
        const auto bound = matchesGraph(pack, fixture.context());
        if(!bound.has_value())
        {
            std::cout << "TF32BENCH|" << name << "|refused|-|-|0" << std::endl;
            continue;
        }
        void* x = nullptr;
        void* w = nullptr;
        void* y = nullptr;
        ASSERT_EQ(hipMalloc(&x, static_cast<size_t>(bytesOf(spec.x))), hipSuccess);
        ASSERT_EQ(hipMalloc(&w, static_cast<size_t>(bytesOf(spec.w))), hipSuccess);
        ASSERT_EQ(hipMalloc(&y, static_cast<size_t>(bytesOf(spec.y))), hipSuccess);
        // 0x3c3c3c3c: about 0.0115 in every element, finite and nonzero.
        ASSERT_EQ(hipMemset(x, 0x3c, static_cast<size_t>(bytesOf(spec.x))), hipSuccess);
        ASSERT_EQ(hipMemset(w, 0x3c, static_cast<size_t>(bytesOf(spec.w))), hipSuccess);
        ASSERT_EQ(hipMemset(y, 0x3c, static_cast<size_t>(bytesOf(spec.y))), hipSuccess);
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{conv_naive_test::X_UID, x},
            hipdnnPluginDeviceBuffer_t{conv_naive_test::W_UID, w},
            hipdnnPluginDeviceBuffer_t{conv_naive_test::Y_UID, y}};
        std::map<std::string, std::pair<float, std::string>> best;
        std::map<std::string, int> counts;
        const Handle handle;
        const auto stream = handle.getStream();
        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        ASSERT_EQ(hipEventCreate(&start), hipSuccess);
        ASSERT_EQ(hipEventCreate(&stop), hipSuccess);
        for(const auto& kernel : kernels)
        {
            if(!matchesKernel(pack, fixture.context(), kernel, *bound))
            {
                continue;
            }
            const auto form = formOf(kernel);
            ++counts[form];
            const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
            const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
            void* workspace = nullptr;
            if(workspaceBytes > 0 && hipMalloc(&workspace, workspaceBytes) != hipSuccess)
            {
                std::cout << "TF32BENCH-SKIP|" << name << "|" << kernel.name << "|workspace "
                          << workspaceBytes << std::endl;
                continue;
            }
            auto timed = [&] {
                EXPECT_EQ(hipEventRecord(start, stream), hipSuccess);
                handler.launch(handle,
                               *prepared,
                               buffers.data(),
                               static_cast<uint32_t>(buffers.size()),
                               workspace);
                EXPECT_EQ(hipEventRecord(stop, stream), hipSuccess);
                EXPECT_EQ(hipEventSynchronize(stop), hipSuccess);
                float ms = 0.0f;
                EXPECT_EQ(hipEventElapsedTime(&ms, start, stop), hipSuccess);
                return ms;
            };
            float ms = timed(); // warm-up
            auto& slot = best[form];
            // Far slower than the best so far: one run says enough.
            if(slot.second.empty() || ms < 4.0f * slot.first)
            {
                for(int i = 0; i < 3; ++i)
                {
                    ms = std::min(ms, timed());
                }
            }
            if(slot.second.empty() || ms < slot.first)
            {
                slot = {ms, kernel.name};
            }
            if(workspace != nullptr)
            {
                EXPECT_EQ(hipFree(workspace), hipSuccess);
            }
        }
        for(const auto& [form, slot] : best)
        {
            std::cout << "TF32BENCH|" << name << "|" << form << "|" << slot.first << "|"
                      << slot.second << "|" << counts[form] << std::endl;
        }
        EXPECT_EQ(hipEventDestroy(start), hipSuccess);
        EXPECT_EQ(hipEventDestroy(stop), hipSuccess);
        EXPECT_EQ(hipFree(x), hipSuccess);
        EXPECT_EQ(hipFree(w), hipSuccess);
        EXPECT_EQ(hipFree(y), hipSuccess);
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_tf32_timing

#endif
