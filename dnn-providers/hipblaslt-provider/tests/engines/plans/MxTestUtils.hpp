// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

// Helpers shared by the MX plan, plan builder and scale-layout tests.
namespace hipblaslt_plugin_test
{

struct DeviceFree
{
    void operator()(void* ptr) const
    {
        static_cast<void>(hipFree(ptr));
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;

inline DevicePtr deviceAlloc(size_t bytes)
{
    void* ptr = nullptr;
    EXPECT_EQ(hipMalloc(&ptr, bytes == 0 ? 1 : bytes), hipSuccess);
    return DevicePtr(ptr);
}

inline DevicePtr upload(const std::vector<uint8_t>& bytes)
{
    auto ptr = deviceAlloc(bytes.size());
    EXPECT_EQ(hipMemcpy(ptr.get(), bytes.data(), bytes.size(), hipMemcpyHostToDevice), hipSuccess);
    return ptr;
}

// Row-major [rows][cols] bytes rewritten column-major.
inline std::vector<uint8_t> transposeHost(const std::vector<uint8_t>& in, size_t rows, size_t cols)
{
    std::vector<uint8_t> out(in.size());
    for(size_t r = 0; r < rows; ++r)
    {
        for(size_t c = 0; c < cols; ++c)
        {
            out[c * rows + r] = in[r * cols + c];
        }
    }
    return out;
}

// Physical order of our A and B. hipBLAS sees the transposed pair (row-major
// swap): MN_MN -> NT (the original provider layout), K_K -> TN (the only layout
// with Tensile MX kernels), K_MN -> NN, MN_K -> TT.
enum class MxOperandLayout
{
    MN_MN,
    K_K,
    K_MN,
    MN_K,
};

inline const char* toString(MxOperandLayout layout)
{
    switch(layout)
    {
    case MxOperandLayout::MN_MN:
        return "MN_MN";
    case MxOperandLayout::K_K:
        return "K_K";
    case MxOperandLayout::K_MN:
        return "K_MN";
    case MxOperandLayout::MN_K:
        return "MN_K";
    default:
        return "?";
    }
}

// M = N = 32, K = 128 MX graph with the given strides for A [32, 128], B [128, 32]
// and their scales [32, 4] and [4, 32]; C is a packed row-major [32, 32].
inline flatbuffers::FlatBufferBuilder
    createMxGraphWithStrides(const std::vector<int64_t>& aStrides,
                             const std::vector<int64_t>& bStrides,
                             const std::vector<int64_t>& scaleAStrides,
                             const std::vector<int64_t>& scaleBStrides,
                             hipdnn_flatbuffers_sdk::data_objects::DataType cType
                             = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                             hipdnn_flatbuffers_sdk::data_objects::DataType xTypeA
                             = hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                             hipdnn_flatbuffers_sdk::data_objects::DataType xTypeB
                             = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
{
    using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;
    return hipdnn_test_sdk::utilities::createValidMxMatmulGraph({32, 128},
                                                                aStrides,
                                                                {128, 32},
                                                                bStrides,
                                                                {32, 32},
                                                                {32, 1},
                                                                {32, 4},
                                                                {4, 32},
                                                                xTypeA,
                                                                cType,
                                                                DT::FP8_E8M0,
                                                                DT::FP8_E8M0,
                                                                DT::FLOAT,
                                                                32,
                                                                false,
                                                                false,
                                                                xTypeB,
                                                                scaleAStrides,
                                                                scaleBStrides);
}

// M = N = 32, K = 128 MX graph in the given operand layout. Scales are
// K-block-contiguous ([M, K/32] row-major, [K/32, N] column-major) unless
// mnContiguousScales is set.
inline flatbuffers::FlatBufferBuilder
    createMxGraphWithLayout(MxOperandLayout layout,
                            hipdnn_flatbuffers_sdk::data_objects::DataType xTypeA
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3,
                            hipdnn_flatbuffers_sdk::data_objects::DataType cType
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                            hipdnn_flatbuffers_sdk::data_objects::DataType xTypeB
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
                            bool mnContiguousScales = false)
{
    const bool aKContiguous = layout == MxOperandLayout::K_K || layout == MxOperandLayout::K_MN;
    const bool bKContiguous = layout == MxOperandLayout::K_K || layout == MxOperandLayout::MN_K;
    const std::vector<int64_t> aStrides
        = aKContiguous ? std::vector<int64_t>{128, 1} : std::vector<int64_t>{1, 32};
    const std::vector<int64_t> bStrides
        = bKContiguous ? std::vector<int64_t>{1, 128} : std::vector<int64_t>{32, 1};
    const std::vector<int64_t> scaleAStrides
        = mnContiguousScales ? std::vector<int64_t>{1, 32} : std::vector<int64_t>{4, 1};
    const std::vector<int64_t> scaleBStrides
        = mnContiguousScales ? std::vector<int64_t>{32, 1} : std::vector<int64_t>{1, 4};
    return createMxGraphWithStrides(
        aStrides, bStrides, scaleAStrides, scaleBStrides, cType, xTypeA, xTypeB);
}

} // namespace hipblaslt_plugin_test
