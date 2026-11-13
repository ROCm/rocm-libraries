// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMatmul.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "engines/rocblas/MatmulTestGraphs.hpp"
#include "engines/rocblas/RocblasMatmulPlanBuilder.hpp"

/**
 * @file TestGpuRocblasMatmulPlan.cpp
 * @brief The rocBLAS matmul plan on the device against hipDNN's CPU reference: every storage
 *        order, batching, broadcast, and fp32/fp16/bf16 inputs, with fp32 outputs from fp16 and
 *        bf16 inputs.
 */
namespace
{

using namespace hipblaslt_plugin;
using namespace hipblaslt_plugin::test;
using hipdnn_data_sdk::utilities::Tensor;

using Half = hipdnn_data_sdk::types::half;
using BFloat16 = hipdnn_data_sdk::types::bfloat16;

struct MatmulCase
{
    std::string name;
    int64_t m;
    int64_t n;
    int64_t k;
    std::vector<int64_t> batch;
    bool aColumnMajor;
    bool bColumnMajor;
    bool cColumnMajor;
    bool broadcastA = false;
    bool broadcastB = false;
};

std::vector<MatmulCase> matmulCases()
{
    std::vector<MatmulCase> cases;
    for(int order = 0; order < 8; ++order)
    {
        const bool a = (order & 1) != 0;
        const bool b = (order & 2) != 0;
        const bool c = (order & 4) != 0;
        cases.push_back({std::string("Order") + (a ? "A" : "a") + (b ? "B" : "b") + (c ? "C" : "c"),
                         33,
                         17,
                         29,
                         {},
                         a,
                         b,
                         c});
    }
    cases.push_back({"Batched", 16, 24, 40, {2, 3}, false, true, false});
    cases.push_back({"BroadcastA", 12, 20, 9, {4}, true, false, false, true, false});
    cases.push_back({"BroadcastB", 12, 20, 9, {4}, false, false, true, false, true});
    return cases;
}

template <typename T>
hipdnn_flatbuffers_sdk::data_objects::DataType dataTypeOf()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    }
    else
    {
        return hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    }
}

template <typename T>
float storageEpsilon()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1.2e-7f;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return 9.8e-4f;
    }
    else
    {
        return 7.9e-3f;
    }
}

/// Every multi-index of @p dims, last axis fastest.
std::vector<std::vector<int64_t>> allIndices(const std::vector<int64_t>& dims)
{
    std::vector<std::vector<int64_t>> indices;
    std::vector<int64_t> index(dims.size(), 0);
    while(true)
    {
        indices.push_back(index);
        size_t axis = dims.size();
        while(axis > 0)
        {
            --axis;
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
            if(axis == 0)
            {
                return indices;
            }
        }
    }
}

template <typename TIn, typename TOut>
void runCase(const MatmulCase& matmul)
{
    auto spec = matmulSpec(matmul.m,
                           matmul.n,
                           matmul.k,
                           matmul.batch,
                           matmul.aColumnMajor,
                           matmul.bColumnMajor,
                           matmul.cColumnMajor,
                           matmul.broadcastA,
                           matmul.broadcastB);
    spec.a.dataType = spec.b.dataType = dataTypeOf<TIn>();
    spec.c.dataType = dataTypeOf<TOut>();
    const MatmulGraph graph(spec);

    const RocblasMatmulPlanBuilder builder;
    HipdnnEnginePluginHandle handle;
    ASSERT_TRUE(builder.isApplicable(handle, graph.graph()));
    HipdnnEnginePluginExecutionContext context;
    builder.buildPlan(handle, graph.graph(), context);

    Tensor<TIn> a(spec.a.dims, spec.a.strides);
    Tensor<TIn> b(spec.b.dims, spec.b.strides);
    Tensor<TOut> c(spec.c.dims, spec.c.strides);
    Tensor<TOut> reference(spec.c.dims, spec.c.strides);
    a.fillTensorWithRandomValues(-1.0f, 1.0f, 11);
    b.fillTensorWithRandomValues(-1.0f, 1.0f, 12);
    c.fillWithSentinelValue();

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    handle.setStream(stream);
    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
        hipdnnPluginDeviceBuffer_t{A_UID, a.rawDeviceData()},
        hipdnnPluginDeviceBuffer_t{B_UID, b.rawDeviceData()},
        hipdnnPluginDeviceBuffer_t{C_UID, c.rawDeviceData()}};
    context.plan().execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
    c.markDeviceModified();

    // Bring every host copy current: the reads below go through const accessors, which refuse
    // to migrate memory the device last touched.
    static_cast<void>(a.rawHostData());
    static_cast<void>(b.rawHostData());
    static_cast<void>(c.rawHostData());

    hipdnn_test_sdk::utilities::CpuFpReferenceMatmul::matmul(a, b, reference);

    // Each output is a k-term sum of products bounded by 1, rounded once into TOut.
    const float bound = 1e-6f * static_cast<float>(matmul.k);
    int64_t mismatches = 0;
    for(const auto& index : allIndices(spec.c.dims))
    {
        const float got = static_cast<float>(c.getHostValue(index));
        const float want = static_cast<float>(reference.getHostValue(index));
        const float tolerance = bound + 2.0f * storageEpsilon<TOut>() * (std::fabs(want) + 1e-3f);
        if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
        {
            ADD_FAILURE() << "got " << got << ", want " << want << ", tolerance " << tolerance;
        }
    }
    EXPECT_EQ(mismatches, 0);
}

class TestGpuRocblasMatmulPlan : public ::testing::TestWithParam<MatmulCase>
{
};

/// A bf16 matmul whose reduction walks a 67 M-element leading dimension (a stored column-major,
/// 8193 * 8193 rows, k = 16): rocBLAS would cut k into chunks of 15 and 1 and round the first to
/// bf16 before adding the second, so the engine declines it (hipBLASLt serves it). An fp32 c, or
/// a stored row-major (k unit stride), is kept. Host-only: nothing runs.
TEST(TestRocblasMatmulLimits, DeclinesAReductionRocblasWouldRoundBetweenChunks)
{
    const int64_t rows = int64_t{8193} * 8193;
    const RocblasMatmulPlanBuilder builder;
    HipdnnEnginePluginHandle handle;
    for(const bool aColumnMajor : {true, false})
    {
        for(const auto outType : {hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT})
        {
            auto spec = matmulSpec(rows, 16, 16, {}, aColumnMajor, false, true);
            spec.a.dataType = spec.b.dataType
                = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
            spec.c.dataType = outType;
            const MatmulGraph graph(spec);
            const bool declined
                = aColumnMajor && outType != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
            EXPECT_EQ(builder.isApplicable(handle, graph.graph()), !declined)
                << "a column-major " << aColumnMajor << ", fp32 c "
                << (outType == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
        }
    }
}

TEST_P(TestGpuRocblasMatmulPlan, Fp32)
{
    SKIP_IF_NO_DEVICES();
    runCase<float, float>(GetParam());
}

TEST_P(TestGpuRocblasMatmulPlan, Fp16)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half, Half>(GetParam());
}

TEST_P(TestGpuRocblasMatmulPlan, Bfp16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16, BFloat16>(GetParam());
}

TEST_P(TestGpuRocblasMatmulPlan, Fp16InFp32Out)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half, float>(GetParam());
}

TEST_P(TestGpuRocblasMatmulPlan, Bfp16InFp32Out)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuRocblasMatmulPlan,
                         ::testing::ValuesIn(matmulCases()),
                         [](const auto& info) { return info.param.name; });

} // namespace
