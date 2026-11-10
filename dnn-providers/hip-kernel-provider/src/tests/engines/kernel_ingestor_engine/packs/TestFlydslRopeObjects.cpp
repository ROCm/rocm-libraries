// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRope.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRopeTestGraphs.hpp"

/**
 * @file TestFlydslRopeObjects.cpp
 * @brief Every FlyDSL RoPE object this device's shard carries, launched through the pack's
 *        own matchers and dispatch handler against hipDNN's CPU reference
 *        (CpuFpReferenceRope) over the shapes the op exists for: Qwen3's partial rotary
 *        (D = 256, the leading 64 rotated, BSHD strides, per-sequence and per-batch freqs,
 *        angles up to thousands of radians), cuDNN's trailing default offset, full rotation
 *        in place, a middle segment, a head dim wider than the workgroup, and a strided
 *        head dim. Outputs start as NaN, so an element nothing writes fails.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using hipdnn_data_sdk::utilities::Tensor;
using Bf16 = hipdnn_data_sdk::types::bfloat16;
using F16 = hipdnn_data_sdk::types::half;

/// One launch: a graph spec (dtype filled in per object) and how its data is drawn.
struct RopeCase
{
    std::string name;
    RopeGraphSpec spec;
    /// Angles are drawn from [0, maxAngle].
    float maxAngle = 3.0F;
    /// y is x's own buffer.
    bool inPlace = false;
};

std::vector<RopeCase> ropeCases()
{
    std::vector<RopeCase> cases;
    {
        // Qwen3 prefill: BSHD, D = 256, the leading 64 rotated, positions into the
        // thousands so sin/cos take a real range reduction.
        RopeCase c{"QwenPrefill", {}, 4000.0F};
        c.spec.xDims = {2, 16, 37, 256};
        c.spec.freqsDims = {1, 1, 37, 64};
        cases.push_back(c);
    }
    {
        // Qwen3 per-batch freqs (M-RoPE builds them per token), the KV-head count.
        RopeCase c{"QwenPerBatchFreqs", {}, 300.0F};
        c.spec.xDims = {2, 4, 5, 256};
        c.spec.freqsDims = {2, 1, 5, 64};
        cases.push_back(c);
    }
    {
        // cuDNN's default: rope_offset absent, so the trailing r dims rotate. BHSD, scaled.
        RopeCase c{"CudnnTrailingDefault", {}};
        c.spec.xDims = {1, 4, 9, 128};
        c.spec.xStrides = flydslRopePackedStrides(c.spec.xDims);
        c.spec.freqsDims = {1, 1, 9, 64};
        c.spec.ropeOffset = std::nullopt;
        c.spec.outputScale = 0.5F;
        cases.push_back(c);
    }
    {
        RopeCase c{"FullRotationInPlace", {}, 3.0F, true};
        c.spec.xDims = {3, 2, 5, 64};
        c.spec.xStrides = flydslRopePackedStrides(c.spec.xDims);
        c.spec.freqsDims = {3, 1, 5, 64};
        c.spec.ropeDim = 0;
        c.spec.outputScale = 1.25F;
        cases.push_back(c);
    }
    {
        // Rank 3, a segment in the middle of an odd-width head dim.
        RopeCase c{"MiddleSegmentRank3", {}};
        c.spec.xDims = {3, 7, 70};
        c.spec.freqsDims = {1, 7, 10};
        c.spec.ropeDim = 10;
        c.spec.ropeOffset = 30;
        cases.push_back(c);
    }
    {
        // Wider than the workgroup, with passthrough on both sides of the segment.
        RopeCase c{"WideHeadDim", {}};
        c.spec.xDims = {1, 1, 3, 1000};
        c.spec.freqsDims = {1, 1, 3, 512};
        c.spec.ropeDim = 512;
        c.spec.ropeOffset = 100;
        c.spec.outputScale = 2.0F;
        cases.push_back(c);
    }
    {
        // The head dim outermost in memory: every element a strided access.
        RopeCase c{"StridedHeadDim", {}};
        c.spec.xDims = {1, 3, 7, 70};
        c.spec.xStrides = std::vector<int64_t>{21, 7, 1, 21};
        c.spec.freqsDims = {1, 3, 7, 10};
        c.spec.ropeDim = 10;
        c.spec.ropeOffset = 3;
        cases.push_back(c);
    }
    return cases;
}

std::string stringField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<std::string>(kernel.metadata, field);
    return value == nullptr ? std::string() : *value;
}

double toleranceFor(const std::string& dtype)
{
    if(dtype == "bf16")
    {
        return 1e-2;
    }
    return dtype == "f16" ? 2e-3 : 1e-5;
}

/// Every logical index of @p dims, row-major.
std::vector<std::vector<int64_t>> logicalIndices(const std::vector<int64_t>& dims)
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

template <typename T>
void expectObjectMatches(const KernelDefinition& kernel,
                         DataType dataType,
                         const RopeCase& ropeCase)
{
    SCOPED_TRACE(kernel.name + " on " + ropeCase.name);
    const auto dtype = stringField(kernel, "dtype");
    auto spec = ropeCase.spec;
    spec.dataType = dataType;

    const GraphFixture fixture(buildFlydslRopeGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_ROPE, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    ASSERT_TRUE(matchesKernel(FLYDSL_ROPE, fixture.context(), kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_ROPE);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    const auto xStrides = flydslRopeXStrides(spec);
    const auto freqsStrides = flydslRopePackedStrides(spec.freqsDims);
    Tensor<T> x(spec.xDims, xStrides);
    Tensor<float> freqs(spec.freqsDims, freqsStrides);
    Tensor<T> y(spec.xDims, xStrides);
    Tensor<T> expected(spec.xDims, xStrides);
    x.fillWithRandomValues(T(-1.0F), T(1.0F), 1234U);
    freqs.fillWithRandomValues(0.0F, ropeCase.maxAngle, 4321U);
    y.fillWithSentinelValue();

    const auto ropeDim = spec.ropeDim;
    hipdnn_test_sdk::utilities::CpuFpReferenceRope::forward<T>(
        x, freqs, expected, spec.outputScale, ropeDim, spec.ropeOffset);

    auto& output = ropeCase.inPlace ? x : y;
    const std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_ROPE_X_UID, x.memory().deviceData()},
        {FLYDSL_ROPE_FREQS_UID, freqs.memory().deviceData()},
        {FLYDSL_ROPE_Y_UID, output.memory().deviceData()}};
    const Handle handle;
    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    output.memory().markDeviceModified();
    // getHostValue() is const and will not migrate; this non-const access copies y back.
    static_cast<void>(output.memory().hostData());

    const double tolerance = toleranceFor(dtype);
    size_t mismatches = 0;
    for(const auto& index : logicalIndices(spec.xDims))
    {
        const auto got = static_cast<double>(static_cast<float>(output.getHostValue(index)));
        const auto want = static_cast<double>(static_cast<float>(expected.getHostValue(index)));
        if(!std::isfinite(got) || std::abs(got - want) > tolerance * (1.0 + std::abs(want)))
        {
            if(++mismatches <= 5)
            {
                std::string where;
                for(const auto i : index)
                {
                    where += std::to_string(i) + ",";
                }
                ADD_FAILURE() << "y[" << where << "] = " << got << ", expected " << want;
            }
        }
    }
    EXPECT_EQ(mismatches, 0U);
}

TEST(TestGpuFlydslRopeObjects, EveryObjectMatchesTheCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto kernels = stagedKernels(FLYDSL_ROPE.engineName);
    if(kernels.empty())
    {
        reportUnstaged("rope", "this build staged no RoPE objects for this device");
        return;
    }

    for(const auto& kernel : kernels)
    {
        const auto dtype = stringField(kernel, "dtype");
        for(const auto& ropeCase : ropeCases())
        {
            if(dtype == "bf16")
            {
                expectObjectMatches<Bf16>(kernel, DataType::BFLOAT16, ropeCase);
            }
            else if(dtype == "f16")
            {
                expectObjectMatches<F16>(kernel, DataType::HALF, ropeCase);
            }
            else if(dtype == "f32")
            {
                expectObjectMatches<float>(kernel, DataType::FLOAT, ropeCase);
            }
            else
            {
                ADD_FAILURE() << kernel.name << " declares dtype '" << dtype << "'";
            }
        }
    }
    RecordProperty("objects", static_cast<int>(kernels.size()));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
