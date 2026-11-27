// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "utils/GpuTestUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace test_gpu_utils;

namespace
{

constexpr float FP32_TOLERANCE = 2e-4f;
constexpr float FP16_TOLERANCE = 3e-2f;
constexpr float GUARD_VALUE = 12288.0f;
constexpr size_t GUARD_ELEMENTS = 16;

size_t elementSize(DataType type)
{
    return type == DataType::HALF ? sizeof(hipdnn_data_sdk::types::half) : sizeof(float);
}

// Host values rounded to `type`, as raw bytes.
std::vector<char> encode(const std::vector<float>& values, DataType type)
{
    std::vector<char> bytes(values.size() * elementSize(type));
    for(size_t i = 0; i < values.size(); ++i)
    {
        if(type == DataType::HALF)
        {
            const hipdnn_data_sdk::types::half h(values[i]);
            std::memcpy(bytes.data() + i * sizeof(h), &h, sizeof(h));
        }
        else
        {
            std::memcpy(bytes.data() + i * sizeof(float), &values[i], sizeof(float));
        }
    }
    return bytes;
}

std::vector<float> decode(const std::vector<char>& bytes, DataType type)
{
    std::vector<float> values(bytes.size() / elementSize(type));
    for(size_t i = 0; i < values.size(); ++i)
    {
        if(type == DataType::HALF)
        {
            hipdnn_data_sdk::types::half h;
            std::memcpy(&h, bytes.data() + i * sizeof(h), sizeof(h));
            values[i] = static_cast<float>(h);
        }
        else
        {
            std::memcpy(&values[i], bytes.data() + i * sizeof(float), sizeof(float));
        }
    }
    return values;
}

// Round-trips through `type` so the reference sees the values the device sees.
std::vector<float> rounded(const std::vector<float>& values, DataType type)
{
    return decode(encode(values, type), type);
}

// `matrixDims` with the batch dimensions in front.
std::vector<int64_t> batched(const std::vector<int64_t>& batchDims, std::vector<int64_t> matrixDims)
{
    matrixDims.insert(matrixDims.begin(), batchDims.begin(), batchDims.end());
    return matrixDims;
}

// Row-major packed strides of `dims`.
std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size() - 1; i-- > 0;)
    {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

// The explicit strides, or the packed ones when none are given.
std::vector<int64_t> stridesOrPacked(const std::vector<int64_t>& strides,
                                     const std::vector<int64_t>& dims)
{
    return strides.empty() ? packedStrides(dims) : strides;
}

// Element offset of every element of a tensor with these dims and strides, in logical
// row-major order.
std::vector<size_t> elementOffsets(const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& strides)
{
    size_t count = 1;
    for(const auto dim : dims)
    {
        count *= static_cast<size_t>(dim);
    }
    std::vector<size_t> offsets(count);
    for(size_t linear = 0; linear < count; ++linear)
    {
        size_t rest = linear;
        size_t offset = 0;
        for(size_t d = dims.size(); d-- > 0;)
        {
            const auto extent = static_cast<size_t>(dims[d]);
            offset += (rest % extent) * static_cast<size_t>(strides[d]);
            rest /= extent;
        }
        offsets[linear] = offset;
    }
    return offsets;
}

// Number of elements a tensor with these offsets spans.
size_t spanOf(const std::vector<size_t>& offsets)
{
    return *std::max_element(offsets.begin(), offsets.end()) + 1;
}

// Lays the logical values out at `offsets`; every element they do not reach holds GUARD_VALUE.
std::vector<float> scatter(const std::vector<float>& values, const std::vector<size_t>& offsets)
{
    std::vector<float> laidOut(spanOf(offsets), GUARD_VALUE);
    for(size_t i = 0; i < values.size(); ++i)
    {
        laidOut[offsets[i]] = values[i];
    }
    return laidOut;
}

// One matmul, optionally followed by a bias ADD and a RELU_FWD, described by the dtype of every
// tensor, the strides of A, B and the matmul output, and the ReLU clip bounds.
struct MatmulCase
{
    int64_t m = 8;
    int64_t k = 16;
    int64_t n = 12;
    // Leading batch dimensions shared by A, B and the outputs; empty is a plain 2-D matmul.
    // Every strides vector below then spans the batch dimensions too.
    std::vector<int64_t> batchDims;
    DataType ioType = DataType::FLOAT;
    DataType intermediateType = DataType::FLOAT;
    // Strides of the A and B operands; empty means row-major.
    std::vector<int64_t> aStrides;
    std::vector<int64_t> bStrides;
    // Strides of the matmul output; empty lets the frontend infer them.
    std::vector<int64_t> cStrides;
    // Strides of the final output when it is not the matmul output; empty means `cStrides`
    // when set and packed otherwise.
    std::vector<int64_t> finalStrides;
    // Data type of the matmul output when a bias or activation follows.
    DataType matmulOutType = DataType::FLOAT;
    bool withBias = false;
    // Data type of the bias output when an activation follows.
    DataType biasOutType = DataType::FLOAT;
    bool withRelu = false;
    // Clip bounds of the RELU_FWD; unset leaves the attribute unset.
    std::optional<float> reluLowerClip;
    std::optional<float> reluUpperClip;
    DataType finalType = DataType::FLOAT;
};

class IntegrationGpuMatmulApplicability : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        const auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        auto [handle, err] = createHipdnnHandle();
        ASSERT_TRUE(err.is_good()) << err.err_msg;
        _handle = std::move(handle);
    }

    // Builds the graph on the forced hipBLASLt engine. Returns the name of the first stage that
    // failed, with its message, or an empty string when the graph is ready to execute.
    std::string buildGraph(Graph& graph, const MatmulCase& c) const
    {
        graph.set_name("MatmulApplicability");
        graph.set_io_data_type(c.ioType)
            .set_intermediate_data_type(c.intermediateType)
            .set_compute_data_type(DataType::FLOAT);

        const auto aDims = batched(c.batchDims, {c.m, c.k});
        const auto bDims = batched(c.batchDims, {c.k, c.n});
        const auto outDims = batched(c.batchDims, {c.m, c.n});
        const auto a = std::make_shared<TensorAttributes>(
            makeTensorAttributes("a", c.ioType, aDims, stridesOrPacked(c.aStrides, aDims))
                .set_uid(UID_A));
        const auto b = std::make_shared<TensorAttributes>(
            makeTensorAttributes("b", c.ioType, bDims, stridesOrPacked(c.bStrides, bDims))
                .set_uid(UID_B));

        auto out = graph.matmul(a, b, MatmulAttributes());
        const bool fused = c.withBias || c.withRelu;
        out->set_data_type(fused ? c.matmulOutType : c.finalType);
        if(!c.cStrides.empty())
        {
            out->set_dim(outDims).set_stride(c.cStrides);
        }

        if(c.withBias)
        {
            std::vector<int64_t> biasDims(outDims.size(), 1);
            biasDims.back() = c.n;
            const auto bias = std::make_shared<TensorAttributes>(
                makeTensorAttributes("bias", DataType::FLOAT, biasDims, packedStrides(biasDims))
                    .set_uid(UID_BIAS));
            PointwiseAttributes biasAttrs;
            biasAttrs.set_mode(PointwiseMode::ADD);
            out = graph.pointwise(out, bias, biasAttrs);
            out->set_data_type(c.withRelu ? c.biasOutType : c.finalType);
        }
        if(c.withRelu)
        {
            PointwiseAttributes reluAttrs;
            reluAttrs.set_mode(PointwiseMode::RELU_FWD);
            if(c.reluLowerClip.has_value())
            {
                reluAttrs.set_relu_lower_clip(*c.reluLowerClip);
            }
            if(c.reluUpperClip.has_value())
            {
                reluAttrs.set_relu_upper_clip(*c.reluUpperClip);
            }
            out = graph.pointwise(out, reluAttrs);
            out->set_data_type(c.finalType);
        }
        if(!c.finalStrides.empty())
        {
            out->set_dim(outDims).set_stride(c.finalStrides);
        }
        out->set_output(true).set_uid(UID_C);

        auto result = graph.validate();
        if(result.is_bad())
        {
            return "validate: " + result.err_msg;
        }
        result = graph.build_operation_graph(*_handle);
        if(result.is_bad())
        {
            return "build_operation_graph: " + result.err_msg;
        }
        result = graph.create_execution_plan_ext(
            hipdnn_data_sdk::utilities::engineNameOrIdToId("HIPBLASLT_ENGINE"), {});
        if(result.is_bad())
        {
            return "create_execution_plan_ext: " + result.err_msg;
        }
        result = graph.check_support();
        if(result.is_bad())
        {
            return "check_support: " + result.err_msg;
        }
        result = graph.build_plans();
        if(result.is_bad())
        {
            return "build_plans: " + result.err_msg;
        }
        return {};
    }

    // Expects the graph to be declined at a stage before build_plans.
    void expectDeclinedBeforeBuildPlans(const MatmulCase& c) const
    {
        Graph graph;
        const std::string stage = buildGraph(graph, c);
        EXPECT_FALSE(stage.empty()) << "the provider accepted a graph it cannot compute correctly";
        EXPECT_NE(stage.rfind("build_plans", 0), 0U)
            << "declined only at build_plans, after the engine was selected: " << stage;
    }

    // Expects the engine to decline the graph: validate accepts it and build_plans is never
    // reached, so a frontend rejection or a late failure cannot satisfy the test.
    void expectDeclinedByEngine(const MatmulCase& c) const
    {
        Graph graph;
        const std::string stage = buildGraph(graph, c);
        EXPECT_FALSE(stage.empty()) << "the provider accepted a graph it cannot compute correctly";
        EXPECT_NE(stage.rfind("validate", 0), 0U)
            << "rejected by the frontend before the engine saw the graph: " << stage;
        EXPECT_NE(stage.rfind("build_plans", 0), 0U)
            << "declined only at build_plans, after the engine was selected: " << stage;
    }

    // Builds, executes and compares against a CPU reference computed from the rounded inputs.
    // Every output element outside the output tensor's own elements (the guard elements past its
    // extent and any padding between batches) must keep its guard value.
    void expectMatchesReference(const MatmulCase& c, float tolerance) const
    {
        Graph graph;
        const std::string stage = buildGraph(graph, c);
        ASSERT_TRUE(stage.empty()) << stage;

        const auto m = static_cast<size_t>(c.m);
        const auto k = static_cast<size_t>(c.k);
        const auto n = static_cast<size_t>(c.n);

        const auto aDims = batched(c.batchDims, {c.m, c.k});
        const auto bDims = batched(c.batchDims, {c.k, c.n});
        const auto outDims = batched(c.batchDims, {c.m, c.n});
        const auto aOffsets = elementOffsets(aDims, stridesOrPacked(c.aStrides, aDims));
        const auto bOffsets = elementOffsets(bDims, stridesOrPacked(c.bStrides, bDims));
        const auto outStrides
            = !c.finalStrides.empty() ? c.finalStrides : stridesOrPacked(c.cStrides, outDims);
        const auto outOffsets = elementOffsets(outDims, outStrides);
        size_t batch = 1;
        for(const auto dim : c.batchDims)
        {
            batch *= static_cast<size_t>(dim);
        }

        // A and B in logical row-major order; the device copies lay them out by their strides.
        // The reference reads them back through the offsets, so a stride-0 operand, whose logical
        // elements share one storage slot, is computed from the value that slot holds.
        const auto a = rounded(randomUniform(batch * m * k, 11U), c.ioType);
        const auto b = rounded(randomUniform(batch * k * n, 22U), c.ioType);
        const auto bias = rounded(randomUniform(n, 33U), DataType::FLOAT);
        const auto laidOutA = scatter(a, aOffsets);
        const auto laidOutB = scatter(b, bOffsets);
        std::vector<float> aEff(a.size());
        std::vector<float> bEff(b.size());
        for(size_t idx = 0; idx < a.size(); ++idx)
        {
            aEff[idx] = laidOutA[aOffsets[idx]];
        }
        for(size_t idx = 0; idx < b.size(); ++idx)
        {
            bEff[idx] = laidOutB[bOffsets[idx]];
        }

        std::vector<float> reference(batch * m * n);
        for(size_t bi = 0; bi < batch; ++bi)
        {
            for(size_t i = 0; i < m; ++i)
            {
                for(size_t j = 0; j < n; ++j)
                {
                    float acc = 0.0f;
                    for(size_t p = 0; p < k; ++p)
                    {
                        acc += aEff[(bi * m + i) * k + p] * bEff[(bi * k + p) * n + j];
                    }
                    if(c.withBias)
                    {
                        acc += bias[j];
                    }
                    if(c.withRelu)
                    {
                        acc = c.reluUpperClip.has_value()
                                  ? std::min(std::max(acc, c.reluLowerClip.value_or(0.0f)),
                                             *c.reluUpperClip)
                                  : std::max(acc, 0.0f);
                    }
                    reference[(bi * m + i) * n + j] = acc;
                }
            }
        }

        // Output extent in elements, honoring the output strides.
        const size_t extent = spanOf(outOffsets);
        const size_t outBytes = (extent + GUARD_ELEMENTS) * elementSize(c.finalType);

        const auto aBytes = encode(laidOutA, c.ioType);
        const auto bBytes = encode(laidOutB, c.ioType);
        const auto biasBytes = encode(bias, DataType::FLOAT);
        const std::vector<float> outInit(extent + GUARD_ELEMENTS, GUARD_VALUE);
        const auto outInitBytes = encode(outInit, c.finalType);

        const auto dA = deviceAlloc(aBytes.size());
        const auto dB = deviceAlloc(bBytes.size());
        const auto dBias = deviceAlloc(biasBytes.size());
        const auto dC = deviceAlloc(outBytes);
        ASSERT_TRUE(dA && dB && dBias && dC) << "device allocation failed";
        ASSERT_EQ(hipMemcpy(dA.get(), aBytes.data(), aBytes.size(), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(dB.get(), bBytes.data(), bBytes.size(), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(dBias.get(), biasBytes.data(), biasBytes.size(), hipMemcpyHostToDevice),
                  hipSuccess);
        ASSERT_EQ(hipMemcpy(dC.get(), outInitBytes.data(), outBytes, hipMemcpyHostToDevice),
                  hipSuccess);

        int64_t workspaceSize = 0;
        ASSERT_TRUE(graph.get_workspace_size(workspaceSize).is_good());
        const auto dWorkspace
            = workspaceSize > 0 ? deviceAlloc(static_cast<size_t>(workspaceSize)) : nullptr;
        ASSERT_TRUE(workspaceSize == 0 || dWorkspace) << "workspace allocation failed";

        std::unordered_map<int64_t, void*> pack
            = {{UID_A, dA.get()}, {UID_B, dB.get()}, {UID_C, dC.get()}};
        if(c.withBias)
        {
            pack[UID_BIAS] = dBias.get();
        }
        const auto execErr = graph.execute(*_handle, pack, dWorkspace.get());
        ASSERT_TRUE(execErr.is_good()) << execErr.err_msg;
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<char> outBytesHost(outBytes);
        ASSERT_EQ(hipMemcpy(outBytesHost.data(), dC.get(), outBytes, hipMemcpyDeviceToHost),
                  hipSuccess);
        const auto result = decode(outBytesHost, c.finalType);

        std::vector<bool> isOutputElement(result.size(), false);
        for(size_t linear = 0; linear < reference.size(); ++linear)
        {
            const float expected = reference[linear];
            const float got = result[outOffsets[linear]];
            isOutputElement[outOffsets[linear]] = true;
            ASSERT_LE(std::fabs(got - expected), tolerance + tolerance * std::fabs(expected))
                << "mismatch at (" << linear / (m * n) << "," << (linear / n) % m << ","
                << linear % n << "): got " << got << ", expected " << expected;
        }
        for(size_t g = 0; g < result.size(); ++g)
        {
            if(!isOutputElement[g])
            {
                ASSERT_EQ(result[g], GUARD_VALUE)
                    << "write outside the output tensor's elements at element " << g;
            }
        }
    }

    HipdnnHandlePtr _handle;
};

MatmulCase narrowIntermediateCaseMatmulOut()
{
    // Matmul output is a virtual HALF tensor consumed by a bias ADD; the final output is FLOAT.
    MatmulCase c;
    c.m = 4;
    c.k = 256;
    c.n = 4;
    c.ioType = DataType::HALF;
    c.intermediateType = DataType::HALF;
    c.matmulOutType = DataType::HALF;
    c.withBias = true;
    c.finalType = DataType::FLOAT;
    return c;
}

MatmulCase narrowIntermediateCaseBiasOut()
{
    // Matmul output is FLOAT; the bias output is a virtual HALF tensor consumed by a RELU_FWD.
    MatmulCase c;
    c.m = 4;
    c.k = 256;
    c.n = 4;
    c.ioType = DataType::HALF;
    c.intermediateType = DataType::FLOAT;
    c.matmulOutType = DataType::FLOAT;
    c.withBias = true;
    c.biasOutType = DataType::HALF;
    c.withRelu = true;
    c.finalType = DataType::FLOAT;
    return c;
}

} // namespace

// C = A * B is computed as C^T = B^T * A^T; a column-major output would be written transposed.
// Fails if the provider accepts a column-major output.
TEST_F(IntegrationGpuMatmulApplicability, ColumnMajorOutputDeclines)
{
    MatmulCase c;
    c.m = 8;
    c.k = 16;
    c.n = 12;
    c.cStrides = {1, 8};
    expectDeclinedBeforeBuildPlans(c);
}

// A size-1 dimension's stride is arbitrary; [M,1] with strides [1,M] is a contiguous column.
// Fails if the output layout is derived from the non-canonical stride and written wrongly.
class IntegrationGpuMatmulApplicabilityUnitColumn : public IntegrationGpuMatmulApplicability,
                                                    public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(IntegrationGpuMatmulApplicabilityUnitColumn,
       UnitColumnOutputWithNonCanonicalStrideMatchesReference)
{
    const int64_t m = GetParam();
    MatmulCase c;
    c.m = m;
    c.k = 16;
    c.n = 1;
    c.cStrides = {1, m};
    expectMatchesReference(c, FP32_TOLERANCE);
}

INSTANTIATE_TEST_SUITE_P(IntegrationGpuMatmulApplicability,
                         IntegrationGpuMatmulApplicabilityUnitColumn,
                         ::testing::Values<int64_t>(8, 33));

// A broadcast operand has a zero leading dimension: every row (or column) reads the same storage.
// Fails if the provider declines a zero leading dimension on A or B, or reads the operand with a
// non-zero leading dimension.
struct ZeroLeadingDimensionParam
{
    const char* name;
    DataType ioType;
    std::vector<int64_t> aStrides;
    std::vector<int64_t> bStrides;
};

class IntegrationGpuMatmulApplicabilityZeroLd
    : public IntegrationGpuMatmulApplicability,
      public ::testing::WithParamInterface<ZeroLeadingDimensionParam>
{
};

TEST_P(IntegrationGpuMatmulApplicabilityZeroLd, ZeroLeadingDimensionOperandMatchesReference)
{
    const auto& p = GetParam();
    MatmulCase c;
    c.m = 8;
    c.k = 16;
    c.n = 12;
    c.ioType = p.ioType;
    c.finalType = p.ioType;
    c.aStrides = p.aStrides;
    c.bStrides = p.bStrides;
    expectMatchesReference(c, p.ioType == DataType::HALF ? FP16_TOLERANCE : FP32_TOLERANCE);
}

INSTANTIATE_TEST_SUITE_P(
    IntegrationGpuMatmulApplicability,
    IntegrationGpuMatmulApplicabilityZeroLd,
    ::testing::Values(ZeroLeadingDimensionParam{"ARowMajorFloat", DataType::FLOAT, {0, 1}, {}},
                      ZeroLeadingDimensionParam{"AColumnMajorFloat", DataType::FLOAT, {1, 0}, {}},
                      ZeroLeadingDimensionParam{"BRowMajorFloat", DataType::FLOAT, {}, {0, 1}},
                      ZeroLeadingDimensionParam{"BColumnMajorFloat", DataType::FLOAT, {}, {1, 0}},
                      ZeroLeadingDimensionParam{"ARowMajorHalf", DataType::HALF, {0, 1}, {}},
                      ZeroLeadingDimensionParam{"AColumnMajorHalf", DataType::HALF, {1, 0}, {}},
                      ZeroLeadingDimensionParam{"BRowMajorHalf", DataType::HALF, {}, {0, 1}},
                      ZeroLeadingDimensionParam{"BColumnMajorHalf", DataType::HALF, {}, {1, 0}}),
    [](const ::testing::TestParamInfo<ZeroLeadingDimensionParam>& info) {
        return std::string(info.param.name);
    });

// An output with a zero leading dimension would have every row written over the same storage.
// The frontend accepts the stride (it only rejects negative ones), so the decline must come from
// the engine, after validate.
// Fails if the provider accepts a zero-leading-dimension output.
TEST_F(IntegrationGpuMatmulApplicability, ZeroLeadingDimensionOutputDeclines)
{
    MatmulCase c;
    c.m = 8;
    c.k = 16;
    c.n = 12;
    c.cStrides = {0, 1};
    expectDeclinedByEngine(c);
}

// The engine runs the epilogue on the FP32 accumulator, so a virtual intermediate narrower than
// FLOAT cannot be honored, whatever the final output type.
// Fails if the provider accepts such a fused graph.
TEST_F(IntegrationGpuMatmulApplicability, NarrowFusedMatmulIntermediateDeclines)
{
    expectDeclinedBeforeBuildPlans(narrowIntermediateCaseMatmulOut());
}

TEST_F(IntegrationGpuMatmulApplicability, NarrowFusedBiasIntermediateDeclines)
{
    expectDeclinedBeforeBuildPlans(narrowIntermediateCaseBiasOut());
}

// A narrow intermediate equal to the narrow final output is declined too.
// Fails if the provider exempts an intermediate that matches the final output type.
TEST_F(IntegrationGpuMatmulApplicability, NarrowFusedIntermediateEqualToOutputDeclines)
{
    auto c = narrowIntermediateCaseMatmulOut();
    c.finalType = DataType::HALF;
    expectDeclinedBeforeBuildPlans(c);
}

// Activation only: a narrow matmul output feeding a lone RELU_FWD, with no bias node.
// Fails if the intermediate check runs only when the graph has a bias.
TEST_F(IntegrationGpuMatmulApplicability, NarrowFusedIntermediateBeforeActivationOnlyDeclines)
{
    auto c = narrowIntermediateCaseMatmulOut();
    c.withBias = false;
    c.withRelu = true;
    c.finalType = DataType::HALF;
    expectDeclinedBeforeBuildPlans(c);
}

// Positive control for the decline rule: a FLOAT intermediate under HALF io builds and matches.
TEST_F(IntegrationGpuMatmulApplicability, FloatFusedIntermediateWithHalfOutputMatchesReference)
{
    MatmulCase c;
    c.m = 8;
    c.k = 16;
    c.n = 12;
    c.ioType = DataType::HALF;
    c.intermediateType = DataType::FLOAT;
    c.matmulOutType = DataType::FLOAT;
    c.withBias = true;
    c.finalType = DataType::HALF;
    expectMatchesReference(c, FP16_TOLERANCE);
}

// With K == 1, A [8,1] with strides {7,7} has neither stride equal to 1, so a rule that reads only
// the strides rejects it; the size-1 trailing dimension makes it row-major regardless.
// Fails if the provider declines an operand whose layout only the size-1 rule can classify.
TEST_F(IntegrationGpuMatmulApplicability, OperandWithUnitDimNonCanonicalStrideMatchesReference)
{
    MatmulCase c;
    c.m = 8;
    c.k = 1;
    c.n = 12;
    c.aStrides = {7, 7};
    c.bStrides = {12, 1};
    c.cStrides = {12, 1};
    expectMatchesReference(c, FP32_TOLERANCE);
}

// With K == 1, B [1,12] with strides {99,100} has a size-1 row dimension and a last stride that
// is not 1, so it is classified column-major and transposed. Exercises the column-major branch of
// the size-1 rule on the B operand end to end.
// Fails if the provider declines the operand or misreads it after a wrong transpose.
TEST_F(IntegrationGpuMatmulApplicability, SizeOneRowBOperandWithNonCanonicalStridesMatchesReference)
{
    MatmulCase c;
    c.m = 8;
    c.k = 1;
    c.n = 12;
    c.bStrides = {99, 100};
    c.cStrides = {12, 1};
    expectMatchesReference(c, FP32_TOLERANCE);
}

// A clamp with a non-zero lower bound maps to the clamp epilogue with both activation arguments.
// The inputs are uniform in [-1,1) over K = 32, so results fall below -0.25 and above 0.5.
// Fails if the lower or upper bound is not applied to the executed plan (e.g. lost when the
// plan is cloned).
TEST_F(IntegrationGpuMatmulApplicability, ClampedReluFusedOutputMatchesReference)
{
    MatmulCase c;
    c.m = 16;
    c.k = 32;
    c.n = 16;
    c.withRelu = true;
    c.reluLowerClip = -0.25f;
    c.reluUpperClip = 0.5f;
    expectMatchesReference(c, FP32_TOLERANCE);
}

// A batch axis of extent one never advances, so a fused final output [2,1,4,5] with strides
// [20,5,5,1] has batch stride 20 whatever the singleton axis carries.
// Fails if the singleton axis stride (5) is taken as the C and D batch stride.
TEST_F(IntegrationGpuMatmulApplicability, FusedFinalWithSingletonBatchAxisMatchesReference)
{
    MatmulCase c;
    c.m = 4;
    c.k = 8;
    c.n = 5;
    c.batchDims = {2, 1};
    c.cStrides = {20, 20, 5, 1};
    c.finalStrides = {20, 5, 5, 1};
    c.withRelu = true;
    expectMatchesReference(c, FP32_TOLERANCE);
}

// A [4,1,4,8] with strides [32,7,8,1]: the stride of the singleton axis (7) is not the batch
// stride (32).
// Fails if the A batch offset is read from the singleton axis.
TEST_F(IntegrationGpuMatmulApplicability, SingletonBatchAxisStrideOnAMatchesReference)
{
    MatmulCase c;
    c.m = 4;
    c.k = 8;
    c.n = 5;
    c.batchDims = {4, 1};
    c.aStrides = {32, 7, 8, 1};
    c.cStrides = {20, 20, 5, 1};
    expectMatchesReference(c, FP32_TOLERANCE);
}

// A fused final output [2,4,5] with strides [27,5,1] is padded between batches while the
// virtual matmul output is packed: D must be written at batch stride 27, and the seven padding
// elements between the batches must stay untouched.
// Fails if D is written at the matmul output's batch stride (20), which overwrites the padding
// and misplaces the second batch.
TEST_F(IntegrationGpuMatmulApplicability, FusedFinalWithPaddedBatchStrideMatchesReference)
{
    MatmulCase c;
    c.m = 4;
    c.k = 8;
    c.n = 5;
    c.batchDims = {2};
    c.cStrides = {20, 5, 1};
    c.finalStrides = {27, 5, 1};
    c.withRelu = true;
    expectMatchesReference(c, FP32_TOLERANCE);
}

// The same padded final output under HALF io with a FLOAT intermediate: the FP32-only
// intermediate rule and the final output's own batch stride apply together.
// Fails if either rule declines the graph or D is written at the matmul output's batch stride.
TEST_F(IntegrationGpuMatmulApplicability, HalfFusedFinalWithPaddedBatchStrideMatchesReference)
{
    MatmulCase c;
    c.m = 4;
    c.k = 8;
    c.n = 5;
    c.batchDims = {2};
    c.ioType = DataType::HALF;
    c.intermediateType = DataType::FLOAT;
    c.matmulOutType = DataType::FLOAT;
    c.cStrides = {20, 5, 1};
    c.finalStrides = {27, 5, 1};
    c.withRelu = true;
    c.finalType = DataType::HALF;
    expectMatchesReference(c, FP16_TOLERANCE);
}

// A final output [2,4,5] with strides [10,5,1] has a batch stride below one matrix's span (4 rows
// of stride 5 and 5 columns = 20 elements): batch 0 row 2 and batch 1 row 0 share element 10. The
// virtual matmul output is packed, so only the final output overlaps.
// Fails if the provider accepts a fused final output whose batches overlap.
TEST_F(IntegrationGpuMatmulApplicability, FusedFinalWithOverlappingBatchStrideDeclines)
{
    MatmulCase c;
    c.m = 4;
    c.k = 8;
    c.n = 5;
    c.batchDims = {2};
    c.cStrides = {20, 5, 1};
    c.finalStrides = {10, 5, 1};
    c.withRelu = true;
    expectDeclinedByEngine(c);
}

// A plain matmul output [2,4,5] with strides [10,5,1] overlaps its batches the same way.
// Fails if the provider accepts a matmul output whose batch stride is below one matrix's span.
TEST_F(IntegrationGpuMatmulApplicability, OverlappingBatchStrideOutputDeclines)
{
    MatmulCase c;
    c.m = 4;
    c.k = 8;
    c.n = 5;
    c.batchDims = {2};
    c.cStrides = {10, 5, 1};
    expectDeclinedByEngine(c);
}

// A matmul output [4,5] with strides [4,1] has a row stride below the column count: row 0 column 4
// and row 1 column 0 share element 4.
// Fails if the provider accepts an output whose row stride is smaller than its column count.
TEST_F(IntegrationGpuMatmulApplicability, OverlappingRowsOutputDeclines)
{
    MatmulCase c;
    c.m = 4;
    c.k = 8;
    c.n = 5;
    c.cStrides = {4, 1};
    expectDeclinedByEngine(c);
}
