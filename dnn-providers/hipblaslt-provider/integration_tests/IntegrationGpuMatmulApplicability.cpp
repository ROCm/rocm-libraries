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
#include <random>
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

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

constexpr int64_t UID_A = 1;
constexpr int64_t UID_B = 2;
constexpr int64_t UID_BIAS = 3;
constexpr int64_t UID_C = 4;

constexpr float FP32_TOLERANCE = 2e-4f;
constexpr float FP16_TOLERANCE = 3e-2f;
constexpr float GUARD_VALUE = 12288.0f;
constexpr size_t GUARD_ELEMENTS = 16;

struct HipFreeDeleter
{
    void operator()(void* ptr) const
    {
        // Best-effort release; a failed free cannot be acted on here.
        std::ignore = hipFree(ptr);
    }
};

using DevicePtr = std::unique_ptr<void, HipFreeDeleter>;

DevicePtr deviceAlloc(size_t bytes)
{
    void* ptr = nullptr;
    if(hipMalloc(&ptr, bytes) != hipSuccess)
    {
        return nullptr;
    }
    return DevicePtr(ptr);
}

std::vector<float> randomUniform(size_t count, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> values(count);
    for(auto& v : values)
    {
        v = dist(gen);
    }
    return values;
}

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

// One matmul, optionally followed by a bias ADD and a RELU_FWD, described by the dtype of every
// tensor and the strides of the matmul output.
struct MatmulCase
{
    int64_t m = 8;
    int64_t k = 16;
    int64_t n = 12;
    DataType ioType = DataType::FLOAT;
    DataType intermediateType = DataType::FLOAT;
    // Strides of the matmul output; empty lets the frontend infer them.
    std::vector<int64_t> cStrides;
    // Data type of the matmul output when a bias or activation follows.
    DataType matmulOutType = DataType::FLOAT;
    bool withBias = false;
    // Data type of the bias output when an activation follows.
    DataType biasOutType = DataType::FLOAT;
    bool withRelu = false;
    DataType finalType = DataType::FLOAT;
};

class IntegrationGpuMatmulApplicability : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        auto pluginPath = std::filesystem::weakly_canonical(
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

        const std::vector<int64_t> aDims = {c.m, c.k};
        const std::vector<int64_t> bDims = {c.k, c.n};
        auto a = std::make_shared<TensorAttributes>(
            makeTensorAttributes("a", c.ioType, aDims, {c.k, 1}).set_uid(UID_A));
        auto b = std::make_shared<TensorAttributes>(
            makeTensorAttributes("b", c.ioType, bDims, {c.n, 1}).set_uid(UID_B));

        auto out = graph.matmul(a, b, MatmulAttributes());
        const bool fused = c.withBias || c.withRelu;
        out->set_data_type(fused ? c.matmulOutType : c.finalType);
        if(!c.cStrides.empty())
        {
            out->set_dim({c.m, c.n}).set_stride(c.cStrides);
        }

        if(c.withBias)
        {
            const std::vector<int64_t> biasDims = {1, c.n};
            auto bias = std::make_shared<TensorAttributes>(
                makeTensorAttributes("bias", DataType::FLOAT, biasDims, {c.n, 1})
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
            out = graph.pointwise(out, reluAttrs);
            out->set_data_type(c.finalType);
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

    // Builds, executes and compares against a CPU reference computed from the rounded inputs.
    // The output buffer carries guard elements past its extent that must stay untouched.
    void expectMatchesReference(const MatmulCase& c, float tolerance) const
    {
        Graph graph;
        const std::string stage = buildGraph(graph, c);
        ASSERT_TRUE(stage.empty()) << stage;

        const auto m = static_cast<size_t>(c.m);
        const auto k = static_cast<size_t>(c.k);
        const auto n = static_cast<size_t>(c.n);

        const auto a = rounded(randomUniform(m * k, 11U), c.ioType);
        const auto b = rounded(randomUniform(k * n, 22U), c.ioType);
        const auto bias = rounded(randomUniform(n, 33U), DataType::FLOAT);

        std::vector<float> reference(m * n);
        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                float acc = 0.0f;
                for(size_t p = 0; p < k; ++p)
                {
                    acc += a[i * k + p] * b[p * n + j];
                }
                if(c.withBias)
                {
                    acc += bias[j];
                }
                if(c.withRelu)
                {
                    acc = std::max(acc, 0.0f);
                }
                reference[i * n + j] = acc;
            }
        }

        // Output extent in elements, honoring the output strides.
        const std::vector<int64_t> outStrides
            = c.cStrides.empty() ? std::vector<int64_t>{c.n, 1} : c.cStrides;
        const size_t extent = (m - 1) * static_cast<size_t>(outStrides[0])
                              + (n - 1) * static_cast<size_t>(outStrides[1]) + 1;
        const size_t outBytes = (extent + GUARD_ELEMENTS) * elementSize(c.finalType);

        const auto aBytes = encode(a, c.ioType);
        const auto bBytes = encode(b, c.ioType);
        const auto biasBytes = encode(bias, DataType::FLOAT);
        const std::vector<float> outInit(extent + GUARD_ELEMENTS, GUARD_VALUE);
        const auto outInitBytes = encode(outInit, c.finalType);

        auto dA = deviceAlloc(aBytes.size());
        auto dB = deviceAlloc(bBytes.size());
        auto dBias = deviceAlloc(biasBytes.size());
        auto dC = deviceAlloc(outBytes);
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
        auto dWorkspace
            = workspaceSize > 0 ? deviceAlloc(static_cast<size_t>(workspaceSize)) : nullptr;
        ASSERT_TRUE(workspaceSize == 0 || dWorkspace) << "workspace allocation failed";

        std::unordered_map<int64_t, void*> pack
            = {{UID_A, dA.get()}, {UID_B, dB.get()}, {UID_C, dC.get()}};
        if(c.withBias)
        {
            pack[UID_BIAS] = dBias.get();
        }
        auto execErr = graph.execute(*_handle, pack, dWorkspace.get());
        ASSERT_TRUE(execErr.is_good()) << execErr.err_msg;
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<char> outBytesHost(outBytes);
        ASSERT_EQ(hipMemcpy(outBytesHost.data(), dC.get(), outBytes, hipMemcpyDeviceToHost),
                  hipSuccess);
        const auto result = decode(outBytesHost, c.finalType);

        for(size_t i = 0; i < m; ++i)
        {
            for(size_t j = 0; j < n; ++j)
            {
                const float expected = reference[i * n + j];
                const float got = result[i * static_cast<size_t>(outStrides[0])
                                         + j * static_cast<size_t>(outStrides[1])];
                ASSERT_LE(std::fabs(got - expected), tolerance + tolerance * std::fabs(expected))
                    << "mismatch at (" << i << "," << j << "): got " << got << ", expected "
                    << expected;
            }
        }
        for(size_t g = extent; g < extent + GUARD_ELEMENTS; ++g)
        {
            ASSERT_EQ(result[g], GUARD_VALUE) << "write past the output extent at element " << g;
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

// Positive control: the same shapes with a row-major output build and match the reference.
TEST_F(IntegrationGpuMatmulApplicability, RowMajorOutputMatchesReference)
{
    MatmulCase c;
    c.m = 8;
    c.k = 16;
    c.n = 12;
    c.cStrides = {12, 1};
    expectMatchesReference(c, FP32_TOLERANCE);
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

// The engine adds the bias to the FP32 accumulator and rounds once, so a virtual intermediate
// narrower than FLOAT (and different from the final output) cannot be honored.
// Fails if the provider accepts such a fused graph.
TEST_F(IntegrationGpuMatmulApplicability, NarrowFusedMatmulIntermediateDeclines)
{
    expectDeclinedBeforeBuildPlans(narrowIntermediateCaseMatmulOut());
}

TEST_F(IntegrationGpuMatmulApplicability, NarrowFusedBiasIntermediateDeclines)
{
    expectDeclinedBeforeBuildPlans(narrowIntermediateCaseBiasOut());
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
