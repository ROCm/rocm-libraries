// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/BatchnormInferencePointwiseTestGraphs.hpp"

/**
 * @file TestBatchnormInferencePointwisePack.cpp
 * @brief The fused two-node fixture engine (ALMIOPEN-3070): its descriptor set, its graph
 *        match over [BatchnormInference, Pointwise] and every refusal, the kernel-scoped
 *        activation matcher, and numerics through the dispatch handler and the engine.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr std::string_view ACTIVATION_FIELD = "activation";
constexpr std::string_view ACTIVATION_TOKEN = "batchnorm_inference_pointwise.activation";
constexpr std::string_view ALPHA_TOKEN = "batchnorm_inference_pointwise.alpha";
constexpr std::string_view BETA_TOKEN = "batchnorm_inference_pointwise.beta";
constexpr std::string_view INTERMEDIATE_TOKEN = "batchnorm_inference_pointwise.intermediate.uid";

/// The four activations the engine's kernels apply, with the graph that asks for each.
struct ActivationCase
{
    const char* name;
    data_objects::PointwiseMode mode;
    std::optional<float> lowerClip;
    std::optional<float> upperClip;
};

const std::array<ActivationCase, 4>& activationCases()
{
    static const std::array<ActivationCase, 4> s_cases
        = {ActivationCase{"IDENTITY", data_objects::PointwiseMode::IDENTITY, {}, {}},
           ActivationCase{"RELU", data_objects::PointwiseMode::RELU_FWD, {}, {}},
           ActivationCase{"CLIPPED_RELU", data_objects::PointwiseMode::RELU_FWD, {}, 1.0f},
           ActivationCase{"CLAMP", data_objects::PointwiseMode::RELU_FWD, -0.5f, 0.75f}};
    return s_cases;
}

BatchnormInferencePointwiseGraph graphFor(const ActivationCase& activation)
{
    BatchnormInferencePointwiseGraph spec;
    spec.mode = activation.mode;
    spec.lowerClip = activation.lowerClip;
    spec.upperClip = activation.upperClip;
    return spec;
}

std::optional<BoundTokens> match(const BatchnormInferencePointwiseGraph& spec,
                                 const hipdnn_plugin_sdk::ingestor::DeviceProperties& properties
                                 = testDeviceProperties())
{
    const GraphFixture fixture(buildBatchnormInferencePointwiseGraph(spec), properties);
    return matchesGraph(BATCHNORM_INFERENCE_POINTWISE, fixture.context());
}

std::optional<std::string> boundString(const BoundTokens& bound, std::string_view token)
{
    const auto entry = bound.find(std::string(token));
    if(entry == bound.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&entry->second);
    return value == nullptr ? std::nullopt : std::make_optional(*value);
}

std::optional<double> boundDouble(const BoundTokens& bound, std::string_view token)
{
    const auto entry = bound.find(std::string(token));
    if(entry == bound.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<double>(&entry->second);
    return value == nullptr ? std::nullopt : std::make_optional(*value);
}

/// A loaded kernel descriptor's string metadata field.
std::string stringField(const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel,
                        std::string_view field)
{
    return std::get<std::string>(kernel.metadata.at(std::string(field)));
}

/// The loaded kernel whose `activation` metadata is @p activation, as the engine hands it to
/// matchers and dispatch.
KernelDefinition loadedKernel(const std::string& activation)
{
    const auto& set = loadedSet(BATCHNORM_INFERENCE_POINTWISE.engineName);
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            if(stringField(kernel, ACTIVATION_FIELD) == activation)
            {
                KernelDefinition definition;
                definition.kernelId = kernel.id;
                definition.packId = pack.id;
                definition.dispatchId = pack.dispatchId;
                definition.source = kernel.source;
                definition.metadata = kernel.metadata;
                definition.priority = kernel.priority;
                definition.arch = kernel.arch.empty() ? pack.arch : kernel.arch;
                definition.originDirectory = kernel.originDirectory;
                definition.name = kernel.name;
                definition.treeRoot = kernel.treeRoot;
                return definition;
            }
        }
    }
    throw std::runtime_error("no loaded kernel applies " + activation);
}

// ---------------------------------------------------------------------------
// The descriptor set
// ---------------------------------------------------------------------------

TEST(TestBatchnormInferencePointwisePack, ShipsOneFloatKernelPerActivation)
{
    const auto& set = loadedSet(BATCHNORM_INFERENCE_POINTWISE.engineName);

    ASSERT_EQ(distinctPackIdCount(set), 1U);
    std::vector<std::string> activations;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            EXPECT_EQ(kernel.source.kind,
                      hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE);
            EXPECT_EQ(kernel.source.sourceFile, "kernels/BatchnormInferencePointwise.cpp");
            EXPECT_EQ(stringField(kernel, DTYPE_FIELD), "FLOAT");
            activations.push_back(stringField(kernel, ACTIVATION_FIELD));
        }
    }
    std::sort(activations.begin(), activations.end());
    activations.erase(std::unique(activations.begin(), activations.end()), activations.end());
    EXPECT_THAT(activations, ::testing::ElementsAre("CLAMP", "CLIPPED_RELU", "IDENTITY", "RELU"));
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, BATCHNORM_INFERENCE_POINTWISE.graphMatcher);
}

// ---------------------------------------------------------------------------
// Graph match: acceptances
// ---------------------------------------------------------------------------

TEST(TestBatchnormInferencePointwiseGraphMatch, BindsEveryOperandAcrossBothNodes)
{
    const auto bound = match({});
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, BATCHNORM_INFERENCE_POINTWISE.inputAToken), BNPW_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, "batchnorm_inference_pointwise.mean.uid"), BNPW_MEAN_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, "batchnorm_inference_pointwise.inv_variance.uid"),
              BNPW_INV_VARIANCE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, "batchnorm_inference_pointwise.scale.uid"), BNPW_SCALE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, "batchnorm_inference_pointwise.bias.uid"), BNPW_BIAS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, INTERMEDIATE_TOKEN), BNPW_INTERMEDIATE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, BATCHNORM_INFERENCE_POINTWISE.outputToken), BNPW_OUTPUT_UID);
}

TEST(TestBatchnormInferencePointwiseGraphMatch, ClassifiesEachActivationAndBindsItsBounds)
{
    for(const auto& activation : activationCases())
    {
        SCOPED_TRACE(activation.name);
        const auto bound = match(graphFor(activation));
        ASSERT_TRUE(bound.has_value());

        EXPECT_EQ(boundString(*bound, ACTIVATION_TOKEN), activation.name);
        EXPECT_EQ(boundDouble(*bound, ALPHA_TOKEN),
                  activation.lowerClip.has_value() && activation.upperClip.has_value()
                      ? static_cast<double>(*activation.lowerClip)
                      : 0.0);
        EXPECT_EQ(boundDouble(*bound, BETA_TOKEN),
                  activation.upperClip.has_value() ? static_cast<double>(*activation.upperClip)
                                                   : 0.0);
    }
}

TEST(TestBatchnormInferencePointwiseGraphMatch, AcceptsAZeroLowerClipAsPlainRelu)
{
    BatchnormInferencePointwiseGraph spec;
    spec.lowerClip = 0.0f;
    const auto bound = match(spec);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(boundString(*bound, ACTIVATION_TOKEN), "RELU");
}

// ---------------------------------------------------------------------------
// Graph match: refusals
// ---------------------------------------------------------------------------

struct RefusalCase
{
    const char* name;
    BatchnormInferencePointwiseGraph spec;
};

BatchnormInferencePointwiseGraph with(void (*edit)(BatchnormInferencePointwiseGraph&))
{
    BatchnormInferencePointwiseGraph spec;
    edit(spec);
    return spec;
}

class TestBatchnormInferencePointwiseGraphRefusal : public ::testing::TestWithParam<RefusalCase>
{
};

TEST_P(TestBatchnormInferencePointwiseGraphRefusal, Declines)
{
    EXPECT_FALSE(match(GetParam().spec).has_value());
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestBatchnormInferencePointwiseGraphRefusal,
    ::testing::Values(
        RefusalCase{"BatchnormAlone", with([](auto& s) { s.batchnormOnly = true; })},
        RefusalCase{"NodesInTheOtherOrder", with([](auto& s) { s.pointwiseFirst = true; })},
        RefusalCase{"PointwiseNotFedByTheBatchnorm",
                    with([](auto& s) { s.pointwiseReadsInput = true; })},
        RefusalCase{"IntermediateNotVirtual", with([](auto& s) { s.intermediateVirtual = false; })},
        RefusalCase{"OutputVirtual", with([](auto& s) { s.outputVirtual = true; })},
        RefusalCase{"BinaryPointwise", with([](auto& s) {
                        s.mode = data_objects::PointwiseMode::ADD;
                        s.binaryPointwise = true;
                    })},
        RefusalCase{"UnsupportedActivation",
                    with([](auto& s) { s.mode = data_objects::PointwiseMode::SIGMOID_FWD; })},
        RefusalCase{"LeakyRelu", with([](auto& s) { s.lowerClipSlope = 0.1f; })},
        RefusalCase{"NonZeroLowerClipWithoutUpper", with([](auto& s) { s.lowerClip = 0.5f; })},
        RefusalCase{"HalfPrecision",
                    with([](auto& s) { s.dataType = data_objects::DataType::HALF; })},
        RefusalCase{"RankFive", with([](auto& s) { s.dims = {2, 3, 4, 5, 6}; })},
        RefusalCase{"ChannelTensorNotPerChannel",
                    with([](auto& s) { s.channelDims = {1, 3, 4, 1}; })}),
    [](const ::testing::TestParamInfo<RefusalCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Kernel-scoped matcher
// ---------------------------------------------------------------------------

TEST(TestBatchnormInferencePointwiseKernelMatch, AdmitsOnlyTheKernelForTheGraphsActivation)
{
    for(const auto& graphActivation : activationCases())
    {
        SCOPED_TRACE(graphActivation.name);
        const GraphFixture fixture(
            buildBatchnormInferencePointwiseGraph(graphFor(graphActivation)));
        const auto bound = matchesGraph(BATCHNORM_INFERENCE_POINTWISE, fixture.context());
        ASSERT_TRUE(bound.has_value());

        for(const auto& kernelActivation : activationCases())
        {
            EXPECT_EQ(matchesKernel(BATCHNORM_INFERENCE_POINTWISE,
                                    fixture.context(),
                                    loadedKernel(kernelActivation.name),
                                    *bound),
                      std::string_view(kernelActivation.name) == graphActivation.name)
                << "kernel " << kernelActivation.name;
        }
    }
}

TEST(TestBatchnormInferencePointwiseKernelMatch, DeclinesWithoutTheGraphMatchsTokens)
{
    const GraphFixture fixture(buildBatchnormInferencePointwiseGraph());
    EXPECT_FALSE(
        matchesKernel(BATCHNORM_INFERENCE_POINTWISE, fixture.context(), loadedKernel("RELU")));
}

// ---------------------------------------------------------------------------
// Numerics on device
// ---------------------------------------------------------------------------

/// Deterministic operands for the default [2, 3, 4, 5] graph, on host and device.
class FusedOperands
{
public:
    static constexpr size_t N = 2;
    static constexpr size_t C = 3;
    static constexpr size_t SPATIAL = 4 * 5;
    static constexpr size_t COUNT = N * C * SPATIAL;

    FusedOperands()
    {
        _x.resize(COUNT);
        for(size_t i = 0; i < COUNT; ++i)
        {
            _x[i] = 3.0f * std::sin(0.37f * static_cast<float>(i));
        }
        for(size_t c = 0; c < C; ++c)
        {
            _mean.push_back(0.5f * static_cast<float>(c) - 0.5f);
            _invVariance.push_back(0.5f + 0.25f * static_cast<float>(c));
            _scale.push_back(1.5f - 0.5f * static_cast<float>(c));
            _bias.push_back(0.1f * static_cast<float>(c) - 0.2f);
        }

        upload(_x, _dx);
        upload(_mean, _dmean);
        upload(_invVariance, _dinvVariance);
        upload(_scale, _dscale);
        upload(_bias, _dbias);
        EXPECT_EQ(hipSuccess, hipMalloc(&_dout, COUNT * sizeof(float)));
        EXPECT_EQ(hipSuccess, hipMemset(_dout, 0xFF, COUNT * sizeof(float)));
    }

    ~FusedOperands()
    {
        for(void* buffer : {_dx, _dmean, _dinvVariance, _dscale, _dbias, _dout})
        {
            static_cast<void>(hipFree(buffer));
        }
    }

    FusedOperands(const FusedOperands&) = delete;
    FusedOperands& operator=(const FusedOperands&) = delete;

    /// No buffer for the intermediate: the fused kernel never materialises it.
    std::array<hipdnnPluginDeviceBuffer_t, 6> buffers() const
    {
        return {hipdnnPluginDeviceBuffer_t{BNPW_X_UID, _dx},
                hipdnnPluginDeviceBuffer_t{BNPW_MEAN_UID, _dmean},
                hipdnnPluginDeviceBuffer_t{BNPW_INV_VARIANCE_UID, _dinvVariance},
                hipdnnPluginDeviceBuffer_t{BNPW_SCALE_UID, _dscale},
                hipdnnPluginDeviceBuffer_t{BNPW_BIAS_UID, _dbias},
                hipdnnPluginDeviceBuffer_t{BNPW_OUTPUT_UID, _dout}};
    }

    std::vector<float> output() const
    {
        std::vector<float> result(COUNT);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(result.data(), _dout, COUNT * sizeof(float), hipMemcpyDeviceToHost));
        return result;
    }

    std::vector<float> reference(const ActivationCase& activation) const
    {
        std::vector<float> result(COUNT);
        for(size_t i = 0; i < COUNT; ++i)
        {
            const size_t c = (i / SPATIAL) % C;
            float value = _scale[c] * (_x[i] - _mean[c]) * _invVariance[c] + _bias[c];
            const std::string_view name = activation.name;
            if(name == "RELU")
            {
                value = std::max(value, 0.0f);
            }
            else if(name == "CLIPPED_RELU")
            {
                value = std::min(std::max(value, 0.0f), *activation.upperClip);
            }
            else if(name == "CLAMP")
            {
                value = std::min(std::max(value, *activation.lowerClip), *activation.upperClip);
            }
            result[i] = value;
        }
        return result;
    }

private:
    static void upload(const std::vector<float>& host, void*& device)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&device, host.size() * sizeof(float)));
        EXPECT_EQ(
            hipSuccess,
            hipMemcpy(device, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
    }

    std::vector<float> _x;
    std::vector<float> _mean;
    std::vector<float> _invVariance;
    std::vector<float> _scale;
    std::vector<float> _bias;
    void* _dx = nullptr;
    void* _dmean = nullptr;
    void* _dinvVariance = nullptr;
    void* _dscale = nullptr;
    void* _dbias = nullptr;
    void* _dout = nullptr;
};

void expectMatchesReference(const std::vector<float>& actual, const std::vector<float>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "element " << i;
    }
}

TEST(TestBatchnormInferencePointwiseDevice, DispatchMatchesTheReferenceForEveryActivation)
{
    SKIP_IF_NO_DEVICES();

    const auto& handler = dispatchHandler(BATCHNORM_INFERENCE_POINTWISE);
    for(const auto& activation : activationCases())
    {
        SCOPED_TRACE(activation.name);
        const GraphFixture fixture(buildBatchnormInferencePointwiseGraph(graphFor(activation)),
                                   currentDeviceProperties());
        const auto bound = matchesGraph(BATCHNORM_INFERENCE_POINTWISE, fixture.context());
        ASSERT_TRUE(bound.has_value());

        const auto prepared
            = handler.prepare(fixture.context(), *bound, loadedKernel(activation.name));
        ASSERT_NE(prepared, nullptr);

        const FusedOperands operands;
        const auto buffers = operands.buffers();
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(handle.getStream()));

        expectMatchesReference(operands.output(), operands.reference(activation));
    }
}

TEST(TestBatchnormInferencePointwiseDevice, TheEngineServesAndExecutesTheFusedGraph)
{
    SKIP_IF_NO_DEVICES();

    const int64_t engineId
        = hipdnn_data_sdk::utilities::engineNameToId(BATCHNORM_INFERENCE_POINTWISE.engineName);
    core::Container container;
    auto& engineManager = container.getEngineManager();
    Handle handle;

    for(const auto& activation : activationCases())
    {
        SCOPED_TRACE(activation.name);
        const auto builder = buildBatchnormInferencePointwiseGraph(graphFor(activation));
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());

        const auto applicable = engineManager.getApplicableEngineIds(handle, graph);
        ASSERT_NE(std::find(applicable.begin(), applicable.end(), engineId), applicable.end());

        hipdnn_test_sdk::utilities::MockEngineConfig engineConfig;
        EXPECT_CALL(engineConfig, isValid()).WillRepeatedly(::testing::Return(false));
        EXPECT_CALL(engineConfig, engineId()).WillRepeatedly(::testing::Return(engineId));

        Context context;
        ASSERT_NO_THROW(
            engineManager.initializeExecutionContext(handle, graph, engineConfig, context));
        ASSERT_TRUE(context.hasValidPlan());
        const auto& plan
            = dynamic_cast<const hipdnn_plugin_sdk::ingestor::GenericPlan<Handle>&>(context.plan());
        EXPECT_EQ(plan.kernel().getStringMetadata(std::string(ACTIVATION_FIELD)), activation.name);

        const FusedOperands operands;
        const auto buffers = operands.buffers();
        ASSERT_NO_THROW(context.plan().execute(
            handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(handle.getStream()));

        expectMatchesReference(operands.output(), operands.reference(activation));
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
