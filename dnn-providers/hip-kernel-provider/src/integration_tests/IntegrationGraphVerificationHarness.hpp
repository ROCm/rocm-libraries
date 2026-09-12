// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include <limits>
#include <memory>
#include <unordered_map>

namespace hip_kernel_provider::test_utilities
{

/// Verification state belongs to this graph and survives repeated executions.
/// The graph (whether stack allocated or shared-owned) must outlive its context.
class GraphVerificationContext
{
public:
    explicit GraphVerificationContext(hipdnn_frontend::graph::Graph& graph)
        : _graph(graph)
    {
    }

    GraphVerificationContext(const GraphVerificationContext&) = delete;
    GraphVerificationContext& operator=(const GraphVerificationContext&) = delete;

private:
    template <typename DataType, typename TestCaseType>
    friend class IntegrationGraphVerificationHarness;

    struct Registration
    {
        float absoluteTolerance = 0.0f;
        float relativeTolerance = 0.0f;
        hipdnn_frontend::DataType validatorType = hipdnn_frontend::DataType::NOT_SET;
        std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation> validator;
    };

    hipdnn_frontend::graph::Graph& _graph;
    std::unordered_map<std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>, Registration>
        _registrations;
};

// NOLINTBEGIN (portability-template-virtual-member-function)
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
protected:
    static constexpr float DEFAULT_MIN = -1.0f;
    static constexpr float DEFAULT_MAX = 1.0f;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

protected:
    /// Builds the context's graph and validates every current nonvirtual output.
    void verifyGraph(GraphVerificationContext& context, unsigned int seed)
    {
        auto result = context._graph.build(_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        ASSERT_NO_FATAL_FAILURE(verifyBuiltGraph(context, seed));
    }

    /// Uses the same verification path after a caller builds its engine-pinned plans.
    void verifyBuiltGraph(GraphVerificationContext& context, unsigned int seed)
    {
        auto& graph = context._graph;
        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle;
        hipdnn_test_sdk::utilities::GraphTensorBundle cpuBundle;
        std::vector<OutputTensor> outputs;

        generateBundles(graph, cpuBundle, gpuBundle, outputs);

        initializeBundle(graph, gpuBundle, seed);
        initializeBundle(graph, cpuBundle, seed);

        ASSERT_NO_FATAL_FAILURE(executeGpuGraph(_handle, graph, gpuBundle));
        ASSERT_NO_FATAL_FAILURE(executeCpuGraph(graph, cpuBundle));

        ASSERT_NO_FATAL_FAILURE(resolveOutputValidators(context, outputs));

        HIPDNN_PLUGIN_LOG_INFO("Validating " << outputs.size() << " output tensors");

        for(const auto& output : outputs)
        {
            auto& cpuTensor = cpuBundle.tensors.at(output.uid);
            auto& gpuTensor = gpuBundle.tensors.at(output.uid);
            gpuTensor->markDeviceModified();

            const auto& registration = context._registrations.at(output.attr);
            const bool valid = registration.validator->allClose(*cpuTensor, *gpuTensor);
            ASSERT_TRUE(valid) << "Mismatch found in tensor with id: " << output.uid
                               << ", name: " << output.attr->get_name();
        }
    }

    /// Registers each nonvirtual output in this graph's context.
    ///
    /// @p epsilonMultiple is expressed in epsilons of THIS FIXTURE'S element type, not in
    /// absolute units: a K-term sum needs ~K of them, an elementwise op needs one, and
    /// the same number then means the same thing in a FLOAT fixture and a HALF one.
    /// Hardcoding an absolute float epsilon is how a HALF comparison ends up ~4000x
    /// tighter than the type can represent.
    void registerValidatorsForOutputs(GraphVerificationContext& context,
                                      float epsilonMultiple = 1.0f)
    {
        const float tolerance
            = epsilonMultiple * static_cast<float>(std::numeric_limits<DataType>::epsilon());
        context._graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(!tensorAttr->get_is_virtual())
                {
                    registerValidator(context, tensorAttr, tolerance);
                }
            }
        });
    }

    void registerValidator(GraphVerificationContext& context,
                           const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& attr,
                           float tolerance)
    {
        registerValidator(context, attr, tolerance, tolerance);
    }

    void registerValidator(GraphVerificationContext& context,
                           const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& attr,
                           float absoluteTolerance,
                           float relativeTolerance)
    {
        ASSERT_NE(attr, nullptr);
        bool belongsToGraph = false;
        context._graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& output : node.getNodeOutputTensorAttributes())
            {
                belongsToGraph = belongsToGraph || output == attr;
            }
        });
        ASSERT_TRUE(belongsToGraph) << "Validator output does not belong to the context's graph";

        auto& registration = context._registrations[attr];
        if(registration.absoluteTolerance != absoluteTolerance
           || registration.relativeTolerance != relativeTolerance)
        {
            registration.validator.reset();
        }
        registration.absoluteTolerance = absoluteTolerance;
        registration.relativeTolerance = relativeTolerance;
    }

    struct OutputTensor
    {
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr;
        int64_t uid;
        hipdnn_frontend::DataType dataType;
    };

    // Check the entire current output set before any comparator reads its storage.
    void resolveOutputValidators(GraphVerificationContext& context,
                                 const std::vector<OutputTensor>& outputs)
    {
        ASSERT_FALSE(outputs.empty())
            << "At least one output tensor id must be specified for validation.";
        for(const auto& output : outputs)
        {
            const auto it = context._registrations.find(output.attr);
            ASSERT_NE(it, context._registrations.end())
                << "No validator registered for tensor with id: " << output.uid
                << ", name: " << output.attr->get_name();
            ASSERT_EQ(output.attr->get_uid(), output.uid)
                << "Output UID changed after bundle creation";
            ASSERT_EQ(output.attr->get_data_type(), output.dataType)
                << "Output data type changed after bundle creation";

            auto& registration = it->second;
            if(!registration.validator || registration.validatorType != output.dataType)
            {
                registration.validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
                    hipdnn_test_sdk::utilities::frontendToSdkDataType(output.dataType),
                    registration.absoluteTolerance,
                    registration.relativeTolerance);
                registration.validatorType = output.dataType;
            }
        }
    }

    void generateBundles(hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& cpuBundle,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle,
                         std::vector<OutputTensor>& outputs)
    {
        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(!tensorAttr->get_is_virtual())
                {
                    tryAddTensorToBundles(tensorAttr, cpuBundle, gpuBundle);
                    outputs.push_back(
                        {tensorAttr, tensorAttr->get_uid(), tensorAttr->get_data_type()});
                }
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                tryAddTensorToBundles(tensorAttr, cpuBundle, gpuBundle);
            }
        });
    }

    /// Seeds every tensor in @p bundle alike.
    ///
    /// A suite whose operation can confuse two identically-seeded operands overrides this
    /// rather than changing it here: the base is shared with suites that have their own
    /// seeding contract and their own tolerances calibrated against it.
    virtual void initializeBundle([[maybe_unused]] const hipdnn_frontend::graph::Graph& graph,
                                  hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                                  unsigned int seed)
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, DEFAULT_MIN, DEFAULT_MAX, seed);
        }
    }

    virtual hipStream_t stream() const
    {
        return _stream;
    }

    // Exposed to subclasses driving Graph staging calls directly (build_operation_graph,
    // create_execution_plans, get_ranked_engine_ids, etc.), not only via verifyGraph().
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;

private:
    void executeGpuGraph(hipdnnHandle_t handle,
                         hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        int64_t workspaceSize;
        auto result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0) << result.err_msg;
        hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack = bundle.toDeviceVariantPack();
        result = graph.execute(handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
    }

    void executeCpuGraph(hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        auto [serializedGraph, serErr] = graph.to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
            serializedGraph.data(), serializedGraph.size(), bundle.toHostVariantPack());
    }

    void tryAddTensorToBundles(
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& tensorAttr,
        hipdnn_test_sdk::utilities::GraphTensorBundle& cpuBundle,
        hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle)
    {
        int64_t tensorId = tensorAttr->get_uid();

        if(tensorAttr->get_is_virtual()
           || cpuBundle.tensors.find(tensorId) != cpuBundle.tensors.end())
        {
            return;
        }

        cpuBundle.addTensor(*tensorAttr,
                            hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr));
        gpuBundle.addTensor(*tensorAttr,
                            hipdnn_test_sdk::utilities::createTensorFromAttribute(*tensorAttr));
    }
};

// NOLINTEND (portability-template-virtual-member-function)

} // namespace hip_kernel_provider::test_utilities
