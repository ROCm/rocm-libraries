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
#include <string>
#include <unordered_map>
#include <vector>

namespace hip_kernel_provider::test_utilities
{

/// Per-graph verification state that survives repeated executions. The graph must
/// outlive its context.
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
        /// A caller-supplied validator is never rebuilt from the tolerances above; any
        /// later registration for the same output is rejected.
        bool suppliedByCaller = false;
        std::shared_ptr<hipdnn_test_sdk::utilities::IReferenceValidation> validator;
    };

    hipdnn_frontend::graph::Graph& _graph;
    std::unordered_map<std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>, Registration>
        _registrations;
};

// The tensors and output validators of one verification. It keeps no reference to a graph,
// a context or a handle. It stays valid after all three are destroyed.
struct VerificationRun
{
    struct Output
    {
        int64_t uid = 0;
        std::string name;
        std::shared_ptr<hipdnn_test_sdk::utilities::IReferenceValidation> validator;
    };

    hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle;
    hipdnn_test_sdk::utilities::GraphTensorBundle cpuBundle;
    std::vector<Output> outputs;
    // The first executeAndValidate call computes the CPU reference and sets this flag.
    bool referenceReady = false;
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
        VerificationRun run;
        ASSERT_NO_FATAL_FAILURE(prepareVerification(context, seed, run));
        ASSERT_NO_FATAL_FAILURE(executeAndValidate(run, context._graph, _handle));
    }

    // Split in two so a test can work out the right answers once, then check several runs
    // against them, even runs of a plan loaded after the original graph and handle are gone.

    // Creates and seeds the tensors of the context's graph and resolves a validator for each
    // output. It seeds the GPU tensors before the CPU tensors. It runs no graph.
    void prepareVerification(GraphVerificationContext& context,
                             unsigned int seed,
                             VerificationRun& run)
    {
        run = VerificationRun{};
        auto& graph = context._graph;
        std::vector<OutputTensor> outputs;

        generateBundles(graph, run.cpuBundle, run.gpuBundle, outputs);

        initializeBundle(graph, run.gpuBundle, seed);
        initializeBundle(graph, run.cpuBundle, seed);

        ASSERT_NO_FATAL_FAILURE(resolveOutputValidators(context, outputs));

        for(const auto& output : outputs)
        {
            run.gpuBundle.outputTensorIds.insert(output.uid);
            run.outputs.push_back({output.uid,
                                   output.attr->get_name(),
                                   context._registrations.at(output.attr).validator});
        }
    }

    // Executes graph on the GPU and compares each output with the CPU reference. The first
    // call computes the reference from its graph, after the GPU run. Later calls use that
    // reference again and do not serialize graph. Before a later call executes, it fills each
    // GPU output with a sentinel, so an output that the execute does not write fails.
    void executeAndValidate(VerificationRun& run,
                            hipdnn_frontend::graph::Graph& graph,
                            hipdnnHandle_t handle)
    {
        ASSERT_FALSE(run.outputs.empty())
            << "The run has no outputs. Call prepareVerification() first.";

        if(run.referenceReady)
        {
            run.gpuBundle.sentinelFillOutputTensors();
        }
        ASSERT_NO_FATAL_FAILURE(executeGpuGraph(handle, graph, run.gpuBundle));

        if(!run.referenceReady)
        {
            ASSERT_NO_FATAL_FAILURE(executeCpuGraph(graph, run.cpuBundle));
            run.referenceReady = true;
        }

        HIPDNN_PLUGIN_LOG_INFO("Validating " << run.outputs.size() << " output tensors");

        for(const auto& output : run.outputs)
        {
            auto& cpuTensor = run.cpuBundle.tensors.at(output.uid);
            auto& gpuTensor = run.gpuBundle.tensors.at(output.uid);
            gpuTensor->markDeviceModified();

            const bool valid = output.validator->allClose(*cpuTensor, *gpuTensor);
            ASSERT_TRUE(valid) << "Mismatch found in tensor with id: " << output.uid
                               << ", name: " << output.name;
        }
    }

    /// Registers each nonvirtual output in this graph's context.
    ///
    /// @p epsilonMultiple is expressed in epsilons of this fixture's element type, so the
    /// same value means the same thing for FLOAT and HALF fixtures: a K-term sum needs
    /// ~K of them, an elementwise op one.
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

    void assertOutputBelongsToGraph(
        GraphVerificationContext& context,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& attr)
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
    }

    /// Refuses any registration that would displace a caller-supplied comparator and
    /// answers whether it was refused. Fails nonfatally so a suite sees every offending
    /// output, not only the first.
    static bool refuseDuplicateOfCallerValidator(
        const GraphVerificationContext::Registration& registration,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& attr)
    {
        if(!registration.suppliedByCaller)
        {
            return false;
        }
        ADD_FAILURE() << "Duplicate validator for tensor " << attr->get_uid() << " ("
                      << attr->get_name() << "); keeping first registration";
        return true;
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
        ASSERT_NO_FATAL_FAILURE(assertOutputBelongsToGraph(context, attr));

        auto& registration = context._registrations[attr];
        if(refuseDuplicateOfCallerValidator(registration, attr))
        {
            return;
        }
        if(registration.absoluteTolerance != absoluteTolerance
           || registration.relativeTolerance != relativeTolerance)
        {
            registration.validator.reset();
        }
        registration.absoluteTolerance = absoluteTolerance;
        registration.relativeTolerance = relativeTolerance;
    }

    /// Registers a caller-built comparator for one output whose correct values the
    /// tolerance-based default validator cannot express. One comparator per output: a
    /// second registration is rejected.
    void registerValidator(
        GraphVerificationContext& context,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& attr,
        std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation> validator)
    {
        ASSERT_NE(validator, nullptr);
        ASSERT_NO_FATAL_FAILURE(assertOutputBelongsToGraph(context, attr));

        auto& registration = context._registrations[attr];
        if(refuseDuplicateOfCallerValidator(registration, attr))
        {
            return;
        }
        registration.validator = std::move(validator);
        registration.suppliedByCaller = true;
    }

    struct OutputTensor
    {
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr;
        int64_t uid;
        hipdnn_frontend::DataType dataType;
    };

    // Check the entire output set before any comparator reads its storage.
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
            if(registration.suppliedByCaller)
            {
                continue;
            }
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

    /// Seeds every tensor in @p bundle alike. A suite whose operation can confuse two
    /// identically-seeded operands overrides this instead of changing it: other suites
    /// calibrate their tolerances against this seeding.
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

        // execute() only enqueues. The readback copies on the tensor's own stream, which
        // is not ordered against a caller-selected stream such as hipStreamPerThread.
        ASSERT_EQ(hipStreamSynchronize(stream()), hipSuccess);
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
