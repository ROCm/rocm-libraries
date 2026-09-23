// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <filesystem>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "../tests/common/TestWorkarounds.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

namespace
{

// Generic regression suite for runtime pass-by-value scalars in the MIOpen provider.
// It exercises every op the MIOpen provider serves that consumes a pass-by-value scalar
// tensor: batchnorm-inference-with-variance epsilon and batchnorm-forward-training epsilon.
//
// Each op is tested with the same two-part oracle. For a given scalar value the op is run
// twice: once with the scalar baked as a compile-time constant, once delivered at execute as
// a HOST scalar (set_as_runtime_parameter(), no baked value). The runtime host scalar must
// produce the SAME output as the equal-valued compile-time constant -- proving the provider
// reads the delivered scalar rather than ignoring or defaulting it. Then the op is run with
// two materially different runtime values, which must produce DIFFERENT output -- proving the
// delivered value actually flows through.
//
// Momentum (the other pass-by-value scalar) is intentionally NOT covered: MIOpen's batchnorm plan
// builder declines forward-training graphs that carry running statistics
// (MiopenBatchnormPlanBuilder::isApplicable returns false when prev/next running mean+variance
// and momentum are all set), so MIOpen never executes the momentum path and such a test would
// only SKIP. Momentum runtime coverage lives in the hip-kernel provider, whose hip_mlops
// engine does serve full-training running-stats graphs. Layernorm and rmsnorm are likewise
// absent because MIOpen has no plans for those ops.
class IntegrationGpuPassByValue : public ::testing::Test
{
protected:
    enum class PbvMode
    {
        COMPILE_TIME,
        RUNTIME_USER
    };

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
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    // Builds a FLOAT scalar (1-element) tensor as either a baked compile-time constant or a
    // pure runtime-user-supplied parameter (no baked value), per `mode`.
    static std::shared_ptr<TensorAttributes>
        makeScalarTensor(const std::string& name, PbvMode mode, float value)
    {
        auto attr = std::make_shared<TensorAttributes>();
        attr->set_name(name).set_dim({1}).set_stride({1}).set_data_type(
            hipdnn_frontend::DataType::FLOAT);
        if(mode == PbvMode::COMPILE_TIME)
        {
            attr->set_compile_time_constant(value);
        }
        else
        {
            attr->set_as_runtime_parameter();
        }
        return attr;
    }

    // Shared build/fill/execute/read runner. Builds `graphObj`, allocates device tensors for
    // every non-virtual tensor except the pass-by-value scalars in `runtimeHostScalars`
    // (whose uids are delivered as HOST pointers only in RUNTIME_USER mode), fills tensors
    // with fixed-seed data (tensors in `nonNegInputs` in [0.1, 1.0] so variance-like inputs
    // stay valid, everything else in [-1, 1]), executes, and returns the host copy of the
    // tensors in `outputs`, concatenated in order.
    //
    // Tensors are passed as attribute handles (not uids): a graph tensor's uid is only
    // assigned during build(), so callers cannot capture uids up front -- they are read here,
    // after build().
    std::vector<float> buildFillExecuteRead(
        Graph& graphObj,
        PbvMode mode,
        const std::vector<std::pair<std::shared_ptr<TensorAttributes>, float>>& runtimeHostScalars,
        const std::vector<std::shared_ptr<TensorAttributes>>& nonNegInputs,
        const std::vector<std::shared_ptr<TensorAttributes>>& outputs,
        unsigned int seed)
    {
        auto result = graphObj.build(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        if(result.code != ErrorCode::OK)
        {
            return {};
        }

        // Resolve uids now that build() has assigned them.
        std::unordered_map<int64_t, float> runtimeHostScalarsByUid;
        for(const auto& [scalarAttr, value] : runtimeHostScalars)
        {
            runtimeHostScalarsByUid.emplace(scalarAttr->get_uid(), value);
        }
        std::vector<int64_t> nonNegUids;
        nonNegUids.reserve(nonNegInputs.size());
        for(const auto& attr : nonNegInputs)
        {
            nonNegUids.push_back(attr->get_uid());
        }

        GraphTensorBundle bundle;
        graphObj.visit([&](const INode& node) {
            auto addTensor = [&](const std::shared_ptr<TensorAttributes>& tensorAttr) {
                if(tensorAttr->get_is_virtual()
                   || bundle.tensors.find(tensorAttr->get_uid()) != bundle.tensors.end()
                   || runtimeHostScalarsByUid.find(tensorAttr->get_uid())
                          != runtimeHostScalarsByUid.end())
                {
                    return;
                }
                bundle.addTensor(*tensorAttr, createTensorFromAttribute(*tensorAttr));
            };
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                addTensor(tensorAttr);
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                addTensor(tensorAttr);
            }
        });

        for(auto& [uid, tensor] : bundle.tensors)
        {
            if(std::find(nonNegUids.begin(), nonNegUids.end(), uid) != nonNegUids.end())
            {
                bundle.randomizeTensor(uid, 0.1f, 1.0f, seed);
            }
            else
            {
                bundle.randomizeTensor(uid, -1.0f, 1.0f, seed);
            }
        }

        int64_t workspaceSize = 0;
        result = graphObj.get_workspace_size(workspaceSize);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        const Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack = bundle.toDeviceVariantPack();

        // Host scalar storage must outlive execute(); keep it stable for the whole call.
        std::vector<float> hostScalarStorage;
        hostScalarStorage.reserve(runtimeHostScalarsByUid.size());
        if(mode == PbvMode::RUNTIME_USER)
        {
            for(const auto& [uid, value] : runtimeHostScalarsByUid)
            {
                hostScalarStorage.push_back(value);
                // Pure runtime-user-supplied: deliver the scalar as a HOST pointer, matching
                // the frontend's requirement that pass-by-value uids carry host-readable data.
                variantPack[uid] = &hostScalarStorage.back();
            }
        }

        result = graphObj.execute(_handle, variantPack, workspace.get());
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        if(result.code != ErrorCode::OK)
        {
            return {};
        }

        const hipError_t syncStatus = hipStreamSynchronize(_stream);
        EXPECT_EQ(syncStatus, hipSuccess);
        if(syncStatus != hipSuccess)
        {
            return {};
        }

        std::vector<float> out;
        for(const auto& outputAttr : outputs)
        {
            auto& tensor = bundle.tensors.at(outputAttr->get_uid());
            tensor->markDeviceModified();
            const auto* hostData = static_cast<const float*>(tensor->rawHostData());
            out.insert(out.end(), hostData, hostData + tensor->elementCount());
        }
        return out;
    }

    // --- Per-op graph builders + runners ---

    // Batchnorm inference with variance: epsilon enters y directly via 1/sqrt(var+eps).
    std::vector<float> runBnInferenceVarianceEpsilon(PbvMode mode, float epsilonValue)
    {
        constexpr unsigned int SEED = 42;
        const std::vector<int64_t> dims = {2, 8, 4, 4};
        const std::vector<int64_t> derivedDims = {1, 8, 1, 1};

        Graph graphObj;
        graphObj.set_name("BnInferenceVarianceEpsilonPbv");
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(hipdnn_frontend::DataType::FLOAT);

        auto xTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("X", dims, generateStrides(dims, TensorLayout::NCHW.strideOrder)));
        auto meanTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("mean", derivedDims, generateStrides(derivedDims)));
        auto varianceTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("variance", derivedDims, generateStrides(derivedDims)));
        auto scaleTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("scale", derivedDims, generateStrides(derivedDims)));
        auto biasTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("bias", derivedDims, generateStrides(derivedDims)));
        auto epsilonTensorAttr = makeScalarTensor("epsilon", mode, epsilonValue);

        const BatchnormInferenceAttributesVarianceExt bnAttrs;
        auto yTensorAttr = graphObj.batchnorm_inference_variance_ext(xTensorAttr,
                                                                     meanTensorAttr,
                                                                     varianceTensorAttr,
                                                                     scaleTensorAttr,
                                                                     biasTensorAttr,
                                                                     epsilonTensorAttr,
                                                                     bnAttrs);
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);

        return buildFillExecuteRead(graphObj,
                                    mode,
                                    {{epsilonTensorAttr, epsilonValue}},
                                    {varianceTensorAttr},
                                    {yTensorAttr},
                                    SEED);
    }

    // Batchnorm forward training WITHOUT running stats: epsilon enters inv_variance
    // (1/sqrt(var+eps)) and therefore y.
    std::vector<float> runBnTrainingEpsilon(PbvMode mode, float epsilonValue)
    {
        constexpr unsigned int SEED = 11;
        const std::vector<int64_t> dims = {2, 8, 4, 4};
        const std::vector<int64_t> derivedDims = {1, 8, 1, 1};

        Graph graphObj;
        graphObj.set_name("BnTrainingEpsilonPbv");
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(hipdnn_frontend::DataType::FLOAT);

        auto xTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("X", dims, generateStrides(dims, TensorLayout::NCHW.strideOrder)));
        auto scaleTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("scale", derivedDims, generateStrides(derivedDims)));
        auto biasTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("bias", derivedDims, generateStrides(derivedDims)));
        auto epsilonTensorAttr = makeScalarTensor("epsilon", mode, epsilonValue);

        BatchnormAttributes bnAttrs;
        bnAttrs.set_epsilon(epsilonTensorAttr);

        auto [yTensorAttr, meanTensorAttr, invVarianceTensorAttr, nextMean, nextVar]
            = graphObj.batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);
        yTensorAttr->set_output(true);
        yTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);
        meanTensorAttr->set_output(true);
        meanTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);
        invVarianceTensorAttr->set_output(true);
        invVarianceTensorAttr->set_data_type(hipdnn_frontend::DataType::FLOAT);

        return buildFillExecuteRead(
            graphObj, mode, {{epsilonTensorAttr, epsilonValue}}, {}, {invVarianceTensorAttr}, SEED);
    }

    // Asserts the runtime-user-supplied scalar reproduces the compile-time-constant output
    // and that two different runtime values yield different output.
    static void expectScalarFlowsThrough(const std::vector<float>& compileTime,
                                         const std::vector<float>& runtimeSame,
                                         const std::vector<float>& runtimeDifferent,
                                         const char* scalarName)
    {
        ASSERT_EQ(compileTime.size(), runtimeSame.size());
        ASSERT_EQ(compileTime.size(), runtimeDifferent.size());
        ASSERT_FALSE(compileTime.empty());

        for(size_t i = 0; i < compileTime.size(); ++i)
        {
            EXPECT_NEAR(compileTime[i], runtimeSame[i], 1e-4f)
                << scalarName << ": runtime host scalar diverged from equal compile-time constant "
                << "at element " << i;
        }

        bool foundDifference = false;
        for(size_t i = 0; i < runtimeSame.size(); ++i)
        {
            if(std::abs(runtimeSame[i] - runtimeDifferent[i]) > 1e-3f)
            {
                foundDifference = true;
                break;
            }
        }
        EXPECT_TRUE(foundDifference)
            << scalarName << ": differing runtime values produced identical output; the scalar "
            << "may not be flowing through";
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

} // namespace

TEST_F(IntegrationGpuPassByValue, BatchnormInferenceVarianceEpsilon)
{
    expectScalarFlowsThrough(runBnInferenceVarianceEpsilon(PbvMode::COMPILE_TIME, 1e-3f),
                             runBnInferenceVarianceEpsilon(PbvMode::RUNTIME_USER, 1e-3f),
                             runBnInferenceVarianceEpsilon(PbvMode::RUNTIME_USER, 5.0f),
                             "batchnorm-inference-variance epsilon");
}

TEST_F(IntegrationGpuPassByValue, BatchnormTrainingEpsilon)
{
    expectScalarFlowsThrough(runBnTrainingEpsilon(PbvMode::COMPILE_TIME, 1e-3f),
                             runBnTrainingEpsilon(PbvMode::RUNTIME_USER, 1e-3f),
                             runBnTrainingEpsilon(PbvMode::RUNTIME_USER, 5.0f),
                             "batchnorm-training epsilon");
}
