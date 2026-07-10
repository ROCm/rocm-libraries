// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <filesystem>
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

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

namespace hip_kernel_provider::batchnorm::test
{

namespace
{

// Regression coverage for RFC 0016 runtime pass-by-value scalars: proves that a
// batchnorm-inference-with-variance graph's epsilon, delivered at execute time as a
// HOST scalar (set_as_runtime_parameter(), no baked value), is actually read and used
// by the hip-kernel provider rather than ignored or defaulted.
class IntegrationGpuBatchnormPassByValueEpsilon : public ::testing::Test
{
protected:
    enum class EpsilonMode
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

    // Builds a batchnorm-inference-with-variance graph with epsilon created per `mode`,
    // fills every tensor (except epsilon) with the same fixed-seed data every call, executes,
    // and returns the host copy of `y`. Epsilon enters the output directly as
    // `1/sqrt(var+eps)`, giving a clean numeric oracle for whether the delivered scalar
    // was actually used.
    std::vector<float> runInferenceVariance(EpsilonMode mode, float epsilonValue)
    {
        constexpr unsigned int SEED = 42;
        const std::vector<int64_t> dims = {2, 8, 4, 4};
        const std::vector<int64_t> derivedDims = {1, 8, 1, 1};
        const TensorLayout layout = TensorLayout::NCHW;

        Graph graphObj;
        graphObj.set_name("BatchnormPassByValueEpsilonTest");
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(hipdnn_frontend::DataType::FLOAT);

        auto xTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("X", dims, generateStrides(dims, layout.strideOrder)));
        auto meanTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("mean", derivedDims, generateStrides(derivedDims)));
        auto varianceTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("variance", derivedDims, generateStrides(derivedDims)));
        auto scaleTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("scale", derivedDims, generateStrides(derivedDims)));
        auto biasTensorAttr = std::make_shared<TensorAttributes>(
            makeTensorAttributes("bias", derivedDims, generateStrides(derivedDims)));

        auto epsilonTensorAttr = std::make_shared<TensorAttributes>();
        epsilonTensorAttr->set_name("epsilon").set_dim({1}).set_stride({1}).set_data_type(
            hipdnn_frontend::DataType::FLOAT);
        if(mode == EpsilonMode::COMPILE_TIME)
        {
            epsilonTensorAttr->set_compile_time_constant(epsilonValue);
        }
        else
        {
            epsilonTensorAttr->set_as_runtime_parameter();
        }

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

        auto result = graphObj.build(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        if(result.code != ErrorCode::OK)
        {
            return {};
        }

        GraphTensorBundle bundle;
        graphObj.visit([&](const INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(!tensorAttr->get_is_virtual()
                   && bundle.tensors.find(tensorAttr->get_uid()) == bundle.tensors.end())
                {
                    bundle.tensors.insert(
                        {tensorAttr->get_uid(), createTensorFromAttribute(*tensorAttr)});
                }
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                if(!tensorAttr->get_is_virtual()
                   && bundle.tensors.find(tensorAttr->get_uid()) == bundle.tensors.end()
                   && tensorAttr->get_uid() != epsilonTensorAttr->get_uid())
                {
                    bundle.tensors.insert(
                        {tensorAttr->get_uid(), createTensorFromAttribute(*tensorAttr)});
                }
            }
        });

        for(auto& [uid, tensor] : bundle.tensors)
        {
            if(uid == varianceTensorAttr->get_uid())
            {
                // Variance must be non-negative and bounded so 1e-3 vs 5.0 epsilon
                // produce a clearly distinguishable inverse-sqrt result.
                bundle.randomizeTensor(uid, 0.1f, 1.0f, SEED);
            }
            else
            {
                bundle.randomizeTensor(uid, -1.0f, 1.0f, SEED);
            }
        }

        int64_t workspaceSize = 0;
        result = graphObj.get_workspace_size(workspaceSize);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        const Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack = bundle.toDeviceVariantPack();

        float hostEpsilon = epsilonValue;
        if(mode == EpsilonMode::RUNTIME_USER)
        {
            // Pure runtime-user-supplied: deliver the scalar as a HOST pointer, matching
            // the frontend's requirement that pass-by-value uids carry host-readable data.
            variantPack[epsilonTensorAttr->get_uid()] = &hostEpsilon;
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

        auto& yTensor = bundle.tensors.at(yTensorAttr->get_uid());
        yTensor->markDeviceModified();
        const auto* hostData = static_cast<const float*>(yTensor->rawHostData());
        return {hostData, hostData + yTensor->elementCount()};
    }

    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
};

} // namespace

TEST_F(IntegrationGpuBatchnormPassByValueEpsilon, RuntimeUserSuppliedMatchesCompileTimeConstant)
{
    auto compileTimeResult = runInferenceVariance(EpsilonMode::COMPILE_TIME, 1e-3f);
    auto runtimeUserResult = runInferenceVariance(EpsilonMode::RUNTIME_USER, 1e-3f);

    ASSERT_EQ(compileTimeResult.size(), runtimeUserResult.size());
    ASSERT_FALSE(compileTimeResult.empty());
    for(size_t i = 0; i < compileTimeResult.size(); ++i)
    {
        EXPECT_NEAR(compileTimeResult[i], runtimeUserResult[i], 1e-4f)
            << "Mismatch at element " << i;
    }
}

TEST_F(IntegrationGpuBatchnormPassByValueEpsilon, RuntimeUserSuppliedValueActuallyFlowsThrough)
{
    auto smallEpsilonResult = runInferenceVariance(EpsilonMode::RUNTIME_USER, 1e-3f);
    auto largeEpsilonResult = runInferenceVariance(EpsilonMode::RUNTIME_USER, 5.0f);

    ASSERT_EQ(smallEpsilonResult.size(), largeEpsilonResult.size());
    ASSERT_FALSE(smallEpsilonResult.empty());

    bool foundDifference = false;
    for(size_t i = 0; i < smallEpsilonResult.size(); ++i)
    {
        if(std::abs(smallEpsilonResult[i] - largeEpsilonResult[i]) > 1e-3f)
        {
            foundDifference = true;
            break;
        }
    }
    EXPECT_TRUE(foundDifference)
        << "Runtime epsilon value did not affect output; scalar may not be flowing through";
}

} // namespace hip_kernel_provider::batchnorm::test
