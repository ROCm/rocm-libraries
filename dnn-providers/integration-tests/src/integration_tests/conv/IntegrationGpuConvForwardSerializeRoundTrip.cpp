// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU serialize/deserialize round-trip for conv forward. The test branches on the
// observed result of to_binary() rather than predicting the engine's behavior: if
// the plan was serializable it round-trips and the lifted graph executes directly;
// otherwise to_binary() emits a graph-only blob plus a degradation warning and the
// lifted graph is freshly built before executing. The presence of the warning and
// the presence of a restored plan are cross-checked for consistency. Either way the
// lifted graph's GPU output is compared against a reference computed from the
// original graph.

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>

#include "common/ConvolutionCommon.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;
using namespace test_conv_common;

namespace
{

// Mirrors IntegrationGpuConvForward.cpp's case tuple (its definition lives in an
// anonymous namespace and is not reachable across translation units).
using ConvFwdTestCase = std::tuple<TensorLayout, test_conv_common::ConvTestCase>;

template <typename DataType>
class ConvForwardSerializeRoundTrip
    : public IntegrationGraphVerificationHarness<DataType, ConvFwdTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> y;
    };

    // Builds the identical conv-fprop graph as IntegrationGpuConvForward.cpp.
    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const ConvFwdTestCase& tc)
    {
        const auto& [layout, testCase] = tc;

        hipdnn_frontend::graph::Graph graphObj;
        graphObj.set_name("ConvolutionForwardSerializeRoundTrip");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        auto xAttr = graph::makeTensorAttributes(
            "x", testCase.xDims, generateStrides(testCase.xDims, layout.strideOrder));
        auto xTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(xAttr));

        auto wAttr = graph::makeTensorAttributes(
            "w", testCase.wDims, generateStrides(testCase.wDims, layout.strideOrder));
        auto wTensorAttr = std::make_shared<graph::TensorAttributes>(std::move(wAttr));

        graph::ConvFpropAttributes convAttrs;
        convAttrs.set_pre_padding(testCase.convPrePadding);
        convAttrs.set_post_padding(testCase.convPostPadding);
        convAttrs.set_stride(testCase.convStride);
        convAttrs.set_dilation(testCase.convDilation);

        auto yAttr = graphObj.conv_fprop(xTensorAttr, wTensorAttr, convAttrs);
        yAttr->set_output(true);

        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            throw std::runtime_error("Failed to validate graph: " + validateResult.get_message());
        }

        auto buildResult = graphObj.build_operation_graph(handle);
        if(buildResult.is_bad())
        {
            throw std::runtime_error("Failed to build operation graph: "
                                     + buildResult.get_message());
        }

        return std::make_pair(std::move(graphObj), GraphOutputs{yAttr});
    }

protected:
    void runGraphTest() override
    {
        // rocBLAS/Tensile heap-buffer-overflow on gfx90a; CK ASAN stall on gfx942
        SKIP_IF_ASAN();

        const auto& testCase = this->GetParam();
        const auto& [layout, convTestCase] = testCase;

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->setTestCaseLayout(layout.name);
        this->setTestCaseNote(convTestCase.note);

        // GPU and reference bundles share a seed so their inputs match.
        GraphTensorBundle gpuBundle;
        GraphTensorBundle refBundle;
        this->generateBundles(graphObj, refBundle, gpuBundle);
        this->initializeBundle(graphObj, gpuBundle, convTestCase.seed);
        this->initializeBundle(graphObj, refBundle, convTestCase.seed);

        // Finalize a plan on the original graph, then compute the reference.
        ASSERT_EQ(graphObj.build(getSharedHandle()).code, hipdnn_frontend::ErrorCode::OK);
        ASSERT_NO_FATAL_FAILURE(this->executeReferenceGraph(graphObj, refBundle));

        // Branch on the observed outcome of to_binary() rather than predicting it
        // from the ranked engine, which can diverge from the engine the built plan
        // actually used. A SYNC WARN recorder around to_binary() captures whether
        // the plan-degradation warning fired; the recorder must stay alive across
        // the branch below so its logs remain available.
        auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
        auto cbErr = setUserLogCallback(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                                        HIPDNN_SEV_WARN,
                                        LogCallbackMode::SYNC,
                                        this);
        ASSERT_EQ(cbErr.code, ErrorCode::OK);
        auto lvlErr = setGlobalLogLevel(HIPDNN_SEV_WARN);
        ASSERT_EQ(lvlErr.code, ErrorCode::OK);

        auto [blob, serErr] = graphObj.to_binary();
        ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
        ASSERT_FALSE(blob.empty());

        const bool warned = recorder.hasLogContaining(
            HIPDNN_SEV_WARN, "does not support execution plan serialization");

        // Stop recording before the rest of the test runs; the recorder object
        // itself stays in scope for getRecordedLogsAsString() below.
        setUserLogCallback(IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
                           HIPDNN_SEV_OFF,
                           LogCallbackMode::SYNC,
                           this);

        hipdnn_tests::TestableGraphLifting lifted;
        ASSERT_EQ(lifted.deserialize(getSharedHandle(), blob).code, hipdnn_frontend::ErrorCode::OK);

        if(lifted.hasExecutionPlan())
        {
            // Plan was serializable: it round-tripped and is ready to execute
            // directly. No degradation warning should have fired.
            EXPECT_FALSE(warned) << "Plan was serialized, so no degradation warning is expected.";
        }
        else
        {
            // Graph-only: serialize warned and dropped the plan; rebuild before exec.
            EXPECT_TRUE(warned)
                << "Graph-only serialization must warn that the plan was not captured.\n"
                << "Recorded logs:\n"
                << recorder.getRecordedLogsAsString();
            ASSERT_EQ(lifted.build(getSharedHandle()).code, hipdnn_frontend::ErrorCode::OK);
        }

        // Execute the deserialized graph and compare to the reference.
        ASSERT_NO_FATAL_FAILURE(this->executeGpuGraph(getSharedHandle(), lifted, gpuBundle));

        const bool referenceUsesDevice
            = IntegrationGraphVerificationHarness<DataType, ConvFwdTestCase>::getReferenceExecutor()
                  .requiresDeviceMemory();

        const int64_t yUid = outputs.y->get_uid();
        ASSERT_GT(gpuBundle.outputTensorIds.count(yUid), 0u)
            << "Output tensor was not registered in the bundle";

        auto& refTensor = refBundle.tensors.at(yUid);
        auto& gpuTensor = gpuBundle.tensors.at(yUid);

        // Mark device data dirty so host access triggers a device-to-host sync
        // before the comparison reads it.
        gpuTensor->markDeviceModified();
        if(referenceUsesDevice)
        {
            refTensor->markDeviceModified();
        }

        const float tolerance = this->getTolerance(graphObj, outputs.y);
        auto validator = createAllCloseValidator(
            frontendToSdkDataType(outputs.y->get_data_type()), tolerance, tolerance);
        ASSERT_TRUE(validator->allClose(*refTensor, *gpuTensor))
            << "Mismatch in output tensor after serialize round-trip (uid: " << yUid << ")";
    }
};

using IntegrationGpuConvFwdSerializeRoundTripFp32 = ConvForwardSerializeRoundTrip<float>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuConvFwdSerializeRoundTripFp32);
TEST_P(IntegrationGpuConvFwdSerializeRoundTripFp32, SerializeRoundTripExecutesCorrectly)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    IntegrationGpuConvFwdSerializeRoundTripFp32,
    testing::Combine(testing::Values(TensorLayout::NCHW, TensorLayout::NHWC),
                     testing::ValuesIn(test_conv_common::getConvTestCases4D())));
