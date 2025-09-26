// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace ::testing;

template <typename InputDataType, typename ScaleBiasDataType, typename MeanVarianceDataType>
struct BatchnormTrainTensorBundle
{
    BatchnormTrainTensorBundle(const std::vector<int64_t>& dims,
                               unsigned int seed = 1,
                               const TensorLayout& layout = TensorLayout::NCHW,
                               bool useOptionalTensors = false)
        : derivedDims(getDerivedShape(dims))
        , xTensor(dims, layout)
        , scaleTensor(derivedDims)
        , biasTensor(derivedDims)
        , meanTensor(derivedDims)
        , invVarianceTensor(derivedDims)
        , epsilonTensor({1})
        //, momentumTensor({1})
        , yTensor(dims, layout)
    {
        xTensor.fillWithRandomValues(
            static_cast<InputDataType>(-1.0f), static_cast<InputDataType>(1.0f), seed);

        scaleTensor.fillWithRandomValues(
            static_cast<ScaleBiasDataType>(-0.1f), static_cast<ScaleBiasDataType>(0.1f), seed);
        biasTensor.fillWithRandomValues(
            static_cast<ScaleBiasDataType>(-0.1f), static_cast<ScaleBiasDataType>(0.1f), seed);

        meanTensor.fillWithRandomValues(static_cast<MeanVarianceDataType>(-0.1f),
                                        static_cast<MeanVarianceDataType>(0.1f),
                                        seed);

        invVarianceTensor.fillWithRandomValues(
            static_cast<MeanVarianceDataType>(1.9f), static_cast<MeanVarianceDataType>(2.0f), seed);

        epsilonTensor.fillWithValue(static_cast<MeanVarianceDataType>(1e-5f));

        if(useOptionalTensors)
        {
            momentumTensor = Tensor<MeanVarianceDataType>({1});
            momentumTensor->fillWithValue(static_cast<MeanVarianceDataType>(0.1f));

            prevRunningMeanTensor = Tensor<MeanVarianceDataType>(derivedDims);
            prevRunningMeanTensor->fillWithRandomValues(static_cast<MeanVarianceDataType>(-0.1f),
                                                        static_cast<MeanVarianceDataType>(0.1f),
                                                        seed);

            prevRunningVarianceTensor = Tensor<MeanVarianceDataType>(derivedDims);
            prevRunningVarianceTensor->fillWithRandomValues(static_cast<MeanVarianceDataType>(1.9f),
                                                            static_cast<MeanVarianceDataType>(2.0f),
                                                            seed);

            nextRunningMeanTensor = Tensor<MeanVarianceDataType>(derivedDims);
            nextRunningVarianceTensor = Tensor<MeanVarianceDataType>(derivedDims);
        }
    }

    std::unordered_map<int64_t, void*> createVariantPack(
        const hipdnn_frontend::graph::TensorAttributes& xTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& scaleTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& biasTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& meanTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& invVarianceTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& epsilonTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& yTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& momentumTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& prevRunningMeanTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>&
            prevRunningVarianceTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& nextRunningMeanTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>&
            nextRunningVarianceTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;

        variantPack[xTensorAttr.get_uid()] = xTensor.memory().hostData();
        variantPack[scaleTensorAttr.get_uid()] = scaleTensor.memory().hostData();
        variantPack[biasTensorAttr.get_uid()] = biasTensor.memory().hostData();
        variantPack[meanTensorAttr.get_uid()] = meanTensor.memory().hostData();
        variantPack[invVarianceTensorAttr.get_uid()] = invVarianceTensor.memory().hostData();
        variantPack[epsilonTensorAttr.get_uid()] = epsilonTensor.memory().hostData();
        variantPack[yTensorAttr.get_uid()] = yTensor.memory().hostData();

        //optionals
        if(momentumTensorAttr != nullptr)
        {
            variantPack[momentumTensorAttr->get_uid()] = momentumTensor.value().memory().hostData();
        }

        if(prevRunningMeanTensorAttr != nullptr)
        {
            variantPack[prevRunningMeanTensorAttr->get_uid()]
                = prevRunningMeanTensor.value().memory().hostData();
        }

        if(prevRunningVarianceTensorAttr != nullptr)
        {
            variantPack[prevRunningVarianceTensorAttr->get_uid()]
                = prevRunningVarianceTensor.value().memory().hostData();
        }

        if(nextRunningMeanTensorAttr != nullptr)
        {
            variantPack[nextRunningMeanTensorAttr->get_uid()]
                = nextRunningMeanTensor.value().memory().hostData();
        }

        if(nextRunningVarianceTensorAttr != nullptr)
        {
            variantPack[nextRunningVarianceTensorAttr->get_uid()]
                = nextRunningVarianceTensor.value().memory().hostData();
        }

        return variantPack;
    }

    std::vector<int64_t> derivedDims;
    Tensor<InputDataType> xTensor;
    Tensor<ScaleBiasDataType> scaleTensor;
    Tensor<ScaleBiasDataType> biasTensor;
    Tensor<MeanVarianceDataType> meanTensor;
    Tensor<MeanVarianceDataType> invVarianceTensor;
    Tensor<MeanVarianceDataType> epsilonTensor;
    Tensor<InputDataType> yTensor;

    std::optional<Tensor<MeanVarianceDataType>> momentumTensor;
    std::optional<Tensor<MeanVarianceDataType>> prevRunningMeanTensor;
    std::optional<Tensor<MeanVarianceDataType>> prevRunningVarianceTensor;
    std::optional<Tensor<MeanVarianceDataType>> nextRunningMeanTensor;
    std::optional<Tensor<MeanVarianceDataType>> nextRunningVarianceTensor;
};

class TestCpuReferenceGraphExecutor
{
private:
public:
    static flatbuffers::FlatBufferBuilder createValidBatchnormFwdInferenceGraph(
        std::vector<int64_t> strides,
        std::vector<int64_t> dims,
        bool hasOptionalAttributes = true,
        hipdnn_sdk::data_objects::DataType inputDataType = DataType::FLOAT,
        hipdnn_sdk::data_objects::DataType scaleBiasDataType = DataType::FLOAT,
        hipdnn_sdk::data_objects::DataType meanVarianceDataType = DataType::FLOAT)
    {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>>
            tensorAttributes;

        std::vector<int64_t> derivedStrides = {1, strides[1], 1, 1};
        std::vector<int64_t> derivedDims = {1, dims[1], 1, 1};

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 1, "x", inputDataType, &strides, &dims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 2, "y", inputDataType, &strides, &dims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 3, "scale", scaleBiasDataType, &derivedStrides, &derivedDims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 4, "bias", scaleBiasDataType, &derivedStrides, &derivedDims));

        if(hasOptionalAttributes)
        {
            tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 5, "est_mean", meanVarianceDataType, &derivedStrides, &derivedDims));

            tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 6, "est_variance", meanVarianceDataType, &derivedStrides, &derivedDims));
        }

        auto bnormAttributes
            = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(builder,
                                                                           1, // x uid
                                                                           5, // mean uid
                                                                           6, // inv_variance uid
                                                                           3, // scale uid
                                                                           4, // bias uid
                                                                           2 // y uid
            );

        std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
        auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            "batchnorm",
            hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
            bnormAttributes.Union());
        nodes.push_back(node);

        auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                       "test",
                                                                       DataType::FLOAT,
                                                                       DataType::HALF,
                                                                       DataType::BFLOAT16,
                                                                       &tensorAttributes,
                                                                       &nodes);
        builder.Finish(graphOffset);
        return builder;
    }

    template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
    static void runBatchnormFwdTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                    hipdnn_sdk::data_objects::DataType scaleBiasDataType,
                                    hipdnn_sdk::data_objects::DataType meanVarianceDataType)
    {
        unsigned int seed = std::random_device{}();

        std::vector<int64_t> dims = {1, 3, 14, 14};

        std::vector<int64_t> derivedDims = {1, dims[1]};

        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        TensorLayout layout = TensorLayout::NCHW;

        PinnedTensor<InputType> xTensor(dims, layout);
        deviceBuffers.push_back(generateRandomHostBuffer(
            xTensor, 1, static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed));

        PinnedTensor<InputType> yTensor(dims, layout);
        deviceBuffers.push_back(generateEmptyHostBuffer(yTensor, 2));

        PinnedTensor<ScaleBiasType> scaleTensor(derivedDims);
        deviceBuffers.push_back(generateRandomHostBuffer(scaleTensor,
                                                         3,
                                                         static_cast<ScaleBiasType>(0.0f),
                                                         static_cast<ScaleBiasType>(1.0f),
                                                         seed));

        PinnedTensor<ScaleBiasType> biasTensor(derivedDims);
        deviceBuffers.push_back(generateRandomHostBuffer(biasTensor,
                                                         4,
                                                         static_cast<ScaleBiasType>(0.0f),
                                                         static_cast<ScaleBiasType>(1.0f),
                                                         seed));

        PinnedTensor<MeanVarianceType> meanTensor(derivedDims);
        deviceBuffers.push_back(generateRandomHostBuffer(meanTensor,
                                                         5,
                                                         static_cast<MeanVarianceType>(0.0f),
                                                         static_cast<MeanVarianceType>(1.0f),
                                                         seed));

        PinnedTensor<MeanVarianceType> varianceTensor(derivedDims);
        deviceBuffers.push_back(generateRandomHostBuffer(varianceTensor,
                                                         6,
                                                         static_cast<MeanVarianceType>(0.1f),
                                                         static_cast<MeanVarianceType>(1.0f),
                                                         seed));

        auto batchnormBuilder
            = TestCpuReferenceGraphExecutor::createValidBatchnormFwdInferenceGraph(
                xTensor.strides(),
                xTensor.dims(),
                true,
                inputDataType,
                scaleBiasDataType,
                meanVarianceDataType);

        auto batchnormGraph = batchnormBuilder.GetBufferPointer();

        std::unordered_map<int64_t, void*> variantPack;
        for(const auto& deviceBuffer : deviceBuffers)
        {
            variantPack[deviceBuffer.uid] = deviceBuffer.ptr;
        }

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            batchnormGraph, batchnormBuilder.GetSize(), variantPack);
    }

    template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
    static void runBatchnormBwdTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                    hipdnn_sdk::data_objects::DataType scaleBiasDataType,
                                    hipdnn_sdk::data_objects::DataType meanVarianceDataType)
    {
        unsigned int seed = std::random_device{}();

        std::vector<int64_t> dims = {1, 3, 14, 14};

        std::vector<int64_t> derivedDims = {1, dims[1]};

        std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

        TensorLayout layout = TensorLayout::NCHW;

        PinnedTensor<InputType> xTensor(dims, layout);
        deviceBuffers.push_back(generateRandomHostBuffer(
            xTensor, 1, static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed));

        PinnedTensor<InputType> dyTensor(dims, layout);
        deviceBuffers.push_back(generateRandomHostBuffer(
            dyTensor, 2, static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed));

        PinnedTensor<InputType> dxTensor(dims, layout);
        deviceBuffers.push_back(generateEmptyHostBuffer(dxTensor, 3));

        PinnedTensor<ScaleBiasType> scaleTensor(derivedDims);
        deviceBuffers.push_back(generateRandomHostBuffer(scaleTensor,
                                                         4,
                                                         static_cast<ScaleBiasType>(-0.1f),
                                                         static_cast<ScaleBiasType>(0.1f),
                                                         seed));

        PinnedTensor<ScaleBiasType> dscaleTensor(derivedDims);
        deviceBuffers.push_back(generateEmptyHostBuffer(dscaleTensor, 5));

        PinnedTensor<ScaleBiasType> dbiasTensor(derivedDims);
        deviceBuffers.push_back(generateEmptyHostBuffer(dbiasTensor, 6));

        PinnedTensor<MeanVarianceType> meanTensor(derivedDims);
        deviceBuffers.push_back(generateRandomHostBuffer(meanTensor,
                                                         7,
                                                         static_cast<MeanVarianceType>(-0.1f),
                                                         static_cast<MeanVarianceType>(0.1f),
                                                         seed));

        PinnedTensor<MeanVarianceType> invVarianceTensor(derivedDims);
        deviceBuffers.push_back(generateRandomHostBuffer(invVarianceTensor,
                                                         8,
                                                         static_cast<MeanVarianceType>(1.9f),
                                                         static_cast<MeanVarianceType>(2.0f),
                                                         seed));

        auto batchnormBuilder
            = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph(dyTensor.strides(),
                                                                       dyTensor.dims(),
                                                                       true,
                                                                       inputDataType,
                                                                       scaleBiasDataType,
                                                                       meanVarianceDataType);

        auto batchnormGraph = batchnormBuilder.GetBufferPointer();

        std::unordered_map<int64_t, void*> variantPack;
        for(const auto& deviceBuffer : deviceBuffers)
        {
            variantPack[deviceBuffer.uid] = deviceBuffer.ptr;
        }

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            batchnormGraph, batchnormBuilder.GetSize(), variantPack);
    }

    static hipdnn_frontend::DataType fromSdkType(const hipdnn_sdk::data_objects::DataType& type)
    {
        switch(type)
        {
        case hipdnn_sdk::data_objects::DataType::FLOAT:
            return hipdnn_frontend::DataType::FLOAT;
        case hipdnn_sdk::data_objects::DataType::HALF:
            return hipdnn_frontend::DataType::HALF;
        case hipdnn_sdk::data_objects::DataType::BFLOAT16:
            return hipdnn_frontend::DataType::BFLOAT16;
        case hipdnn_sdk::data_objects::DataType::DOUBLE:
            return hipdnn_frontend::DataType::DOUBLE;
        case hipdnn_sdk::data_objects::DataType::UINT8:
            return hipdnn_frontend::DataType::UINT8;
        case hipdnn_sdk::data_objects::DataType::INT32:
            return hipdnn_frontend::DataType::INT32;
        default:
            return hipdnn_frontend::DataType::NOT_SET;
        }
    }

    template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
    static void runBatchnormTrainTest(hipdnn_sdk::data_objects::DataType inputDataType,
                                      hipdnn_sdk::data_objects::DataType scaleBiasDataType,
                                      hipdnn_sdk::data_objects::DataType meanVarianceDataType,
                                      bool useOptionalTensors = false)
    {
        std::vector<int64_t> dims = {1, 3, 14, 14};
        BatchnormTrainTensorBundle<InputType, ScaleBiasType, MeanVarianceType> graphTensorBundle(
            dims, 1, TensorLayout::NCHW, useOptionalTensors);

        auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
        graph->set_name("BatchnormTrainTest");

        int64_t uid = 1;
        auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "X", fromSdkType(inputDataType), graphTensorBundle.xTensor);
        xAttr.set_uid(uid++);
        auto xTensorAttr
            = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

        auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "scale", fromSdkType(scaleBiasDataType), graphTensorBundle.scaleTensor);
        scaleAttr.set_uid(uid++);
        auto scaleTensorAttr
            = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

        auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "bias", fromSdkType(scaleBiasDataType), graphTensorBundle.biasTensor);
        biasAttr.set_uid(uid++);
        auto biasTensorAttr
            = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(biasAttr));

        auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
        epsilonTensor->set_uid(uid++)
            .set_name("EpsilonTensor")
            .set_data_type(fromSdkType(meanVarianceDataType))
            .set_dim({1})
            .set_stride({1});

        hipdnn_frontend::graph::BatchnormAttributes bnAttrs;
        bnAttrs.set_name("batchnorm_fwd_train");
        bnAttrs.set_epsilon(epsilonTensor);

        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> momentumTensorAttr;
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> prevRunningMeanTensorAttr;
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> prevRunningVarianceTensorAttr;
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> nextRunningMeanTensorAttr;
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> nextRunningVarianceTensorAttr;
        if(useOptionalTensors)
        {
            auto momentumAttr = hipdnn_frontend::graph::makeTensorAttributes(
                "momentum",
                fromSdkType(meanVarianceDataType),
                graphTensorBundle.momentumTensor.value());
            momentumAttr.set_uid(uid++);
            momentumTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
                std::move(momentumAttr));

            auto prevRunningMeanAttr = hipdnn_frontend::graph::makeTensorAttributes(
                "prev_running_mean",
                fromSdkType(meanVarianceDataType),
                graphTensorBundle.prevRunningMeanTensor.value());
            prevRunningMeanAttr.set_uid(uid++);
            prevRunningMeanTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
                std::move(prevRunningMeanAttr));

            auto prevRunningVarianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
                "prev_running_variance",
                fromSdkType(meanVarianceDataType),
                graphTensorBundle.prevRunningVarianceTensor.value());
            prevRunningVarianceAttr.set_uid(uid++);
            prevRunningVarianceTensorAttr
                = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
                    std::move(prevRunningVarianceAttr));

            bnAttrs.set_momentum(momentumTensorAttr);
            bnAttrs.set_prev_running_mean(prevRunningMeanTensorAttr);
            bnAttrs.set_prev_running_variance(prevRunningVarianceTensorAttr);
        }

        auto outputTensorsAttr
            = graph->batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

        auto& yTensorAttr = outputTensorsAttr[0];
        if(!yTensorAttr->has_uid())
        {
            yTensorAttr->set_uid(uid++);
        }
        yTensorAttr->set_data_type(fromSdkType(inputDataType));

        auto& meanTensorAttr = outputTensorsAttr[1];
        if(!meanTensorAttr->has_uid())
        {
            meanTensorAttr->set_uid(uid++);
        }
        meanTensorAttr->set_data_type(fromSdkType(meanVarianceDataType));

        auto& invVarianceTensorAttr = outputTensorsAttr[2];
        if(!invVarianceTensorAttr->has_uid())
        {
            invVarianceTensorAttr->set_uid(uid++);
        }
        invVarianceTensorAttr->set_data_type(fromSdkType(meanVarianceDataType));

        if(useOptionalTensors)
        {
            nextRunningMeanTensorAttr = outputTensorsAttr[3];
            if(!nextRunningMeanTensorAttr->has_uid())
            {
                nextRunningMeanTensorAttr->set_uid(uid++);
            }
            nextRunningMeanTensorAttr->set_data_type(fromSdkType(meanVarianceDataType));

            nextRunningVarianceTensorAttr = outputTensorsAttr[4];
            if(!nextRunningVarianceTensorAttr->has_uid())
            {
                nextRunningVarianceTensorAttr->set_uid(uid++);
            }
            nextRunningVarianceTensorAttr->set_data_type(fromSdkType(meanVarianceDataType));
        }

        auto result = graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

        auto variantPack = graphTensorBundle.createVariantPack(*xTensorAttr,
                                                               *scaleTensorAttr,
                                                               *biasTensorAttr,
                                                               *meanTensorAttr,
                                                               *invVarianceTensorAttr,
                                                               *epsilonTensor,
                                                               *yTensorAttr,
                                                               momentumTensorAttr,
                                                               prevRunningMeanTensorAttr,
                                                               prevRunningVarianceTensorAttr,
                                                               nextRunningMeanTensorAttr,
                                                               nextRunningVarianceTensorAttr);

        hipdnn_sdk::test_utilities::CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), variantPack);
    }

    template <typename T>
    static hipdnnPluginDeviceBuffer_t generateRandomHostBuffer(
        TensorBase<T>& tensor, int uid, T min, T max, unsigned int seed = 0)
    {
        tensor.fillWithRandomValues(min, max, seed);
        hipdnnPluginDeviceBuffer_t buffer;
        buffer.uid = uid;
        buffer.ptr = tensor.memory().hostData();
        return buffer;
    }

    template <typename T>
    static hipdnnPluginDeviceBuffer_t generateEmptyHostBuffer(TensorBase<T>& tensor, int uid)
    {
        hipdnnPluginDeviceBuffer_t buffer;
        buffer.uid = uid;
        buffer.ptr = tensor.memory().hostData();
        return buffer;
    }
};

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllHalfs)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest<half, half, half>(
        DataType::HALF, DataType::HALF, DataType::HALF);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormFwdInferenceAllBFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormFwdTest<hip_bfloat16, hip_bfloat16, hip_bfloat16>(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, SignaturesThatDontExist)
{
    EXPECT_THROW((TestCpuReferenceGraphExecutor::runBatchnormFwdTest<float, half, half>(
                     DataType::FLOAT, DataType::HALF, DataType::HALF)),
                 std::runtime_error);

    EXPECT_THROW((TestCpuReferenceGraphExecutor::runBatchnormFwdTest<float, half, float>(
                     DataType::FLOAT, DataType::HALF, DataType::FLOAT)),
                 std::runtime_error);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllHalfs)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<half, half, half>(
        DataType::HALF, DataType::HALF, DataType::HALF);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormBwdAllBFloat16)
{
    TestCpuReferenceGraphExecutor::runBatchnormBwdTest<hip_bfloat16, hip_bfloat16, hip_bfloat16>(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
}

TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllFloats)
{
    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    TestCpuReferenceGraphExecutor::runBatchnormTrainTest<float, float, float>(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, true);
}

// TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllHalfs)
// {
//     TestCpuReferenceGraphExecutor::runBatchnormTrainTest<half, half, half>(
//         DataType::HALF, DataType::HALF, DataType::HALF);
// }

// TEST(TestCpuReferenceGraphExecutor, BatchnormTrainAllBFloat16)
// {
//     TestCpuReferenceGraphExecutor::runBatchnormTrainTest<hip_bfloat16, hip_bfloat16, hip_bfloat16>(
//         DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16);
// }
