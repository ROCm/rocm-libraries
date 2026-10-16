// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// What a finalized operation descriptor records for lowering to hipDNN: the
// target node, slot maps, norm dispatch and derived values. White-box: reads
// the C++ objects behind the C handles.
#include "CudnnShimV8OperationFixture.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace hipdnn_shim_test;
namespace backend = hipdnn_frontend::compatibility::cudnn_frontend::detail::backend;
using backend::OperationTarget;
using backend::PointwiseModeClass;
using hipdnn_frontend::compatibility::cudnn_frontend::DataType_t;
using hipdnn_frontend::compatibility::cudnn_frontend::NormFwdPhase_t;
using hipdnn_frontend::compatibility::cudnn_frontend::PointwiseMode_t;

struct Binding
{
    cudnnBackendAttributeName_t id;
    cudnnBackendDescriptor_t value;
};

class TestCudnnShimV8OperationLowering : public ShimOperationsFixture
{
protected:
    static backend::Descriptor& object(cudnnBackendDescriptor_t descriptor)
    {
        return *backend::liveHandle(descriptor)->object;
    }

    template <typename Op>
    static Op& as(cudnnBackendDescriptor_t descriptor, cudnnBackendDescriptorType_t type)
    {
        return backend::descriptorCast<Op>(object(descriptor), type);
    }

    // Finalizes `row` plus `extra`, with `bindings` in place of the fixture's
    // referents for those attributes.
    cudnnBackendDescriptor_t finalized(const Row& row,
                                       const std::vector<Binding>& bindings = {},
                                       const std::vector<Slot>& extra = {})
    {
        const auto bound = [&](cudnnBackendAttributeName_t id) {
            return std::any_of(bindings.begin(), bindings.end(), [&](const Binding& binding) {
                return binding.id == id;
            });
        };
        cudnnBackendDescriptor_t op = create(row.type);
        for(const std::vector<Slot>* slots : {&row.required, &extra})
        {
            for(const Slot& slot : *slots)
            {
                if(!bound(slot.attr.id))
                {
                    EXPECT_EQ(setSlot(op, slot, row), CUDNN_STATUS_SUCCESS) << slot.attr.name;
                }
            }
        }
        for(const Binding& binding : bindings)
        {
            EXPECT_EQ(setDescriptor(op, binding.id, binding.value), CUDNN_STATUS_SUCCESS);
        }
        EXPECT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS) << lastError();
        return op;
    }
};

TEST_F(TestCudnnShimV8OperationLowering, TargetPerOperation)
{
    const std::vector<std::pair<const char*, OperationTarget>> expected = {
        {"ConvolutionForward", OperationTarget::CONVOLUTION_FPROP},
        {"ConvolutionBackwardData", OperationTarget::CONVOLUTION_DGRAD},
        {"ConvolutionBackwardFilter", OperationTarget::CONVOLUTION_WGRAD},
        {"PointwiseMathUnary", OperationTarget::POINTWISE},
        {"PointwiseMathBinary", OperationTarget::POINTWISE},
        {"PointwiseBinarySelect", OperationTarget::POINTWISE},
        {"PointwiseIdentity", OperationTarget::POINTWISE},
        {"PointwiseActivationForward", OperationTarget::POINTWISE},
        {"PointwiseActivationBackward", OperationTarget::POINTWISE},
        {"Matmul", OperationTarget::MATMUL},
        {"Reduction", OperationTarget::REDUCTION},
        {"ResampleForward", OperationTarget::RESAMPLE_FWD},
        {"ResampleBackward", OperationTarget::RESAMPLE_BWD},
        {"NormForwardBatchTraining", OperationTarget::BATCHNORM},
        {"NormForwardBatchInference", OperationTarget::BATCHNORM_INFERENCE},
        {"NormForwardLayerTraining", OperationTarget::LAYERNORM},
        {"NormForwardLayerInference", OperationTarget::LAYERNORM},
        {"NormForwardRmsTraining", OperationTarget::RMSNORM},
        {"NormForwardRmsInference", OperationTarget::RMSNORM},
        {"NormBackwardBatch", OperationTarget::BATCHNORM_BACKWARD},
        {"NormBackwardLayer", OperationTarget::LAYERNORM_BACKWARD},
        {"NormBackwardRms", OperationTarget::RMSNORM_BACKWARD},
    };
    ASSERT_EQ(expected.size(), operationRows().size());
    std::set<OperationTarget> seen;
    for(const auto& [label, target] : expected)
    {
        SCOPED_TRACE(label);
        cudnnBackendDescriptor_t op = create(operationRow(label).type);
        EXPECT_EQ(backend::operationCast(object(op)).target(), OperationTarget::NOT_SET);
        op = finalized(operationRow(label));
        EXPECT_EQ(backend::operationCast(object(op)).target(), target);
        seen.insert(target);
    }
    EXPECT_EQ(seen.size(), 15U);
}

TEST_F(TestCudnnShimV8OperationLowering, OperationCastRejectsNonOperation)
{
    try
    {
        backend::operationCast(object(tensor()));
        ADD_FAILURE() << "operationCast accepted a TENSOR descriptor";
    }
    catch(const backend::ShimError& error)
    {
        EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    }
}

TEST_F(TestCudnnShimV8OperationLowering, UnsupportedOperationHasNoTarget)
{
    cudnnBackendDescriptor_t op = create(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_NOT_SUPPORTED);
    lastError();
    EXPECT_EQ(backend::operationCast(object(op)).target(), OperationTarget::NOT_SET);
}

TEST_F(TestCudnnShimV8OperationLowering, ConvolutionSlotsKeepIdentity)
{
    struct ConvolutionCase
    {
        const char* row;
        cudnnBackendAttributeName_t convolution;
        std::array<cudnnBackendAttributeName_t, 3> tensors;
    };
    const std::vector<ConvolutionCase> cases = {
        {"ConvolutionForward",
         CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
         {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
          CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
          CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y}},
        {"ConvolutionBackwardData",
         CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
          CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
          CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX}},
        {"ConvolutionBackwardFilter",
         CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
          CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
          CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW}},
    };
    for(const ConvolutionCase& convolutionCase : cases)
    {
        SCOPED_TRACE(convolutionCase.row);
        cudnnBackendDescriptor_t conv = convolution();
        const std::array<cudnnBackendDescriptor_t, 3> tensors = {tensor(), tensor(), tensor()};
        cudnnBackendDescriptor_t op = finalized(operationRow(convolutionCase.row),
                                                {{convolutionCase.convolution, conv},
                                                 {convolutionCase.tensors[0], tensors[0]},
                                                 {convolutionCase.tensors[1], tensors[1]},
                                                 {convolutionCase.tensors[2], tensors[2]}});
        const auto& base = static_cast<const backend::ConvolutionOperationBase&>(
            backend::operationCast(object(op)));
        EXPECT_EQ(base.convolution().get(), &object(conv));
        EXPECT_EQ(base.convolution()->spatialDims(), 2);

        const backend::Descriptor* first = nullptr;
        const backend::Descriptor* second = nullptr;
        const backend::Descriptor* output = nullptr;
        if(convolutionCase.convolution == CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC)
        {
            const auto& fprop = as<backend::ConvolutionForwardOperation>(
                op, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
            first = fprop.x().get();
            second = fprop.w().get();
            output = fprop.y().get();
        }
        else if(convolutionCase.convolution == CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC)
        {
            const auto& dgrad = as<backend::ConvolutionBackwardDataOperation>(
                op, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
            first = dgrad.dy().get();
            second = dgrad.w().get();
            output = dgrad.dx().get();
        }
        else
        {
            const auto& wgrad = as<backend::ConvolutionBackwardFilterOperation>(
                op, CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
            first = wgrad.dy().get();
            second = wgrad.x().get();
            output = wgrad.dw().get();
        }
        EXPECT_EQ(first, &object(tensors[0]));
        EXPECT_EQ(second, &object(tensors[1]));
        EXPECT_EQ(output, &object(tensors[2]));
    }
}

TEST_F(TestCudnnShimV8OperationLowering, PointwiseForwardSlotMap)
{
    struct ForwardCase
    {
        cudnnPointwiseMode_t mode;
        PointwiseMode_t lowered;
        PointwiseModeClass modeClass;
        std::int64_t ports;
    };
    for(const ForwardCase& forwardCase :
        {ForwardCase{CUDNN_POINTWISE_SQRT, PointwiseMode_t::SQRT, PointwiseModeClass::MATH, 2},
         ForwardCase{CUDNN_POINTWISE_ERF, PointwiseMode_t::ERF, PointwiseModeClass::MATH, 2},
         ForwardCase{CUDNN_POINTWISE_ADD, PointwiseMode_t::ADD, PointwiseModeClass::MATH, 3},
         ForwardCase{CUDNN_POINTWISE_BINARY_SELECT,
                     PointwiseMode_t::BINARY_SELECT,
                     PointwiseModeClass::MATH,
                     4},
         ForwardCase{
             CUDNN_POINTWISE_IDENTITY, PointwiseMode_t::IDENTITY, PointwiseModeClass::IDENTITY, 2},
         ForwardCase{
             CUDNN_POINTWISE_EXP, PointwiseMode_t::EXP, PointwiseModeClass::ACTIVATION_FORWARD, 2}})
    {
        SCOPED_TRACE(static_cast<int>(forwardCase.mode));
        cudnnBackendDescriptor_t x = tensor();
        cudnnBackendDescriptor_t b = tensor();
        cudnnBackendDescriptor_t t = tensor();
        cudnnBackendDescriptor_t y = tensor();
        std::vector<Binding> bindings = {{CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x},
                                         {CUDNN_ATTR_OPERATION_POINTWISE_YDESC, y}};
        if(forwardCase.ports >= 3)
        {
            bindings.push_back({CUDNN_ATTR_OPERATION_POINTWISE_BDESC, b});
        }
        if(forwardCase.ports == 4)
        {
            bindings.push_back({CUDNN_ATTR_OPERATION_POINTWISE_TDESC, t});
        }
        cudnnBackendDescriptor_t op
            = finalized(pointwiseRow("", forwardCase.mode, pointwiseSlots(false, false)), bindings);
        const auto& lowered
            = as<backend::PointwiseOperation>(op, CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        EXPECT_EQ(lowered.pointwise()->mode(), forwardCase.lowered);
        EXPECT_EQ(lowered.modeClass(), forwardCase.modeClass);
        EXPECT_EQ(lowered.portCount(), forwardCase.ports);
        EXPECT_EQ(lowered.in0().get(), &object(x));
        EXPECT_EQ(lowered.in1().get(), forwardCase.ports >= 3 ? &object(b) : nullptr);
        EXPECT_EQ(lowered.in2().get(), forwardCase.ports == 4 ? &object(t) : nullptr);
        EXPECT_EQ(lowered.out0().get(), &object(y));
    }
}

TEST_F(TestCudnnShimV8OperationLowering, PointwiseActivationBackwardSlotMap)
{
    cudnnBackendDescriptor_t x = tensor();
    cudnnBackendDescriptor_t dy = tensor();
    cudnnBackendDescriptor_t dx = tensor();
    cudnnBackendDescriptor_t op = finalized(operationRow("PointwiseActivationBackward"),
                                            {{CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x},
                                             {CUDNN_ATTR_OPERATION_POINTWISE_DYDESC, dy},
                                             {CUDNN_ATTR_OPERATION_POINTWISE_DXDESC, dx}});
    const auto& lowered
        = as<backend::PointwiseOperation>(op, CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    EXPECT_EQ(lowered.pointwise()->mode(), PointwiseMode_t::RELU_BWD);
    EXPECT_EQ(lowered.modeClass(), PointwiseModeClass::ACTIVATION_BACKWARD);
    EXPECT_EQ(lowered.portCount(), 3);
    EXPECT_EQ(lowered.in0().get(), &object(dy));
    EXPECT_EQ(lowered.in1().get(), &object(x));
    EXPECT_EQ(lowered.in2(), nullptr);
    EXPECT_EQ(lowered.out0().get(), &object(dx));
}

TEST_F(TestCudnnShimV8OperationLowering, LayerNormDimCountFromScale)
{
    struct DimCase
    {
        std::vector<std::int64_t> scale;
        std::int64_t expected;
    };
    for(const DimCase& dimCase : {DimCase{{1, 8, 3, 3}, 3}, DimCase{{8, 3, 3}, 3}})
    {
        SCOPED_TRACE(std::to_string(dimCase.scale.size()) + "-D scale");
        cudnnBackendDescriptor_t forward
            = finalized(operationRow("NormForwardLayerTraining"),
                        {{FWD_X.id, tensor({2, 8, 3, 3})}, {FWD_SCALE.id, tensor(dimCase.scale)}});
        const auto& lowered = as<backend::NormForwardOperation>(
            forward, CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
        EXPECT_EQ(lowered.normalizedDimCount(), dimCase.expected);
        EXPECT_EQ(lowered.phase(), NormFwdPhase_t::TRAINING);
        EXPECT_EQ(backend::NormForwardOperation::computeType(), DataType_t::FLOAT);

        cudnnBackendDescriptor_t backward
            = finalized(operationRow("NormBackwardLayer"),
                        {{BWD_X.id, tensor({2, 8, 3, 3})}, {BWD_SCALE.id, tensor(dimCase.scale)}});
        EXPECT_EQ(as<backend::NormBackwardOperation>(
                      backward, CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
                      .normalizedDimCount(),
                  dimCase.expected);
    }
}

TEST_F(TestCudnnShimV8OperationLowering, BatchNormDropsEpsilon)
{
    const Row& inference = operationRow("NormForwardBatchInference");
    cudnnBackendDescriptor_t forward = finalized(inference, {}, inference.optional);
    const auto& lowered = as<backend::NormForwardOperation>(
        forward, CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
    EXPECT_EQ(lowered.epsilon(), nullptr);
    EXPECT_NE(lowered.mean(), nullptr);
    EXPECT_NE(lowered.invVariance(), nullptr);
    EXPECT_EQ(lowered.phase(), NormFwdPhase_t::INFERENCE);

    const Row& batchBackward = operationRow("NormBackwardBatch");
    cudnnBackendDescriptor_t backward = finalized(batchBackward, {}, batchBackward.optional);
    EXPECT_EQ(as<backend::NormBackwardOperation>(backward,
                                                 CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
                  .epsilon(),
              nullptr);

    const Row& layerBackward = operationRow("NormBackwardLayer");
    cudnnBackendDescriptor_t layer = finalized(layerBackward, {}, layerBackward.optional);
    EXPECT_NE(
        as<backend::NormBackwardOperation>(layer, CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
            .epsilon(),
        nullptr);
}

TEST_F(TestCudnnShimV8OperationLowering, BatchNormRunningStatsKeepIdentity)
{
    std::vector<Binding> bindings;
    bindings.reserve(RUNNING_STATS.size());
    for(const Attr& attr : RUNNING_STATS)
    {
        bindings.push_back({attr.id, tensor()});
    }
    cudnnBackendDescriptor_t op = finalized(operationRow("NormForwardBatchTraining"), bindings);
    const auto& lowered
        = as<backend::NormForwardOperation>(op, CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
    EXPECT_EQ(lowered.inputRunningMean().get(), &object(bindings[0].value));
    EXPECT_EQ(lowered.inputRunningVariance().get(), &object(bindings[1].value));
    EXPECT_EQ(lowered.expAvgFactor().get(), &object(bindings[2].value));
    EXPECT_EQ(lowered.outputRunningMean().get(), &object(bindings[3].value));
    EXPECT_EQ(lowered.outputRunningVariance().get(), &object(bindings[4].value));
    EXPECT_EQ(lowered.mean(), nullptr);
    EXPECT_NE(lowered.epsilon(), nullptr);
}

TEST_F(TestCudnnShimV8OperationLowering, RmsNormHasNoMean)
{
    const Row& forwardRow = operationRow("NormForwardRmsTraining");
    cudnnBackendDescriptor_t invRms = tensor();
    cudnnBackendDescriptor_t forward
        = finalized(forwardRow, {{FWD_INV_VARIANCE.id, invRms}}, {tensorSlot(FWD_BIAS)});
    const auto& lowered = as<backend::NormForwardOperation>(
        forward, CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
    EXPECT_EQ(lowered.mean(), nullptr);
    EXPECT_EQ(lowered.invVariance().get(), &object(invRms));
    EXPECT_NE(lowered.bias(), nullptr);
    EXPECT_EQ(lowered.normalizedDimCount(), 0);

    cudnnBackendDescriptor_t backward = finalized(operationRow("NormBackwardRms"));
    const auto& loweredBackward = as<backend::NormBackwardOperation>(
        backward, CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR);
    EXPECT_EQ(loweredBackward.mean(), nullptr);
    EXPECT_NE(loweredBackward.invVariance(), nullptr);
    EXPECT_EQ(loweredBackward.dbias(), nullptr);
}

TEST_F(TestCudnnShimV8OperationLowering, ResampleIndexFollowsPresence)
{
    const Row& forwardRow = operationRow("ResampleForward");
    EXPECT_EQ(as<backend::ResampleForwardOperation>(finalized(forwardRow),
                                                    CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                  .index(),
              nullptr);
    cudnnBackendDescriptor_t forwardIndex = tensor();
    EXPECT_EQ(
        as<backend::ResampleForwardOperation>(
            finalized(forwardRow, {{CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC, forwardIndex}}),
            CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
            .index()
            .get(),
        &object(forwardIndex));

    cudnnBackendDescriptor_t backwardIndex = tensor();
    EXPECT_EQ(as<backend::ResampleBackwardOperation>(
                  finalized(operationRow("ResampleBackward"),
                            {{CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC, backwardIndex}}),
                  CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                  .index()
                  .get(),
              &object(backwardIndex));

    cudnnBackendDescriptor_t averagePool = create(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR);
    ASSERT_EQ(setDescriptor(averagePool,
                            CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC,
                            resample(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING)),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(setDescriptor(averagePool, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, tensor()),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(setDescriptor(averagePool, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, tensor()),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(averagePool), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(as<backend::ResampleBackwardOperation>(
                  averagePool, CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                  .index(),
              nullptr);
}

TEST_F(TestCudnnShimV8OperationLowering, ReductionDeterminismRecorded)
{
    const Row& row = operationRow("Reduction");
    const auto deterministic = [&](cudnnBackendDescriptor_t op) {
        return as<backend::ReductionOperation>(op, CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
            .reduction()
            ->isDeterministic();
    };
    EXPECT_FALSE(deterministic(finalized(row)));

    cudnnBackendDescriptor_t reductionDescriptor = create(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
    ASSERT_EQ(set(reductionDescriptor,
                  CUDNN_ATTR_REDUCTION_OPERATOR,
                  CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                  CUDNN_REDUCE_TENSOR_ADD),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(reductionDescriptor,
                  CUDNN_ATTR_REDUCTION_COMP_TYPE,
                  CUDNN_TYPE_DATA_TYPE,
                  CUDNN_DATA_FLOAT),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(
        set(reductionDescriptor, CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC, CUDNN_TYPE_BOOLEAN, true),
        CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(reductionDescriptor), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_TRUE(deterministic(
        finalized(row, {{CUDNN_ATTR_OPERATION_REDUCTION_DESC, reductionDescriptor}})));
}

} // namespace
