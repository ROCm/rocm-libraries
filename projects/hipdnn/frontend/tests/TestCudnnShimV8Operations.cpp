// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Operation descriptor finalize rules of the cuDNN v8 shim, driven only through
// the C API. Expectations are written from the cuDNN attribute contract, not
// from the implementation, so attribute and type names are spelled out here.
#include "CudnnShimV8OperationFixture.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace hipdnn_shim_test;

class TestCudnnShimV8Operations : public ShimOperationsFixture
{
};

class TestCudnnShimV8OperationRows : public ShimOperationsFixture,
                                     public ::testing::WithParamInterface<Row>
{
};

TEST_P(TestCudnnShimV8OperationRows, RequiredAttributesSuffice)
{
    EXPECT_EQ(cudnnBackendFinalize(build(GetParam())), CUDNN_STATUS_SUCCESS) << lastError();
}

TEST_P(TestCudnnShimV8OperationRows, OptionalAttributesAccepted)
{
    const Row& row = GetParam();
    EXPECT_EQ(cudnnBackendFinalize(build(row, std::nullopt, row.optional)), CUDNN_STATUS_SUCCESS)
        << lastError();
}

TEST_P(TestCudnnShimV8OperationRows, MissingRequiredNamed)
{
    const Row& row = GetParam();
    for(const Slot& slot : row.required)
    {
        SCOPED_TRACE(slot.attr.name);
        expectRejected(build(row, slot.attr.id), CUDNN_STATUS_BAD_PARAM, slot.attr);
    }
}

INSTANTIATE_TEST_SUITE_P(Table,
                         TestCudnnShimV8OperationRows,
                         ::testing::ValuesIn(operationRows()),
                         [](const ::testing::TestParamInfo<Row>& info) {
                             return std::string(info.param.label);
                         });

TEST_F(TestCudnnShimV8Operations, UnsupportedTypesRejectedUnconditionally)
{
    struct Unsupported
    {
        cudnnBackendDescriptorType_t type;
        const char* name;
    };
    const std::vector<Unsupported> types = {
        {CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR,
         "CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR,
         "CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
         "CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
         "CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR, "CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
         "CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR,
         "CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR,
         "CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
         "CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR, "CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR, "CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR, "CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR, "CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR"},
        {CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR,
         "CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR"},
    };
    for(const Unsupported& unsupported : types)
    {
        SCOPED_TRACE(unsupported.name);
        expectRejected(create(unsupported.type),
                       CUDNN_STATUS_NOT_SUPPORTED,
                       std::vector<std::string>{unsupported.name});
    }
}

// Upstream's builder sets GEN_STATS attributes; the operation type is rejected
// before any of them is looked at.
TEST_F(TestCudnnShimV8Operations, UnsupportedTypeIgnoresSetAttributes)
{
    cudnnBackendDescriptor_t op = create(CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR);
    ASSERT_EQ(setDescriptor(op, CUDNN_ATTR_OPERATION_GENSTATS_XDESC, tensor()),
              CUDNN_STATUS_SUCCESS);
    expectRejected(op,
                   CUDNN_STATUS_NOT_SUPPORTED,
                   std::vector<std::string>{"CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR"});
}

// Port count alone does not decide the mode class.
TEST_F(TestCudnnShimV8Operations, PointwiseModeClassesFinalize)
{
    struct ModeClass
    {
        std::vector<cudnnPointwiseMode_t> modes;
        std::vector<Slot> slots;
    };
    // COS, MOD and POW have no hipDNN mapping, so their POINTWISE descriptor
    // already fails to finalize.
    const std::vector<ModeClass> classes = {
        {{CUDNN_POINTWISE_ADD,
          CUDNN_POINTWISE_MUL,
          CUDNN_POINTWISE_DIV,
          CUDNN_POINTWISE_SUB,
          CUDNN_POINTWISE_ADD_SQUARE,
          CUDNN_POINTWISE_MIN,
          CUDNN_POINTWISE_MAX,
          CUDNN_POINTWISE_CMP_EQ,
          CUDNN_POINTWISE_CMP_NEQ,
          CUDNN_POINTWISE_CMP_GT,
          CUDNN_POINTWISE_CMP_GE,
          CUDNN_POINTWISE_CMP_LT,
          CUDNN_POINTWISE_CMP_LE,
          CUDNN_POINTWISE_LOGICAL_AND,
          CUDNN_POINTWISE_LOGICAL_OR},
         pointwiseSlots(true, false)},
        {{CUDNN_POINTWISE_SQRT,         CUDNN_POINTWISE_RSQRT,
          CUDNN_POINTWISE_SIN,          CUDNN_POINTWISE_TAN,
          CUDNN_POINTWISE_LOG,          CUDNN_POINTWISE_NEG,
          CUDNN_POINTWISE_ABS,          CUDNN_POINTWISE_CEIL,
          CUDNN_POINTWISE_FLOOR,        CUDNN_POINTWISE_LOGICAL_NOT,
          CUDNN_POINTWISE_GEN_INDEX,    CUDNN_POINTWISE_ERF,
          CUDNN_POINTWISE_RECIPROCAL,   CUDNN_POINTWISE_IDENTITY,
          CUDNN_POINTWISE_RELU_FWD,     CUDNN_POINTWISE_TANH_FWD,
          CUDNN_POINTWISE_SIGMOID_FWD,  CUDNN_POINTWISE_ELU_FWD,
          CUDNN_POINTWISE_GELU_FWD,     CUDNN_POINTWISE_GELU_APPROX_TANH_FWD,
          CUDNN_POINTWISE_SOFTPLUS_FWD, CUDNN_POINTWISE_SWISH_FWD,
          CUDNN_POINTWISE_EXP},
         pointwiseSlots(false, false)},
        {{CUDNN_POINTWISE_BINARY_SELECT}, pointwiseSlots(true, true)},
        {{CUDNN_POINTWISE_RELU_BWD,
          CUDNN_POINTWISE_TANH_BWD,
          CUDNN_POINTWISE_SIGMOID_BWD,
          CUDNN_POINTWISE_ELU_BWD,
          CUDNN_POINTWISE_GELU_BWD,
          CUDNN_POINTWISE_GELU_APPROX_TANH_BWD,
          CUDNN_POINTWISE_SOFTPLUS_BWD,
          CUDNN_POINTWISE_SWISH_BWD},
         pointwiseBackwardSlots()},
    };
    for(const ModeClass& modeClass : classes)
    {
        for(const cudnnPointwiseMode_t mode : modeClass.modes)
        {
            SCOPED_TRACE(static_cast<int>(mode));
            const Row row = pointwiseRow("", mode, modeClass.slots);
            EXPECT_EQ(cudnnBackendFinalize(build(row)), CUDNN_STATUS_SUCCESS) << lastError();
        }
    }
}

TEST_F(TestCudnnShimV8Operations, BinarySelectWithoutTDescRejected)
{
    expectRejected(
        build(operationRow("PointwiseBinarySelect"), CUDNN_ATTR_OPERATION_POINTWISE_TDESC),
        CUDNN_STATUS_BAD_PARAM,
        SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_TDESC));
}

TEST_F(TestCudnnShimV8Operations, ActivationBackwardRejectsForwardSlots)
{
    for(const Attr& attr : {SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_YDESC),
                            SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_BDESC),
                            SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_TDESC)})
    {
        SCOPED_TRACE(attr.name);
        expectRejected(
            build(operationRow("PointwiseActivationBackward"), attr), CUDNN_STATUS_BAD_PARAM, attr);
    }
}

TEST_F(TestCudnnShimV8Operations, ForwardClassesRejectGradientSlots)
{
    for(const char* label : {"PointwiseMathUnary",
                             "PointwiseMathBinary",
                             "PointwiseBinarySelect",
                             "PointwiseIdentity",
                             "PointwiseActivationForward"})
    {
        for(const Attr& attr : {SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_DYDESC),
                                SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_DXDESC)})
        {
            SCOPED_TRACE(std::string(label) + " " + attr.name);
            expectRejected(build(operationRow(label), attr), CUDNN_STATUS_BAD_PARAM, attr);
        }
    }
}

TEST_F(TestCudnnShimV8Operations, TDescNeedsFourPorts)
{
    const Attr tdesc = SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_TDESC);
    for(const char* label : {"PointwiseMathUnary",
                             "PointwiseMathBinary",
                             "PointwiseIdentity",
                             "PointwiseActivationForward"})
    {
        SCOPED_TRACE(label);
        expectRejected(build(operationRow(label), tdesc), CUDNN_STATUS_BAD_PARAM, tdesc);
    }
}

TEST_F(TestCudnnShimV8Operations, TwoPortClassesRejectBDesc)
{
    const Attr bdesc = SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_BDESC);
    for(const cudnnPointwiseMode_t mode : {CUDNN_POINTWISE_SQRT,
                                           CUDNN_POINTWISE_GEN_INDEX,
                                           CUDNN_POINTWISE_IDENTITY,
                                           CUDNN_POINTWISE_RELU_FWD,
                                           CUDNN_POINTWISE_EXP})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        expectRejected(build(pointwiseRow("", mode, pointwiseSlots(false, false)), bdesc),
                       CUDNN_STATUS_BAD_PARAM,
                       bdesc);
    }
}

struct ScaleCase
{
    const char* row;
    Attr attr;
    double identity;
};

const std::vector<ScaleCase>& scaleCases()
{
    static const std::vector<ScaleCase> s_cases = {
        {"ConvolutionForward", SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA), 1.0},
        {"ConvolutionForward", SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA), 0.0},
        {"ConvolutionBackwardData",
         SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA),
         1.0},
        {"ConvolutionBackwardData", SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA), 0.0},
        {"ConvolutionBackwardFilter",
         SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA),
         1.0},
        {"ConvolutionBackwardFilter",
         SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA),
         0.0},
        {"PointwiseMathBinary", SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1), 1.0},
        {"PointwiseMathBinary", SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2), 1.0},
        {"ResampleForward", SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA), 1.0},
        {"ResampleForward", SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA), 0.0},
        {"ResampleBackward", SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA), 1.0},
        {"ResampleBackward", SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA), 0.0},
    };
    return s_cases;
}

TEST_F(TestCudnnShimV8Operations, ScaleOtherThanIdentityNotSupported)
{
    for(const ScaleCase& scale : scaleCases())
    {
        SCOPED_TRACE(scale.attr.name);
        const double other = scale.identity + 0.5;

        cudnnBackendDescriptor_t asFloat = build(operationRow(scale.row));
        ASSERT_EQ(set(asFloat, scale.attr.id, CUDNN_TYPE_FLOAT, static_cast<float>(other)),
                  CUDNN_STATUS_SUCCESS);
        expectRejected(asFloat, CUDNN_STATUS_NOT_SUPPORTED, scale.attr);

        cudnnBackendDescriptor_t asDouble = build(operationRow(scale.row));
        ASSERT_EQ(set(asDouble, scale.attr.id, CUDNN_TYPE_DOUBLE, other), CUDNN_STATUS_SUCCESS);
        expectRejected(asDouble, CUDNN_STATUS_NOT_SUPPORTED, scale.attr);
    }
}

TEST_F(TestCudnnShimV8Operations, IdentityScalesAccepted)
{
    for(const ScaleCase& scale : scaleCases())
    {
        SCOPED_TRACE(scale.attr.name);

        cudnnBackendDescriptor_t asFloat = build(operationRow(scale.row));
        ASSERT_EQ(set(asFloat, scale.attr.id, CUDNN_TYPE_FLOAT, static_cast<float>(scale.identity)),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendFinalize(asFloat), CUDNN_STATUS_SUCCESS) << lastError();

        cudnnBackendDescriptor_t asDouble = build(operationRow(scale.row));
        ASSERT_EQ(set(asDouble, scale.attr.id, CUDNN_TYPE_DOUBLE, scale.identity),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendFinalize(asDouble), CUDNN_STATUS_SUCCESS) << lastError();
    }
}

TEST_F(TestCudnnShimV8Operations, AllScalesAtIdentityAccepted)
{
    cudnnBackendDescriptor_t op = build(operationRow("PointwiseMathBinary"));
    ASSERT_EQ(set(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1.0F),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(set(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, CUDNN_TYPE_DOUBLE, 1.0),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS) << lastError();
}

TEST_F(TestCudnnShimV8Operations, NormModesWithoutLoweringNotSupported)
{
    for(const cudnnBackendNormMode_t mode :
        {CUDNN_INSTANCE_NORM, CUDNN_GROUP_NORM, CUDNN_ADA_LAYER_NORM})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        for(const cudnnBackendNormFwdPhase_t phase :
            {CUDNN_NORM_FWD_TRAINING, CUDNN_NORM_FWD_INFERENCE})
        {
            cudnnBackendDescriptor_t forward
                = create(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR);
            ASSERT_EQ(set(forward, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, CUDNN_TYPE_NORM_MODE, mode),
                      CUDNN_STATUS_SUCCESS);
            ASSERT_EQ(
                set(forward, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, CUDNN_TYPE_NORM_FWD_PHASE, phase),
                CUDNN_STATUS_SUCCESS);
            expectRejected(
                forward, CUDNN_STATUS_NOT_SUPPORTED, SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_MODE));
        }
        cudnnBackendDescriptor_t backward
            = create(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR);
        ASSERT_EQ(set(backward, CUDNN_ATTR_OPERATION_NORM_BWD_MODE, CUDNN_TYPE_NORM_MODE, mode),
                  CUDNN_STATUS_SUCCESS);
        expectRejected(
            backward, CUDNN_STATUS_NOT_SUPPORTED, SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_MODE));
    }
}

TEST_F(TestCudnnShimV8Operations, NormPeerStatsNotSupported)
{
    for(const Row& row : operationRows())
    {
        const bool forward = row.type == CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
        if(!forward && row.type != CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
        {
            continue;
        }
        SCOPED_TRACE(row.label);
        const Attr& peerStats = forward ? FWD_PEER_STATS : BWD_PEER_STATS;
        cudnnBackendDescriptor_t op = build(row);
        ASSERT_EQ(setArray(op,
                           peerStats.id,
                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                           std::vector<cudnnBackendDescriptor_t>{tensor()}),
                  CUDNN_STATUS_SUCCESS);
        expectRejected(op, CUDNN_STATUS_NOT_SUPPORTED, peerStats);
    }
}

TEST_F(TestCudnnShimV8Operations, EmptyPeerStatsAccepted)
{
    cudnnBackendDescriptor_t op = build(operationRow("NormForwardBatchTraining"));
    ASSERT_EQ(setArray(op,
                       FWD_PEER_STATS.id,
                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                       std::vector<cudnnBackendDescriptor_t>{}),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS) << lastError();
}

TEST_F(TestCudnnShimV8Operations, NormSavedStatsBothOrNone)
{
    struct Pair
    {
        const char* row;
        Attr mean;
        Attr invVariance;
    };
    for(const Pair& pair : {Pair{"NormForwardBatchTraining", FWD_MEAN, FWD_INV_VARIANCE},
                            Pair{"NormForwardLayerTraining", FWD_MEAN, FWD_INV_VARIANCE},
                            Pair{"NormForwardLayerInference", FWD_MEAN, FWD_INV_VARIANCE},
                            Pair{"NormBackwardLayer", BWD_MEAN, BWD_INV_VARIANCE}})
    {
        SCOPED_TRACE(pair.row);
        expectRejected(
            build(operationRow(pair.row), pair.mean), CUDNN_STATUS_BAD_PARAM, pair.invVariance);
        expectRejected(
            build(operationRow(pair.row), pair.invVariance), CUDNN_STATUS_BAD_PARAM, pair.mean);
    }
}

TEST_F(TestCudnnShimV8Operations, RunningStatsAllOrNone)
{
    const Row& row = operationRow("NormForwardBatchTraining");
    for(const Attr& present : RUNNING_STATS)
    {
        SCOPED_TRACE(std::string("only ") + present.name);
        std::vector<std::string> missing;
        for(const Attr& attr : RUNNING_STATS)
        {
            if(attr.id != present.id)
            {
                missing.emplace_back(attr.name);
            }
        }
        expectRejected(build(row, present), CUDNN_STATUS_BAD_PARAM, missing);
    }
    for(const Attr& absent : RUNNING_STATS)
    {
        SCOPED_TRACE(std::string("all but ") + absent.name);
        std::vector<Slot> others;
        for(const Attr& attr : RUNNING_STATS)
        {
            if(attr.id != absent.id)
            {
                others.push_back(tensorSlot(attr));
            }
        }
        expectRejected(build(row, std::nullopt, others), CUDNN_STATUS_BAD_PARAM, absent);
    }
}

TEST_F(TestCudnnShimV8Operations, RunningStatsOutsideBatchTrainingNotSupported)
{
    for(const char* label : {"NormForwardBatchInference",
                             "NormForwardLayerTraining",
                             "NormForwardLayerInference",
                             "NormForwardRmsTraining",
                             "NormForwardRmsInference"})
    {
        for(const Attr& attr : RUNNING_STATS)
        {
            SCOPED_TRACE(std::string(label) + " " + attr.name);
            expectRejected(build(operationRow(label), attr), CUDNN_STATUS_NOT_SUPPORTED, attr);
        }
    }
}

TEST_F(TestCudnnShimV8Operations, RmsMeanNotSupported)
{
    expectRejected(build(operationRow("NormForwardRmsTraining"), FWD_MEAN),
                   CUDNN_STATUS_NOT_SUPPORTED,
                   FWD_MEAN);
    expectRejected(build(operationRow("NormForwardRmsInference"), FWD_MEAN),
                   CUDNN_STATUS_NOT_SUPPORTED,
                   FWD_MEAN);
    expectRejected(
        build(operationRow("NormBackwardRms"), BWD_MEAN), CUDNN_STATUS_NOT_SUPPORTED, BWD_MEAN);
}

TEST_F(TestCudnnShimV8Operations, MatmulOverridesNotSupported)
{
    for(const Attr& attr : {SHIM_ATTR(CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC),
                            SHIM_ATTR(CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC),
                            SHIM_ATTR(CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC)})
    {
        SCOPED_TRACE(attr.name);
        expectRejected(build(operationRow("Matmul"), attr), CUDNN_STATUS_NOT_SUPPORTED, attr);
    }
}

TEST_F(TestCudnnShimV8Operations, ResampleBackwardForwardTensorsNotSupported)
{
    for(const Attr& attr : {SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC),
                            SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC)})
    {
        SCOPED_TRACE(attr.name);
        expectRejected(
            build(operationRow("ResampleBackward"), attr), CUDNN_STATUS_NOT_SUPPORTED, attr);
    }
}

TEST_F(TestCudnnShimV8Operations, ConvolutionTensorRankMismatchNamed)
{
    for(const char* label :
        {"ConvolutionForward", "ConvolutionBackwardData", "ConvolutionBackwardFilter"})
    {
        const Row& row = operationRow(label);
        for(const Slot& slot : row.required)
        {
            if(slot.ref != Ref::TENSOR)
            {
                continue;
            }
            SCOPED_TRACE(slot.attr.name);
            cudnnBackendDescriptor_t op = build(row, slot.attr.id);
            ASSERT_EQ(setDescriptor(op, slot.attr.id, tensor({2, 3, 4})), CUDNN_STATUS_SUCCESS);
            expectRejected(op, CUDNN_STATUS_BAD_PARAM, slot.attr);
        }
    }
}

TEST_F(TestCudnnShimV8Operations, ConvolutionSpatialDimsMismatchRejected)
{
    for(const char* label :
        {"ConvolutionForward", "ConvolutionBackwardData", "ConvolutionBackwardFilter"})
    {
        SCOPED_TRACE(label);
        const Row& row = operationRow(label);
        const Slot& convSlot = row.required.front();
        cudnnBackendDescriptor_t op = build(row, convSlot.attr.id);
        ASSERT_EQ(setDescriptor(op, convSlot.attr.id, convolution(3)), CUDNN_STATUS_SUCCESS);
        expectRejected(op,
                       CUDNN_STATUS_BAD_PARAM,
                       std::vector<std::string>{"CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS"});
    }
}

// Upstream's Errata.h reads these back into descriptors it created, then asks
// for 5 dimensions.
TEST_F(TestCudnnShimV8Operations, InPlaceTensorGetFillsCallerDescriptor)
{
    struct Fill
    {
        const char* row;
        Attr attr;
    };
    const std::vector<Fill> fills = {
        {"ConvolutionForward", SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X)},
        {"ConvolutionForward", SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W)},
        {"ConvolutionBackwardFilter", SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X)},
        {"ConvolutionBackwardFilter", SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW)},
        {"ConvolutionBackwardData", SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX)},
        {"ConvolutionBackwardData", SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W)},
    };
    const std::vector<std::int64_t> dims = {7, 6, 5, 4};
    for(const Fill& fill : fills)
    {
        SCOPED_TRACE(fill.attr.name);
        cudnnBackendDescriptor_t op = build(operationRow(fill.row), fill.attr.id);
        ASSERT_EQ(setDescriptor(op, fill.attr.id, tensor(dims)), CUDNN_STATUS_SUCCESS);
        ASSERT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS) << lastError();

        cudnnBackendDescriptor_t target = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        std::int64_t count = 0;
        ASSERT_EQ(cudnnBackendGetAttribute(op,
                                           fill.attr.id,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &count,
                                           static_cast<void*>(&target)),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        EXPECT_EQ(count, 1);

        std::array<std::int64_t, 5> filled = {-1, -1, -1, -1, -1};
        ASSERT_EQ(cudnnBackendGetAttribute(target,
                                           CUDNN_ATTR_TENSOR_DIMENSIONS,
                                           CUDNN_TYPE_INT64,
                                           static_cast<std::int64_t>(filled.size()),
                                           &count,
                                           filled.data()),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        EXPECT_EQ(count, 4);
        EXPECT_EQ(std::vector<std::int64_t>(filled.begin(), filled.begin() + 4), dims);
        EXPECT_EQ(filled[4], -1);
    }
}

TEST_F(TestCudnnShimV8Operations, ResampleBackwardMaxPoolNeedsIndex)
{
    const Attr index = SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC);
    expectRejected(
        build(operationRow("ResampleBackward"), index.id), CUDNN_STATUS_BAD_PARAM, index);
}

TEST_F(TestCudnnShimV8Operations, ResampleBackwardAveragePoolIndexOptional)
{
    cudnnBackendDescriptor_t op = create(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR);
    ASSERT_EQ(setDescriptor(op,
                            CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC,
                            resample(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING)),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(setDescriptor(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC, tensor()),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(setDescriptor(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC, tensor()),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS) << lastError();
}

TEST_F(TestCudnnShimV8Operations, NanScaleNotSupported)
{
    const Attr alpha1 = SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1);
    const Row& row = operationRow("PointwiseMathBinary");

    cudnnBackendDescriptor_t asFloat = build(row);
    ASSERT_EQ(set(asFloat, alpha1.id, CUDNN_TYPE_FLOAT, std::numeric_limits<float>::quiet_NaN()),
              CUDNN_STATUS_SUCCESS);
    expectRejected(asFloat, CUDNN_STATUS_NOT_SUPPORTED, alpha1);

    cudnnBackendDescriptor_t asDouble = build(row);
    ASSERT_EQ(set(asDouble, alpha1.id, CUDNN_TYPE_DOUBLE, std::numeric_limits<double>::quiet_NaN()),
              CUDNN_STATUS_SUCCESS);
    expectRejected(asDouble, CUDNN_STATUS_NOT_SUPPORTED, alpha1);
}

// Finalize checks, in order: operation type, mode, required attributes,
// attributes the mode does not use, alpha/beta, attributes that cannot be
// lowered. The first failure decides the status.
TEST_F(TestCudnnShimV8Operations, MissingRequiredPrecedesScale)
{
    const Attr y = SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y);
    cudnnBackendDescriptor_t op = build(operationRow("ConvolutionForward"), y.id);
    ASSERT_EQ(set(op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_DOUBLE, 2.0),
              CUDNN_STATUS_SUCCESS);
    expectRejected(op, CUDNN_STATUS_BAD_PARAM, y);
}

TEST_F(TestCudnnShimV8Operations, UnusedSlotPrecedesScale)
{
    const Attr y = SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_YDESC);
    cudnnBackendDescriptor_t op = build(operationRow("PointwiseActivationBackward"), y);
    ASSERT_EQ(set(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 2.0F),
              CUDNN_STATUS_SUCCESS);
    expectRejected(op, CUDNN_STATUS_BAD_PARAM, y);
}

TEST_F(TestCudnnShimV8Operations, MissingIndexPrecedesScale)
{
    const Attr index = SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC);
    cudnnBackendDescriptor_t op = build(operationRow("ResampleBackward"), index.id);
    ASSERT_EQ(set(op, CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA, CUDNN_TYPE_DOUBLE, 0.5),
              CUDNN_STATUS_SUCCESS);
    expectRejected(op, CUDNN_STATUS_BAD_PARAM, index);
}

TEST_F(TestCudnnShimV8Operations, ScalePrecedesUnlowerableTensor)
{
    const Attr beta = SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA);
    cudnnBackendDescriptor_t op = build(operationRow("ResampleBackward"),
                                        SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC));
    ASSERT_EQ(set(op, beta.id, CUDNN_TYPE_DOUBLE, 0.5), CUDNN_STATUS_SUCCESS);
    expectRejected(op, CUDNN_STATUS_NOT_SUPPORTED, beta);
}

TEST_F(TestCudnnShimV8Operations, MissingRequiredPrecedesUnlowerableTensor)
{
    expectRejected(
        build(operationRow("NormForwardRmsTraining"), FWD_EPSILON.id, {tensorSlot(FWD_MEAN)}),
        CUDNN_STATUS_BAD_PARAM,
        FWD_EPSILON);
}

TEST_F(TestCudnnShimV8Operations, FailedFinalizeCanBeRetried)
{
    const Attr y = SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y);
    cudnnBackendDescriptor_t op = build(operationRow("ConvolutionForward"), y.id);
    expectRejected(op, CUDNN_STATUS_BAD_PARAM, y);

    cudnnBackendDescriptor_t target = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    std::int64_t count = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&target)),
              CUDNN_STATUS_NOT_INITIALIZED);
    lastError();

    ASSERT_EQ(setDescriptor(op, y.id, tensor()), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS) << lastError();
    EXPECT_EQ(cudnnBackendGetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&target)),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(count, 1);
}

// The operation keeps its referents alive after the caller destroys their
// handles, as upstream builders do once the operation is built.
TEST_F(TestCudnnShimV8Operations, ReferentHandlesMayBeDestroyedFirst)
{
    const std::vector<std::int64_t> dims = {7, 6, 5, 4};
    cudnnBackendDescriptor_t op = create(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    const std::vector<std::pair<cudnnBackendAttributeName_t, cudnnBackendDescriptor_t>> referents
        = {{CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, convolution()},
           {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensor(dims)},
           {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, tensor()},
           {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, tensor()}};
    for(const auto& [name, referent] : referents)
    {
        ASSERT_EQ(setDescriptor(op, name, referent), CUDNN_STATUS_SUCCESS);
        release(referent);
    }
    ASSERT_EQ(cudnnBackendFinalize(op), CUDNN_STATUS_SUCCESS) << lastError();

    cudnnBackendDescriptor_t target = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    std::int64_t count = 0;
    ASSERT_EQ(cudnnBackendGetAttribute(op,
                                       CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&target)),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    std::array<std::int64_t, 4> filled{};
    ASSERT_EQ(cudnnBackendGetAttribute(target,
                                       CUDNN_ATTR_TENSOR_DIMENSIONS,
                                       CUDNN_TYPE_INT64,
                                       static_cast<std::int64_t>(filled.size()),
                                       &count,
                                       filled.data()),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(std::vector<std::int64_t>(filled.begin(), filled.end()), dims);
}

} // namespace
