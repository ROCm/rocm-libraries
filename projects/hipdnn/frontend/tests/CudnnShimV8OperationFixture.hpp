// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// C API helpers shared by the cuDNN v8 shim operation descriptor tests: a
// fixture that builds finalized referents, and the attributes each operation
// type requires or accepts, spelled out as the oracle for error messages.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>

#include "fake_backend/MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// Pairs an attribute with its spelling, the oracle for error messages.
#define SHIM_ATTR(id) \
    Attr              \
    {                 \
        (id), #id     \
    }

namespace hipdnn_shim_test
{

struct Attr
{
    cudnnBackendAttributeName_t id;
    const char* name;
};

enum class Ref
{
    TENSOR,
    CONVOLUTION,
    MATMUL,
    REDUCTION,
    RESAMPLE,
    POINTWISE,
    NORM_MODE,
    NORM_PHASE,
};

struct Slot
{
    Attr attr;
    Ref ref;
};

struct Row
{
    const char* label;
    cudnnBackendDescriptorType_t type;
    std::vector<Slot> required;
    std::vector<Slot> optional;
    std::optional<cudnnPointwiseMode_t> pointwiseMode;
    std::optional<cudnnBackendNormMode_t> normMode;
    std::optional<cudnnBackendNormFwdPhase_t> phase;
};

// NOLINTNEXTLINE(readability-identifier-naming): gtest finds the printer by this name.
inline void PrintTo(const Row& row, std::ostream* out)
{
    *out << row.label;
}

inline Slot tensorSlot(Attr attr)
{
    return {attr, Ref::TENSOR};
}

inline std::vector<Slot> pointwiseSlots(bool b, bool t)
{
    std::vector<Slot> slots
        = {{SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR), Ref::POINTWISE},
           tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_XDESC))};
    if(b)
    {
        slots.push_back(tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_BDESC)));
    }
    if(t)
    {
        slots.push_back(tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_TDESC)));
    }
    slots.push_back(tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_YDESC)));
    return slots;
}

inline std::vector<Slot> pointwiseBackwardSlots()
{
    return {{SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR), Ref::POINTWISE},
            tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_XDESC)),
            tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_DYDESC)),
            tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_POINTWISE_DXDESC))};
}

inline Row pointwiseRow(const char* label, cudnnPointwiseMode_t mode, std::vector<Slot> required)
{
    return {label,
            CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
            std::move(required),
            {},
            mode,
            std::nullopt,
            std::nullopt};
}

inline Row normForwardRow(const char* label,
                          cudnnBackendNormMode_t mode,
                          cudnnBackendNormFwdPhase_t phase,
                          const std::vector<Attr>& requiredTensors,
                          const std::vector<Attr>& optionalTensors)
{
    Row row{label,
            CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
            {{SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_MODE), Ref::NORM_MODE},
             {SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_PHASE), Ref::NORM_PHASE}},
            {},
            std::nullopt,
            mode,
            phase};
    for(const Attr& attr : requiredTensors)
    {
        row.required.push_back(tensorSlot(attr));
    }
    for(const Attr& attr : optionalTensors)
    {
        row.optional.push_back(tensorSlot(attr));
    }
    return row;
}

inline Row normBackwardRow(const char* label,
                           cudnnBackendNormMode_t mode,
                           const std::vector<Attr>& requiredTensors,
                           const std::vector<Attr>& optionalTensors)
{
    Row row{label,
            CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
            {{SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_MODE), Ref::NORM_MODE}},
            {},
            std::nullopt,
            mode,
            std::nullopt};
    for(const Attr& attr : requiredTensors)
    {
        row.required.push_back(tensorSlot(attr));
    }
    for(const Attr& attr : optionalTensors)
    {
        row.optional.push_back(tensorSlot(attr));
    }
    return row;
}

inline const Attr FWD_X = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_XDESC);
inline const Attr FWD_Y = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_YDESC);
inline const Attr FWD_SCALE = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC);
inline const Attr FWD_BIAS = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC);
inline const Attr FWD_EPSILON = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC);
inline const Attr FWD_MEAN = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC);
inline const Attr FWD_INV_VARIANCE = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC);
inline const Attr FWD_PEER_STATS = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS);
inline const std::array<Attr, 5> RUNNING_STATS = {
    SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC),
    SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC),
    SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC),
    SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC),
    SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC),
};
inline const Attr BWD_X = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_XDESC);
inline const Attr BWD_DY = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC);
inline const Attr BWD_SCALE = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC);
inline const Attr BWD_MEAN = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC);
inline const Attr BWD_INV_VARIANCE = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC);
inline const Attr BWD_DX = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC);
inline const Attr BWD_DSCALE = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC);
inline const Attr BWD_DBIAS = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC);
inline const Attr BWD_EPSILON = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC);
inline const Attr BWD_PEER_STATS = SHIM_ATTR(CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS);

inline std::vector<Attr> withRunningStats(std::vector<Attr> attrs)
{
    attrs.insert(attrs.end(), RUNNING_STATS.begin(), RUNNING_STATS.end());
    return attrs;
}

// One row per operation type and mode class (or norm mode and phase) the shim
// lowers: the attributes finalize requires, and the optional ones it accepts.
inline const std::vector<Row>& operationRows()
{
    static const std::vector<Row> s_rows = {
        {"ConvolutionForward",
         CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
         {{SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC), Ref::CONVOLUTION},
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y))},
         {},
         std::nullopt,
         std::nullopt,
         std::nullopt},
        {"ConvolutionBackwardData",
         CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
         {{SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC), Ref::CONVOLUTION},
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY))},
         {},
         std::nullopt,
         std::nullopt,
         std::nullopt},
        {"ConvolutionBackwardFilter",
         CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
         {{SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC), Ref::CONVOLUTION},
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY))},
         {},
         std::nullopt,
         std::nullopt,
         std::nullopt},
        pointwiseRow("PointwiseMathUnary", CUDNN_POINTWISE_SQRT, pointwiseSlots(false, false)),
        pointwiseRow("PointwiseMathBinary", CUDNN_POINTWISE_ADD, pointwiseSlots(true, false)),
        pointwiseRow(
            "PointwiseBinarySelect", CUDNN_POINTWISE_BINARY_SELECT, pointwiseSlots(true, true)),
        pointwiseRow("PointwiseIdentity", CUDNN_POINTWISE_IDENTITY, pointwiseSlots(false, false)),
        pointwiseRow(
            "PointwiseActivationForward", CUDNN_POINTWISE_RELU_FWD, pointwiseSlots(false, false)),
        pointwiseRow(
            "PointwiseActivationBackward", CUDNN_POINTWISE_RELU_BWD, pointwiseBackwardSlots()),
        {"Matmul",
         CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
         {{SHIM_ATTR(CUDNN_ATTR_OPERATION_MATMUL_DESC), Ref::MATMUL},
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_MATMUL_ADESC)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_MATMUL_BDESC)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_MATMUL_CDESC))},
         {},
         std::nullopt,
         std::nullopt,
         std::nullopt},
        {"Reduction",
         CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
         {{SHIM_ATTR(CUDNN_ATTR_OPERATION_REDUCTION_DESC), Ref::REDUCTION},
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_REDUCTION_XDESC)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_REDUCTION_YDESC))},
         {},
         std::nullopt,
         std::nullopt,
         std::nullopt},
        {"ResampleForward",
         CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
         {{SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC), Ref::RESAMPLE},
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC))},
         {tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC))},
         std::nullopt,
         std::nullopt,
         std::nullopt},
        {"ResampleBackward",
         CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
         {{SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC), Ref::RESAMPLE},
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC)),
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC)),
          // The RESAMPLE referent is a max pool, which needs the index.
          tensorSlot(SHIM_ATTR(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC))},
         {},
         std::nullopt,
         std::nullopt,
         std::nullopt},
        normForwardRow("NormForwardBatchTraining",
                       CUDNN_BATCH_NORM,
                       CUDNN_NORM_FWD_TRAINING,
                       {FWD_X, FWD_SCALE, FWD_BIAS, FWD_EPSILON, FWD_Y},
                       withRunningStats({FWD_MEAN, FWD_INV_VARIANCE})),
        normForwardRow("NormForwardBatchInference",
                       CUDNN_BATCH_NORM,
                       CUDNN_NORM_FWD_INFERENCE,
                       {FWD_X, FWD_MEAN, FWD_INV_VARIANCE, FWD_SCALE, FWD_BIAS, FWD_Y},
                       {FWD_EPSILON}),
        normForwardRow("NormForwardLayerTraining",
                       CUDNN_LAYER_NORM,
                       CUDNN_NORM_FWD_TRAINING,
                       {FWD_X, FWD_SCALE, FWD_BIAS, FWD_EPSILON, FWD_Y},
                       {FWD_MEAN, FWD_INV_VARIANCE}),
        normForwardRow("NormForwardLayerInference",
                       CUDNN_LAYER_NORM,
                       CUDNN_NORM_FWD_INFERENCE,
                       {FWD_X, FWD_SCALE, FWD_BIAS, FWD_EPSILON, FWD_Y},
                       {FWD_MEAN, FWD_INV_VARIANCE}),
        normForwardRow("NormForwardRmsTraining",
                       CUDNN_RMS_NORM,
                       CUDNN_NORM_FWD_TRAINING,
                       {FWD_X, FWD_SCALE, FWD_EPSILON, FWD_Y},
                       {FWD_BIAS, FWD_INV_VARIANCE}),
        normForwardRow("NormForwardRmsInference",
                       CUDNN_RMS_NORM,
                       CUDNN_NORM_FWD_INFERENCE,
                       {FWD_X, FWD_SCALE, FWD_EPSILON, FWD_Y},
                       {FWD_BIAS, FWD_INV_VARIANCE}),
        normBackwardRow(
            "NormBackwardBatch",
            CUDNN_BATCH_NORM,
            {BWD_X, BWD_DY, BWD_SCALE, BWD_MEAN, BWD_INV_VARIANCE, BWD_DX, BWD_DSCALE, BWD_DBIAS},
            {BWD_EPSILON}),
        normBackwardRow("NormBackwardLayer",
                        CUDNN_LAYER_NORM,
                        {BWD_X, BWD_DY, BWD_SCALE, BWD_DX, BWD_DSCALE, BWD_DBIAS},
                        {BWD_MEAN, BWD_INV_VARIANCE, BWD_EPSILON}),
        normBackwardRow("NormBackwardRms",
                        CUDNN_RMS_NORM,
                        {BWD_X, BWD_DY, BWD_SCALE, BWD_INV_VARIANCE, BWD_DX, BWD_DSCALE},
                        {BWD_DBIAS}),
    };
    return s_rows;
}

inline const Row& operationRow(const std::string& label)
{
    const std::vector<Row>& rows = operationRows();
    const auto found = std::find_if(
        rows.begin(), rows.end(), [&](const Row& row) { return row.label == label; });
    if(found == rows.end())
    {
        ADD_FAILURE() << "no row " << label;
        return rows.front();
    }
    return *found;
}

inline std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

class ShimOperationsFixture : public ShimMockBackendFixture
{
protected:
    std::vector<cudnnBackendDescriptor_t> _owned;
    std::int64_t _nextUid = 1;

    void SetUp() override
    {
        ShimMockBackendFixture::SetUp();
        lastError();
    }

    void TearDown() override
    {
        for(cudnnBackendDescriptor_t descriptor : _owned)
        {
            cudnnBackendDestroyDescriptor(descriptor);
        }
        _owned.clear();
        ShimMockBackendFixture::TearDown();
    }

    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t descriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(type, &descriptor), CUDNN_STATUS_SUCCESS);
        _owned.push_back(descriptor);
        return descriptor;
    }

    // Destroys the caller's handle early; the object may outlive it.
    void release(cudnnBackendDescriptor_t descriptor)
    {
        _owned.erase(std::remove(_owned.begin(), _owned.end(), descriptor), _owned.end());
        EXPECT_EQ(cudnnBackendDestroyDescriptor(descriptor), CUDNN_STATUS_SUCCESS);
    }

    template <typename T>
    static cudnnStatus_t set(cudnnBackendDescriptor_t descriptor,
                             cudnnBackendAttributeName_t name,
                             cudnnBackendAttributeType_t type,
                             const T& value)
    {
        return cudnnBackendSetAttribute(
            descriptor, name, type, 1, static_cast<const void*>(&value));
    }

    template <typename T>
    static cudnnStatus_t setArray(cudnnBackendDescriptor_t descriptor,
                                  cudnnBackendAttributeName_t name,
                                  cudnnBackendAttributeType_t type,
                                  const std::vector<T>& values)
    {
        // A non-null pointer even for count 0, as upstream's fixed arrays give.
        static const T s_empty{};
        return cudnnBackendSetAttribute(descriptor,
                                        name,
                                        type,
                                        static_cast<std::int64_t>(values.size()),
                                        values.empty() ? static_cast<const void*>(&s_empty)
                                                       : static_cast<const void*>(values.data()));
    }

    static cudnnStatus_t setDescriptor(cudnnBackendDescriptor_t descriptor,
                                       cudnnBackendAttributeName_t name,
                                       cudnnBackendDescriptor_t value)
    {
        return set(descriptor, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, value);
    }

    static void expectRejected(cudnnBackendDescriptor_t descriptor,
                               cudnnStatus_t status,
                               const std::vector<std::string>& anyOf)
    {
        EXPECT_EQ(cudnnBackendFinalize(descriptor), status);
        const std::string message = lastError();
        EXPECT_TRUE(std::any_of(anyOf.begin(), anyOf.end(), [&](const std::string& name) {
            return message.find(name) != std::string::npos;
        })) << message;
    }

    static void
        expectRejected(cudnnBackendDescriptor_t descriptor, cudnnStatus_t status, const Attr& attr)
    {
        expectRejected(descriptor, status, std::vector<std::string>{attr.name});
    }

    // Finalized and packed, with a fresh UID.
    cudnnBackendDescriptor_t tensor(const std::vector<std::int64_t>& dims = {2, 3, 4, 5})
    {
        std::vector<std::int64_t> strides(dims.size(), 1);
        for(std::size_t i = dims.size() - 1; i > 0; --i)
        {
            strides[i - 1] = strides[i] * dims[i];
        }
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        EXPECT_EQ(
            set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(setArray(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dims),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(setArray(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, strides),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, _nextUid++),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(set(descriptor,
                      CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                      CUDNN_TYPE_INT64,
                      static_cast<std::int64_t>(16)),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
        return descriptor;
    }

    cudnnBackendDescriptor_t convolution(std::int64_t spatialDims = 2)
    {
        const auto count = static_cast<std::size_t>(spatialDims);
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        EXPECT_EQ(set(descriptor,
                      CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                      CUDNN_TYPE_DATA_TYPE,
                      CUDNN_DATA_FLOAT),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(set(descriptor,
                      CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                      CUDNN_TYPE_CONVOLUTION_MODE,
                      CUDNN_CROSS_CORRELATION),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            set(descriptor, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, spatialDims),
            CUDNN_STATUS_SUCCESS);
        const std::vector<std::int64_t> zeros(count, 0);
        const std::vector<std::int64_t> ones(count, 1);
        EXPECT_EQ(
            setArray(descriptor, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, zeros),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            setArray(descriptor, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, zeros),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(setArray(descriptor, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, ones),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            setArray(descriptor, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, ones),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
        return descriptor;
    }

    cudnnBackendDescriptor_t pointwise(cudnnPointwiseMode_t mode)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        EXPECT_EQ(set(descriptor, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, mode),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            set(descriptor, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
        return descriptor;
    }

    cudnnBackendDescriptor_t matmul()
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        EXPECT_EQ(
            set(descriptor, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
        return descriptor;
    }

    cudnnBackendDescriptor_t reduction()
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        EXPECT_EQ(set(descriptor,
                      CUDNN_ATTR_REDUCTION_OPERATOR,
                      CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                      CUDNN_REDUCE_TENSOR_ADD),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            set(descriptor, CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
        return descriptor;
    }

    cudnnBackendDescriptor_t resample(cudnnResampleMode_t mode = CUDNN_RESAMPLE_MAXPOOL)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        EXPECT_EQ(set(descriptor, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, mode),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            set(descriptor, CUDNN_ATTR_RESAMPLE_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(set(descriptor,
                      CUDNN_ATTR_RESAMPLE_PADDING_MODE,
                      CUDNN_TYPE_PADDING_MODE,
                      CUDNN_NEG_INF_PAD),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(set(descriptor,
                      CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS,
                      CUDNN_TYPE_INT64,
                      static_cast<std::int64_t>(2)),
                  CUDNN_STATUS_SUCCESS);
        const std::vector<cudnnFraction_t> windows(2, {2, 1});
        const std::vector<cudnnFraction_t> zeros(2, {0, 1});
        const std::vector<cudnnFraction_t> strides(2, {2, 1});
        EXPECT_EQ(
            setArray(descriptor, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_FRACTION, windows),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            setArray(descriptor, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, CUDNN_TYPE_FRACTION, zeros),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(
            setArray(descriptor, CUDNN_ATTR_RESAMPLE_POST_PADDINGS, CUDNN_TYPE_FRACTION, zeros),
            CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(setArray(descriptor, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_FRACTION, strides),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS);
        return descriptor;
    }

    cudnnStatus_t setSlot(cudnnBackendDescriptor_t op, const Slot& slot, const Row& row)
    {
        switch(slot.ref)
        {
        case Ref::NORM_MODE:
            return set(op, slot.attr.id, CUDNN_TYPE_NORM_MODE, row.normMode.value());
        case Ref::NORM_PHASE:
            return set(op, slot.attr.id, CUDNN_TYPE_NORM_FWD_PHASE, row.phase.value());
        case Ref::CONVOLUTION:
            return setDescriptor(op, slot.attr.id, convolution());
        case Ref::MATMUL:
            return setDescriptor(op, slot.attr.id, matmul());
        case Ref::REDUCTION:
            return setDescriptor(op, slot.attr.id, reduction());
        case Ref::RESAMPLE:
            return setDescriptor(op, slot.attr.id, resample());
        case Ref::POINTWISE:
            return setDescriptor(op, slot.attr.id, pointwise(row.pointwiseMode.value()));
        case Ref::TENSOR:
        default:
            return setDescriptor(op, slot.attr.id, tensor());
        }
    }

    // The row's required attributes minus `skip`, plus `extra`; not finalized.
    cudnnBackendDescriptor_t build(const Row& row,
                                   std::optional<cudnnBackendAttributeName_t> skip = std::nullopt,
                                   const std::vector<Slot>& extra = {})
    {
        cudnnBackendDescriptor_t op = create(row.type);
        for(const Slot& slot : row.required)
        {
            if(slot.attr.id != skip)
            {
                EXPECT_EQ(setSlot(op, slot, row), CUDNN_STATUS_SUCCESS) << slot.attr.name;
            }
        }
        for(const Slot& slot : extra)
        {
            EXPECT_EQ(setSlot(op, slot, row), CUDNN_STATUS_SUCCESS) << slot.attr.name;
        }
        return op;
    }

    cudnnBackendDescriptor_t build(const Row& row, const Attr& extraTensor)
    {
        return build(row, std::nullopt, {tensorSlot(extraTensor)});
    }
};

} // namespace hipdnn_shim_test
