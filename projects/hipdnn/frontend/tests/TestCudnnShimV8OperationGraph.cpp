// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// OPERATIONGRAPH lowering and engine index space against the mock hipDNN
// backend: what reaches hipDNN, in which order, and how engines are indexed.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

namespace backend = hipdnn_frontend::compatibility::cudnn_frontend::detail::backend;

using Slots = std::vector<std::pair<cudnnBackendAttributeName_t, cudnnBackendDescriptor_t>>;

struct TensorSpec
{
    std::int64_t uid;
    std::vector<std::int64_t> dims;
    bool isVirtual = false;
    bool byValue = false;
    std::int64_t alignment = 16;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    cudnnBackendDescriptor_t raggedOffset = nullptr;
};

// One norm operation per lowered target, and the hipDNN slot each of its
// tensor UIDs must land in.
struct NormCase
{
    const char* label;
    std::function<cudnnBackendDescriptor_t()> build;
    hipdnnBackendDescriptorType_t operationType;
    hipdnnBackendAttributeName_t computeType;
    std::vector<std::pair<hipdnnBackendAttributeName_t, std::int64_t>> slots;
};

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

class TestCudnnShimV8OperationGraph : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    std::vector<cudnnBackendDescriptor_t> _owned;

    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        lastError();
    }

    void TearDown() override
    {
        for(cudnnBackendDescriptor_t descriptor : _owned)
        {
            cudnnBackendDestroyDescriptor(descriptor);
        }
        _owned.clear();
        ShimV8MockBackendFixture::TearDown();
    }

    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t descriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(type, &descriptor), CUDNN_STATUS_SUCCESS);
        _owned.push_back(descriptor);
        return descriptor;
    }

    template <typename T>
    static void set(cudnnBackendDescriptor_t descriptor,
                    cudnnBackendAttributeName_t name,
                    cudnnBackendAttributeType_t type,
                    const T& value)
    {
        EXPECT_EQ(
            cudnnBackendSetAttribute(descriptor, name, type, 1, static_cast<const void*>(&value)),
            CUDNN_STATUS_SUCCESS)
            << lastError();
    }

    template <typename T>
    static void setArray(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t name,
                         cudnnBackendAttributeType_t type,
                         const std::vector<T>& values)
    {
        EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                           name,
                                           type,
                                           static_cast<std::int64_t>(values.size()),
                                           static_cast<const void*>(values.data())),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
    }

    static void finalize(cudnnBackendDescriptor_t descriptor)
    {
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS) << lastError();
    }

    cudnnBackendDescriptor_t tensor(const TensorSpec& spec)
    {
        std::vector<std::int64_t> strides(spec.dims.size(), 1);
        for(std::size_t i = spec.dims.size() - 1; i > 0; --i)
        {
            strides[i - 1] = strides[i] * spec.dims[i];
        }
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, spec.dataType);
        setArray(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, spec.dims);
        setArray(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, strides);
        set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, spec.uid);
        set(descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, spec.alignment);
        set(descriptor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, spec.isVirtual);
        set(descriptor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, spec.byValue);
        if(spec.raggedOffset != nullptr)
        {
            set(descriptor,
                CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                spec.raggedOffset);
        }
        finalize(descriptor);
        return descriptor;
    }

    // A [1,1,1,1] compile-time constant; the value is copied at finalize.
    template <typename T>
    cudnnBackendDescriptor_t constant(std::int64_t uid, cudnnDataType_t dataType, T value)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, dataType);
        setArray(descriptor,
                 CUDNN_ATTR_TENSOR_DIMENSIONS,
                 CUDNN_TYPE_INT64,
                 std::vector<std::int64_t>{1, 1, 1, 1});
        setArray(descriptor,
                 CUDNN_ATTR_TENSOR_STRIDES,
                 CUDNN_TYPE_INT64,
                 std::vector<std::int64_t>{1, 1, 1, 1});
        set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, uid);
        set(descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, std::int64_t{16});
        set(descriptor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, true);
        set(descriptor,
            CUDNN_ATTR_TENSOR_CONSTANT_VALUE,
            CUDNN_TYPE_VOID_PTR,
            static_cast<const void*>(&value));
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t scalar(std::int64_t uid)
    {
        return tensor({uid, {1, 1, 1, 1}, false, true});
    }

    cudnnBackendDescriptor_t operation(cudnnBackendDescriptorType_t type, const Slots& slots)
    {
        cudnnBackendDescriptor_t op = create(type);
        for(const auto& [name, value] : slots)
        {
            set(op, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, value);
        }
        return op;
    }

    cudnnBackendDescriptor_t convolution(const std::vector<std::int64_t>& prePadding,
                                         const std::vector<std::int64_t>& postPadding,
                                         const std::vector<std::int64_t>& strides,
                                         const std::vector<std::int64_t>& dilations)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        set(descriptor,
            CUDNN_ATTR_CONVOLUTION_CONV_MODE,
            CUDNN_TYPE_CONVOLUTION_MODE,
            CUDNN_CROSS_CORRELATION);
        set(descriptor,
            CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
            CUDNN_TYPE_INT64,
            static_cast<std::int64_t>(prePadding.size()));
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, prePadding);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, postPadding);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, dilations);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, strides);
        finalize(descriptor);
        return descriptor;
    }

    // A 3x3-filter convolution with unit padding, stride and dilation.
    cudnnBackendDescriptor_t sameConvolution()
    {
        return convolution({1, 1}, {1, 1}, {1, 1}, {1, 1});
    }

    cudnnBackendDescriptor_t finalizedOperation(cudnnBackendDescriptorType_t type,
                                                const Slots& slots)
    {
        cudnnBackendDescriptor_t op = operation(type, slots);
        finalize(op);
        return op;
    }

    cudnnBackendDescriptor_t convolutionForward(cudnnBackendDescriptor_t x,
                                                cudnnBackendDescriptor_t w,
                                                cudnnBackendDescriptor_t y,
                                                cudnnBackendDescriptor_t convolutionDescriptor)
    {
        return finalizedOperation(
            CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
            {{CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, convolutionDescriptor},
             {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, x},
             {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, w},
             {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, y}});
    }

    cudnnBackendDescriptor_t pointwiseDescriptor(cudnnPointwiseMode_t mode,
                                                 std::optional<double> lowerClip = std::nullopt)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, mode);
        set(descriptor, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        // Upstream sends every parameter of the mode, defaults included.
        set(descriptor,
            CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
            CUDNN_TYPE_DOUBLE,
            lowerClip.value_or(0.0));
        set(descriptor,
            CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
            CUDNN_TYPE_DOUBLE,
            std::numeric_limits<double>::max());
        set(descriptor, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE, CUDNN_TYPE_DOUBLE, 0.0);
        finalize(descriptor);
        return descriptor;
    }

    // x [op b] -> y
    cudnnBackendDescriptor_t pointwise(cudnnPointwiseMode_t mode,
                                       cudnnBackendDescriptor_t x,
                                       cudnnBackendDescriptor_t b,
                                       cudnnBackendDescriptor_t y,
                                       std::optional<double> lowerClip = std::nullopt)
    {
        Slots slots
            = {{CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pointwiseDescriptor(mode, lowerClip)},
               {CUDNN_ATTR_OPERATION_POINTWISE_XDESC, x},
               {CUDNN_ATTR_OPERATION_POINTWISE_YDESC, y}};
        if(b != nullptr)
        {
            slots.emplace_back(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, b);
        }
        cudnnBackendDescriptor_t op
            = operation(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, slots);
        finalize(op);
        return op;
    }

    cudnnBackendDescriptor_t normForward(cudnnBackendNormMode_t mode,
                                         cudnnBackendNormFwdPhase_t phase,
                                         const Slots& slots)
    {
        cudnnBackendDescriptor_t op
            = operation(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR, slots);
        set(op, CUDNN_ATTR_OPERATION_NORM_FWD_MODE, CUDNN_TYPE_NORM_MODE, mode);
        set(op, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE, CUDNN_TYPE_NORM_FWD_PHASE, phase);
        finalize(op);
        return op;
    }

    cudnnBackendDescriptor_t normBackward(cudnnBackendNormMode_t mode, const Slots& slots)
    {
        cudnnBackendDescriptor_t op
            = operation(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR, slots);
        set(op, CUDNN_ATTR_OPERATION_NORM_BWD_MODE, CUDNN_TYPE_NORM_MODE, mode);
        finalize(op);
        return op;
    }

    // Not finalized.
    cudnnBackendDescriptor_t graph(const std::vector<cudnnBackendDescriptor_t>& ops)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, _handle);
        setArray(descriptor, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, ops);
        return descriptor;
    }

    cudnnBackendDescriptor_t finalizedGraph(const std::vector<cudnnBackendDescriptor_t>& ops)
    {
        cudnnBackendDescriptor_t descriptor = graph(ops);
        finalize(descriptor);
        return descriptor;
    }

    // conv(x:1, w:2) -> 3 (virtual)
    cudnnBackendDescriptor_t smallConvolution(std::int64_t yUid = 3, bool yVirtual = true)
    {
        return convolutionForward(tensor({1, {1, 4, 8, 8}}),
                                  tensor({2, {8, 4, 3, 3}}),
                                  tensor({yUid, {1, 8, 8, 8}, yVirtual}),
                                  sameConvolution());
    }

    static backend::OperationGraphDescriptor& shimGraph(cudnnBackendDescriptor_t descriptor)
    {
        return backend::descriptorCast<backend::OperationGraphDescriptor>(
            *backend::liveHandle(descriptor)->object, CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    }

    static std::int64_t engineCount(cudnnBackendDescriptor_t descriptor)
    {
        std::int64_t count = -1;
        std::int64_t written = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(descriptor,
                                           CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &written,
                                           static_cast<void*>(&count)),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        return count;
    }

    // Calls on hipDNN descriptors of `type` that set `name`.
    std::vector<hipdnn_shim_test::CapturedSetAttribute>
        calls(hipdnnBackendDescriptorType_t type, hipdnnBackendAttributeName_t name) const
    {
        std::vector<hipdnn_shim_test::CapturedSetAttribute> result = setAttributesOn(type);
        result.erase(std::remove_if(result.begin(),
                                    result.end(),
                                    [name](const hipdnn_shim_test::CapturedSetAttribute& call) {
                                        return call.name != name;
                                    }),
                     result.end());
        return result;
    }

    std::vector<hipdnn_shim_test::CapturedSetAttribute>
        callsNamed(hipdnnBackendAttributeName_t name) const
    {
        std::vector<hipdnn_shim_test::CapturedSetAttribute> result = setAttributes();
        result.erase(std::remove_if(result.begin(),
                                    result.end(),
                                    [name](const hipdnn_shim_test::CapturedSetAttribute& call) {
                                        return call.name != name;
                                    }),
                     result.end());
        return result;
    }

    std::optional<hipdnn_shim_test::CapturedSetAttribute>
        callOn(hipdnnBackendDescriptor_t descriptor, hipdnnBackendAttributeName_t name) const
    {
        for(const hipdnn_shim_test::CapturedSetAttribute& call : setAttributesOf(descriptor))
        {
            if(call.name == name)
            {
                return call;
            }
        }
        return std::nullopt;
    }

    // UID of the hipDNN tensor an operation slot refers to, or -1 if unset.
    std::int64_t slotUid(hipdnnBackendDescriptor_t op, hipdnnBackendAttributeName_t slot) const
    {
        const auto reference = callOn(op, slot);
        if(!reference)
        {
            return -1;
        }
        const auto uid
            = callOn(reference->value<hipdnnBackendDescriptor_t>(), HIPDNN_ATTR_TENSOR_UNIQUE_ID);
        return uid ? uid->value<std::int64_t>() : -1;
    }

    // The hipDNN graph's OPS array, as each operation's pointwise mode (or -1
    // for a non-pointwise operation).
    std::vector<int> pointwiseModesInGraphOrder() const
    {
        const auto ops
            = calls(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, HIPDNN_ATTR_OPERATIONGRAPH_OPS);
        EXPECT_EQ(ops.size(), 1U);
        std::vector<int> modes;
        if(ops.empty())
        {
            return modes;
        }
        for(hipdnnBackendDescriptor_t op : ops.front().values<hipdnnBackendDescriptor_t>())
        {
            const auto mode = callOn(op, HIPDNN_ATTR_POINTWISE_MODE);
            modes.push_back(mode ? static_cast<int>(mode->value<hipdnnPointwiseMode_t>()) : -1);
        }
        return modes;
    }

    // The hipDNN graph's OPS array, as the UID each operation writes through
    // `outputSlot` (or -1).
    std::vector<std::int64_t> outputUidsInGraphOrder(hipdnnBackendAttributeName_t outputSlot) const
    {
        const auto ops
            = calls(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, HIPDNN_ATTR_OPERATIONGRAPH_OPS);
        std::vector<std::int64_t> uids;
        if(ops.empty())
        {
            return uids;
        }
        for(hipdnnBackendDescriptor_t op : ops.front().values<hipdnnBackendDescriptor_t>())
        {
            uids.push_back(slotUid(op, outputSlot));
        }
        return uids;
    }

    // Every tensor has its own UID, and tensors that share a shape are the ones
    // a swapped slot would confuse.
    std::vector<NormCase> normCases()
    {
        return {
            {"batch training",
             [this] {
                 return normForward(
                     CUDNN_BATCH_NORM,
                     CUDNN_NORM_FWD_TRAINING,
                     {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, tensor({2, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, tensor({3, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(4)},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, tensor({5, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, tensor({6, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                       tensor({7, {1, 3, 1, 1}})}});
             },
             HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT,
             HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, 2},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, 3},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, 4},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, 5},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, 6},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT, 7}}},
            {"batch inference",
             [this] {
                 return normForward(
                     CUDNN_BATCH_NORM,
                     CUDNN_NORM_FWD_INFERENCE,
                     {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, tensor({2, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, tensor({3, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, tensor({4, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, tensor({5, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, tensor({6, {2, 3, 4, 4}})}});
             },
             HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT,
             HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT, 2},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT, 3},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT, 4},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT, 5},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT, 6}}},
            {"batch backward",
             [this] {
                 return normBackward(
                     CUDNN_BATCH_NORM,
                     {{CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, tensor({2, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, tensor({3, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, tensor({4, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, tensor({5, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, tensor({6, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, tensor({7, {1, 3, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, tensor({8, {1, 3, 1, 1}})}});
             },
             HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT,
             HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, 2},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, 3},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, 4},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, 5},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, 6},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, 7},
              {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, 8}}},
            {"layer training",
             [this] {
                 return normForward(
                     CUDNN_LAYER_NORM,
                     CUDNN_NORM_FWD_TRAINING,
                     {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, tensor({2, {1, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, tensor({3, {1, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(4)},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, tensor({5, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, tensor({6, {2, 1, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                       tensor({7, {2, 1, 1, 1}})}});
             },
             HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT,
             HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT, 2},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT, 3},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT, 4},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT, 5},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT, 6},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT, 7}}},
            {"layer backward",
             [this] {
                 return normBackward(
                     CUDNN_LAYER_NORM,
                     {{CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, tensor({2, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, tensor({3, {1, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, tensor({4, {2, 1, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, tensor({5, {2, 1, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, tensor({6, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, tensor({7, {1, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, tensor({8, {1, 3, 4, 4}})}});
             },
             HIPDNN_BACKEND_OPERATION_LAYERNORM_BACKWARD_DESCRIPTOR_EXT,
             HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY_EXT, 2},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE_EXT, 3},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN_EXT, 4},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE_EXT, 5},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX_EXT, 6},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE_EXT, 7},
              {HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS_EXT, 8}}},
            {"rms training",
             [this] {
                 return normForward(
                     CUDNN_RMS_NORM,
                     CUDNN_NORM_FWD_TRAINING,
                     {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, tensor({2, {1, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(3)},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, tensor({4, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC, tensor({5, {2, 1, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, tensor({6, {1, 3, 4, 4}})}});
             },
             HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT,
             HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT, 2},
              {HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT, 3},
              {HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT, 4},
              {HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT, 5},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT, 6}}},
            {"rms backward",
             [this] {
                 return normBackward(
                     CUDNN_RMS_NORM,
                     {{CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, tensor({2, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, tensor({3, {1, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, tensor({4, {2, 1, 1, 1}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, tensor({5, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, tensor({6, {1, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, tensor({7, {1, 3, 4, 4}})}});
             },
             HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT,
             HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT,
             {{HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT, 1},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT, 2},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT, 3},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT, 4},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT, 5},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT, 6},
              {HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT, 7}}},
        };
    }

    cudnnBackendDescriptor_t
        reduction(std::int64_t xUid, std::int64_t yUid, std::optional<bool> isDeterministic)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        set(descriptor,
            CUDNN_ATTR_REDUCTION_OPERATOR,
            CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
            CUDNN_REDUCE_TENSOR_ADD);
        set(descriptor, CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        if(isDeterministic)
        {
            set(descriptor,
                CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC,
                CUDNN_TYPE_BOOLEAN,
                *isDeterministic);
        }
        finalize(descriptor);
        return finalizedOperation(
            CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
            {{CUDNN_ATTR_OPERATION_REDUCTION_DESC, descriptor},
             {CUDNN_ATTR_OPERATION_REDUCTION_XDESC, tensor({xUid, {2, 3, 4, 5}})},
             {CUDNN_ATTR_OPERATION_REDUCTION_YDESC, tensor({yUid, {2, 3, 1, 1}})}});
    }

    // 2x2 max pooling with stride 2 and no padding.
    cudnnBackendDescriptor_t maxPool()
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, CUDNN_RESAMPLE_MAXPOOL);
        set(descriptor, CUDNN_ATTR_RESAMPLE_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        set(descriptor,
            CUDNN_ATTR_RESAMPLE_PADDING_MODE,
            CUDNN_TYPE_PADDING_MODE,
            CUDNN_NEG_INF_PAD);
        set(descriptor, CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, CUDNN_TYPE_INT64, std::int64_t{2});
        const std::vector<cudnnFraction_t> twos(2, {2, 1});
        const std::vector<cudnnFraction_t> zeros(2, {0, 1});
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_FRACTION, twos);
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, CUDNN_TYPE_FRACTION, zeros);
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_POST_PADDINGS, CUDNN_TYPE_FRACTION, zeros);
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_FRACTION, twos);
        finalize(descriptor);
        return descriptor;
    }

    // The hipDNN tensor created for `uid` (the last one, if several graphs used it).
    hipdnnBackendDescriptor_t hipdnnTensor(std::int64_t uid) const
    {
        hipdnnBackendDescriptor_t found = nullptr;
        for(const auto& call : callsNamed(HIPDNN_ATTR_TENSOR_UNIQUE_ID))
        {
            if(call.value<std::int64_t>() == uid)
            {
                found = call.descriptor;
            }
        }
        EXPECT_NE(found, nullptr) << "no hipDNN tensor with UID " << uid;
        return found;
    }

    // The hipDNN pointwise operation whose mode is `mode`.
    hipdnnBackendDescriptor_t pointwiseOperation(hipdnnPointwiseMode_t mode) const
    {
        for(hipdnnBackendDescriptor_t op : created(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR))
        {
            const auto set = callOn(op, HIPDNN_ATTR_POINTWISE_MODE);
            if(set && set->value<hipdnnPointwiseMode_t>() == mode)
            {
                return op;
            }
        }
        ADD_FAILURE() << "no pointwise operation with mode " << mode;
        return nullptr;
    }

    hipdnnBackendDescriptor_t onlyCreated(hipdnnBackendDescriptorType_t type) const
    {
        const std::vector<hipdnnBackendDescriptor_t> all = created(type);
        EXPECT_EQ(all.size(), 1U);
        return all.empty() ? nullptr : all.front();
    }
};

TEST_F(TestCudnnShimV8OperationGraph, ConvolutionParametersReachOperation)
{
    // Pre and post paddings differ per axis but keep the per-axis sums, so Y
    // is still [1,8,4,6].
    cudnnBackendDescriptor_t conv = convolutionForward(tensor({1, {1, 4, 8, 8}}),
                                                       tensor({2, {8, 4, 3, 3}}),
                                                       tensor({3, {1, 8, 4, 6}}),
                                                       convolution({1, 0}, {1, 2}, {2, 1}, {1, 2}));
    finalizedGraph({conv});

    hipdnnBackendDescriptor_t op
        = onlyCreated(HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    EXPECT_EQ(callOn(op, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS)->values<std::int64_t>(),
              (std::vector<std::int64_t>{1, 0}));
    EXPECT_EQ(callOn(op, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS)->values<std::int64_t>(),
              (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(callOn(op, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES)->values<std::int64_t>(),
              (std::vector<std::int64_t>{2, 1}));
    EXPECT_EQ(callOn(op, HIPDNN_ATTR_CONVOLUTION_DILATIONS)->values<std::int64_t>(),
              (std::vector<std::int64_t>{1, 2}));
    EXPECT_EQ(callOn(op, HIPDNN_ATTR_CONVOLUTION_CONV_MODE)->value<hipdnnConvolutionMode_t>(),
              HIPDNN_CROSS_CORRELATION);
    EXPECT_EQ(callOn(op, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE)->value<hipdnnDataType_t>(),
              HIPDNN_DATA_FLOAT);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X), 1);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W), 2);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y), 3);

    hipdnnBackendDescriptor_t graphDescriptor
        = onlyCreated(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    EXPECT_EQ(callOn(graphDescriptor, HIPDNN_ATTR_OPERATIONGRAPH_HANDLE)->value<hipdnnHandle_t>(),
              _handle);
}

TEST_F(TestCudnnShimV8OperationGraph, PointwiseForwardSlotsMapInOrder)
{
    cudnnBackendDescriptor_t add = pointwise(CUDNN_POINTWISE_ADD,
                                             tensor({10, {2, 3, 4, 5}}),
                                             tensor({11, {1, 3, 1, 1}}),
                                             tensor({12, {2, 3, 4, 5}}));
    finalizedGraph({add});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT), 10);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT), 11);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT), -1);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT), 12);
    EXPECT_EQ(callOn(op, HIPDNN_ATTR_POINTWISE_MODE)->value<hipdnnPointwiseMode_t>(),
              HIPDNN_POINTWISE_ADD);
    EXPECT_EQ(callOn(op, HIPDNN_ATTR_POINTWISE_MATH_PREC)->value<hipdnnDataType_t>(),
              HIPDNN_DATA_FLOAT);
}

TEST_F(TestCudnnShimV8OperationGraph, PointwiseBackwardSlotsMapGradientFirst)
{
    cudnnBackendDescriptor_t reluBackward
        = operation(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                    {{CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                      pointwiseDescriptor(CUDNN_POINTWISE_RELU_BWD)},
                     {CUDNN_ATTR_OPERATION_POINTWISE_DYDESC, tensor({20, {2, 3, 4, 5}})},
                     {CUDNN_ATTR_OPERATION_POINTWISE_XDESC, tensor({21, {2, 3, 4, 5}})},
                     {CUDNN_ATTR_OPERATION_POINTWISE_DXDESC, tensor({22, {2, 3, 4, 5}})}});
    finalize(reluBackward);
    finalizedGraph({reluBackward});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT), 20);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT), 21);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT), 22);
}

TEST_F(TestCudnnShimV8OperationGraph, DefaultReluSendsOnlyLowerClip)
{
    finalizedGraph({pointwise(
        CUDNN_POINTWISE_RELU_FWD, tensor({1, {2, 3, 4, 5}}), nullptr, tensor({2, {2, 3, 4, 5}}))});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    const auto lowerClip = callOn(op, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP);
    ASSERT_TRUE(lowerClip);
    EXPECT_EQ(lowerClip->type, HIPDNN_TYPE_FLOAT);
    EXPECT_EQ(lowerClip->value<float>(), 0.0F);
    EXPECT_FALSE(callOn(op, HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP));
    EXPECT_FALSE(callOn(op, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE));
    EXPECT_TRUE(callOn(op, HIPDNN_ATTR_POINTWISE_MODE));
}

TEST_F(TestCudnnShimV8OperationGraph, OutOfRangeClipSaturates)
{
    finalizedGraph({pointwise(CUDNN_POINTWISE_RELU_FWD,
                              tensor({1, {2, 3, 4, 5}}),
                              nullptr,
                              tensor({2, {2, 3, 4, 5}}),
                              -1e300)});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    const auto clip = callOn(op, HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP);
    ASSERT_TRUE(clip);
    EXPECT_EQ(clip->type, HIPDNN_TYPE_FLOAT);
    EXPECT_EQ(clip->value<float>(), -std::numeric_limits<float>::max());
    EXPECT_FALSE(callOn(op, HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP));
}

TEST_F(TestCudnnShimV8OperationGraph, SharedUidLowersToOneTensor)
{
    // Two shim tensors with UID 3: conv's output and relu's input.
    cudnnBackendDescriptor_t conv = smallConvolution();
    cudnnBackendDescriptor_t relu = pointwise(CUDNN_POINTWISE_RELU_FWD,
                                              tensor({3, {1, 8, 8, 8}, true}),
                                              nullptr,
                                              tensor({4, {1, 8, 8, 8}}));
    finalizedGraph({conv, relu});

    const auto uids = callsNamed(HIPDNN_ATTR_TENSOR_UNIQUE_ID);
    std::vector<std::int64_t> values;
    values.reserve(uids.size());
    for(const auto& call : uids)
    {
        values.push_back(call.value<std::int64_t>());
    }
    std::sort(values.begin(), values.end());
    EXPECT_EQ(values, (std::vector<std::int64_t>{1, 2, 3, 4}));
}

TEST_F(TestCudnnShimV8OperationGraph, ConflictingUidRejectedWithUid)
{
    cudnnBackendDescriptor_t conv = smallConvolution(37);
    cudnnBackendDescriptor_t relu = pointwise(CUDNN_POINTWISE_RELU_FWD,
                                              tensor({37, {1, 8, 8, 4}, true}),
                                              nullptr,
                                              tensor({4, {1, 8, 8, 4}}));
    cudnnBackendDescriptor_t descriptor = graph({conv, relu});
    EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_BAD_PARAM);
    const std::string message = lastError();
    EXPECT_NE(message.find("UID 37"), std::string::npos) << message;
    EXPECT_NE(message.find("dimensions"), std::string::npos) << message;
    EXPECT_TRUE(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8OperationGraph, OrderedChainKeepsOrder)
{
    cudnnBackendDescriptor_t conv = smallConvolution();
    cudnnBackendDescriptor_t add = pointwise(CUDNN_POINTWISE_ADD,
                                             tensor({3, {1, 8, 8, 8}, true}),
                                             tensor({5, {1, 8, 1, 1}}),
                                             tensor({6, {1, 8, 8, 8}, true}));
    cudnnBackendDescriptor_t relu = pointwise(CUDNN_POINTWISE_RELU_FWD,
                                              tensor({6, {1, 8, 8, 8}, true}),
                                              nullptr,
                                              tensor({7, {1, 8, 8, 8}}));
    finalizedGraph({conv, add, relu});

    EXPECT_EQ(pointwiseModesInGraphOrder(),
              (std::vector<int>{-1, HIPDNN_POINTWISE_ADD, HIPDNN_POINTWISE_RELU_FWD}));
}

TEST_F(TestCudnnShimV8OperationGraph, UnorderedChainSortedStably)
{
    cudnnBackendDescriptor_t conv = smallConvolution();
    cudnnBackendDescriptor_t add = pointwise(CUDNN_POINTWISE_ADD,
                                             tensor({3, {1, 8, 8, 8}, true}),
                                             tensor({5, {1, 8, 1, 1}}),
                                             tensor({6, {1, 8, 8, 8}, true}));
    cudnnBackendDescriptor_t relu = pointwise(CUDNN_POINTWISE_RELU_FWD,
                                              tensor({6, {1, 8, 8, 8}, true}),
                                              nullptr,
                                              tensor({7, {1, 8, 8, 8}}));
    // An independent op stays ahead of the chain it was listed before.
    cudnnBackendDescriptor_t neg = pointwise(
        CUDNN_POINTWISE_NEG, tensor({8, {2, 3, 4, 5}}), nullptr, tensor({9, {2, 3, 4, 5}}));
    finalizedGraph({neg, relu, add, conv});

    EXPECT_EQ(pointwiseModesInGraphOrder(),
              (std::vector<int>{
                  HIPDNN_POINTWISE_NEG, -1, HIPDNN_POINTWISE_ADD, HIPDNN_POINTWISE_RELU_FWD}));
}

TEST_F(TestCudnnShimV8OperationGraph, DisconnectedComponentsAccepted)
{
    cudnnBackendDescriptor_t first = pointwise(
        CUDNN_POINTWISE_ABS, tensor({1, {2, 3, 4, 5}}), nullptr, tensor({2, {2, 3, 4, 5}}));
    cudnnBackendDescriptor_t second = pointwise(
        CUDNN_POINTWISE_NEG, tensor({3, {2, 3, 4, 5}}), nullptr, tensor({4, {2, 3, 4, 5}}));
    finalizedGraph({second, first});

    EXPECT_EQ(outputUidsInGraphOrder(HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT),
              (std::vector<std::int64_t>{4, 2}));
}

TEST_F(TestCudnnShimV8OperationGraph, CycleRejected)
{
    cudnnBackendDescriptor_t first = pointwise(CUDNN_POINTWISE_ABS,
                                               tensor({1, {2, 3, 4, 5}, true}),
                                               nullptr,
                                               tensor({2, {2, 3, 4, 5}, true}));
    cudnnBackendDescriptor_t second = pointwise(CUDNN_POINTWISE_NEG,
                                                tensor({2, {2, 3, 4, 5}, true}),
                                                nullptr,
                                                tensor({1, {2, 3, 4, 5}, true}));
    cudnnBackendDescriptor_t descriptor = graph({first, second});
    EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("cycle"), std::string::npos);
}

TEST_F(TestCudnnShimV8OperationGraph, AlignmentForwardedOnlyBelowSixteen)
{
    finalizedGraph({pointwise(CUDNN_POINTWISE_ADD,
                              tensor({1, {2, 3, 4, 5}, false, false, 32}),
                              tensor({2, {2, 3, 4, 5}, false, false, 4}),
                              tensor({3, {2, 3, 4, 5}}))});

    const auto alignments = callsNamed(HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT);
    ASSERT_EQ(alignments.size(), 1U);
    EXPECT_EQ(alignments.front().value<std::int64_t>(), 4);
    EXPECT_EQ(
        callOn(alignments.front().descriptor, HIPDNN_ATTR_TENSOR_UNIQUE_ID)->value<std::int64_t>(),
        2);
}

TEST_F(TestCudnnShimV8OperationGraph, ScalarsLowerAsRuntimePassByValue)
{
    cudnnBackendDescriptor_t layer
        = normForward(CUDNN_LAYER_NORM,
                      CUDNN_NORM_FWD_INFERENCE,
                      {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, tensor({2, {1, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, tensor({3, {1, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(4)},
                       {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, tensor({5, {2, 3, 4, 4}})}});
    finalizedGraph({layer});

    const auto byValue = callsNamed(HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT);
    ASSERT_EQ(byValue.size(), 1U);
    EXPECT_EQ(
        callOn(byValue.front().descriptor, HIPDNN_ATTR_TENSOR_UNIQUE_ID)->value<std::int64_t>(), 4);
}

TEST_F(TestCudnnShimV8OperationGraph, NormOperationsUseSynthesizedComputeType)
{
    for(const NormCase& item : normCases())
    {
        SCOPED_TRACE(item.label);
        finalizedGraph({item.build()});
        const std::vector<hipdnnBackendDescriptor_t> ops = created(item.operationType);
        ASSERT_FALSE(ops.empty());
        const auto computeType = callOn(ops.back(), item.computeType);
        ASSERT_TRUE(computeType);
        EXPECT_EQ(computeType->value<hipdnnDataType_t>(), HIPDNN_DATA_FLOAT);
    }
}

TEST_F(TestCudnnShimV8OperationGraph, NormSlotsMapByUid)
{
    for(const NormCase& item : normCases())
    {
        SCOPED_TRACE(item.label);
        finalizedGraph({item.build()});
        const std::vector<hipdnnBackendDescriptor_t> ops = created(item.operationType);
        ASSERT_FALSE(ops.empty());
        for(const auto& [slot, uid] : item.slots)
        {
            EXPECT_EQ(slotUid(ops.back(), slot), uid) << "hipDNN attribute " << slot;
        }
    }
}

TEST_F(TestCudnnShimV8OperationGraph, LayerNormSendsNormalizedDimCount)
{
    finalizedGraph(
        {normForward(CUDNN_LAYER_NORM,
                     CUDNN_NORM_FWD_INFERENCE,
                     {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, tensor({2, {1, 1, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, tensor({3, {1, 1, 4, 4}})},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(4)},
                      {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, tensor({5, {2, 3, 4, 4}})}})});

    const auto count = callsNamed(HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT);
    ASSERT_EQ(count.size(), 1U);
    EXPECT_EQ(count.front().value<std::int64_t>(), 2);
}

TEST_F(TestCudnnShimV8OperationGraph, ExpAvgFactorBecomesMomentum)
{
    finalizedGraph({normForward(
        CUDNN_BATCH_NORM,
        CUDNN_NORM_FWD_TRAINING,
        {{CUDNN_ATTR_OPERATION_NORM_FWD_XDESC, tensor({1, {2, 3, 4, 4}})},
         {CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC, tensor({2, {1, 3, 1, 1}})},
         {CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC, tensor({3, {1, 3, 1, 1}})},
         {CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC, scalar(4)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_YDESC, tensor({5, {2, 3, 4, 4}})},
         {CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC, tensor({6, {1, 3, 1, 1}})},
         {CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC, tensor({7, {1, 3, 1, 1}})},
         {CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC, scalar(8)},
         {CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC, tensor({9, {1, 3, 1, 1}})},
         {CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC, tensor({10, {1, 3, 1, 1}})}})});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT), 8);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT), 4);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT), 6);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT), 7);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT), 9);
    EXPECT_EQ(slotUid(op, HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT), 10);
}

TEST_F(TestCudnnShimV8OperationGraph, FrontendRejectionIsBadParam)
{
    // ADD whose second input does not broadcast to the output.
    cudnnBackendDescriptor_t add = pointwise(CUDNN_POINTWISE_ADD,
                                             tensor({1, {2, 3, 4, 5}}),
                                             tensor({2, {2, 7, 4, 5}}),
                                             tensor({3, {2, 3, 4, 5}}));
    cudnnBackendDescriptor_t descriptor = graph({add});
    EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR"), std::string::npos);
}

TEST_F(TestCudnnShimV8OperationGraph, MissingHandleRejected)
{
    cudnnBackendDescriptor_t relu = pointwise(
        CUDNN_POINTWISE_RELU_FWD, tensor({1, {2, 3, 4, 5}}), nullptr, tensor({2, {2, 3, 4, 5}}));
    cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    setArray(descriptor,
             CUDNN_ATTR_OPERATIONGRAPH_OPS,
             CUDNN_TYPE_BACKEND_DESCRIPTOR,
             std::vector<cudnnBackendDescriptor_t>{relu});
    EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_OPERATIONGRAPH_HANDLE"), std::string::npos);
}

TEST_F(TestCudnnShimV8OperationGraph, EmptyOpsRejected)
{
    cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    set(descriptor, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, _handle);
    EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_OPERATIONGRAPH_OPS"), std::string::npos);
}

TEST_F(TestCudnnShimV8OperationGraph, OpsSetReplacesEarlierOps)
{
    cudnnBackendDescriptor_t abs = pointwise(
        CUDNN_POINTWISE_ABS, tensor({1, {2, 3, 4, 5}}), nullptr, tensor({2, {2, 3, 4, 5}}));
    cudnnBackendDescriptor_t neg = pointwise(
        CUDNN_POINTWISE_NEG, tensor({3, {2, 3, 4, 5}}), nullptr, tensor({4, {2, 3, 4, 5}}));
    cudnnBackendDescriptor_t descriptor = graph({abs, neg});
    setArray(descriptor,
             CUDNN_ATTR_OPERATIONGRAPH_OPS,
             CUDNN_TYPE_BACKEND_DESCRIPTOR,
             std::vector<cudnnBackendDescriptor_t>{neg});
    finalize(descriptor);
    EXPECT_EQ(pointwiseModesInGraphOrder(), (std::vector<int>{HIPDNN_POINTWISE_NEG}));
}

TEST_F(TestCudnnShimV8OperationGraph, ShapeFlagsNotSupported)
{
    for(const cudnnBackendAttributeName_t flag :
        {CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED,
         CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED})
    {
        cudnnBackendDescriptor_t relu = pointwise(CUDNN_POINTWISE_RELU_FWD,
                                                  tensor({1, {2, 3, 4, 5}}),
                                                  nullptr,
                                                  tensor({2, {2, 3, 4, 5}}));
        cudnnBackendDescriptor_t accepted = graph({relu});
        set(accepted, flag, CUDNN_TYPE_BOOLEAN, false);
        finalize(accepted);

        cudnnBackendDescriptor_t rejected = graph({relu});
        set(rejected, flag, CUDNN_TYPE_BOOLEAN, true);
        EXPECT_EQ(cudnnBackendFinalize(rejected), CUDNN_STATUS_NOT_SUPPORTED);
        EXPECT_NE(lastError().find(flag == CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED
                                       ? "IS_DYNAMIC_SHAPE_ENABLED"
                                       : "IS_OVERRIDE_SHAPE_ENABLED"),
                  std::string::npos);
    }
}

TEST_F(TestCudnnShimV8OperationGraph, FinalizeRunsNoHeuristics)
{
    installHeuristicResults({-2});
    finalizedGraph({smallConvolution(3, false)});

    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).empty());
    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).empty());
    EXPECT_TRUE(getAttributeNames().empty());
    EXPECT_EQ(finalized().back(), onlyCreated(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR));
}

TEST_F(TestCudnnShimV8OperationGraph, EngineIndexFollowsSignedId)
{
    installHeuristicResults({5, -2, 300, -40});
    cudnnBackendDescriptor_t descriptor = finalizedGraph({smallConvolution(3, false)});

    std::int64_t count = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(descriptor,
                                       CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                       CUDNN_TYPE_INT64,
                                       0,
                                       &count,
                                       nullptr),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(engineCount(descriptor), 4);

    const backend::OperationGraphDescriptor& shim = shimGraph(descriptor);
    EXPECT_EQ(shim.rankedEngineIds(), (std::vector<std::int64_t>{5, -2, 300, -40}));
    const std::vector<std::int64_t> byIndex = {-40, -2, 5, 300};
    for(std::size_t index = 0; index < byIndex.size(); ++index)
    {
        const auto signedIndex = static_cast<std::int64_t>(index);
        EXPECT_EQ(shim.engineIdAt(signedIndex), byIndex[index]);
        EXPECT_EQ(shim.engineIndexOf(byIndex[index]), signedIndex);
        EXPECT_EQ(engineIdOf(shim.defaultEngineConfig(byIndex[index])), byIndex[index]);
        EXPECT_EQ(engineIdOf(shim.defaultEngine(byIndex[index])), byIndex[index]);
        EXPECT_EQ(typeOf(shim.defaultEngine(byIndex[index])), HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
    }
    EXPECT_EQ(shim.engineIdAt(4), std::nullopt);
    EXPECT_EQ(shim.engineIdAt(-1), std::nullopt);
    EXPECT_EQ(shim.engineIndexOf(7), std::nullopt);
    EXPECT_EQ(shim.defaultEngineConfig(7), nullptr);
    EXPECT_EQ(shim.handle(), _handle);

    hipdnnBackendDescriptor_t heuristic = onlyCreated(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    EXPECT_EQ(callOn(heuristic, HIPDNN_ATTR_ENGINEHEUR_MODE)->value<hipdnnBackendHeurMode_t>(),
              HIPDNN_HEUR_MODE_FALLBACK);
    EXPECT_EQ(callOn(heuristic, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH)
                  ->value<hipdnnBackendDescriptor_t>(),
              shim.hipdnnGraph());
}

TEST_F(TestCudnnShimV8OperationGraph, EnginesEnumeratedOnceAcrossThreads)
{
    using ::testing::_;
    installHeuristicResults({3, 1, 2});
    cudnnBackendDescriptor_t descriptor = finalizedGraph({smallConvolution(3, false)});

    // A slow heuristic create keeps the first sweep open while the others arrive.
    ON_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillByDefault([this](hipdnnBackendDescriptorType_t type, hipdnnBackendDescriptor_t* out) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            *out = newMockDescriptor(type);
            return HIPDNN_STATUS_SUCCESS;
        });

    constexpr std::size_t THREAD_COUNT = 8;
    std::atomic<bool> start{false};
    std::array<std::int64_t, THREAD_COUNT> counts{};
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(std::size_t i = 0; i < THREAD_COUNT; ++i)
    {
        threads.emplace_back([&, i] {
            while(!start.load())
            {
                std::this_thread::yield();
            }
            counts.at(i) = engineCount(descriptor);
        });
    }
    start.store(true);
    for(std::thread& thread : threads)
    {
        thread.join();
    }

    for(const std::int64_t count : counts)
    {
        EXPECT_EQ(count, 3);
    }
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).size(), 1U);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), 3U);
}

TEST_F(TestCudnnShimV8OperationGraph, FailingConfigDropsEngine)
{
    installHeuristicResults({7, -1, 4}, {-1});
    cudnnBackendDescriptor_t descriptor = finalizedGraph({smallConvolution(3, false)});

    EXPECT_EQ(engineCount(descriptor), 2);
    const backend::OperationGraphDescriptor& shim = shimGraph(descriptor);
    EXPECT_EQ(shim.rankedEngineIds(), (std::vector<std::int64_t>{7, 4}));
    EXPECT_EQ(shim.engineIdAt(0), 4);
    EXPECT_EQ(shim.engineIdAt(1), 7);
    EXPECT_EQ(shim.engineIndexOf(-1), std::nullopt);
}

TEST_F(TestCudnnShimV8OperationGraph, NoApplicableEngineCountsZero)
{
    cudnnBackendDescriptor_t descriptor = finalizedGraph({smallConvolution(3, false)});
    EXPECT_EQ(engineCount(descriptor), 0);
    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8OperationGraph, FailedEnumerationRetried)
{
    using ::testing::_;
    installHeuristicResults({-2});
    cudnnBackendDescriptor_t descriptor = finalizedGraph({smallConvolution(3, false)});

    int attempts = 0;
    ON_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillByDefault(
            [this, &attempts](hipdnnBackendDescriptorType_t type, hipdnnBackendDescriptor_t* out) {
                if(attempts++ == 0)
                {
                    return HIPDNN_STATUS_INTERNAL_ERROR;
                }
                *out = newMockDescriptor(type);
                return HIPDNN_STATUS_SUCCESS;
            });

    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(descriptor,
                                       CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       nullptr,
                                       static_cast<void*>(&count)),
              CUDNN_STATUS_INTERNAL_ERROR);
    EXPECT_NE(lastError().find("engine heuristic"), std::string::npos);
    EXPECT_EQ(engineCount(descriptor), 1);
    EXPECT_EQ(attempts, 2);
}

TEST_F(TestCudnnShimV8OperationGraph, EngineCountNeedsFinalize)
{
    cudnnBackendDescriptor_t descriptor = graph({smallConvolution(3, false)});
    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(descriptor,
                                       CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       nullptr,
                                       static_cast<void*>(&count)),
              CUDNN_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                       CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                       CUDNN_TYPE_INT64,
                                       1,
                                       static_cast<const void*>(&count)),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8OperationGraph, ConvolutionBackwardSlotsMap)
{
    cudnnBackendDescriptor_t dgrad = finalizedOperation(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, sameConvolution()},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, tensor({1, {1, 8, 8, 8}})},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, tensor({2, {8, 4, 3, 3}})},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, tensor({3, {1, 4, 8, 8}})}});
    cudnnBackendDescriptor_t wgrad = finalizedOperation(
        CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, sameConvolution()},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, tensor({4, {1, 4, 8, 8}})},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, tensor({5, {1, 8, 8, 8}})},
         {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, tensor({6, {8, 4, 3, 3}})}});
    finalizedGraph({dgrad, wgrad});

    hipdnnBackendDescriptor_t data
        = onlyCreated(HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
    EXPECT_EQ(slotUid(data, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY), 1);
    EXPECT_EQ(slotUid(data, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W), 2);
    EXPECT_EQ(slotUid(data, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX), 3);
    hipdnnBackendDescriptor_t filter
        = onlyCreated(HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
    EXPECT_EQ(slotUid(filter, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X), 4);
    EXPECT_EQ(slotUid(filter, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY), 5);
    EXPECT_EQ(slotUid(filter, HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW), 6);
}

TEST_F(TestCudnnShimV8OperationGraph, ReductionForwardsDeterminism)
{
    finalizedGraph({reduction(1, 2, std::nullopt), reduction(3, 4, true)});

    const std::vector<hipdnnBackendDescriptor_t> ops
        = created(HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
    ASSERT_EQ(ops.size(), 2U);
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_REDUCTION_YDESC), 2);
    EXPECT_FALSE(callOn(ops[0], HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC)->value<bool>());
    EXPECT_EQ(slotUid(ops[1], HIPDNN_ATTR_OPERATION_REDUCTION_YDESC), 4);
    EXPECT_TRUE(callOn(ops[1], HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC)->value<bool>());
}

TEST_F(TestCudnnShimV8OperationGraph, ResampleForwardSendsPaddingModeAndIndex)
{
    cudnnBackendDescriptor_t indexed
        = finalizedOperation(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
                             {{CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, maxPool()},
                              {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, tensor({1, {1, 4, 8, 8}})},
                              {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, tensor({2, {1, 4, 4, 4}})},
                              {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                               tensor({3, {1, 4, 4, 4}, false, false, 16, CUDNN_DATA_INT32})}});
    cudnnBackendDescriptor_t plain = finalizedOperation(
        CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
        {{CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, maxPool()},
         {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC, tensor({4, {1, 4, 8, 8}})},
         {CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC, tensor({5, {1, 4, 4, 4}})}});
    finalizedGraph({indexed, plain});

    const std::vector<hipdnnBackendDescriptor_t> ops
        = created(HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
    ASSERT_EQ(ops.size(), 2U);
    for(hipdnnBackendDescriptor_t op : ops)
    {
        EXPECT_EQ(callOn(op, HIPDNN_ATTR_RESAMPLE_PADDING_MODE)->value<hipdnnPaddingMode_t>(),
                  HIPDNN_PADDING_NEG_INF_PAD);
    }
    EXPECT_TRUE(callOn(ops[0], HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT)->value<bool>());
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC), 3);
    EXPECT_EQ(slotUid(ops[1], HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC), 5);
    EXPECT_FALSE(callOn(ops[1], HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT));
    EXPECT_FALSE(callOn(ops[1], HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC));
}

TEST_F(TestCudnnShimV8OperationGraph, LayerNormBackwardSendsNormalizedDimCount)
{
    finalizedGraph(
        {normBackward(CUDNN_LAYER_NORM,
                      {{CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, tensor({2, {2, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, tensor({3, {1, 1, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, tensor({4, {2, 3, 1, 1}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, tensor({5, {2, 3, 1, 1}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, tensor({6, {2, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, tensor({7, {1, 1, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, tensor({8, {1, 1, 4, 4}})}})});

    const auto count = callsNamed(HIPDNN_ATTR_LAYERNORM_BACKWARD_NORMALIZED_DIM_COUNT_EXT);
    ASSERT_EQ(count.size(), 1U);
    EXPECT_EQ(count.front().value<std::int64_t>(), 2);
}

TEST_F(TestCudnnShimV8OperationGraph, RmsNormBackwardDbiasFollowsPresence)
{
    const auto build = [this](bool withDbias) {
        Slots slots = {{CUDNN_ATTR_OPERATION_NORM_BWD_XDESC, tensor({1, {2, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC, tensor({2, {2, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC, tensor({3, {1, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC, tensor({4, {2, 1, 1, 1}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC, tensor({5, {2, 3, 4, 4}})},
                       {CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC, tensor({6, {1, 3, 4, 4}})}};
        if(withDbias)
        {
            slots.emplace_back(CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC, tensor({7, {1, 3, 4, 4}}));
        }
        return normBackward(CUDNN_RMS_NORM, slots);
    };
    finalizedGraph({build(true)});
    finalizedGraph({build(false)});

    const std::vector<hipdnnBackendDescriptor_t> ops
        = created(HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT);
    ASSERT_EQ(ops.size(), 2U);
    EXPECT_EQ(slotUid(ops[0], HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT), 7);
    EXPECT_EQ(slotUid(ops[1], HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT), -1);
    EXPECT_EQ(slotUid(ops[1], HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT), 6);
}

TEST_F(TestCudnnShimV8OperationGraph, ConstantTensorLowersAsValue)
{
    const auto add = [this](cudnnBackendDescriptor_t constantOperand, std::int64_t firstUid) {
        return pointwise(CUDNN_POINTWISE_ADD,
                         tensor({firstUid, {2, 3, 4, 5}}),
                         constantOperand,
                         tensor({firstUid + 1, {2, 3, 4, 5}}));
    };
    finalizedGraph({add(constant(10, CUDNN_DATA_FLOAT, 2.5F), 1),
                    add(constant(20, CUDNN_DATA_INT32, std::int32_t{7}), 3),
                    add(constant(30, CUDNN_DATA_BOOLEAN, true), 5)});

    hipdnnBackendDescriptor_t floatConstant = hipdnnTensor(10);
    EXPECT_EQ(callOn(floatConstant, HIPDNN_ATTR_TENSOR_VALUE_EXT)->values<float>(),
              (std::vector<float>{2.5F}));
    EXPECT_EQ(callOn(floatConstant, HIPDNN_ATTR_TENSOR_DATA_TYPE)->value<hipdnnDataType_t>(),
              HIPDNN_DATA_FLOAT);
    EXPECT_EQ(callOn(hipdnnTensor(20), HIPDNN_ATTR_TENSOR_VALUE_EXT)->values<std::int32_t>(),
              (std::vector<std::int32_t>{7}));
    EXPECT_EQ(callOn(hipdnnTensor(30), HIPDNN_ATTR_TENSOR_VALUE_EXT)->values<std::uint8_t>(),
              (std::vector<std::uint8_t>{1}));
    for(const std::int64_t uid : {10, 20, 30})
    {
        SCOPED_TRACE(uid);
        EXPECT_FALSE(callOn(hipdnnTensor(uid), HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT));
        EXPECT_EQ(callOn(hipdnnTensor(uid), HIPDNN_ATTR_TENSOR_DIMENSIONS)->values<std::int64_t>(),
                  (std::vector<std::int64_t>{1, 1, 1, 1}));
    }

    cudnnBackendDescriptor_t unsupported
        = graph({add(constant(40, CUDNN_DATA_INT8, std::int8_t{1}), 7)});
    EXPECT_EQ(cudnnBackendFinalize(unsupported), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(lastError().find("CUDNN_ATTR_TENSOR_CONSTANT_VALUE"), std::string::npos);
}

TEST_F(TestCudnnShimV8OperationGraph, RaggedOffsetLinksAuxiliaryTensor)
{
    cudnnBackendDescriptor_t offsets
        = tensor({9, {3, 1, 1, 1}, false, false, 16, CUDNN_DATA_INT32});
    finalizedGraph(
        {pointwise(CUDNN_POINTWISE_ABS,
                   tensor({1, {2, 3, 4, 5}, false, false, 16, CUDNN_DATA_FLOAT, offsets}),
                   nullptr,
                   tensor({2, {2, 3, 4, 5}}))});

    hipdnnBackendDescriptor_t op = onlyCreated(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    const auto input = callOn(op, HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT);
    ASSERT_TRUE(input);
    const auto ragged
        = callOn(input->value<hipdnnBackendDescriptor_t>(), HIPDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC);
    ASSERT_TRUE(ragged);
    EXPECT_EQ(ragged->value<hipdnnBackendDescriptor_t>(), hipdnnTensor(9));
}

TEST_F(TestCudnnShimV8OperationGraph, SameUidAlignmentUsesSmallest)
{
    cudnnBackendDescriptor_t conv = convolutionForward(tensor({1, {1, 4, 8, 8}}),
                                                       tensor({2, {8, 4, 3, 3}}),
                                                       tensor({3, {1, 8, 8, 8}, true, false, 8}),
                                                       sameConvolution());
    cudnnBackendDescriptor_t relu = pointwise(CUDNN_POINTWISE_RELU_FWD,
                                              tensor({3, {1, 8, 8, 8}, true, false, 4}),
                                              nullptr,
                                              tensor({4, {1, 8, 8, 8}}));
    finalizedGraph({conv, relu});

    const auto alignments = callsNamed(HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT);
    ASSERT_EQ(alignments.size(), 1U);
    EXPECT_EQ(alignments.front().value<std::int64_t>(), 4);
    EXPECT_EQ(alignments.front().descriptor, hipdnnTensor(3));
}

TEST_F(TestCudnnShimV8OperationGraph, UidWrittenTwiceRejected)
{
    cudnnBackendDescriptor_t first = pointwise(
        CUDNN_POINTWISE_ABS, tensor({1, {2, 3, 4, 5}}), nullptr, tensor({5, {2, 3, 4, 5}}));
    cudnnBackendDescriptor_t second = pointwise(
        CUDNN_POINTWISE_NEG, tensor({2, {2, 3, 4, 5}}), nullptr, tensor({5, {2, 3, 4, 5}}));
    cudnnBackendDescriptor_t descriptor = graph({first, second});
    EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_BAD_PARAM);
    const std::string message = lastError();
    EXPECT_NE(message.find("UID 5"), std::string::npos) << message;
    EXPECT_NE(message.find("operations 0 and 1"), std::string::npos) << message;
}

TEST_F(TestCudnnShimV8OperationGraph, DuplicateEngineIdKeptOnce)
{
    installHeuristicResults({5, 5, -1});
    cudnnBackendDescriptor_t descriptor = finalizedGraph({smallConvolution(3, false)});

    EXPECT_EQ(engineCount(descriptor), 2);
    EXPECT_EQ(shimGraph(descriptor).rankedEngineIds(), (std::vector<std::int64_t>{5, -1}));
}

TEST_F(TestCudnnShimV8OperationGraph, PointwiseParametersReachOperation)
{
    const auto parameterized = [this](cudnnPointwiseMode_t mode,
                                      cudnnBackendAttributeName_t parameter,
                                      double value,
                                      std::int64_t uid) {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, mode);
        set(descriptor, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        set(descriptor, parameter, CUDNN_TYPE_DOUBLE, value);
        if(mode == CUDNN_POINTWISE_ELU_FWD)
        {
            set(descriptor, CUDNN_ATTR_POINTWISE_AXIS, CUDNN_TYPE_INT64, std::int64_t{2});
        }
        finalize(descriptor);
        return finalizedOperation(
            CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
            {{CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, descriptor},
             {CUDNN_ATTR_OPERATION_POINTWISE_XDESC, tensor({uid, {2, 3, 4, 5}})},
             {CUDNN_ATTR_OPERATION_POINTWISE_YDESC, tensor({uid + 1, {2, 3, 4, 5}})}});
    };
    finalizedGraph(
        {parameterized(CUDNN_POINTWISE_ELU_FWD, CUDNN_ATTR_POINTWISE_ELU_ALPHA, 0.5, 1),
         parameterized(CUDNN_POINTWISE_SOFTPLUS_FWD, CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, 2.5, 3),
         parameterized(CUDNN_POINTWISE_SWISH_FWD, CUDNN_ATTR_POINTWISE_SWISH_BETA, 3.5, 5)});

    const std::array<hipdnnBackendAttributeName_t, 4> parameters
        = {HIPDNN_ATTR_POINTWISE_ELU_ALPHA,
           HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
           HIPDNN_ATTR_POINTWISE_SWISH_BETA,
           HIPDNN_ATTR_POINTWISE_AXIS};
    const auto sent = [&](hipdnnBackendDescriptor_t op) {
        std::vector<hipdnnBackendAttributeName_t> names;
        for(const hipdnnBackendAttributeName_t name : parameters)
        {
            if(callOn(op, name))
            {
                names.push_back(name);
            }
        }
        return names;
    };

    hipdnnBackendDescriptor_t elu = pointwiseOperation(HIPDNN_POINTWISE_ELU_FWD);
    EXPECT_EQ(sent(elu),
              (std::vector<hipdnnBackendAttributeName_t>{HIPDNN_ATTR_POINTWISE_ELU_ALPHA,
                                                         HIPDNN_ATTR_POINTWISE_AXIS}));
    EXPECT_EQ(callOn(elu, HIPDNN_ATTR_POINTWISE_ELU_ALPHA)->value<float>(), 0.5F);
    EXPECT_EQ(callOn(elu, HIPDNN_ATTR_POINTWISE_AXIS)->value<std::int64_t>(), 2);

    hipdnnBackendDescriptor_t softplus = pointwiseOperation(HIPDNN_POINTWISE_SOFTPLUS_FWD);
    EXPECT_EQ(sent(softplus),
              (std::vector<hipdnnBackendAttributeName_t>{HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA}));
    EXPECT_EQ(callOn(softplus, HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA)->value<float>(), 2.5F);

    hipdnnBackendDescriptor_t swish = pointwiseOperation(HIPDNN_POINTWISE_SWISH_FWD);
    EXPECT_EQ(sent(swish),
              (std::vector<hipdnnBackendAttributeName_t>{HIPDNN_ATTR_POINTWISE_SWISH_BETA}));
    EXPECT_EQ(callOn(swish, HIPDNN_ATTR_POINTWISE_SWISH_BETA)->value<float>(), 3.5F);
}

TEST_F(TestCudnnShimV8OperationGraph, CountQueryRunsNoHeuristics)
{
    installHeuristicResults({-2});
    cudnnBackendDescriptor_t descriptor = finalizedGraph({smallConvolution(3, false)});

    for(const std::int64_t requested : {0, 1})
    {
        std::int64_t count = -1;
        EXPECT_EQ(cudnnBackendGetAttribute(descriptor,
                                           CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                           CUDNN_TYPE_INT64,
                                           requested,
                                           &count,
                                           nullptr),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(count, 1);
    }
    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).empty());
    EXPECT_EQ(engineCount(descriptor), 1);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).size(), 1U);
}

} // namespace
