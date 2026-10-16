// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Builds cuDNN v8 backend descriptors through the shim's C API only, for the
// integration suites that run them against the real hipDNN backend.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_shim_v8_test
{

inline std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

inline std::vector<std::int64_t> packedStrides(const std::vector<std::int64_t>& dims)
{
    std::vector<std::int64_t> strides(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

struct TensorOptions
{
    bool isVirtual = false;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    bool isByValue = false;
};

using DescriptorSlot = std::pair<cudnnBackendAttributeName_t, cudnnBackendDescriptor_t>;

// Owns every descriptor it creates and destroys them in reverse order.
class GraphBuilder
{
public:
    GraphBuilder() = default;
    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;
    GraphBuilder(GraphBuilder&&) = delete;
    GraphBuilder& operator=(GraphBuilder&&) = delete;

    ~GraphBuilder()
    {
        for(auto it = _owned.rbegin(); it != _owned.rend(); ++it)
        {
            cudnnBackendDestroyDescriptor(*it);
        }
    }

    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t descriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(type, &descriptor), CUDNN_STATUS_SUCCESS)
            << lastError();
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
            << "attribute " << static_cast<int>(name) << ": " << lastError();
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
            << "attribute " << static_cast<int>(name) << ": " << lastError();
    }

    static void finalize(cudnnBackendDescriptor_t descriptor)
    {
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS) << lastError();
    }

    cudnnBackendDescriptor_t tensor(std::int64_t uid,
                                    const std::vector<std::int64_t>& dims,
                                    const TensorOptions& options = {})
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, options.dataType);
        setArray(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dims);
        setArray(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, packedStrides(dims));
        set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, uid);
        set(descriptor,
            CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
            CUDNN_TYPE_INT64,
            static_cast<std::int64_t>(16));
        if(options.isVirtual)
        {
            set(descriptor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, true);
        }
        if(options.isByValue)
        {
            set(descriptor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, true);
        }
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t virtualTensor(std::int64_t uid, const std::vector<std::int64_t>& dims)
    {
        TensorOptions options;
        options.isVirtual = true;
        return tensor(uid, dims, options);
    }

    // 2-D cross-correlation with symmetric padding.
    cudnnBackendDescriptor_t convolution(std::int64_t padding, std::int64_t stride)
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
            static_cast<std::int64_t>(2));
        const std::vector<std::int64_t> paddings(2, padding);
        const std::vector<std::int64_t> strides(2, stride);
        const std::vector<std::int64_t> dilations(2, 1);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, paddings);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, paddings);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, dilations);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, strides);
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t pointwise(cudnnPointwiseMode_t mode)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, mode);
        set(descriptor, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t matmul()
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t reduction(cudnnReduceTensorOp_t op)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_REDUCTION_OPERATOR, CUDNN_TYPE_REDUCTION_OPERATOR_TYPE, op);
        set(descriptor, CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        finalize(descriptor);
        return descriptor;
    }

    // 2-D pooling with a 2x2 window and stride and no padding. The shim requires
    // a padding mode; max pooling pads with -inf.
    cudnnBackendDescriptor_t resample(cudnnResampleMode_t mode)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, mode);
        set(descriptor, CUDNN_ATTR_RESAMPLE_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        set(descriptor,
            CUDNN_ATTR_RESAMPLE_PADDING_MODE,
            CUDNN_TYPE_PADDING_MODE,
            mode == CUDNN_RESAMPLE_MAXPOOL ? CUDNN_NEG_INF_PAD : CUDNN_ZERO_PAD);
        set(descriptor,
            CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS,
            CUDNN_TYPE_INT64,
            static_cast<std::int64_t>(2));
        const std::vector<cudnnFraction_t> windows(2, {2, 1});
        const std::vector<cudnnFraction_t> zeros(2, {0, 1});
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_FRACTION, windows);
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, CUDNN_TYPE_FRACTION, zeros);
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_POST_PADDINGS, CUDNN_TYPE_FRACTION, zeros);
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_FRACTION, windows);
        finalize(descriptor);
        return descriptor;
    }

    // An operation with its descriptor-valued attributes set; enum-valued ones
    // (norm mode and phase) are the caller's, as is finalize.
    cudnnBackendDescriptor_t operation(cudnnBackendDescriptorType_t type,
                                       const std::vector<DescriptorSlot>& slots)
    {
        cudnnBackendDescriptor_t descriptor = create(type);
        for(const auto& [name, value] : slots)
        {
            set(descriptor, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, value);
        }
        return descriptor;
    }

    cudnnBackendDescriptor_t finalizedOperation(cudnnBackendDescriptorType_t type,
                                                const std::vector<DescriptorSlot>& slots)
    {
        cudnnBackendDescriptor_t descriptor = operation(type, slots);
        finalize(descriptor);
        return descriptor;
    }

    // Not finalized; a null handle leaves HANDLE unset.
    cudnnBackendDescriptor_t graph(cudnnHandle_t handle,
                                   const std::vector<cudnnBackendDescriptor_t>& ops)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        if(handle != nullptr)
        {
            set(descriptor, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, handle);
        }
        setArray(descriptor, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, ops);
        return descriptor;
    }

private:
    std::vector<cudnnBackendDescriptor_t> _owned;
};

} // namespace hipdnn_shim_v8_test
