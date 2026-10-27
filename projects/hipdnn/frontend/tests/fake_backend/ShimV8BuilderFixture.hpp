// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Builder-only helpers for the cuDNN v8 shim tests: tensors, operations and graphs come
// from the v0.x builders, and what reaches hipDNN is read from the mock backend.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "ShimV8MockBackendFixture.hpp"
#include "ShimV8TestSupport.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_shim_test
{

namespace v8fe = hipdnn_frontend::compatibility::cudnn_frontend;

// Reads and clears the shim's last error.
inline std::string shimLastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

struct BuildFailure
{
    bool thrown = false;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    std::string what;
};

template <typename Action>
BuildFailure failureOf(Action&& action)
{
    BuildFailure failure;
    try
    {
        std::forward<Action>(action)();
    }
    catch(v8fe::cudnnException& e)
    {
        failure.thrown = true;
        failure.status = e.getCudnnStatus();
        failure.what = e.what();
    }
    return failure;
}

template <typename Builder>
BuildFailure buildFailure(Builder& builder)
{
    return failureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
}

struct TensorSpec
{
    std::int64_t uid;
    Dims dims;
    bool isVirtual = false;
    bool byValue = false;
    std::int64_t alignment = 16;
    v8fe::DataType_t dataType = v8fe::DataType_t::FLOAT;
};

class ShimV8BuilderFixture : public ShimV8MockBackendFixture
{
protected:
    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        shimLastError();
    }

    // Sets every field of `spec` on `builder`; non-packed strides via setStride after.
    static v8fe::TensorBuilder& describe(v8fe::TensorBuilder& builder, const TensorSpec& spec)
    {
        const Dims strides = packedStrides(spec.dims);
        return builder.setDataType(spec.dataType)
            .setDim(static_cast<std::int64_t>(spec.dims.size()), spec.dims.data())
            .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
            .setId(spec.uid)
            .setAlignment(spec.alignment)
            .setVirtual(spec.isVirtual)
            .setByValue(spec.byValue);
    }

    static v8fe::Tensor tensor(const TensorSpec& spec)
    {
        v8fe::TensorBuilder builder;
        return describe(builder, spec).build();
    }

    static v8fe::Tensor tensor(std::int64_t uid,
                               const Dims& dims,
                               bool isVirtual = false,
                               v8fe::DataType_t dataType = v8fe::DataType_t::FLOAT)
    {
        TensorSpec spec{uid, dims};
        spec.isVirtual = isVirtual;
        spec.dataType = dataType;
        return tensor(spec);
    }

    // A [1,1,1,1] runtime pass-by-value scalar.
    static v8fe::Tensor scalar(std::int64_t uid,
                               v8fe::DataType_t dataType = v8fe::DataType_t::FLOAT)
    {
        TensorSpec spec{uid, {1, 1, 1, 1}};
        spec.byValue = true;
        spec.dataType = dataType;
        return tensor(spec);
    }

    // Two spatial dims with unit padding, stride and dilation: a 3x3 filter keeps the size.
    static v8fe::ConvDesc convDesc(v8fe::DataType_t computeType = v8fe::DataType_t::FLOAT,
                                   cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION)
    {
        const std::array<std::int64_t, 2> ones = {1, 1};
        return v8fe::ConvDescBuilder()
            .setComputeType(computeType)
            .setMathMode(mode)
            .setSpatialDimCount(2)
            .setPrePadding(2, ones.data())
            .setPostPadding(2, ones.data())
            .setDilation(2, ones.data())
            .setSpatialStride(2, ones.data())
            .build();
    }

    static v8fe::Operation convolutionForward(const v8fe::Tensor& x,
                                              const v8fe::Tensor& w,
                                              const v8fe::Tensor& y,
                                              const v8fe::ConvDesc& conv)
    {
        return v8fe::OperationBuilder(
                   v8fe::DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            .setxDesc(x)
            .setwDesc(w)
            .setyDesc(y)
            .setcDesc(conv)
            .build();
    }

    static v8fe::PointWiseDesc pointwiseDesc(v8fe::PointwiseMode_t mode,
                                             v8fe::DataType_t computeType = v8fe::DataType_t::FLOAT)
    {
        return v8fe::PointWiseDescBuilder().setMode(mode).setComputeType(computeType).build();
    }

    static v8fe::Operation pointwise(const v8fe::PointWiseDesc& desc,
                                     const v8fe::Tensor& x,
                                     const v8fe::Tensor& y,
                                     const v8fe::Tensor* b = nullptr,
                                     const v8fe::Tensor* t = nullptr)
    {
        v8fe::OperationBuilder builder(v8fe::DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
        builder.setpwDesc(desc).setxDesc(x).setyDesc(y);
        if(b != nullptr)
        {
            builder.setbDesc(*b);
        }
        if(t != nullptr)
        {
            builder.settDesc(*t);
        }
        return builder.build();
    }

    // The ops are copied into the builder, so `ops` may be a temporary.
    v8fe::OperationGraphBuilder& describe(v8fe::OperationGraphBuilder& builder,
                                          std::vector<const v8fe::Operation*> ops) const
    {
        return builder.setHandle(_handle).setOperationGraph(static_cast<std::int64_t>(ops.size()),
                                                            ops.data());
    }

    v8fe::OperationGraph graph(const std::vector<const v8fe::Operation*>& ops) const
    {
        v8fe::OperationGraphBuilder builder;
        return describe(builder, ops).build();
    }

    v8fe::OperationGraph graphOf(const v8fe::Operation& op) const
    {
        return graph({&op});
    }

    BuildFailure graphFailure(const std::vector<const v8fe::Operation*>& ops) const
    {
        v8fe::OperationGraphBuilder builder;
        describe(builder, ops);
        return buildFailure(builder);
    }
};

} // namespace hipdnn_shim_test
