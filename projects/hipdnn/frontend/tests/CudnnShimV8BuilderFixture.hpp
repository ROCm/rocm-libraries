// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Builder-only helpers for the cuDNN v8 shim tests: tensors and graphs come from
// the v0.x builders, and what reaches hipDNN is read from the mock backend.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_shim_test
{

namespace v8fe = hipdnn_frontend::compatibility::cudnn_frontend;

using Dims = std::vector<std::int64_t>;

inline Dims packedStrides(const Dims& dims)
{
    Dims strides(dims.size(), 1);
    for(std::size_t i = dims.size(); i > 1; --i)
    {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

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

    BuildFailure graphFailure(const std::vector<const v8fe::Operation*>& ops) const
    {
        v8fe::OperationGraphBuilder builder;
        describe(builder, ops);
        return buildFailure(builder);
    }

    // SetAttribute calls on hipDNN descriptors of `type` that set `name`.
    std::vector<CapturedSetAttribute> calls(hipdnnBackendDescriptorType_t type,
                                            hipdnnBackendAttributeName_t name) const
    {
        std::vector<CapturedSetAttribute> result = setAttributesOn(type);
        result.erase(
            std::remove_if(result.begin(),
                           result.end(),
                           [name](const CapturedSetAttribute& call) { return call.name != name; }),
            result.end());
        return result;
    }

    std::vector<CapturedSetAttribute> callsNamed(hipdnnBackendAttributeName_t name) const
    {
        std::vector<CapturedSetAttribute> result = setAttributes();
        result.erase(
            std::remove_if(result.begin(),
                           result.end(),
                           [name](const CapturedSetAttribute& call) { return call.name != name; }),
            result.end());
        return result;
    }

    std::optional<CapturedSetAttribute> callOn(hipdnnBackendDescriptor_t descriptor,
                                               hipdnnBackendAttributeName_t name) const
    {
        for(const CapturedSetAttribute& call : setAttributesOf(descriptor))
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

    // The hipDNN tensor created for `uid` (the last one, if several graphs used it).
    hipdnnBackendDescriptor_t hipdnnTensor(std::int64_t uid) const
    {
        hipdnnBackendDescriptor_t found = nullptr;
        for(const CapturedSetAttribute& call : callsNamed(HIPDNN_ATTR_TENSOR_UNIQUE_ID))
        {
            if(call.value<std::int64_t>() == uid)
            {
                found = call.descriptor;
            }
        }
        EXPECT_NE(found, nullptr) << "no hipDNN tensor with UID " << uid;
        return found;
    }

    hipdnnBackendDescriptor_t onlyCreated(hipdnnBackendDescriptorType_t type) const
    {
        const std::vector<hipdnnBackendDescriptor_t> all = created(type);
        EXPECT_EQ(all.size(), 1U);
        return all.empty() ? nullptr : all.front();
    }

    // The OPS array of the last hipDNN OPERATIONGRAPH, in order.
    std::vector<hipdnnBackendDescriptor_t> graphOps() const
    {
        const std::vector<CapturedSetAttribute> ops
            = calls(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, HIPDNN_ATTR_OPERATIONGRAPH_OPS);
        if(ops.empty())
        {
            return {};
        }
        return ops.back().values<hipdnnBackendDescriptor_t>();
    }
};

} // namespace hipdnn_shim_test
