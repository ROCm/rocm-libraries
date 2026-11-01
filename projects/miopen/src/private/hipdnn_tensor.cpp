// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_tensor.hpp"

#include "miopen_impl.h"

#include <algorithm>

namespace miopen {
namespace wrapper {
namespace hipdnn {

namespace fe = hipdnn_frontend;

namespace {

size_t TypeSize(miopenDataType_t type)
{
    switch(type)
    {
    case miopenInt8:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz: return 1;
    case miopenHalf:
    case miopenBFloat16: return 2;
    case miopenFloat:
    case miopenInt32: return 4;
    case miopenDouble:
    case miopenInt64: return 8;
    }
    return 0;
}

} // namespace

// miopenGetTensorDescriptor fills caller-provided int arrays, so the size query
// has to come first.
miopenStatus_t ReadTensor(miopenTensorDescriptor_t desc, TensorInfo& out)
{
    int size = 0;
    if(miopenGetTensorDescriptorSize_impl(desc, &size) != miopenStatusSuccess || size <= 0)
        return miopenStatusInvalidValue;

    std::vector<int> dims(static_cast<size_t>(size));
    std::vector<int> strides(static_cast<size_t>(size));
    miopenDataType_t dataType{};
    if(miopenGetTensorDescriptor_impl(desc, &dataType, dims.data(), strides.data()) !=
       miopenStatusSuccess)
        return miopenStatusInvalidValue;

    out.dataType = dataType;
    out.dims.assign(dims.begin(), dims.end());
    out.strides.assign(strides.begin(), strides.end());

    // MIOpen stores no dim or stride below 1, so one here was cut down by the
    // int getter.
    const auto belowOne = [](int value) { return value < 1; };
    if(std::any_of(dims.begin(), dims.end(), belowOne) ||
       std::any_of(strides.begin(), strides.end(), belowOne))
    {
        out.truncated = true;
        return miopenStatusSuccess;
    }

    // No getter exposes the vector length, but MIOpen's byte count is
    // typeSize * (vectorLength + sum of (dim - 1) * stride), so it can be solved for.
    // The sum is unsigned, as in MIOpen, so a cut-down value gives a wrong
    // answer rather than signed overflow.
    size_t numBytes       = 0;
    const size_t typeSize = TypeSize(dataType);
    if(miopenGetTensorNumBytes_impl(desc, &numBytes) != miopenStatusSuccess || typeSize == 0)
        return miopenStatusInvalidValue;
    uint64_t span = 0;
    for(size_t i = 0; i < dims.size(); ++i)
        span += static_cast<uint64_t>(dims[i] - 1) * static_cast<uint64_t>(strides[i]);
    const uint64_t vectorLength = numBytes / typeSize - span;
    // MIOpen allows only these, so any other answer means a value was cut down.
    if(vectorLength != 1 && vectorLength != 4 && vectorLength != 8)
    {
        out.truncated = true;
        return miopenStatusSuccess;
    }
    out.vectorLength = static_cast<int64_t>(vectorLength);
    return miopenStatusSuccess;
}

miopenStatus_t ReadActivationMode(miopenActivationDescriptor_t desc, miopenActivationMode_t& mode)
{
    // The getter just needs somewhere to write these.
    double alpha = 0.0;
    double beta  = 0.0;
    double gamma = 0.0;
    return miopenGetActivationDescriptor_impl(desc, &mode, &alpha, &beta, &gamma);
}

bool ScalarEquals(const void* value, miopenDataType_t type, double expected)
{
    if(value == nullptr)
        return false;
    if(type == miopenDouble)
        return *static_cast<const double*>(value) == expected;
    return static_cast<double>(*static_cast<const float*>(value)) == expected;
}

const char* CheckTensorLayouts(const std::vector<TensorInfo>& tensors)
{
    for(const TensorInfo& tensor : tensors)
    {
        if(tensor.truncated)
            return "tensors with a dim or stride above INT_MAX are not forwarded to hipDNN";
        // hipDNN describes a tensor only by dims and strides, which cannot
        // express a vector packed into each element.
        if(tensor.vectorLength != 1)
            return "vectorized tensor layouts (NCHWc4, NCHWc8, CHWNc4, CHWNc8) are not "
                   "forwarded to hipDNN";
    }
    return nullptr;
}

void AppendList(std::vector<int64_t>& out, const std::vector<int64_t>& values)
{
    out.push_back(static_cast<int64_t>(values.size()));
    out.insert(out.end(), values.begin(), values.end());
}

void AppendTensor(std::vector<int64_t>& out, const TensorInfo& tensor)
{
    out.push_back(static_cast<int64_t>(tensor.dataType));
    AppendList(out, tensor.dims);
    AppendList(out, tensor.strides);
}

std::shared_ptr<fe::graph::TensorAttributes>
MakeTensor(const TensorInfo& info, fe::DataType dataType, int64_t uid)
{
    return fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                        .set_dim(info.dims)
                                        .set_stride(info.strides)
                                        .set_data_type(dataType)
                                        .set_uid(uid));
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
