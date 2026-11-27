// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_tensor.hpp"

#include "miopen_impl.h"

#include <algorithm>
#include <cstdint>

namespace miopen {
namespace wrapper {
namespace hipdnn {

namespace fe = hipdnn_frontend;

miopenStatus_t ReadTensor(miopenTensorDescriptor_t desc, TensorInfo& out)
{
    int size = 0;
    if(miopenGetTensorDescriptorSize_impl(desc, &size) != miopenStatusSuccess || size <= 0)
        return miopenStatusInvalidValue;

    // Read straight into `out`, since this runs on every forwarded call: a size_t may be
    // accessed through its signed counterpart, and no length or stride reaches 2^63.
    static_assert(sizeof(size_t) == sizeof(int64_t));
    out.dims.resize(static_cast<size_t>(size));
    out.strides.resize(static_cast<size_t>(size));
    size_t vectorLength = 1;
    if(miopenGetTensorDescriptorDetails_impl(desc,
                                             &out.dataType,
                                             &out.castType,
                                             reinterpret_cast<size_t*>(out.dims.data()),
                                             reinterpret_cast<size_t*>(out.strides.data()),
                                             &vectorLength) != miopenStatusSuccess)
        return miopenStatusInvalidValue;

    out.vectorLength = static_cast<int64_t>(vectorLength);
    return miopenStatusSuccess;
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
    out.push_back(tensor.alignment);
}

int64_t PointerAlignment(const void* pointer)
{
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if(address == 0)
        return 16;
    return static_cast<int64_t>(std::min<std::uintptr_t>(address & (~address + 1), 16));
}

std::shared_ptr<fe::graph::TensorAttributes>
MakeTensor(const TensorInfo& info, fe::DataType dataType, int64_t uid)
{
    return fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                        .set_dim(info.dims)
                                        .set_stride(info.strides)
                                        .set_data_type(dataType)
                                        .set_alignment(info.alignment)
                                        .set_uid(uid));
}

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
