// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenReduceTensorDescriptor.hpp"
#include "MiopenUtils.hpp"

namespace miopen_plugin
{

MiopenReduceTensorDescriptor::MiopenReduceTensorDescriptor(miopenReduceTensorOp_t reduceOp)
{
    THROW_ON_MIOPEN_FAILURE(miopenCreateReduceTensorDescriptor(&_descriptor));

    try
    {
        // MIOPEN_32BIT_INDICES is a miopenIndicesType_t (not the pooling miopenIndexType_t); it
        // is ignored under NO_INDICES.
        THROW_ON_MIOPEN_FAILURE(miopenSetReduceTensorDescriptor(_descriptor,
                                                                reduceOp,
                                                                miopenFloat,
                                                                MIOPEN_PROPAGATE_NAN,
                                                                MIOPEN_REDUCE_TENSOR_NO_INDICES,
                                                                MIOPEN_32BIT_INDICES));
    }
    catch(...)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyReduceTensorDescriptor(_descriptor));
        throw;
    }
}

MiopenReduceTensorDescriptor::MiopenReduceTensorDescriptor(
    MiopenReduceTensorDescriptor&& other) noexcept
    : _descriptor(other._descriptor)
{
    other._descriptor = nullptr;
}

MiopenReduceTensorDescriptor&
    MiopenReduceTensorDescriptor::operator=(MiopenReduceTensorDescriptor&& other) noexcept
{
    if(this == &other)
    {
        return *this;
    }

    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyReduceTensorDescriptor(_descriptor));
    }

    _descriptor = other._descriptor;
    other._descriptor = nullptr;
    return *this;
}

MiopenReduceTensorDescriptor::~MiopenReduceTensorDescriptor()
{
    if(_descriptor != nullptr)
    {
        LOG_ON_MIOPEN_FAILURE(miopenDestroyReduceTensorDescriptor(_descriptor));
    }
}

miopenReduceTensorDescriptor_t MiopenReduceTensorDescriptor::reduceTensorDescriptor() const
{
    return _descriptor;
}

} // namespace miopen_plugin
