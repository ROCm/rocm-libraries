// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltMatrixTransformDesc.hpp"
#include "HipblasltUtils.hpp"

namespace hipblaslt_plugin
{

HipblasltMatrixTransformDesc::HipblasltMatrixTransformDesc(hipDataType scaleType,
                                                           hipblasOperation_t transA)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixTransformDescCreate(&_desc, scaleType));
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixTransformDescSetAttribute(
        _desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transA, sizeof(transA)));
}

HipblasltMatrixTransformDesc::HipblasltMatrixTransformDesc(
    HipblasltMatrixTransformDesc&& other) noexcept
    : _desc(other._desc)
{
    other._desc = nullptr;
}

HipblasltMatrixTransformDesc&
    HipblasltMatrixTransformDesc::operator=(HipblasltMatrixTransformDesc&& other) noexcept
{
    if(this != &other)
    {
        if(_desc != nullptr)
        {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatrixTransformDescDestroy(_desc));
        }

        _desc = other._desc;
        other._desc = nullptr;
    }
    return *this;
}

HipblasltMatrixTransformDesc::~HipblasltMatrixTransformDesc()
{
    if(_desc != nullptr)
    {
        LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatrixTransformDescDestroy(_desc));
    }
}

hipblasLtMatrixTransformDesc_t HipblasltMatrixTransformDesc::transformDesc() const
{
    return _desc;
}

} // namespace hipblaslt_plugin
