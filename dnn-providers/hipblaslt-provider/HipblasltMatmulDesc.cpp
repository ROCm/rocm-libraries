// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <limits>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipblasltMatmulDesc.hpp"
#include "HipblasltUtils.hpp"

namespace hipblaslt_plugin
{

HipblasltMatmulDesc::HipblasltMatmulDesc(hipblasOperation_t transA,
                                         hipblasOperation_t transB,
                                         hipblasComputeType_t computeType,
                                         hipDataType scaleType)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescCreate(&_desc, computeType, scaleType));
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)));
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)));
}

HipblasltMatmulDesc::HipblasltMatmulDesc(HipblasltMatmulDesc&& other) noexcept
    : _desc(other._desc)
{
    other._desc = nullptr;
}

HipblasltMatmulDesc& HipblasltMatmulDesc::operator=(HipblasltMatmulDesc&& other) noexcept
{
    if(this != &other)
    {
        if(_desc != nullptr)
        {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescDestroy(_desc));
        }

        _desc = other._desc;
        other._desc = nullptr;
    }
    return *this;
}

HipblasltMatmulDesc::~HipblasltMatmulDesc()
{
    if(_desc != nullptr)
    {
        LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescDestroy(_desc));
    }
}

hipblasLtMatmulDesc_t HipblasltMatmulDesc::matmulDesc() const
{
    return _desc;
}

void HipblasltMatmulDesc::setAScaleMode(hipblasLtMatmulMatrixScale_t scaleMode)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
}

void HipblasltMatmulDesc::setBScaleMode(hipblasLtMatmulMatrixScale_t scaleMode)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
}

void HipblasltMatmulDesc::setAScalePointer(const void* ptr)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, static_cast<const void*>(&ptr), sizeof(ptr)));
}

void HipblasltMatmulDesc::setBScalePointer(const void* ptr)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, static_cast<const void*>(&ptr), sizeof(ptr)));
}

} // namespace hipblaslt_plugin
