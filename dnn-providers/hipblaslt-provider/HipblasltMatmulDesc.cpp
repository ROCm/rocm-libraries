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
    : _transA(transA)
    , _transB(transB)
    , _computeType(computeType)
    , _scaleType(scaleType)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescCreate(&_desc, computeType, scaleType));
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(int32_t)));
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(int32_t)));
}

HipblasltMatmulDesc::HipblasltMatmulDesc(HipblasltMatmulDesc&& other) noexcept
    : _desc(other._desc)
    , _transA(other._transA)
    , _transB(other._transB)
    , _computeType(other._computeType)
    , _scaleType(other._scaleType)
    , _aScaleMode(other._aScaleMode)
    , _bScaleMode(other._bScaleMode)
    , _epilogue(other._epilogue)
    , _epilogueActArg0(other._epilogueActArg0)
    , _epilogueActArg1(other._epilogueActArg1)
    , _biasDataType(other._biasDataType)
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
        _transA = other._transA;
        _transB = other._transB;
        _computeType = other._computeType;
        _scaleType = other._scaleType;
        _aScaleMode = other._aScaleMode;
        _bScaleMode = other._bScaleMode;
        _epilogue = other._epilogue;
        _epilogueActArg0 = other._epilogueActArg0;
        _epilogueActArg1 = other._epilogueActArg1;
        _biasDataType = other._biasDataType;
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

HipblasltMatmulDesc HipblasltMatmulDesc::clone() const
{
    // An empty descriptor carries no configuration (and its mirror fields are
    // unset), so cloning it yields another empty descriptor.
    if(_desc == nullptr)
    {
        return HipblasltMatmulDesc{};
    }

    HipblasltMatmulDesc copy(_transA, _transB, _computeType, _scaleType);
    if(_aScaleMode.has_value())
    {
        copy.setAScaleMode(_aScaleMode.value());
    }
    if(_bScaleMode.has_value())
    {
        copy.setBScaleMode(_bScaleMode.value());
    }
    if(_epilogue.has_value())
    {
        copy.setEpilogue(_epilogue.value());
    }
    if(_epilogueActArg0.has_value())
    {
        copy.setEpilogueActArg0(_epilogueActArg0.value());
    }
    if(_epilogueActArg1.has_value())
    {
        copy.setEpilogueActArg1(_epilogueActArg1.value());
    }
    if(_biasDataType.has_value())
    {
        copy.setBiasDataType(_biasDataType.value());
    }
    return copy;
}

hipblasLtMatmulDesc_t HipblasltMatmulDesc::matmulDesc() const
{
    return _desc;
}

void HipblasltMatmulDesc::setAScaleMode(hipblasLtMatmulMatrixScale_t scaleMode)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    _aScaleMode = scaleMode;
}

void HipblasltMatmulDesc::setBScaleMode(hipblasLtMatmulMatrixScale_t scaleMode)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    _bScaleMode = scaleMode;
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

void HipblasltMatmulDesc::setEpilogue(hipblasLtEpilogue_t epilogue)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    _epilogue = epilogue;
}

void HipblasltMatmulDesc::setEpilogueActArg0(float actArg0)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT, &actArg0, sizeof(actArg0)));
    _epilogueActArg0 = actArg0;
}

void HipblasltMatmulDesc::setEpilogueActArg1(float actArg1)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT, &actArg1, sizeof(actArg1)));
    _epilogueActArg1 = actArg1;
}

void HipblasltMatmulDesc::setBiasDataType(hipDataType biasDataType)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasDataType, sizeof(biasDataType)));
    _biasDataType = biasDataType;
}

void HipblasltMatmulDesc::setBiasPointer(const void* ptr)
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulDescSetAttribute(
        _desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, static_cast<const void*>(&ptr), sizeof(ptr)));
}

} // namespace hipblaslt_plugin
