// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipblaslt/hipblaslt.h>

namespace hipblaslt_plugin
{

class HipblasltMatrixTransformDesc
{
public:
    HipblasltMatrixTransformDesc() = default;
    HipblasltMatrixTransformDesc(hipDataType scaleType, hipblasOperation_t transA);

    HipblasltMatrixTransformDesc(const HipblasltMatrixTransformDesc&) = delete;
    HipblasltMatrixTransformDesc& operator=(const HipblasltMatrixTransformDesc&) = delete;

    HipblasltMatrixTransformDesc(HipblasltMatrixTransformDesc&& other) noexcept;
    HipblasltMatrixTransformDesc& operator=(HipblasltMatrixTransformDesc&& other) noexcept;

    ~HipblasltMatrixTransformDesc();

    hipblasLtMatrixTransformDesc_t transformDesc() const;

private:
    hipblasLtMatrixTransformDesc_t _desc = nullptr;
};

} // namespace hipblaslt_plugin
