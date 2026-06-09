// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipblaslt/hipblaslt.h>

namespace hipblaslt_plugin
{

class HipblasltMatmulDesc
{
public:
    HipblasltMatmulDesc() = default;
    HipblasltMatmulDesc(hipblasOperation_t transA,
                        hipblasOperation_t transB,
                        hipblasComputeType_t computeType,
                        hipDataType scaleType);

    HipblasltMatmulDesc(const HipblasltMatmulDesc&) = delete;
    HipblasltMatmulDesc& operator=(const HipblasltMatmulDesc&) = delete;

    HipblasltMatmulDesc(HipblasltMatmulDesc&& other) noexcept;
    HipblasltMatmulDesc& operator=(HipblasltMatmulDesc&& other) noexcept;

    ~HipblasltMatmulDesc();

    hipblasLtMatmulDesc_t matmulDesc() const;

    void setAScaleMode(hipblasLtMatmulMatrixScale_t scaleMode);
    void setBScaleMode(hipblasLtMatmulMatrixScale_t scaleMode);
    void setAScalePointer(const void* ptr);
    void setBScalePointer(const void* ptr);

private:
    hipblasLtMatmulDesc_t _desc = nullptr;
};

} // namespace hipblaslt_plugin
