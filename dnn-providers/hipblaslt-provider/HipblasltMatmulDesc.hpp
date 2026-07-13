// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

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

    // An independent copy of the configured descriptor, so each execute() can set
    // its own per-call scale pointers without racing other calls on a shared
    // descriptor. The per-call scale pointers themselves are not copied.
    HipblasltMatmulDesc clone() const;

    hipblasLtMatmulDesc_t matmulDesc() const;

    void setAScaleMode(hipblasLtMatmulMatrixScale_t scaleMode);
    void setBScaleMode(hipblasLtMatmulMatrixScale_t scaleMode);
    void setAScalePointer(const void* ptr);
    void setBScalePointer(const void* ptr);

private:
    hipblasLtMatmulDesc_t _desc = nullptr;

    // Mirror of the wrapper-applied configuration, replayed by clone(). Per-call
    // scale pointers are not mirrored: they are set on the clone at execute time.
    hipblasOperation_t _transA;
    hipblasOperation_t _transB;
    hipblasComputeType_t _computeType;
    hipDataType _scaleType;
    std::optional<hipblasLtMatmulMatrixScale_t> _aScaleMode;
    std::optional<hipblasLtMatmulMatrixScale_t> _bScaleMode;
};

} // namespace hipblaslt_plugin
