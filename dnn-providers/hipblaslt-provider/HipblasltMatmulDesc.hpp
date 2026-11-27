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
    // its own per-call scale and bias pointers without racing other calls on a
    // shared descriptor. The clone replays the transposes, compute and scale type,
    // scale modes, epilogue, activation arguments and bias data type. The per-call
    // scale and bias pointers are not copied.
    HipblasltMatmulDesc clone() const;

    hipblasLtMatmulDesc_t matmulDesc() const;

    void setAScaleMode(hipblasLtMatmulMatrixScale_t scaleMode);
    void setBScaleMode(hipblasLtMatmulMatrixScale_t scaleMode);
    void setAScalePointer(const void* ptr);
    void setBScalePointer(const void* ptr);
    void setEpilogue(hipblasLtEpilogue_t epilogue);
    void setEpilogueActArg0(float actArg0);
    void setEpilogueActArg1(float actArg1);
    void setBiasDataType(hipDataType biasDataType);
    void setBiasPointer(const void* ptr);

private:
    hipblasLtMatmulDesc_t _desc = nullptr;

    // Mirror of the wrapper-applied configuration, replayed by clone(). Per-call
    // scale and bias pointers are not mirrored: they are set on the clone at execute time.
    hipblasOperation_t _transA;
    hipblasOperation_t _transB;
    hipblasComputeType_t _computeType;
    hipDataType _scaleType;
    std::optional<hipblasLtMatmulMatrixScale_t> _aScaleMode;
    std::optional<hipblasLtMatmulMatrixScale_t> _bScaleMode;
    std::optional<hipblasLtEpilogue_t> _epilogue;
    std::optional<float> _epilogueActArg0;
    std::optional<float> _epilogueActArg1;
    std::optional<hipDataType> _biasDataType;
};

} // namespace hipblaslt_plugin
