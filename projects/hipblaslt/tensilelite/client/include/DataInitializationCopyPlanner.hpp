// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "InputLayoutPolicy.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace TensileLite::Client::detail
{
    enum class TensorCopyIntent
    {
        InputCopy,
        OutputReset,
    };

    enum class TensorCopyBoundsMode
    {
        Disable,
        NaN,
        GuardPageBack,
    };

    enum class TensorCopyOperationKind
    {
        Plain,
        BadBounds,
        GuardBack,
    };

    enum class TensorBufferRole
    {
        CpuCurrent,
        CpuValid,
        CpuBad,
        GpuCurrent,
        GpuValid,
        GpuBad,
        GpuSlotData,
    };

    enum class TensorBatchRole
    {
        CpuCurrent,
        GpuCurrent,
        GpuSlot,
    };

    struct TensorCopyView
    {
        bool                hasPristine  = false;
        size_t              maxElements  = 0;
        std::vector<size_t> groupedOffsets;
    };

    struct TensorCopyInstruction
    {
        size_t                     tensorIndex    = 0;
        TensorCopyOperationKind     operationKind  = TensorCopyOperationKind::Plain;
        hipMemcpyKind               copyKind       = hipMemcpyHostToHost;
        TensorBufferRole            dstRole        = TensorBufferRole::CpuCurrent;
        TensorBufferRole            srcRole        = TensorBufferRole::CpuValid;
        std::optional<TensorBufferRole> badRole;
        TensorBatchRole             batchRole      = TensorBatchRole::CpuCurrent;
        std::optional<size_t>       gpuTargetSlot;
        size_t                      maxElements    = 0;
        ptrdiff_t                   customPadding  = -1;
        std::vector<size_t>         groupedOffsets;
    };

    std::vector<std::optional<TensorCopyInstruction>> planTensorCopies(
        ContractionProblemGemm const&          problem,
        std::vector<TensorCopyView> const&     views,
        TensorCopyIntent                      intent,
        TensorCopyBoundsMode                  boundsMode,
        hipMemcpyKind                         copyKind,
        std::optional<size_t>                 gpuTargetSlot = std::nullopt);
} // namespace TensileLite::Client::detail
