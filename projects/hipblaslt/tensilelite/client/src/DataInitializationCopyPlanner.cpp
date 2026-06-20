// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DataInitializationCopyPlanner.hpp"

#include <stdexcept>

namespace TensileLite::Client::detail
{
    namespace
    {
        TensorCopyOperationKind operationKindFor(TensorCopyIntent intent,
                                                 TensorCopyBoundsMode boundsMode)
        {
            if(boundsMode == TensorCopyBoundsMode::NaN)
                return TensorCopyOperationKind::BadBounds;

            if(intent == TensorCopyIntent::InputCopy
               && boundsMode == TensorCopyBoundsMode::GuardPageBack)
            {
                return TensorCopyOperationKind::GuardBack;
            }

            return TensorCopyOperationKind::Plain;
        }

        TensorBufferRole destinationRoleFor(hipMemcpyKind copyKind,
                                            std::optional<size_t> gpuTargetSlot)
        {
            switch(copyKind)
            {
            case hipMemcpyHostToHost:
                return TensorBufferRole::CpuCurrent;
            case hipMemcpyHostToDevice:
            case hipMemcpyDeviceToDevice:
                return gpuTargetSlot ? TensorBufferRole::GpuSlotData : TensorBufferRole::GpuCurrent;
            default:
                throw std::runtime_error("Unsupported hipMemcpyKind in planTensorCopies.");
            }
        }

        TensorBufferRole sourceRoleFor(hipMemcpyKind copyKind)
        {
            switch(copyKind)
            {
            case hipMemcpyHostToHost:
            case hipMemcpyHostToDevice:
                return TensorBufferRole::CpuValid;
            case hipMemcpyDeviceToDevice:
                return TensorBufferRole::GpuValid;
            default:
                throw std::runtime_error("Unsupported hipMemcpyKind in planTensorCopies.");
            }
        }

        std::optional<TensorBufferRole> badRoleFor(hipMemcpyKind copyKind,
                                                   TensorCopyBoundsMode boundsMode)
        {
            if(boundsMode != TensorCopyBoundsMode::NaN)
                return std::nullopt;

            switch(copyKind)
            {
            case hipMemcpyHostToHost:
            case hipMemcpyHostToDevice:
                return TensorBufferRole::CpuBad;
            case hipMemcpyDeviceToDevice:
                return TensorBufferRole::GpuBad;
            default:
                throw std::runtime_error("Unsupported hipMemcpyKind in planTensorCopies.");
            }
        }

        TensorBatchRole batchRoleFor(hipMemcpyKind copyKind, std::optional<size_t> gpuTargetSlot)
        {
            switch(copyKind)
            {
            case hipMemcpyHostToHost:
                return TensorBatchRole::CpuCurrent;
            case hipMemcpyHostToDevice:
            case hipMemcpyDeviceToDevice:
                return gpuTargetSlot ? TensorBatchRole::GpuSlot : TensorBatchRole::GpuCurrent;
            default:
                throw std::runtime_error("Unsupported hipMemcpyKind in planTensorCopies.");
            }
        }
    } // namespace

    std::vector<std::optional<TensorCopyInstruction>> planTensorCopies(
        ContractionProblemGemm const&          problem,
        std::vector<TensorCopyView> const&     views,
        TensorCopyIntent                      intent,
        TensorCopyBoundsMode                  boundsMode,
        hipMemcpyKind                         copyKind,
        std::optional<size_t>                 gpuTargetSlot)
    {
        auto const& tensors = problem.tensors();
        if(views.size() != tensors.size())
        {
            throw std::runtime_error("Tensor count mismatch while planning tensor copies.");
        }

        std::vector<std::optional<TensorCopyInstruction>> plan(tensors.size());
        InputLayoutPolicy const layoutPolicy;

        for(size_t tensorIndex = 0; tensorIndex < tensors.size(); ++tensorIndex)
        {
            auto const& view = views[tensorIndex];
            if(!view.hasPristine)
                continue;

            if(intent == TensorCopyIntent::OutputReset && !tensors[tensorIndex].isOutput())
                continue;

            TensorCopyInstruction instruction;
            instruction.tensorIndex   = tensorIndex;
            instruction.operationKind = operationKindFor(intent, boundsMode);
            instruction.copyKind      = copyKind;
            instruction.dstRole       = destinationRoleFor(copyKind, gpuTargetSlot);
            instruction.srcRole       = sourceRoleFor(copyKind);
            instruction.badRole       = badRoleFor(copyKind, boundsMode);
            instruction.batchRole     = batchRoleFor(copyKind, gpuTargetSlot);
            instruction.gpuTargetSlot  = gpuTargetSlot;
            instruction.maxElements    = view.maxElements;
            instruction.groupedOffsets = view.groupedOffsets;

            if(instruction.operationKind == TensorCopyOperationKind::GuardBack)
            {
                auto const swizzlePlan = layoutPolicy.planTensorSwizzle(problem, tensorIndex);
                if(swizzlePlan.enabled)
                {
                    instruction.customPadding = static_cast<ptrdiff_t>(
                        swizzlePlan.allocatedElements
                        - tensors[tensorIndex].totalAllocatedElements());
                }
            }

            plan[tensorIndex] = std::move(instruction);
        }

        return plan;
    }
} // namespace TensileLite::Client::detail
