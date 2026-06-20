// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace TensileLite::Client
{
    inline bool isMXTensor(const TensorDescriptor& tensor, size_t mxBlock)
    {
        if(mxBlock == 0)
            return false;

        auto const dt = tensor.dataType();
        return dt == rocisa::DataType::Float4 || dt == rocisa::DataType::Float8
               || dt == rocisa::DataType::BFloat8;
    }

    inline bool isF6(const TensorDescriptor& tensor)
    {
        auto const dt = tensor.dataType();
        return dt == rocisa::DataType::Float6 || dt == rocisa::DataType::BFloat6;
    }

    inline bool isMXProblemExceptF6(const ContractionProblemGemm& problem)
    {
        bool const isAnyF6 = isF6(problem.a()) || isF6(problem.b());
        return !isAnyF6
               && (isMXTensor(problem.a(), problem.mxBlockA())
                   || isMXTensor(problem.b(), problem.mxBlockB()));
    }

    inline size_t bitWidthForDataType(rocisa::DataType datatype)
    {
        switch(datatype)
        {
        case rocisa::DataType::Double:
            return 64;
        case rocisa::DataType::XFloat32:
        case rocisa::DataType::Float:
            return 32;
        case rocisa::DataType::Half:
        case rocisa::DataType::BFloat16:
            return 16;
        case rocisa::DataType::Int8:
        case rocisa::DataType::Float8_fnuz:
        case rocisa::DataType::BFloat8_fnuz:
        case rocisa::DataType::Float8BFloat8_fnuz:
        case rocisa::DataType::BFloat8Float8_fnuz:
        case rocisa::DataType::Float8:
        case rocisa::DataType::BFloat8:
        case rocisa::DataType::Float8BFloat8:
        case rocisa::DataType::BFloat8Float8:
        case rocisa::DataType::E8:
        case rocisa::DataType::E5M3:
            return 8;
        default:
            throw std::runtime_error("unsupported datatype");
        }
    }

    struct SelectedSolutionLayout
    {
        bool   present             = false;
        int    mxScaleFormat       = -1;
        size_t matrixInstructionK  = 0;
    };

    struct InputLayoutContext
    {
        int                   userMxScaleFormat   = 0;
        bool                  isMxPreswizzleArch  = false;
        SelectedSolutionLayout solution;
    };

    struct MxPreswizzleState
    {
        bool a = false;
        bool b = false;
    };

    enum class TensorUploadLayout
    {
        Plain,
        TensorSwizzle,
        MxCopyCanonical,
        MxUsePreswizzledGpuValid,
        MxKSwizzle,
    };

    struct TensorSwizzlePlan
    {
        bool                   enabled        = false;
        size_t                 miMN          = 16;
        size_t                 miK           = 0;
        size_t                 miKv          = 0;
        size_t                 packK         = 0;
        std::array<size_t, 2>   paddedShape   {0, 0};
        size_t                 bitWidth       = 0;
        size_t                 unrolledSize   = 0;
        size_t                 tiledSize      = 0;
        size_t                 batchCount     = 1;
        size_t                 allocatedElements = 0;
    };

    struct MxInitializationSidePlan
    {
        bool                useGenerator       = false;
        bool                canHostPreswizzle  = false;
        std::vector<size_t> preSwizzle;
        std::vector<size_t> preTile;
    };

    struct MxInitializationPlan
    {
        bool                    useGenerator = false;
        MxInitializationSidePlan a;
        MxInitializationSidePlan b;
    };

    struct MxTensorLayoutPlan
    {
        TensorUploadLayout action      = TensorUploadLayout::Plain;
        bool               isA         = false;
        bool               unrollMajor = false;
        size_t             mxBlock     = 0;
        size_t             dimK        = 0;
    };

    class InputLayoutPolicy
    {
    public:
        static constexpr size_t ordinarySwizzleMiMN = 16;
        static constexpr size_t mxSwizzleTileMN     = 32;
        static constexpr size_t mxSwizzleTileK      = 256 / mxSwizzleTileMN;

        bool hasSpecialInputLayout(ContractionProblemGemm const& problem) const
        {
            return problem.swizzleTensorA() || problem.swizzleTensorB() || problem.mxBlockA() != 0
                   || problem.mxBlockB() != 0;
        }

        TensorSwizzlePlan planTensorSwizzle(ContractionProblemGemm const& problem,
                                            size_t                         tensorIndex) const
        {
            TensorSwizzlePlan plan;

            auto const& desc = problem.tensors().at(tensorIndex);
            bool const isOrdinarySwizzle
                = (tensorIndex == ContractionProblemGemm::TENSOR::A && problem.swizzleTensorA())
                  || (tensorIndex == ContractionProblemGemm::TENSOR::B
                      && problem.swizzleTensorB());
            if(!isOrdinarySwizzle)
            {
                plan.allocatedElements = desc.totalAllocatedElements();
                return plan;
            }

            plan.enabled = true;
            plan.unrolledSize = desc.sizes()[0];
            plan.tiledSize    = desc.sizes()[1];
            plan.batchCount   = desc.sizes().size() > 2 ? desc.sizes()[2] : 1;
            plan.bitWidth = bitWidthForDataType(desc.dataType());

            switch(desc.dataType())
            {
            case rocisa::DataType::Float:
            case rocisa::DataType::Double:
                plan.miK  = 4;
                plan.miKv = 1;
                break;
            case rocisa::DataType::XFloat32:
                plan.miK  = 8;
                plan.miKv = 2;
                break;
            case rocisa::DataType::Half:
            case rocisa::DataType::BFloat16:
                plan.miK  = 16;
                plan.miKv = 4;
                break;
            case rocisa::DataType::Int8:
            case rocisa::DataType::Float8_fnuz:
            case rocisa::DataType::BFloat8_fnuz:
            case rocisa::DataType::Float8BFloat8_fnuz:
            case rocisa::DataType::BFloat8Float8_fnuz:
            case rocisa::DataType::Float8:
            case rocisa::DataType::BFloat8:
            case rocisa::DataType::Float8BFloat8:
            case rocisa::DataType::BFloat8Float8:
            case rocisa::DataType::E8:
            case rocisa::DataType::E5M3:
                plan.miK  = 32;
                plan.miKv = 8;
                break;
            default:
                throw std::runtime_error("unsupported datatype for swizzling");
            }

            plan.packK = static_cast<size_t>(
                16 / plan.miKv / rocisa::GetElementSize(desc.dataType()));

            auto const swizzleK = plan.miK * plan.packK;
            plan.paddedShape[0] = ((plan.tiledSize + plan.miMN - 1) / plan.miMN) * plan.miMN;
            plan.paddedShape[1]
                = ((plan.unrolledSize + swizzleK - 1) / swizzleK) * swizzleK;
            plan.allocatedElements
                = plan.paddedShape[0] * plan.paddedShape[1] * plan.batchCount;

            return plan;
        }

        size_t plannedAllocatedElements(ContractionProblemGemm const& problem,
                                       size_t                         tensorIndex) const
        {
            auto const& desc = problem.tensors().at(tensorIndex);

            if((tensorIndex == ContractionProblemGemm::TENSOR::A && problem.swizzleTensorA())
               || (tensorIndex == ContractionProblemGemm::TENSOR::B && problem.swizzleTensorB()))
            {
                return planTensorSwizzle(problem, tensorIndex).allocatedElements;
            }

            if(tensorIndex == ContractionProblemGemm::TENSOR::MXSA && problem.mxBlockA() != 0)
            {
                bool const unrollMajor = problem.freeIndicesA()[0].i != 0;
                size_t const mxBlock   = problem.mxBlockA();
                size_t const dimK      = 128 / mxBlock;
                return getSwizzledMXTensorNumAllocatedElements(desc, dimK, unrollMajor);
            }

            if(tensorIndex == ContractionProblemGemm::TENSOR::MXSB && problem.mxBlockB() != 0)
            {
                bool const unrollMajor = problem.freeIndicesB()[0].i != 0;
                size_t const mxBlock   = problem.mxBlockB();
                size_t const dimK      = 128 / mxBlock;
                return getSwizzledMXTensorNumAllocatedElements(desc, dimK, unrollMajor);
            }

            return desc.totalAllocatedElements();
        }

        MxInitializationPlan planMxInitialization(ContractionProblemGemm const& problem,
                                                  InputLayoutContext const&      context) const
        {
            MxInitializationPlan plan;
            plan.useGenerator = isMXProblemExceptF6(problem) && context.userMxScaleFormat > 0;
            plan.a.useGenerator
                = plan.useGenerator && isMXTensor(problem.a(), problem.mxBlockA());
            plan.b.useGenerator
                = plan.useGenerator && isMXTensor(problem.b(), problem.mxBlockB());

            if(!plan.useGenerator || !context.solution.present
               || context.solution.matrixInstructionK == 0)
            {
                return plan;
            }

            auto const matrixInstructionK
                = static_cast<size_t>(context.solution.matrixInstructionK);
            constexpr size_t swizzleTileMN = mxSwizzleTileMN;
            constexpr size_t tileK         = mxSwizzleTileK;

            auto buildSidePlan = [&](TensorDescriptor const& scaleTensor,
                                     size_t                  mxBlock,
                                     MxInitializationSidePlan& side) {
                if(mxBlock == 0)
                    return;
                if(matrixInstructionK % mxBlock != 0)
                    return;

                auto const& sizes = scaleTensor.sizes();
                if(sizes[0] % tileK != 0 || sizes[1] % swizzleTileMN != 0)
                    return;

                size_t const subTileK = matrixInstructionK / mxBlock;
                side.canHostPreswizzle = true;
                side.preSwizzle        = {swizzleTileMN, tileK, subTileK};
                side.preTile           = {tileK, swizzleTileMN};
            };

            buildSidePlan(problem.mxsa(), problem.mxBlockA(), plan.a);
            buildSidePlan(problem.mxsb(), problem.mxBlockB(), plan.b);
            return plan;
        }

        bool shouldSkipDefaultInitTensor(size_t tensorIndex,
                                         MxInitializationPlan const& plan) const
        {
            if(!plan.useGenerator)
                return false;

            return tensorIndex == ContractionProblemGemm::TENSOR::A
                   || tensorIndex == ContractionProblemGemm::TENSOR::B
                   || tensorIndex == ContractionProblemGemm::TENSOR::MXSA
                   || tensorIndex == ContractionProblemGemm::TENSOR::MXSB;
        }

        MxTensorLayoutPlan planMxTensorUpload(ContractionProblemGemm const& problem,
                                              size_t                         tensorIndex,
                                              InputLayoutContext const&      context,
                                              MxPreswizzleState const&       preswizzleState) const
        {
            MxTensorLayoutPlan plan;
            plan.isA = tensorIndex == ContractionProblemGemm::TENSOR::MXSA;
            plan.mxBlock = plan.isA ? problem.mxBlockA() : problem.mxBlockB();
            if(plan.mxBlock == 0
               || (tensorIndex != ContractionProblemGemm::TENSOR::MXSA
                   && tensorIndex != ContractionProblemGemm::TENSOR::MXSB))
            {
                return plan;
            }

            plan.unrollMajor
                = plan.isA ? (problem.freeIndicesA()[0].i != 0)
                           : (problem.freeIndicesB()[0].i != 0);
            plan.dimK = 128 / plan.mxBlock;

            if(context.solution.present && context.solution.mxScaleFormat == 0)
            {
                plan.action = TensorUploadLayout::MxCopyCanonical;
                return plan;
            }

            if(context.isMxPreswizzleArch
               && ((plan.isA && preswizzleState.a) || (!plan.isA && preswizzleState.b)))
            {
                plan.action = TensorUploadLayout::MxUsePreswizzledGpuValid;
                return plan;
            }

            if(context.isMxPreswizzleArch)
            {
                plan.action = TensorUploadLayout::MxCopyCanonical;
                return plan;
            }

            plan.action = TensorUploadLayout::MxKSwizzle;
            return plan;
        }

        TensorUploadLayout tensorUploadLayout(ContractionProblemGemm const& problem,
                                              size_t                         tensorIndex,
                                              InputLayoutContext const&      context,
                                              MxPreswizzleState const&       preswizzleState) const
        {
            bool const ordinarySwizzle
                = (tensorIndex == ContractionProblemGemm::TENSOR::A && problem.swizzleTensorA())
                  || (tensorIndex == ContractionProblemGemm::TENSOR::B
                      && problem.swizzleTensorB());
            if(ordinarySwizzle)
                return TensorUploadLayout::TensorSwizzle;

            if(tensorIndex == ContractionProblemGemm::TENSOR::MXSA
               || tensorIndex == ContractionProblemGemm::TENSOR::MXSB)
            {
                return planMxTensorUpload(problem, tensorIndex, context, preswizzleState).action;
            }

            return TensorUploadLayout::Plain;
        }

        bool shouldRefreshMxForSolution(ContractionProblemGemm const& problem,
                                        InputLayoutContext const&      context,
                                        bool                           gpuInputsPrepared) const
        {
            return context.solution.present && context.userMxScaleFormat > 0
                   && isMXProblemExceptF6(problem) && gpuInputsPrepared;
        }

    private:
        static size_t getSwizzledMXTensorNumAllocatedElements(const TensorDescriptor& desc,
                                                              size_t                  dimk,
                                                              bool                    unrollMajor)
        {
            const auto k    = unrollMajor ? desc.sizes()[0] : desc.sizes()[1];
            const auto m_n  = unrollMajor ? desc.sizes()[1] : desc.sizes()[0];
            const auto b    = desc.sizes()[2];
            const auto padk = (k + dimk - 1) / dimk * dimk;
            return padk * m_n * b;
        }
    };
} // namespace TensileLite::Client
