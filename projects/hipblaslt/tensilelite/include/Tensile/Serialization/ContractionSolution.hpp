/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * Copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <functional>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Serialization/Base.hpp>

namespace TensileLite
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<std::shared_ptr<ContractionSolution>, IO>
        {
            static void mapping(IO& io, std::shared_ptr<ContractionSolution>& p)
            {
                PointerMappingTraits<ContractionSolution, IO>::mapping(io, p);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, ContractionSolution& s)
            {
                iot::mapRequired(io, "name", s.solutionName);
                iot::mapRequired(io, "kernelName", s.kernelName);
                iot::mapRequired(io, "index", s.index);

                iot::mapRequired(io, "hardwarePredicate", s.hardwarePredicate);
                iot::mapRequired(io, "problemPredicate", s.problemPredicate);
                iot::mapRequired(io, "taskPredicate", s.taskPredicate);

                iot::mapRequired(io, "debugKernel", s.debugKernel);
                iot::mapOptional(io, "libraryLogicIndex", s.libraryLogicIndex);
                iot::mapOptional(io, "ideals", s.ideals);
                iot::mapOptional(io, "linearModel", s.linearModel);

                iot::mapRequired(io, "sizeMapping", s.sizeMapping);
                iot::mapRequired(io, "internalArgsSupport", s.internalArgsSupport);
                iot::mapRequired(io, "problemType", s.problemType);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<SizeMapping, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, SizeMapping& s)
            {
                iot::mapRequired(io, "waveNum", s.waveNum);

                iot::mapRequired(io, "workGroup", s.workGroupSize);
                iot::mapRequired(io, "threadTile", s.threadTile);
                iot::mapRequired(io, "macroTile", s.macroTile);
                iot::mapRequired(io, "matrixInstruction", s.matrixInstruction);
                iot::mapRequired(io, "grvwA", s.grvwA);
                iot::mapRequired(io, "grvwB", s.grvwB);
                iot::mapRequired(io, "gwvwC", s.gwvwC);
                iot::mapRequired(io, "gwvwD", s.gwvwD);

                iot::mapRequired(io, "staggerU", s.staggerU);
                iot::mapRequired(io, "staggerUMapping", s.staggerUMapping);
                iot::mapRequired(io, "depthU", s.depthU);
                iot::mapRequired(io, "globalSplitUPGR", s.globalSplitUPGR);
                iot::mapRequired(io, "globalSplitU", s.globalSplitU);
                iot::mapRequired(io, "staggerStrideShift", s.staggerStrideShift);
                iot::mapRequired(io, "workGroupMapping", s.workGroupMapping);

                iot::mapOptional(io, "packBatchDims", s.packBatchDims);
                iot::mapOptional(io, "packSummationDims", s.packSummationDims);
                iot::mapOptional(io, "magicDivAlg", s.magicDivAlg);
                iot::mapOptional(io, "streamK", s.streamK);
                iot::mapOptional(io, "streamKAtomic", s.streamKAtomic);
                iot::mapOptional(io, "persistentKernel", s.persistentKernel);
                iot::mapOptional(io, "persistentKernelAlongBatch", s.persistentKernelAlongBatch);
                iot::mapRequired(io, "sourceKernel", s.sourceKernel);

                iot::mapRequired(io, "globalAccumulation", s.globalAccumulation);
                iot::mapRequired(io, "workspaceSizePerElemC", s.workspaceSizePerElemC);
                iot::mapRequired(io, "workspaceSizePerElemBias", s.workspaceSizePerElemBias);

                iot::mapOptional(io, "activationFused", s.activationFused);

                iot::mapOptional(io, "CustomKernelName", s.customKernelName);

                iot::mapRequired(io, "workGroupMappingXCC", s.workGroupMappingXCC);
                iot::mapRequired(io, "workGroupMappingXCCGroup", s.workGroupMappingXCCGroup);

                iot::mapRequired(io, "globalSplitUCoalesced", s.globalSplitUCoalesced);
                iot::mapRequired(io,
                                 "globalSplitUWorkGroupMappingRoundRobin",
                                 s.globalSplitUWorkGroupMappingRoundRobin);
                iot::mapRequired(io, "CUOccupancy", s.CUOccupancy);
                iot::mapRequired(io, "PrefetchGlobalRead", s.PrefetchGlobalRead);
                iot::mapRequired(io, "MathClocksUnrolledLoop", s.MathClocksUnrolledLoop);
                iot::mapRequired(io, "synchronizerSizePerWG", s.synchronizerSizePerWG);
                iot::mapRequired(io, "nonTemporalA", s.nonTemporalA);
                iot::mapRequired(io, "nonTemporalB", s.nonTemporalB);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution::InternalArgsSupport, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, ContractionSolution::InternalArgsSupport& s)
            {
                iot::mapRequired(io, "version", s.version);
                iot::mapRequired(io, "gsu", s.gsu);
                iot::mapRequired(io, "wgm", s.wgm);
                iot::mapRequired(io, "staggerU", s.staggerU);
                iot::mapRequired(io, "useUniversalArgs", s.useUniversalArgs);
                iot::mapRequired(io, "useSFC", s.useSFC);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution::ProblemType, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, ContractionSolution::ProblemType& s)
            {
                iot::mapRequired(io, "operationIdentifier", s.operationIdentifier);

                iot::mapRequired(io, "transA", s.transA);
                iot::mapRequired(io, "transB", s.transB);
                iot::mapRequired(io, "aType", s.aType);
                iot::mapRequired(io, "bType", s.bType);
                iot::mapRequired(io, "cType", s.cType);
                iot::mapRequired(io, "dType", s.dType);
                iot::mapOptional(io, "eType", s.eType);
                iot::mapRequired(io, "computeInputType", s.computeInputType);
                iot::mapRequired(io, "computeType", s.computeType);
                iot::mapOptional(io, "useGradient", s.useGradient);
                iot::mapRequired(io, "useBeta", s.useBeta);
                iot::mapOptional(io, "useBias", s.useBias);
                iot::mapOptional(io, "useE", s.useE);
                iot::mapOptional(io, "useScaleAB", s.useScaleAB);
                iot::mapOptional(io, "useScaleCD", s.useScaleCD);
                iot::mapOptional(io, "useScaleAlphaVec", s.useScaleAlphaVec);
                iot::mapOptional(io, "outputAmaxD", s.outputAmaxD);
                iot::mapRequired(io, "highPrecisionAccumulate", s.highPrecisionAccumulate);
                iot::mapOptional(io, "useInitialStridesAB", s.useInitialStridesAB);
                iot::mapOptional(io, "useInitialStridesCD", s.useInitialStridesCD);
                iot::mapOptional(io, "stridedBatched", s.stridedBatched);
                iot::mapOptional(io, "groupedGemm", s.groupedGemm);
                iot::mapOptional(io, "activationType", s.activationType);
                iot::mapOptional(io, "activationArgLength", s.activationArgLength);
                iot::mapOptional(io, "activationComputeDataType", s.activationComputeDataType);
                iot::mapOptional(io, "activationNoGuard", s.activationNoGuard);
                iot::mapOptional(io, "biasSrcWhiteList", s.biasSrcWhiteList);
                iot::mapOptional(io, "biasDataTypeWhiteList", s.biasDataTypeWhiteList);
                iot::mapOptional(io, "sparse", s.sparse);
                iot::mapOptional(io, "f32XdlMathOp", s.f32XdlMathOp);
                iot::mapOptional(io, "supportDeviceUserArguments", s.supportDeviceUserArguments);
                iot::mapOptional(io, "swizzleTensorA", s.swizzleTensorA);
                iot::mapOptional(io, "swizzleTensorB", s.swizzleTensorB);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution::LinearModel, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, ContractionSolution::LinearModel& s)
            {
                iot::mapOptional(io, "slope", s.slope);
                iot::mapOptional(io, "intercept", s.intercept);
                iot::mapOptional(io, "max", s.max);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<BufferLoadCheckPacket, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, BufferLoadCheckPacket& s)
            {
                iot::mapRequired(io, "ShiftPtrElemA", s.shiftPtrElemA);
                iot::mapRequired(io, "ShiftPtrElemB", s.shiftPtrElemB);
                iot::mapRequired(io, "DUorMT0", s.depthUorMT0);
                iot::mapRequired(io, "DUorMT1", s.depthUorMT1);
            }

            const static bool flow = false;
        };
    } // namespace Serialization
} // namespace TensileLite
