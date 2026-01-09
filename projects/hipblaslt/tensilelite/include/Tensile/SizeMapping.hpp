/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <Tensile/geom.hpp>

namespace TensileLite
{
    struct SizeMapping
    {
        size_t waveNum;

        dim3 workGroupSize;
        dim3 threadTile;
        dim3 macroTile;

        std::array<int, 4> matrixInstruction;
        size_t             grvwA = 1;
        size_t             grvwB = 1;
        size_t             gwvwC = 1;
        size_t             gwvwD = 1;

        size_t  staggerU           = 0;
        size_t  staggerUMapping    = 0;
        size_t  depthU             = 0;
        size_t  globalSplitUPGR    = 0;
        int16_t globalSplitU       = 0;
        size_t  staggerStrideShift = 0;
        int     workGroupMapping   = 0;

        size_t packBatchDims              = 0;
        int    packSummationDims          = 0;
        int    magicDivAlg                = 1;
        int    streamK                    = 0;
        int    streamKAtomic              = 0;
        int    persistentKernel           = 0;
        bool   persistentKernelAlongBatch = false;

        bool sourceKernel = false;

        int    globalAccumulation       = 0;
        size_t workspaceSizePerElemC    = 0;
        size_t workspaceSizePerElemBias = 0;

        bool activationFused = true;

        std::string customKernelName;

        int  workGroupMappingXCC                    = 0;
        int  workGroupMappingXCCGroup               = 0;
        bool globalSplitUCoalesced                  = false;
        bool globalSplitUWorkGroupMappingRoundRobin = false;

        int CUOccupancy            = 0;
        int PrefetchGlobalRead     = 2;
        int MathClocksUnrolledLoop = 0;

        size_t synchronizerSizePerWG = 0;

        int nonTemporalA = 0;
        int nonTemporalB = 0;
        int NonTemporalD = 0;
        int WaveSeparateGlobalReadA = 0;
        int WaveSeparateGlobalReadB = 0;
        int UnrollLoopSwapGlobalReadOrder = 0;
        bool DirectToVgprA = false;
        bool DirectToVgprB = false;
        int NumLoadsCoalescedA = 0;
        int NumLoadsCoalescedB = 0;
        int VectorWidthA = 1;
        int VectorWidthB = 1;
        int LocalSplitU = 1;
        bool DirectToLdsA = false;
        bool DirectToLdsB = false;

        std::array<int, 2> waveGroup;
    };
} // namespace TensileLite

