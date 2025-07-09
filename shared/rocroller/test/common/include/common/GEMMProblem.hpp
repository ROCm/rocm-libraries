/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/BlockScale_fwd.hpp>
#include <string>

struct GEMMProblem
{
    // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
    int   m     = 512;
    int   n     = 512;
    int   k     = 128;
    float alpha = 2.0f;
    float beta  = 0.5f;

    // output macro tile size
    int macM = 64;
    int macN = 64;
    int macK = 64;

    // Wave tile size
    int waveM = 32;
    int waveN = 32;
    int waveK = 2;
    int waveB = 1;

    // Workgroup size
    uint wavefrontSize  = 64;
    uint workgroupSizeX = 2 * wavefrontSize;
    uint workgroupSizeY = 2;

    uint numWGs = 0;

    std::string transA = "N";
    std::string transB = "T";

    // Unroll Sizes
    unsigned int unrollX = 0;
    unsigned int unrollY = 0;
    unsigned int unrollK = 0;

    bool loadLDSA    = true;
    bool loadLDSB    = true;
    bool storeLDSD   = true;
    bool direct2LDSA = false;
    bool direct2LDSB = false;

    bool fuseLoops                 = true;
    bool tailLoops                 = true;
    bool allowAmbiguousMemoryNodes = false;
    bool betaInFma                 = true;
    bool literalStrides            = true;

    bool swizzleScale  = false;
    bool prefetchScale = false;

    bool prefetch          = false;
    int  prefetchInFlight  = 1;
    int  prefetchLDSFactor = 0;
    bool prefetchMixMemOps = false;

    bool packMultipleElementsInto1VGPR = true;

    bool loopOverTiles  = false;
    bool streamK        = false;
    bool streamKTwoTile = false;

    bool splitStoreTileIntoWaveBlocks = false;

    bool loadLDSScaleA = false;
    bool loadLDSScaleB = false;

    std::pair<int, int> workgroupMapping  = {-1, -1};
    bool                workgroupRemapXCC = false;

    rocRoller::Operations::ScaleMode scaleAMode = rocRoller::Operations::ScaleMode::None;
    rocRoller::Operations::ScaleMode scaleBMode = rocRoller::Operations::ScaleMode::None;

    rocRoller::DataType scaleTypeA = rocRoller::DataType::None;
    rocRoller::DataType scaleTypeB = rocRoller::DataType::None;

    int scaleBlockSize = -1;

    auto operator<=>(GEMMProblem const& rhs) const = default;
};
