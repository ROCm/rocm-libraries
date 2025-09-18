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

#include "common/GEMMProblem.hpp"

GEMMProblem setup_GEMMF8_NT()
{
    GEMMProblem gemm;

    // 4x2 jamming
    uint wavesPerWGX = 16;
    uint wavesPerWGY = 2;

    gemm.waveM = 16;
    gemm.waveN = 16;
    gemm.waveK = 32;

    gemm.macM = wavesPerWGX * gemm.waveM;
    gemm.macN = wavesPerWGY * gemm.waveN;
    gemm.macK = 2 * gemm.waveK;

    gemm.loadLDSA = true;
    gemm.loadLDSB = true;

    gemm.workgroupSizeX = 256;
    gemm.workgroupSizeY = 1;

    gemm.m = 33 * gemm.macM;
    gemm.n = 17 * gemm.macN;
    gemm.k = 4 * gemm.macK;

    gemm.alpha = 2.1;
    gemm.beta  = 0.75;

    gemm.transA = "N";
    gemm.transB = "T";

    return gemm;
}

GEMMProblem setup_GEMMF8F6F4(int waveM, int waveN, int waveK)
{
    GEMMProblem gemm;

    // 2x2 jamming
    uint wavesPerWGX = 4;
    uint wavesPerWGY = 4;

    gemm.waveM = waveM;
    gemm.waveN = waveN;
    gemm.waveK = waveK;

    gemm.macM = wavesPerWGX * gemm.waveM;
    gemm.macN = wavesPerWGY * gemm.waveN;
    gemm.macK = 2 * gemm.waveK;

    gemm.loadLDSA  = true;
    gemm.loadLDSB  = true;
    gemm.storeLDSD = false;

    gemm.workgroupSizeX = 256;
    gemm.workgroupSizeY = 1;

    gemm.m = 2 * gemm.macM;
    gemm.n = 3 * gemm.macN;
    gemm.k = 4 * gemm.macK;

    gemm.alpha = 2.1;
    gemm.beta  = 0.75;

    gemm.transA = "T";
    gemm.transB = "N";

    return gemm;
}

GEMMProblem setup_GEMMF8_TN()
{
    GEMMProblem gemm;

    // 1x1 jamming
    uint wavesPerWGX = 4;
    uint wavesPerWGY = 1;

    gemm.waveM = 16;
    gemm.waveN = 16;
    gemm.waveK = 32;

    gemm.macM = wavesPerWGX * gemm.waveM;
    gemm.macN = wavesPerWGY * gemm.waveN;
    gemm.macK = 2 * gemm.waveK;

    gemm.loadLDSA  = true;
    gemm.loadLDSB  = true;
    gemm.storeLDSD = false;

    gemm.workgroupSizeX = 256;
    gemm.workgroupSizeY = 1;

    gemm.m = 33 * gemm.macM;
    gemm.n = 17 * gemm.macN;
    gemm.k = 4 * gemm.macK;

    gemm.alpha = 2.1;
    gemm.beta  = 0.75;

    gemm.transA = "T";
    gemm.transB = "N";

    return gemm;
}
