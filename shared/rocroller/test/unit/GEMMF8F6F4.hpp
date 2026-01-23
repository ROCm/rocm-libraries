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

#include "common/GEMMProblem.hpp"

struct GEMMProblemF8NT : GEMMProblem
{
    GEMMProblemF8NT()
        : GEMMProblem()
    {
        // 4x2 jamming
        const int wavesPerWGX = 16;
        const int wavesPerWGY = 2;

        this->waveM = 16;
        this->waveN = 16;
        this->waveK = 32;

        this->macM = wavesPerWGX * this->waveM;
        this->macN = wavesPerWGY * this->waveN;
        this->macK = 2 * this->waveK;

        this->loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        this->loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;

        this->workgroupSizeX = 256;
        this->workgroupSizeY = 1;

        this->m = 33 * this->macM;
        this->n = 17 * this->macN;
        this->k = 4 * this->macK;

        this->alpha = 2.1;
        this->beta  = 0.75;

        this->transA = "N";
        this->transB = "T";
    }
};

struct GEMMProblemF8F6F4 : GEMMProblem
{
    GEMMProblemF8F6F4(int waveM, int waveN, int waveK)
        : GEMMProblem()
    {
        // 2x2 jamming
        const int wavesPerWGX = 4;
        const int wavesPerWGY = 4;
        this->waveM           = waveM;
        this->waveN           = waveN;
        this->waveK           = waveK;
        this->macM            = wavesPerWGX * this->waveM;
        this->macN            = wavesPerWGY * this->waveN;
        this->macK            = 2 * this->waveK;
        this->loadPathA       = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        this->loadPathB       = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        this->storeLDSD       = false;
        this->workgroupSizeX  = 256;
        this->workgroupSizeY  = 1;
        this->m               = 2 * this->macM;
        this->n               = 3 * this->macN;
        this->k               = 4 * this->macK;
        this->alpha           = 2.1;
        this->beta            = 0.75;
        this->transA          = "T";
        this->transB          = "N";
    }
};

struct GEMMProblemF8TN : GEMMProblem
{
    GEMMProblemF8TN()
        : GEMMProblem()
    {
        // 1x1 jamming
        const int wavesPerWGX = 4;
        const int wavesPerWGY = 1;

        this->waveM = 16;
        this->waveN = 16;
        this->waveK = 32;

        this->macM = wavesPerWGX * this->waveM;
        this->macN = wavesPerWGY * this->waveN;
        this->macK = 2 * this->waveK;

        this->loadPathA = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        this->loadPathB = SolutionParams::LoadPath::BufferToLDSViaVGPR;
        this->storeLDSD = false;

        this->workgroupSizeX = 256;
        this->workgroupSizeY = 1;

        this->m = 33 * this->macM;
        this->n = 17 * this->macN;
        this->k = 4 * this->macK;

        this->alpha = 2.1;
        this->beta  = 0.75;

        this->transA = "T";
        this->transB = "N";
    }
};
