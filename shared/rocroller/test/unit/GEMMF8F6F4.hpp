// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
        this->storePath       = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;
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
        this->storePath = SolutionParams::StorePath::VGPRToGlobalMemoryWithBuffer;

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
