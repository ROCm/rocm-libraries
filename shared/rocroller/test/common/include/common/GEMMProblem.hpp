// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/BlockScale_fwd.hpp>
#include <rocRoller/Parameters/Solution/LDSBankSwizzleMode.hpp>
#include <rocRoller/Parameters/Solution/LoadOption.hpp>
#include <rocRoller/Parameters/Solution/ScaleSkipPermlaneMode.hpp>
#include <rocRoller/Parameters/Solution/StoreOption.hpp>
#include <rocRoller/Parameters/Solution/StreamK.hpp>

#include <string>
#include <vector>

namespace SolutionParams = rocRoller::Parameters::Solution;

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
    unsigned int unrollK = 0;

    SolutionParams::StorePath storePath{
        SolutionParams::StorePath::VGPRToGlobalMemoryViaLDSWithBuffer};
    SolutionParams::LoadPath loadPathA{SolutionParams::LoadPath::BufferToLDSViaVGPR};
    SolutionParams::LoadPath loadPathB{SolutionParams::LoadPath::BufferToLDSViaVGPR};

    bool fuseLoops                 = true;
    bool tailLoops                 = true;
    bool allowAmbiguousMemoryNodes = false;
    bool betaInFma                 = true;
    bool literalStrides            = true;

    bool swizzleScale  = false;
    bool prefetchScale = false;
    // Swizzle tile size
    int swizzleM = 64;
    int swizzleN = 64;
    int swizzleK = 4;
    int swizzleB = 1;

    bool prefetch          = false;
    int  prefetchInFlight  = 1;
    int  prefetchLDSFactor = 0;
    bool prefetchMixMemOps = false;

    bool packMultipleElementsInto1VGPR = true;

    bool loopOverTiles = false;

    rocRoller::StreamKConfig streamK{rocRoller::StreamKMode::None};

    bool splitStoreTileIntoWaveBlocks = false;

    SolutionParams::LoadPath loadScalePathA{SolutionParams::LoadPath::BufferToVGPR};
    SolutionParams::LoadPath loadScalePathB{SolutionParams::LoadPath::BufferToVGPR};

    int  workgroupMappingDim   = -1;
    int  workgroupMappingValue = -1;
    bool workgroupRemapXCC     = false;

    uint workgroupClusterSizeX = 0;
    uint workgroupClusterSizeY = 0;
    uint workgroupClusterSizeZ = 0;

    rocRoller::Operations::ScaleMode scaleAMode = rocRoller::Operations::ScaleMode::None;
    rocRoller::Operations::ScaleMode scaleBMode = rocRoller::Operations::ScaleMode::None;

    rocRoller::DataType scaleTypeA = rocRoller::DataType::None;
    rocRoller::DataType scaleTypeB = rocRoller::DataType::None;

    int scaleBlockSize = -1;

    // LDS bank conflict swizzle
    rocRoller::LDSBankSwizzleMode ldsSwizzleMode = rocRoller::LDSBankSwizzleMode::None;

    // Scale pretile / swizzle (mirrors client TypeParameters)
    rocRoller::ScaleSkipPermlaneMode scaleSkipPermlane = rocRoller::ScaleSkipPermlaneMode::None;
    std::vector<size_t>              scalePretileA;
    std::vector<size_t>              scalePretileB;
    std::vector<size_t>              scaleShuffleTileA;
    std::vector<size_t>              scaleShuffleTileB;

    // Pre-tile A matrix (MxK tile dimensions); kernel expects pre-tiled layout (transA must be T)
    std::vector<size_t> pretileA;
    // Pre-tile B matrix (KxN tile dimensions); kernel expects pre-tiled layout
    std::vector<size_t> pretileB;

    // LDS padding for MATRIX_A and MATRIX_B; default no padding
    std::pair<int, int> padA = {0, 0};
    std::pair<int, int> padB = {0, 0};

    auto operator<=>(GEMMProblem const& rhs) const = default;
};
