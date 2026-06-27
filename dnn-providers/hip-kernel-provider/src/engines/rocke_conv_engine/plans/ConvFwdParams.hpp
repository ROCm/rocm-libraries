// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocke_conv_engine
{

// Tensor UIDs and problem dimensions extracted from the graph at buildPlan time.
// Enough to reconstruct launch arguments at execute time without re-reading the graph.
struct ConvFwdParams
{
    // Tensor UIDs (keys into the device buffer map at execute time)
    int64_t xUid = 0; // input
    int64_t wUid = 0; // weights / filter
    int64_t yUid = 0; // output

    // Problem shape (NHWC layout)
    int N = 0;
    int C = 0;
    int K = 0;
    int Hi = 0;
    int Wi = 0;
    int Y  = 0; // filter height
    int X  = 0; // filter width
    int Ho = 0; // derived: output height
    int Wo = 0; // derived: output width

    // Convolution parameters
    int strideH = 1;
    int strideW = 1;
    int padH    = 0;
    int padW    = 0;
    int dilH    = 1;
    int dilW    = 1;

    // Selected tile geometry (from ConvMLHeuristic)
    int tileM = 64;
    int tileN = 64;
    int tileK = 64;
    int warpM = 2;
    int warpN = 2;

    // Grid dims derived at build time
    unsigned int gridM = 0; // ceil(M / tileM)  where M = N*Ho*Wo
    unsigned int gridN = 0; // ceil(N_gemm / tileN) where N_gemm = K

    // Block size = warpM * warpN * 64
    unsigned int blockSize = 256;

    // Kernel name (for hipModuleGetFunction)
    std::string kernelName;
};

} // namespace rocke_conv_engine
