// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocke_conv_engine
{

// Launch parameters extracted from the graph at buildPlan time.
// Only fields needed by ConvFwdPlan::execute are stored here.
struct ConvFwdParams
{
    // Tensor UIDs (keys into the device buffer map at execute time)
    int64_t xUid = 0; // input
    int64_t wUid = 0; // weights / filter
    int64_t yUid = 0; // output

    // Pre-computed tensor byte sizes (fp16 = 2 bytes per element)
    int aBytes = 0; // N * Hi * Wi * C * 2
    int bBytes = 0; // K * Y * X * C * 2
    int dBytes = 0; // N * Ho * Wo * K * 2

    // Grid dims derived at build time
    unsigned int gridM = 0; // ceil(M / tileM)  where M = N*Ho*Wo
    unsigned int gridN = 0; // ceil(N_gemm / tileN) where N_gemm = K

    // Block size = warpM * warpN * 64
    unsigned int blockSize = 256;

    // Kernel name (for hipModuleGetFunction)
    std::string kernelName;
};

} // namespace rocke_conv_engine
