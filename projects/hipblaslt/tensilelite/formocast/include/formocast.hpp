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

#include <queue>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>

namespace Tensilelite
{
    // Forward declarations
    class Formocast;

    namespace Simulator
    {
        // Load request calculation functions
        double getLoadRequest(double   MTX,
                             double   DU,
                             double   L1CacheLineSize,
                             uint32_t grvw,
                             uint32_t bpe,
                             int      dtv,
                             bool     isTransposed,
                             bool     isSwizzled,
                             uint32_t VW,
                             double   L1BusWidthPerCU,
                             int      NumLoadsCoalesced,
                             uint32_t numWaveX,
                             double&  tcc_ea0_coalesced);

        // GSU overhead calculation functions
        double getMBOverhead(double M, double N, double GlobalSplitU, double NumBatches,
                            uint32_t bpeCompute, uint32_t bpeD, double hbmBandWidth,
                            double L1CacheLineSize, double NumCUs, double boost_frequency,
                            double mem_frequency, double L2WriteArbEff, double L2ReadArbEff,
                            double L3BandWidth, double L1BusWidthPerCU, double L2BusWidthPerCU,
                            double L1WriteBusWidthPerCU, double L2WriteBusWidthPerCU);

        double getMBSKOverhead(double GlobalSplitU, double MT0, double MT1, uint32_t bpeCompute,
                              double NumCUs, uint32_t numWGs, double boost_frequency,
                              double L2ReadArbEff, double L1BusWidthPerCU, double L2BusWidthPerCU,
                              double storeGSU);

        double getLSUOverhead(double MT0, double MT1, double lsu, uint32_t svw, 
                             uint32_t numThreads, uint32_t bpeCompute, double math_frequency);

        // Cache hit rate calculation functions
        struct L1CacheHitRate {
            double tile0HitRate;
            double tile1HitRate;
        };

        struct L2CacheHitRate {
            double totalHitRate;
            double tile0HitRate;
            double tile1HitRate;
        };

        struct L3CacheHitRate {
            double totalHitRate;
            double tile0HitRate;
            double tile1HitRate;
        };

        struct HardwareConstants;  // Forward declaration

        L1CacheHitRate computeL1CacheHitRate(double L1CacheCapacity, double L1CacheLineSize,
                                             double L1BusWidthPerCU, double MT0, double MT1,
                                             uint32_t bpeA, uint32_t bpeB, int NTA, int NTB,
                                             uint32_t GRVWA, uint32_t GRVWB, bool DTVA, bool DTVB,
                                             bool isSwizzleA, bool isSwizzleB, uint32_t VWA, uint32_t VWB,
                                             bool transA, bool transB, double lda, double ldb,
                                             int NLCA, int NLCB, uint32_t threadnum,
                                             uint32_t NumWave0, uint32_t NumWave1, bool isL1FourBank);

        L3CacheHitRate computeL3CacheHitRate(double M, double N, double K, double L3CacheCapacity,
                                             double NumCUs, uint32_t bpeA, uint32_t bpeB,
                                             int NTA, int NTB, int N_WGs_total, int M_WGs_total,
                                             int N_WGs_per_tile, int M_WGs_per_tile);

        L2CacheHitRate computeL2CacheHitRate(uint32_t M, uint32_t N, uint32_t K,
                                             uint32_t MT0, uint32_t MT1, uint32_t depthU,
                                             uint32_t L2CacheCapacity, uint32_t NumCUs, uint32_t NumXCDs,
                                             uint32_t gsu, int32_t wgm, uint32_t batches,
                                             uint32_t bpeA, uint32_t bpeB, int32_t NTA, int32_t NTB,
                                             bool isGSUWGMRR);

        // Store request calculation functions
        double calculateStoreL3Request(double M, double N, double MT0, double MT1, 
                                       double& non_edge_req, double& edge_req);
        double calculateStoreL2Request(double M, double N, double MT0, double MT1, double SVW,
                                       double& non_edge_req, double& edge_req);
        double calculateStoreL1Request(double M, double N, double MT0, double MT1, double SVW,
                                       double& non_edge_req, double& edge_req);

        // FIFO and queue simulation functions
        int checkGlobalReadFIFOFull(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves, bool isStall);
        int checkLocalReadFinished(int currentCycle, std::queue<int>& fifo, int numLR);
        int checkLocalReadFIFOFull(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves, int lrStallLatencyBuffer);
        void pushLocalRead(int currentCycle, std::queue<int>& fifo, int bpr, bool isGfx950);
        int getLocalReadLatency(int baseLatency, int conflictMultiplier, double bankConflict);
        double analyzeBankConflictsFromVGPR(
            const std::vector<std::unordered_map<std::string, int64_t>>& vgprState,
            const std::string& vgprLocalReadAddrA,
            int NUM_THREADS_TO_SIMULATE,
            int NUM_BANKS,
            int BANK_WIDTH,
            int LocalReadBytesA);

    } // namespace Simulator
} // namespace Tensilelite

