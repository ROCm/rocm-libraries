// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <origami/simulator/tensilelite/formocast.hpp>
#include <origami/math.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstring>

namespace origami
{
    namespace simulator
    {
        using math::ceiling_math;
        using math::safe_ceil_div;

        // Load request calculation functions
        /**
         * @brief Calculate L1 load request for memory access
         * @param MTX Macro tile size in the dimension
         * @param DU Depth U (K dimension tile size)
         * @param L1CacheLineSize L1 cache line size
         * @param grvw Global read vector width
         * @param bpe Bytes per element
         * @param dtv Direct to VGPR flag
         * @param isTransposed Whether the matrix is transposed
         * @param isSwizzled Whether the matrix is swizzled (used when isTransposed=true)
         * @param VW Vector width (used when isTransposed=true)
         * @param L1BusWidthPerCU L1 bus width per CU (used when isTransposed=false)
         * @param NumLoadsCoalesced Number of loads coalesced (used when isTransposed=false)
         * @param numWaveX Number of waves in X dimension (used when isTransposed=false)
         * @param tcc_ea0_coalesced Output parameter for TCC coalesced factor
         * @return L1 load request value
         */
        double getL1LoadRequest(double   MTX,
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
                             double&  tcc_ea0_coalesced)
        {
            double L1_req     = 0.0;
            tcc_ea0_coalesced = 1.0;

            if(isTransposed)
            {
                // Transposed matrix load pattern
                if(isSwizzled)
                {
                    L1_req = MTX * DU * bpe / 64;
                    L1_req *= safe_ceil_div(VW, uint32_t(2));
                }
                else
                {
                    L1_req = MTX * DU * bpe / 64;
                    tcc_ea0_coalesced = L1CacheLineSize / (DU * bpe);
                    if(grvw * bpe == 8 || grvw * bpe <= 2)
                        L1_req *= 2;
                    if(dtv)
                        L1_req *= 4 * numWaveX;
                }
            }
            else
            {
                // Non-transposed matrix load pattern (A is N or B is T)
                L1_req = std::ceil(MTX / NumLoadsCoalesced * bpe / L1BusWidthPerCU) * NumLoadsCoalesced * DU;
                if(L1CacheLineSize > MTX / NumLoadsCoalesced * bpe)
                    tcc_ea0_coalesced = 2;
                if((grvw * bpe == 8 || grvw * bpe <= 2) && (MTX / NumLoadsCoalesced * bpe >= L1BusWidthPerCU))
                    L1_req *= 2;
                if(dtv)
                    L1_req *= numWaveX;
            }

            return L1_req;
        }

        // GSU overhead calculation functions
        double getMultipleBufferOverhead(double M, double N, double GlobalSplitU, double NumBatches,
                            uint32_t bpeCompute, uint32_t bpeD, double hbmBandWidth,
                            double L1CacheLineSize, double NumCUs, uint32_t num_tiles, uint32_t CUOccupancy, double boost_frequency,
                            double mem_frequency, double L2WriteArbEff, double L2ReadArbEff,
                            double L3BandWidth, double L1BusWidthPerCU, double L2BusWidthPerCU,
                            double L1WriteBusWidthPerCU, double L2WriteBusWidthPerCU)
        {
            // MB (MultiBuffer) GSU overhead calculation
            double read_l1_req, write_l1_req;
            double scale_occupancy = (num_tiles == 1 && CUOccupancy > 1) ? 0.7 * CUOccupancy : 1;

            auto bpeIn  = bpeCompute;
            auto bpeOut = bpeD;

            // Read write requests for VW=4 dwordx2 Half output
            if (bpeIn == 4 && bpeOut == 2 && ((int)M % 4) == 0)
            {
                read_l1_req = M * N * (bpeIn * 4 + (GlobalSplitU - 2) * 4 + 2) / 64;
                write_l1_req = M * N * bpeOut / 64 * 8;
            }
            else
            {
                // Just to make it work
                read_l1_req = M * N * (bpeIn * 4 + (GlobalSplitU - 2) * 4 + 2) / 64;
                write_l1_req = M * N * bpeOut / 64 * 8;
            }

            double read_l2_req = M * N * bpeIn * GlobalSplitU / L1CacheLineSize;
            double read_l3_req = read_l2_req;
            double write_l2_req = M * N * bpeOut / 64;
            double write_l3_req = write_l2_req;

            double WGs = ceiling_math(M * N, 1024.0);
            WGs = std::min(NumCUs, WGs);

            double cu_freq  = boost_frequency;
            double hbm_freq = mem_frequency;

            // Not sure if this works using the same formula as gemm
            double L2WriteBandWidthPerCU_local = L2WriteArbEff * 128 * 16 / WGs; //58% eff
            double L2BandWidthPerCU_local      = L2ReadArbEff * 128 * 16 / WGs; //90% eff
            double L3BandWidthPerCU_local      = L3BandWidth / WGs;
            double HBMBandWidthPerCU_local     = hbmBandWidth / WGs;

            double GSU_L1_clk  = read_l1_req/WGs * 64 / L1BusWidthPerCU * scale_occupancy;
            double GSU_L2_clk  = read_l2_req/WGs / 2 * 128 / std::min(L2BandWidthPerCU_local, L2BusWidthPerCU) * scale_occupancy;
            double GSU_L3_clk  = read_l3_req/WGs / 2 * 64 / L3BandWidthPerCU_local * scale_occupancy;
            double GSU_hbm_clk = M * N * bpeIn * GlobalSplitU / hbmBandWidth * scale_occupancy;

            double GSU_L1_overall  = GSU_L1_clk / cu_freq;
            double GSU_L2_overall  = GSU_L2_clk / cu_freq;
            double GSU_L3_overall  = GSU_L3_clk / hbm_freq;
            double GSU_hbm_overall = GSU_hbm_clk / hbm_freq;
            double GSU_mem_overall = std::max(GSU_hbm_overall, std::max(GSU_L3_overall, std::max(GSU_L1_overall, GSU_L2_overall)));

            double D_L1_clk = write_l1_req/WGs * 64 / L1WriteBusWidthPerCU;
            double D_L2_clk = write_l2_req/WGs * 64 / std::min(L2WriteBusWidthPerCU, L2WriteBandWidthPerCU_local);
            double D_L3_clk = write_l3_req/WGs * 64 / L3BandWidthPerCU_local;
            double D_hbm_clk = 0 * 64 / HBMBandWidthPerCU_local;
            double store = std::max(D_L1_clk/cu_freq, std::max(D_L2_clk/cu_freq, std::max(D_L3_clk/hbm_freq, D_hbm_clk/hbm_freq)));

            return GSU_mem_overall + store;
        }

        double getMultipleBufferSingleKernelOverhead(double GlobalSplitU, double MT0, double MT1, uint32_t bpeCompute,
                              double NumCUs, uint32_t WGs_per_gsu_XCD, uint32_t num_tiles, uint32_t CUOccupancy, double boost_frequency,
                              double L2ReadArbEff, double L1BusWidthPerCU, double L2BusWidthPerCU,
                              double storeGSU)
        {
            // MBSK (MultiBufferSingleKernel) GSU overhead calculation
            // FIXME: Modify with the MBSK changes.
            // FIXME: add sync overhead.
            double cu_freq  = boost_frequency;
            auto   bpeIn    = bpeCompute;
            double L2BandWidthPerCU_local = L2ReadArbEff * 128 * 16 / WGs_per_gsu_XCD; //90% eff
            double scale_occupancy = (num_tiles == 1 && CUOccupancy > 1) ? 0.7 * CUOccupancy : 1;
            double atomic_overhead = GlobalSplitU * 0.1;

            double GSU_L1_req      = ((GlobalSplitU - 1) * MT0 * MT1 * bpeIn) / 64;
            if (GlobalSplitU > 2)
            {
                GSU_L1_req += (MT0 * MT1 * bpeIn) / 64;
            }
            double GSU_L1_clk      = GSU_L1_req * 64 / L1BusWidthPerCU * scale_occupancy;
            double GSU_L2_req = MT0 * MT1 * bpeIn * GlobalSplitU / 128;
            double GSU_L2_clk = GSU_L2_req * 128 / std::min(L2BandWidthPerCU_local, L2BusWidthPerCU) * scale_occupancy;

            double cost_overhead = 2*1024.0/1900/(GlobalSplitU-1);

            return storeGSU + atomic_overhead + ((std::max(GSU_L1_clk/cu_freq, GSU_L2_clk/cu_freq) + cost_overhead));
        }

        double getLocalSplitKOverhead(double MT0, double MT1, double lsu, uint32_t svw,
                             uint32_t numThreads, uint32_t bpeCompute, double math_frequency)
        {
            if (lsu == 1) return 0.0;

            double lsu_overall = 0.0;
            auto   bpeIn       = bpeCompute;

            // local write
            double lw_cycle    = 8;
            double local_write = MT0 * MT1 / numThreads; // total elements to store per thread.
            double local_write_cycle;
            switch(svw * bpeIn)
            {
            case 32:
                lw_cycle  = 20;
                local_write_cycle = local_write * lw_cycle * 2;
                break;
            case 16:
                lw_cycle  = 20;
                local_write_cycle = local_write * lw_cycle;
                break;
            case 8:
                lw_cycle  = 12;
                local_write_cycle = local_write * lw_cycle;
                break;
            case 4:
                lw_cycle  = 8;
                local_write_cycle = local_write * lw_cycle;
                break;
            default:
                lw_cycle  = 8;
                local_write_cycle = local_write * lw_cycle;
                break;
            }
            local_write_cycle *= 2;

            // local read
            double local_read_cycle = MT0 * MT1 / numThreads / svw * 4;
            local_read_cycle *= 2;

            // reduction
            double reduction_cycle = MT0 * MT1 / numThreads * (lsu - 1) * 4;

            lsu_overall = (local_write_cycle + local_read_cycle + reduction_cycle) / math_frequency;

            return lsu_overall;
        }

        // Cache hit rate calculation functions
        L1CacheHitRate computeL1CacheHitRate(double L1CacheCapacity, double L1CacheLineSize,
                                             double L1BusWidthPerCU, double MT0, double MT1, uint32_t depthU,
                                             uint32_t bpeA, uint32_t bpeB, int NTA, int NTB,
                                             uint32_t GRVWA, uint32_t GRVWB, bool DTVA, bool DTVB,
                                             bool isSwizzleA, bool isSwizzleB, uint32_t VWA, uint32_t VWB,
                                             bool transA, bool transB, double lda, double ldb,
                                             int NLCA, int NLCB, uint32_t threadnum,
                                             uint32_t NumWave0, uint32_t NumWave1, bool isL1FourBank)
        {
            L1CacheHitRate hr;
            double A_L1_hit = 1.0;
            double B_L1_hit = 1.0;
            // Calculate L1 hit rate, assume bpeA==bpeB, TN only
            bool isL1BypassA = (NTA >= 2);
            bool isL1BypassB = (NTB >= 2);

            if(transA)
            {
                //A is T
                if(depthU / NLCA * bpeA < L1CacheLineSize)
                {
                    bool isL1CacheFull = (L1CacheLineSize * threadnum / (depthU / NLCA / GRVWA)) > L1CacheCapacity;
                    if(isL1CacheFull)
                    {
                        A_L1_hit = 1;
                    }
                    else
                    {
                        A_L1_hit = std::ceil(NLCA / (L1CacheLineSize / (depthU / NLCA * bpeA))) / NLCA;
                    }
                } else if((uint32_t)lda % (uint32_t)L1CacheLineSize == 0) {
                    A_L1_hit = 0.5;
                }
                if(GRVWA * bpeA == 8 || GRVWA * bpeA <= 2)
                    A_L1_hit /= 2;
                if(DTVA)
                    A_L1_hit /= 4 * ((MT0 * ceiling_math(depthU * bpeA, L1CacheLineSize) >= L1CacheCapacity) ? 1 : NumWave1);

                A_L1_hit = isL1BypassA ? 0 : 1 - A_L1_hit;
                A_L1_hit = isSwizzleA ? 1 - 1 / safe_ceil_div(VWA, uint32_t(2)) : A_L1_hit;
            }
            else
            {
                //A is N
                uint32_t L1Limit = L1CacheCapacity;
                if(isL1FourBank) {
                    if((uint32_t)lda % 512 == 0)
                        L1Limit /= 4;
                    else if((uint32_t)lda % 256 == 0)
                        L1Limit /= 2;
                }

                if(MT0 / NLCA * bpeA < L1CacheLineSize)
                {
                    bool isL1CacheFull = (L1CacheLineSize * threadnum / (MT0 / NLCA / GRVWA)) > L1Limit;
                    if(isL1CacheFull)
                    {
                        A_L1_hit = 1;
                    }
                    else
                    {
                        A_L1_hit = std::ceil(NLCA / (L1CacheLineSize / (MT0 / NLCA * bpeA))) / NLCA;
                    }
                }
                else
                {
                    A_L1_hit = 0.5;
                }
                if((GRVWA * bpeA == 8 || GRVWA * bpeA <= 2) && (MT0 / NLCA * bpeA) >= L1BusWidthPerCU)
                    A_L1_hit /= 2;
                if(DTVA)
                    A_L1_hit /= NumWave1;
                A_L1_hit = isL1BypassA ? 0: 1 - A_L1_hit;
            }

            if(transB)
            {
                //B is T
                uint32_t L1Limit = L1CacheCapacity;
                if(isL1FourBank) {
                    if((uint32_t)ldb % 512 == 0)
                        L1Limit /= 4;
                    else if((uint32_t)ldb % 256 == 0)
                        L1Limit /= 2;
                }

                if(MT1 / NLCB * bpeB < L1CacheLineSize)
                {
                    bool isL1CacheFull = (L1CacheLineSize * threadnum / (MT1 / NLCB / GRVWB)) > L1Limit;
                    if(isL1CacheFull)
                    {
                        B_L1_hit = 1;
                    }
                    else
                    {
                        B_L1_hit = std::ceil(NLCB / (L1CacheLineSize / (MT1 / NLCB * bpeB))) / NLCB;
                    }
                }
                else
                {
                    B_L1_hit = 0.5;
                }
                if((GRVWB * bpeB == 8 || GRVWB * bpeB <= 2) && (MT1 / NLCB * bpeB) >= L1BusWidthPerCU)
                    B_L1_hit /= 2;
                if(DTVB)
                    B_L1_hit /= NumWave0;
                B_L1_hit = isL1BypassB ? 0: 1 - B_L1_hit;
            }
            else
            {
                //B is N
                if(depthU / NLCB * bpeB < L1CacheLineSize)
                {
                    bool isL1CacheFull = (L1CacheLineSize * threadnum / (depthU / NLCB / GRVWB)) > L1CacheCapacity;
                    if(isL1CacheFull)
                    {
                        B_L1_hit = 1;
                    }
                    else
                    {
                        B_L1_hit = std::ceil(NLCB / (L1CacheLineSize / (depthU / NLCB * bpeB))) / NLCB;
                    }
                } else if((uint32_t)ldb % (uint32_t)L1CacheLineSize == 0) {
                    B_L1_hit = 0.5;
                }
                if(GRVWB * bpeB == 8 || GRVWB * bpeB <= 2)
                    B_L1_hit /= 2;
                if(DTVB)
                    B_L1_hit /= 4 * ((MT1 * ceiling_math(depthU * bpeB, L1CacheLineSize) >= L1CacheCapacity) ? 1 : NumWave0);
                B_L1_hit = isL1BypassB ? 0 : 1 - B_L1_hit;
                B_L1_hit = isSwizzleB ? 1 - 1 / safe_ceil_div(VWB, uint32_t(2)) : B_L1_hit;
            }

            hr.tile0HitRate = A_L1_hit;
            hr.tile1HitRate = B_L1_hit;
            return hr;
        }

        L3CacheHitRate computeL3CacheHitRate(double M, double N, double K, double L3CacheCapacity,
                                             double NumCUs, uint32_t bpeA, uint32_t bpeB,
                                             int NTA, int NTB, int N_WGs_total, int M_WGs_total,
                                             int N_WGs_per_tile, int M_WGs_per_tile)
        {
            L3CacheHitRate hr;
            double A_L3_hit = 0.0;
            double B_L3_hit = 0.0;

            bool isL3BypassA = (NTA > 3) || (NTA == 1);
            if(!isL3BypassA)
            {
                if((M * K * bpeA) + (N * K * bpeB) < L3CacheCapacity)
                {
                    A_L3_hit = 1 - double(1.0 / N_WGs_total);
                }
                else
                {
                    A_L3_hit = 1 - double(M_WGs_per_tile / NumCUs);
                }
            }
            bool isL3BypassB = (NTB > 3) || (NTB == 1);
            if(!isL3BypassB)
            {
                if((M * K * bpeA) + (N * K * bpeB) < L3CacheCapacity)
                {
                    B_L3_hit = 1 - double(1.0 / M_WGs_total);
                }
                else
                {
                    B_L3_hit = 1 - double(N_WGs_per_tile / NumCUs);
                }
            }
            hr.tile0HitRate = A_L3_hit;
            hr.tile1HitRate = B_L3_hit;
            hr.totalHitRate = A_L3_hit * M/(M+N) + B_L3_hit * N/(M+N);
            return hr;
        }

        L2CacheHitRate computeL2CacheHitRate(uint32_t M, uint32_t N, uint32_t K,
                                             uint32_t MT0, uint32_t MT1, uint32_t depthU,
                                             uint32_t L2CacheCapacity, uint32_t NumCUs, uint32_t NumXCDs,
                                             int XCC, int XCCG, uint32_t gsu, int32_t wgm,
                                             uint32_t batches, uint32_t bpeA, uint32_t bpeB, int32_t NTA,
                                             int32_t NTB, bool isGSUWGMRR)
        {
            L2CacheHitRate hitRate{0.0, 0.0, 0.0};

            bool isL2BypassA = (NTA & 0x6) > 0;
            bool isL2BypassB = (NTB & 0x6) > 0;

            // isL2BypassA = true; // calc B only
            // isL2BypassB = true; // calc A only

            if(isL2BypassA && isL2BypassB)
                return hitRate;

            uint32_t wg0 = safe_ceil_div(M, MT0);
            uint32_t wg1 = safe_ceil_div(N, MT1);

            uint32_t MT0_Edge = MT0 - ((wg0 * MT0) - M);
            uint32_t MT1_Edge = MT1 - ((wg1 * MT1) - N);
            if(MT0_Edge == 0)
                MT0_Edge = MT0;
            if(MT1_Edge == 0)
                MT1_Edge = MT1;

            // other info
            uint32_t L2CacheLineSize = 128; //Bytes
            uint32_t L2Capacity      = L2CacheCapacity; // Bytes
            uint32_t gsuMulBatch     = gsu * batches;

            std::vector<uint32_t> arrA(gsuMulBatch * wg0, 0);
            std::vector<uint32_t> arrB(gsuMulBatch * wg1, 0);
            std::vector<uint32_t> arrA_2(gsuMulBatch * wg0, 0);
            std::vector<uint32_t> arrB_2(gsuMulBatch * wg1, 0);

            int WGMXCC  = (XCC > 0) ? XCC : 1;
            int WGMXCCG = (XCCG > 0)? XCCG : NumCUs;
            if(WGMXCC > 0)
                assert((WGMXCCG % WGMXCC) == 0);

            uint32_t totalWGNum  = gsuMulBatch * wg0 * wg1;

            uint64_t aHitElements  = 0;
            uint64_t aMissElements = 0;
            uint64_t bHitElements  = 0;
            uint64_t bMissElements = 0;

            // // put wg into all xcds (xcc-mapped wg)
            std::vector<std::vector<uint32_t>> xcd_wgs(NumXCDs);
            std::vector<std::unordered_set<uint32_t>> xcd_cachedWG_A(NumXCDs);
            std::vector<std::unordered_set<uint32_t>> xcd_cachedWG_B(NumXCDs);
            // TODO- This should be a better way, but currently the prediction results are not better.
            //   Remain this part here, and do more tests as TODO
            // std::vector<uint32_t> xcd_cachedSizes(NumXCDs, 0); // Use this when cache-size-estimation is better
            uint32_t xccIdx = 0;

            for(uint32_t wg = 0; wg < std::min(totalWGNum, 10 * NumCUs); wg++)
            {
                uint32_t xccMappedWGId;

                // ----------------------------------------------------------
                //  XCD0     XCD1   XCD2     ...    XCDN-1  (NumXCD = N)
                //  _____   _____   _____   _____   _____
                // | wg0 | | wgX | |wg2X | |     | |     |  When we enable XCC and XCCG,
                // | wg1 | |     | |     | |     | |     |  this is the result of "XCC/XCCG-remapped" wgIds
                // |     | |     | |     | |     | |     |  (XCCG % XCC) must == 0,
                // |     | |     | |     | |     | |     |  and each XCD has XCCG/XCC (=X) wgs.
                // |wgX-1| |     | |     | |     | |     |
                // |_____|_|_____|_|_____|_|_____|_|_____|  <---- Up to here, there are #-XCCG wgs. Note that X*N = XCCG
                // | wgY | |     | |     | |     | |     |  <---- Note that wgY = wg(XCCG-1)+1
                // |     | |     | |     | |     | |     |
                //
                // ----------------------------------------------------------

                // Sorted-out Formula
                if(WGMXCCG == 0)
                {
                    xccMappedWGId = (wg / WGMXCC) + (wg % WGMXCC) * (totalWGNum / WGMXCC) + std::min((totalWGNum % WGMXCC), (wg % WGMXCC));
                }
                else
                {
                    xccMappedWGId = (wg / WGMXCCG) * WGMXCCG + ((wg % WGMXCCG) / WGMXCC);
                    if(wg > (totalWGNum / WGMXCCG) * WGMXCCG)
                        xccMappedWGId += (wg % WGMXCC) * ((totalWGNum % WGMXCCG) / WGMXCC) + std::min((totalWGNum % WGMXCC), (wg % WGMXCC));
                    else
                        xccMappedWGId += (wg % WGMXCC) * (WGMXCCG / WGMXCC);
                }
                xcd_wgs[xccIdx++].emplace_back(xccMappedWGId);
                xccIdx %= NumXCDs;
            }

            size_t wgPerXCDIter = NumCUs / NumXCDs;

            // std::vector<uint32_t> curXCDWGs;
            for(uint32_t xccIdx = 0; xccIdx < NumXCDs; xccIdx++)
            {
                auto& curXCDWGVec = xcd_wgs[xccIdx];

                // WorkGroupMapping and Calculate L2 Hit
                for(size_t counter = 0; counter < curXCDWGVec.size(); counter++)
                // for(uint32_t xccMappedWGId : curXCDWGs)
                {
                    // xccMappedWGId = idxWG012 = 1D serial ID
                    uint32_t xccMappedWGId      = curXCDWGVec[counter];
                    int32_t  sgprWGM            = wgm;
                    uint32_t sgprNumWorkGroups0 = wg0;
                    uint32_t sgprNumWorkGroups1 = wg1;
                    uint32_t wg2     = xccMappedWGId / (sgprNumWorkGroups0 * sgprNumWorkGroups1 * gsu); //batch
                    uint32_t idxWG01 = xccMappedWGId - (wg2 * sgprNumWorkGroups0 * sgprNumWorkGroups1 * gsu);
                    uint32_t sgprWorkGroup1 = idxWG01 / wg0;
                    uint32_t sgprWorkGroup0 = idxWG01 - (sgprWorkGroup1 * wg0);

                    //go GSUWGMRR
                    uint32_t gsuSumIdx = 0;
                    if(isGSUWGMRR)
                    {
                        gsuSumIdx      = sgprWorkGroup1 / sgprNumWorkGroups1;
                        sgprWorkGroup1 = sgprWorkGroup1 % sgprNumWorkGroups1;
                    }
                    else
                    {
                        gsuSumIdx      = sgprWorkGroup1 % gsu;
                        sgprWorkGroup1 = sgprWorkGroup1 / gsu;
                    }
                    // remapped wgid[0,1]:
                    //  wgm > 1: WGMPositive
                    //  wgm < 0: WGMNegtive
                    //  wgm == 1 or 0: no remapping (default)
                    uint32_t finalwg0 = sgprWorkGroup0;
                    uint32_t finalwg1 = sgprWorkGroup1;
                    if(wgm > 1)
                    {
                        uint32_t v6  = sgprWorkGroup1 / sgprWGM;
                        uint32_t s84 = v6 * sgprWGM;
                        s84          = sgprWorkGroup1 - s84;
                        s84 *= sgprNumWorkGroups0;
                        s84 += sgprWorkGroup0;
                        uint32_t s81 = v6;
                        v6           = sgprNumWorkGroups1 / sgprWGM;
                        uint32_t s82 = v6;
                        uint32_t s83 = sgprWGM * s82;
                        s83          = sgprNumWorkGroups1 - s83;
                        if(s83 == 0)
                            s83 = sgprWGM;
                        if(s81 >= s82)
                            s82 = s83;
                        else
                            s82 = sgprWGM;
                        v6             = s84 / s82;
                        uint32_t v7    = v6 * s82;
                        v7             = s84 - v7;
                        sgprWorkGroup0 = v6;
                        sgprWorkGroup1 = v7;
                        sgprWorkGroup1 = sgprWorkGroup0 * s82;
                        sgprWorkGroup1 = s84 - sgprWorkGroup1;
                        s81 *= sgprWGM;
                        sgprWorkGroup1 += s81;
                        finalwg1 = sgprWorkGroup1;
                        finalwg0 = sgprWorkGroup0;
                    }
                    else if(wgm < 0)
                    {
                        sgprWGM = 0 - sgprWGM;
                        uint32_t v12 = sgprWorkGroup0 / sgprWGM;
                        uint32_t s85 = v12;
                        uint32_t s88 = s85 * sgprWGM;
                        s88          = sgprWorkGroup0 - s88;
                        s88 *= sgprNumWorkGroups1;
                        s88 += sgprWorkGroup1;
                        v12          = sgprNumWorkGroups0 / sgprWGM;
                        uint32_t s86 = v12;
                        uint32_t s87 = sgprWGM * s86;
                        s87          = sgprNumWorkGroups0 - s87;
                        if(s87 == 0)
                            s87 = sgprWGM;
                        if(s85 >= s86)
                            s86 = s87;
                        else
                            s86 = sgprWGM;
                        v12          = s88 / s86;
                        uint32_t v13 = v12 * s86;
                        v13          = s88 - v13;
                        sgprWorkGroup1 = v12;
                        sgprWorkGroup0 = v13;
                        sgprWorkGroup0 = sgprWorkGroup1 * s86;
                        sgprWorkGroup0 = s88 - sgprWorkGroup0;
                        s85 *= sgprWGM;
                        sgprWorkGroup0 += s85;
                        finalwg0 = sgprWorkGroup0;
                        finalwg1 = sgprWorkGroup1;
                    }
                    uint32_t MT_Size0 = (finalwg0 == (wg0 - 1)) ? MT0_Edge : MT0; // Edge?
                    uint32_t MT_Size1 = (finalwg1 == (wg1 - 1)) ? MT1_Edge : MT1; // Edge?
                    uint32_t idxA = (wg2 * gsu + gsuSumIdx) * wg0 + finalwg0;
                    uint32_t idxB = (wg2 * gsu + gsuSumIdx) * wg1 + finalwg1;


                    // clean cache for each WGMXCCG
                    if((counter % wgPerXCDIter) == 0)
                    {
                        // xcd_cachedSizes[xccIdx] = 0;
                        xcd_cachedWG_A[xccIdx].clear();
                        xcd_cachedWG_B[xccIdx].clear();
                    }

                    // A
                    if(!isL2BypassA)
                    {
                        if(xcd_cachedWG_A[xccIdx].count(idxA) != 0)
                        {
                            aHitElements += (MT_Size0 * (K / gsu));
                            xcd_cachedWG_A[xccIdx].insert(idxA);
                        }
                        else
                        {
                            aMissElements += (MT_Size0 * (K / gsu));
                            xcd_cachedWG_A[xccIdx].insert(idxA);
                            // xcd_cachedSizes[xccIdx] += (MT_Size0 * (K / gsu)) * bpeA;
                        }
                    }
                    // B
                    if(!isL2BypassB)
                    {
                        if(xcd_cachedWG_B[xccIdx].count(idxB) != 0)
                        {
                            bHitElements += (MT_Size1 * (K / gsu));
                            xcd_cachedWG_B[xccIdx].insert(idxB);
                        }
                        else
                        {
                            bMissElements += (MT_Size1 * (K / gsu));
                            xcd_cachedWG_B[xccIdx].insert(idxB);
                            // xcd_cachedSizes[xccIdx] += (MT_Size1 * (K / gsu)) * bpeB;
                        }
                    }
                    // // TODO: clean cache when full
                    // //   This should be a better way, but currently the prediction results are not better.
                    // //   Remain this part here as TODO
                    // if(xcd_cachedSizes[xccIdx] > L2CacheCapacity)
                    // {
                    //     // std::cout << "clean cache when: xccId = " << xccIdx << ", with remapped-wgID = " << xccMappedWGId << std::endl;
                    //     xcd_cachedSizes[xccIdx] = 0;
                    //     xcd_cachedWG_A[xccIdx].clear();
                    //     xcd_cachedWG_B[xccIdx].clear();
                    // }
                }
            }

            double hitRateA     = 0.0;
            double hitRateB     = 0.0;
            double totalHitRate = 0.0;

            if(aHitElements > 0)
                hitRateA = double(aHitElements) / double(aHitElements + aMissElements);
            if(bHitElements > 0)
                hitRateB = double(bHitElements) / double(bHitElements + bMissElements);
            if(aHitElements + bHitElements > 0)
                totalHitRate = double(aHitElements + bHitElements)
                               / double(aHitElements + aMissElements + bHitElements + bMissElements);

            hitRate.totalHitRate = totalHitRate;
            hitRate.tile0HitRate = hitRateA;
            hitRate.tile1HitRate = hitRateB;

            return hitRate;
        }

        // Store request calculation functions
        double calculateStoreL3Request(double M, double N, double MT0, double MT1,
                                       double& non_edge_req, double& edge_req)
        {
            double result = 0.0;

            double edge_size     = std::fmod(M, MT0);
            double numWGsNonEdge = std::floor(M / MT0);
            result = N * ((std::floor(M / 32)) + ceiling_math(edge_size / 32));

            double maxMT1              = std::min(N, MT1);
            double nonEdgeRequestPerMT = maxMT1 * std::floor(M / 32);
            double edgeRequestPerMT    = maxMT1 * ceiling_math(edge_size / 32);
            if(numWGsNonEdge > 0.0)
                non_edge_req = nonEdgeRequestPerMT;
            else
                non_edge_req = 0;
            edge_req = edgeRequestPerMT;

            return result;
        }

        double calculateStoreL2Request(double M, double N, double MT0, double MT1, double SVW,
                                       double& non_edge_req, double& edge_req)
        {
            double result = 0.0;

            double edge_size     = std::fmod(M, MT0);
            double numWGsNonEdge = std::floor(M / MT0);
            double M_MOD_16SVW   = std::fmod(M, 16 * SVW);
            double M_MOD_32      = std::fmod(M, 32);
            double M_MOD_8       = std::fmod(M, 8);
            double M_MOD_4       = std::fmod(M, 4);

            double non_edge_0 = MT0 * 2 / 64 * ceiling_math(64 / (16 * 2 * SVW)) *
              ((M_MOD_32 == 0 || M_MOD_32 == 16) ?
               (M_MOD_32 == 16 ? (SVW == 1 ? 1 : SVW == 2 ? (3.0/2) : SVW == 4 ? (5.0/4) : (9.0/8)) : 1) :
               (SVW == 1 ? (3.0/2) : SVW == 2 ? (4.0/2) : SVW == 4 ? (3.0/2) : (5.0/4)));
            double edge_0     = std::floor(edge_size / (16 * SVW)) * (16 * SVW) * 2 / 64
                                 * ceiling_math(64 / (16 * 2 * SVW)) * SVW;
            double edge_1     = std::min(M_MOD_16SVW, SVW) * ceiling_math(M_MOD_16SVW / 32);
            double edge_2     = SVW == 1 ? 0 : (M_MOD_8 == 0 ? 0 : (std::fmod(edge_size, 32) - SVW) / (32 / SVW));
            double edge_3     = SVW == 1 ? 0 : (M_MOD_8 == 0 ? 0 : std::floor(edge_size / (16 * SVW)) * (32 - SVW) / (32 / SVW));
            double edge_4     = SVW == 1 ? (M_MOD_4 == 0 ? 0 : ((std::fmod(edge_size, 32) - SVW) / (32 / SVW))) : 0;
            double edge_5     = SVW == 1 ? (M_MOD_4 == 0 ? 0 : (std::floor(edge_size / 32) * (32 - SVW) / (32 / SVW))) : 0;

            result = N * ((numWGsNonEdge * non_edge_0) + (edge_0) + (edge_1) + (edge_2) + (edge_3) + (edge_4) + (edge_5));

            double maxMT1              = std::min(N, MT1);
            double nonEdgeRequestPerMT = maxMT1 * (non_edge_0);
            double edgeRequestPerMT    = maxMT1 * (edge_0 + edge_1 + edge_2 + edge_3 + edge_4 + edge_5);
            if(numWGsNonEdge > 0.0)
                non_edge_req = nonEdgeRequestPerMT;
            else
                non_edge_req = 0;
            edge_req = edgeRequestPerMT;

            return result;
        }

        double calculateStoreL1Request(double M, double N, double MT0, double MT1, double SVW,
                                       double& non_edge_req, double& edge_req)
        {
            double result = 0.0;

            double edge_size     = std::fmod(M, MT0);
            double numWGsNonEdge = std::floor(M / MT0);

            double non_edge_0
                = MT0 / 16 * (-1) * (SVW == 1 ? 1 : 0) * (std::fmod(M, 4) == 2 ? 1 : 0);
            double non_edge_1
                = MT0 / 16 * (-4) * (SVW == 1 ? 1 : 0) * (std::fmod(M, 16) == 8 ? 1 : 0);
            double non_edge_2
                = MT0 / 16 * (-3) * (SVW == 1 ? 1 : 0) * (std::fmod(M, 4) == 0 ? 1 : 0);
            double non_edge_3
                = MT0 / 16 * (-12) * (SVW == 1 ? 1 : 0) * (std::fmod(M, 16) == 0 ? 1 : 0);
            double edge_0 = (std::floor(edge_size / (16 * SVW)) * 3 * (SVW == 1 ? 1 : 0)
                             * (std::fmod(M, 2) == 1 ? 1 : 0));
            double edge_1 = (std::floor(edge_size / (16 * SVW)) * 2 * (SVW == 1 ? 1 : 0)
                             * (std::fmod(M, 4) == 2 ? 1 : 0));
            double edge_2 = (std::floor(edge_size / (16 * SVW)) * (-4) * (SVW == 1 ? 1 : 0)
                             * (std::fmod(M, 8) == 0 ? 1 : 0));
            double edge_3 = (std::floor(edge_size / (16 * SVW)) * 4 * (SVW == 1 ? 1 : 0)
                             * (std::fmod(M, 16) == 0 ? 1 : 0));
            double edge_4 = (std::floor(edge_size / (16 * SVW)) * (-12 * SVW * SVW)
                             * (SVW == 1 ? 1 : 0) * (std::fmod(M, 16) == 0 ? 1 : 0));

            result += N / 64
                      * ((numWGsNonEdge * non_edge_0) + (numWGsNonEdge * non_edge_1)
                         + (numWGsNonEdge * non_edge_2) + (numWGsNonEdge * non_edge_3) + (edge_0)
                         + (edge_1) + (edge_2) + (edge_3) + (edge_4));

            double non_edge_4 = MT0 / 32 * (SVW == 2 || SVW == 8 ? 139 : (SVW == 4 ? 82 : 0))
                                * (SVW == 1 ? 0 : 1) * (std::fmod(M, 2) == 1 ? 1 : 0);
            double non_edge_5 = MT0 / 16 * (SVW == 2 || SVW == 8 ? 3 : (SVW == 4 ? 2 : 0))
                                * (SVW == 1 ? 0 : 1) * (std::fmod(M, 4) == 2 ? 1 : 0);
            double non_edge_6 = MT0 / 16 * (SVW == 2 || SVW == 8 ? 2 : (SVW == 4 ? 0 : 0))
                                * (SVW == 1 ? 0 : 1) * (std::fmod(M, 8) == 4 ? 1 : 0);
            double non_edge_7 = MT0 / 16 * (SVW == 2 || SVW == 8 ? 4 : (SVW == 4 ? 0 : 0))
                                * (SVW == 1 ? 0 : 1) * (std::fmod(M, 16) == 8 ? 1 : 0);
            double non_edge_8
                = MT0 / 16 * (-4) * (SVW == 1 ? 0 : 1) * (std::fmod(M, 8) == 0 ? 1 : 0);
            double non_edge_9 = MT0 / 16 * (SVW == 4 || SVW == 8 ? 0 : (SVW == 2 ? -8 : 0))
                                * (SVW == 1 ? 0 : 1) * (std::fmod(M, 16 * SVW) == 0 ? 1 : 0);
            double non_edge_10 = MT0 / 16 * (SVW == 4 || SVW == 8 ? -8 : (SVW == 2 ? 0 : 0))
                                 * (SVW == 1 ? 0 : 1) * (std::fmod(M, 4 * SVW) == 0 ? 1 : 0);
            double edge_5 = (std::floor(edge_size / (16 * SVW)) * (-16 * SVW * SVW)
                             * (SVW == 1 ? 0 : 1) * (std::fmod(M, 16 * SVW) == 8 * SVW ? 1 : 0));
            double edge_6 = (std::floor(edge_size / (16 * SVW)) * (-48 * SVW * SVW)
                             * (SVW == 1 ? 0 : 1) * (std::fmod(M, 16 * SVW) == 0 ? 1 : 0));

            result += N / 64
                      * ((numWGsNonEdge * non_edge_4) + (numWGsNonEdge * non_edge_5)
                         + (numWGsNonEdge * non_edge_6) + (numWGsNonEdge * non_edge_7)
                         + (numWGsNonEdge * non_edge_8) + (numWGsNonEdge * non_edge_9)
                         + (numWGsNonEdge * non_edge_10) + (edge_5) + (edge_6));

            double M_MOD_16SVW = std::fmod(M, 16 * SVW);
            double M_MOD_8VW   = std::fmod(M, 8 * SVW);
            double M_MOD_4SVW  = std::fmod(M, 4 * SVW);
            double M_MOD_4     = std::fmod(M, 4);

            double non_edge_11 = MT0 / 16 * (16 - (SVW == 1 ? 1 : 4));
            double edge_7
                = (std::floor(edge_size / (16 * SVW)) * (12 * SVW * SVW) * (SVW == 1 ? 1 : 4));
            double edge_8 = ((M_MOD_16SVW >= 4 * SVW
                                  ? (M_MOD_16SVW - 4 * SVW) * SVW
                                        * (SVW == 1 && M_MOD_4 == 0 ? 1 : 4) * (M_MOD_8VW == 0 ? 0 : 1)
                                  : 0));

            result += N / 64 * ((numWGsNonEdge * non_edge_11) + (edge_7) + (edge_8));

            double non_edge_12 = ((MT0 * 2) / 64) * (SVW == 1 || SVW == 4 ? 2 : 1);
            double edge_9      = (std::floor(edge_size / (16 * SVW)) * (SVW == 1 ? 1 : 4 * SVW));
            double edge_10     = (M_MOD_16SVW < 4 * SVW ? M_MOD_4SVW : 0);
            double edge_11 = (M_MOD_16SVW >= 4 * SVW ? (SVW == 1 && M_MOD_4 == 0 ? 1 : 4 * SVW) : 0);

            result += N * ((numWGsNonEdge * non_edge_12) + (edge_9) + (edge_10) + (edge_11));

            double maxMT1              = std::min(N, MT1);
            double nonEdgeRequestPerMT = maxMT1 / 64
                                             * (non_edge_0 + non_edge_1 + non_edge_2 + non_edge_3
                                                + non_edge_4 + non_edge_5 + non_edge_6 + non_edge_7
                                                + non_edge_8 + non_edge_9 + non_edge_10 + non_edge_11)
                                         + (maxMT1 * non_edge_12);
            double edgeRequestPerMT
                = maxMT1 / 64
                      * (edge_0 + edge_1 + edge_2 + edge_3 + edge_4 + edge_5 + edge_6 + edge_7 + edge_8)
                  + maxMT1 * (edge_9 + edge_10 + edge_11);

            if(numWGsNonEdge > 0.0)
                non_edge_req = nonEdgeRequestPerMT;
            else
                non_edge_req = 0;
            edge_req = edgeRequestPerMT;

            return result;
        }


        int getGlobalReadQueueFullStallCycles(int currentCycle, std::deque<int>& fifo, int bpRead, int numWaves, bool isStall, bool isSgprOffset)
        {
            int extraIssueCycles = 0;
            if (isSgprOffset) {
                extraIssueCycles = 1;
            }
            if (!isStall) {
                return currentCycle + extraIssueCycles;
            }
            int finalCycle = currentCycle;
            // GR FIFO length is 16, stall cycles is 16 cycles.
            const int grFIFOLength = 16;
            int grStallCycles = 4;
            if (bpRead <= 4) {
                grStallCycles = 1;
            }
            // Theoretically, the maximum number of entries that can be retained is max of readStalledLength.
            const int maxRetainedEntries = (grFIFOLength - 4) * numWaves;
            while (fifo.size() > maxRetainedEntries) {
                fifo.pop_front();
            }
            // pop finished GRs
            if (fifo.size() < grFIFOLength) {
                // if FIFO is not empty, set finalCycle to the last cycle + 1. Only 1 GR can be issued in each cycle.
                if (fifo.size() > 0) {
                    finalCycle = std::max(finalCycle + extraIssueCycles, fifo.back() + extraIssueCycles + 1);
                }
                // push all GRs of all waves
                for(auto wave = 0; wave < numWaves; wave += 1) {
                    fifo.push_back(finalCycle + wave * (1 + extraIssueCycles));
                }
            } else {
                // FIFO is full
                // the index of GR which stall happens is relative with the interval of each GRs.
                int intervalOfGRs = (currentCycle - fifo[fifo.size() - numWaves]);
                int readStalledLength = grFIFOLength;
                if (intervalOfGRs > 4)
                {
                    readStalledLength += (intervalOfGRs - 4) * numWaves;
                }
                if(fifo.size() < readStalledLength)
                {
                    // stall is delayed
                    finalCycle = std::max(finalCycle + extraIssueCycles, fifo.back() + extraIssueCycles + 1);
                    // push all GRs of all waves
                    for(auto wave = 0; wave < numWaves; wave += 1) {
                        fifo.push_back(finalCycle + wave * (1 + extraIssueCycles));
                    }
                }
                else{
                    // push all GRs of all waves
                    finalCycle = std::max(finalCycle + extraIssueCycles, fifo.back() + grStallCycles);
                    for(auto wave = 0; wave < numWaves; wave++) {
                        fifo.push_back(finalCycle + wave * grStallCycles);
                    }
                }
            }
            return finalCycle;
        }

        int getLocalReadCompletionCycle(int currentCycle, std::queue<int>& fifo, int numLR)
        {
            if(fifo.size() <= numLR)
                return currentCycle;
            int finalCycle = currentCycle;
            //pop finisned LR
            while(fifo.size() > numLR)
            {
                int oldCycle = fifo.front();
                if (oldCycle < currentCycle)
                    fifo.pop();
                else
                    break;
            }
            // check non-finished LR
            while(fifo.size() > numLR)
            {
                int oldCycle = fifo.front();
                finalCycle = std::max(finalCycle, oldCycle);
                fifo.pop();
            }
            return finalCycle;
        }

        int getLocalReadQueueFullStallCycles(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves, int lrStallLatencyBuffer)
        {
            int lengthOfQueuePerWave = 16 / numWaves;
            int finalCycle = currentCycle + lrStallLatencyBuffer;
            if (fifo.size() < lengthOfQueuePerWave) {
                fifo.push(finalCycle);
                return currentCycle;
            } else {
                int oldCycle = fifo.front();
                if (currentCycle >= oldCycle) {
                    fifo.pop();
                    fifo.push(finalCycle);
                    return currentCycle;
                } else {
                    // stall happens
                    int stallCycles = safe_ceil_div(lrStallLatencyBuffer, lengthOfQueuePerWave);
                    currentCycle = std::max(currentCycle + stallCycles, oldCycle);
                    finalCycle = currentCycle + lrStallLatencyBuffer;
                    fifo.pop();
                    fifo.push(finalCycle);
                    return currentCycle;
                }
            }
        }

        /**
         * @brief Calculate local read latency based on base latency, conflict multiplier, and bank conflicts
         * @param baseLatency Base latency for local read operation (from HardwareConstants: LocalReadBaseLatencyB128/B64/B32)
         * @param conflictMultiplier Multiplier applied to bank conflict penalty (from HardwareConstants: LocalReadConflictMultiplierB128/B64/B32)
         * @param bankConflict Bank conflict factor (typically 1.0 for no conflict, >1.0 for conflicts)
         * @return Calculated latency in cycles (baseLatency + conflict penalty based on bank conflicts)
         */
        int getLocalReadLatency(int baseLatency, int conflictMultiplier, double bankConflict)
        {
            int conflictPenalty = (bankConflict - 1) * conflictMultiplier;
            return baseLatency + conflictPenalty;
        }

        /**
         * @brief Calculate local write latency based on base latency, conflict multiplier, and bank conflicts
         * @param baseLatency Base latency for local write operation (from HardwareConstants: LocalWriteBaseLatencyB128/B64/B32)
         * @param conflictMultiplier Multiplier applied to bank conflict penalty (from HardwareConstants: LocalWriteConflictMultiplierB128/B64/B32)
         * @param bankConflict Bank conflict factor (typically 1.0 for no conflict, >1.0 for conflicts)
         * @return Calculated latency in cycles (baseLatency + conflict penalty based on bank conflicts)
         */
        int getLocalWriteLatency(int baseLatency, int conflictMultiplier, double bankConflict)
        {
            int conflictPenalty = bankConflict * conflictMultiplier;
            return baseLatency + conflictPenalty;
        }

        double analyzeBankConflictsFromVGPR(
            const std::vector<std::unordered_map<std::string, int64_t>>& vgprState,
            const std::string& vgprLocalReadAddrA,
            int NUM_THREADS_TO_SIMULATE,
            int NUM_BANKS,
            int BANK_WIDTH,
            int LocalReadBytesA)
        {
            double ratioA = 1.0;

            // Track bank usage for conflict analysis
            std::unordered_map<int, int> bankUsageA;

            if(!vgprLocalReadAddrA.empty())
            {
                for(int tid = 0; tid < NUM_THREADS_TO_SIMULATE; tid++)
                {
                    if(!vgprLocalReadAddrA.empty() && vgprState[tid].find(vgprLocalReadAddrA) != vgprState[tid].end())
                    {
                        int64_t addrA = vgprState[tid].at(vgprLocalReadAddrA);
                        int64_t startAddr = addrA;
                        int64_t endAddr = addrA + LocalReadBytesA - 1;

                        // Calculate which banks are accessed by the read range [startAddr, endAddr]
                        int startBankA = (startAddr / BANK_WIDTH) % NUM_BANKS;

                        // Number of BANK_WIDTH-sized chunks this read spans
                        int numBanksAccessed = (endAddr / BANK_WIDTH) - (startAddr / BANK_WIDTH) + 1;

                        // Account for all banks accessed by this LocalReadBytesA read
                        for(int i = 0; i < numBanksAccessed; i++)
                        {
                            int bank = (startBankA + i) % NUM_BANKS;
                            bankUsageA[bank]++;
                        }
                    }
                }

                // Analyze bank conflicts
                if(!bankUsageA.empty())
                {
                    int maxUsageA = 0;
                    int totalAccessesA = 0;
                    for(const auto& pair : bankUsageA)
                    {
                        maxUsageA = std::max(maxUsageA, pair.second);
                        totalAccessesA += pair.second;
                    }
                    double avgUsageA = (double)totalAccessesA / NUM_BANKS;
                    ratioA = (avgUsageA > 0) ? (double)maxUsageA / avgUsageA : 1.0;
                }
            }

            return ratioA;
        }
    } // namespace simulator
} // namespace origami

