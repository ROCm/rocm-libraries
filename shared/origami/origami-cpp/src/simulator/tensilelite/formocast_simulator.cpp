// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <origami/simulator/tensilelite/formocast_simulator.hpp>
#include <origami/math.hpp>
#include <origami/simulator/tensilelite/formocast.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <cassert>
#include <cstring>
#include <iomanip>

namespace origami
{
    using math::safe_ceil_div;

    double Formocast::calculateInitialCost(double num_tiles) const
    {
        return hw_consts.initialCost * num_tiles + 1.7*(num_tiles-1);
    }

    static double calculatePrefetchPerformance(int      pgr,
                                         int      grvwa,
                                         int      grvwb,
                                         int      bpeA,
                                         int      bpeB,
                                         uint32_t depthU,
                                         int      waveNum,
                                         double   MT0,
                                         double   MT1,
                                         double   math_frequency,
                                         double   mem_latency,
                                         int      numAccPerWave)
    {
        const double others = 220 + numAccPerWave * 4;

        int numGRA = MT0 * depthU * bpeA / (waveNum * 64) / grvwa;
        int numGRB = MT1 * depthU * bpeB / (waveNum * 64) / grvwb;

        //issue 2nd prefetch
        double grCycles2 = numGRA * 4 / waveNum;
        grCycles2 += numGRB * 4 / waveNum;
        return (grCycles2 + others + 1024*depthU/64) / math_frequency;
    }



    double Formocast::getLoop_time(MemoryAccessCosts& mem, double math, uint32_t loopCnt, double pgr, uint32_t num_tiles, bool large) const
    {
        double loop_overall;
        double mem_overall = mem.mem_overall;
        if(pgr > 1 && loopCnt > 0)
            loop_overall = (large || (mem_overall / math >= 1.5) ? (math + mem_overall) / 2 : math) * (loopCnt - 1) + mem_overall;
        else
            loop_overall = std::max(math, mem.mem_overall) * loopCnt;

        mem.mem_loop_l1_req = mem.mem_l1_req * (loopCnt - 1);
        mem.mem_loop_l2_req = mem.mem_l2_req * (loopCnt - 1);
        mem.mem_loop_l3_req = mem.mem_l3_req * (loopCnt - 1);
        mem.mem_loop_hbm_req = mem.mem_hbm_req * (loopCnt - 1);
        return loop_overall;
    }

    Formocast::HardwareConstants archConstantMap(const unsigned char* magic, size_t magicSize) {
        Formocast::HardwareConstants hw;
        if (magicSize != sizeof(Formocast::HardwareConstants)) {
            std::cerr << "Error: magic number size does not match HardwareConstants size!" << std::endl;
        }
        std::memcpy(&hw, magic, std::min(magicSize, sizeof(Formocast::HardwareConstants)));
        return hw;
    }

    Formocast::HardwareConstants
    Formocast::getHardwareConstants(const hardware_t::architecture_t arch) const
    {
        HardwareConstants hw;
        // TODO: migrate to use the original hardware_t
        if(arch == hardware_t::architecture_t::gfx950)
        {
            unsigned char magic[232] = {0, 0, 0, 0, 0, 0, 224, 64, 0, 0, 0, 0, 0, 0, 80, 65, 0, 0, 0, 0, 0, 0, 176, 65, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 80, 64, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 80, 64, 0, 0, 0, 0, 0, 0, 80, 64, 0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 176, 157, 64, 189, 134, 242, 26, 202, 171, 152, 64, 189, 134, 242, 26, 202, 171, 168, 64, 0, 0, 0, 0, 0, 32, 156, 64, 0, 0, 0, 0, 0, 92, 162, 64, 205, 204, 204, 204, 204, 204, 4, 64, 205, 204, 204, 204, 204, 204, 0, 64, 0, 0, 0, 0, 0, 0, 176, 64, 0, 0, 0, 0, 0, 0, 112, 64, 0, 0, 0, 0, 0, 0, 80, 64, 205, 204, 204, 204, 204, 204, 236, 63, 0, 0, 0, 0, 0, 0, 232, 63, 8, 0, 0, 0, 14, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0};
            hw = archConstantMap(magic, 232);
            hw.architecture = hardware_t::architecture_t::gfx950;
        }
        else if(arch == hardware_t::architecture_t::gfx942)
        {
            unsigned char magic[232] = {0, 0, 0, 0, 0, 0, 224, 64, 0, 0, 0, 0, 0, 0, 80, 65, 0, 0, 0, 0, 0, 0, 176, 65, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 80, 64, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 80, 64, 0, 0, 0, 0, 0, 0, 80, 64, 0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 80, 148, 64, 118, 98, 39, 118, 98, 7, 162, 64, 118, 98, 39, 118, 98, 7, 178, 64, 0, 0, 0, 0, 0, 48, 145, 64, 0, 0, 0, 0, 0, 48, 161, 64, 154, 153, 153, 153, 153, 153, 5, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 160, 64, 0, 0, 0, 0, 0, 0, 115, 64, 0, 0, 0, 0, 0, 0, 80, 64, 205, 204, 204, 204, 204, 204, 236, 63, 143, 194, 245, 40, 92, 143, 226, 63, 8, 0, 0, 0, 10, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0};
            hw = archConstantMap(magic, 232);
            hw.architecture = hardware_t::architecture_t::gfx942;
        }
        else if(arch == hardware_t::architecture_t::gfx1201)
        {
            unsigned char magic[232] = {0, 0, 0, 0, 0, 0, 224, 64, 0, 0, 0, 0, 0, 0, 96, 65, 0, 0, 0, 0, 0, 0, 144, 65, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 80, 64, 0, 0, 0, 0, 0, 0, 96, 64, 0, 0, 0, 0, 0, 0, 228, 63, 0, 0, 0, 0, 0, 168, 147, 64, 20, 174, 71, 225, 122, 132, 78, 64, 104, 145, 237, 124, 63, 119, 123, 64, 0, 0, 0, 0, 0, 92, 162, 64, 0, 0, 0, 0, 0, 136, 163, 64, 51, 51, 51, 51, 51, 51, 45, 64, 205, 204, 204, 204, 204, 204, 44, 64, 0, 0, 0, 0, 0, 0, 160, 64, 0, 0, 0, 0, 0, 0, 80, 64, 0, 0, 0, 0, 0, 0, 64, 64, 205, 204, 204, 204, 204, 204, 236, 63, 0, 0, 0, 0, 0, 0, 232, 63, 1, 0, 0, 0, 14, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 6, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0};
            hw = archConstantMap(magic, 232);
            hw.architecture = hardware_t::architecture_t::gfx1201;
        }
        else
        {
            throw std::runtime_error(
                    "Attempting to retrieve hardware constants for unsupported architecture");
        }

        return hw;
    }

    double Formocast::calculateStorePerformance(double M, double N, double num_tiles, double NumBatches, double MT0, double MT1, uint32_t GWVWD, uint32_t bpeD, const HardwareConstants& hw_consts, uint32_t WGs_per_tile_full, uint32_t WGs_per_tile_XCD_full, double &store, double &store_edge) const
    {
        double D_L1_req = 0.0;
        double D_L2_req = 0.0;
        double D_L3_req = 0.0;
        double D_L1_edge_req, D_L2_edge_req, D_L3_edge_req;
        double total_store_req1
            = simulator::calculateStoreL1Request(M, N, MT0, MT1, GWVWD, D_L1_req, D_L1_edge_req);
        double total_store_req2
            = simulator::calculateStoreL2Request(M, N, MT0, MT1, GWVWD, D_L2_req, D_L2_edge_req);
        double total_store_req3 = simulator::calculateStoreL3Request(M, N, MT0, MT1, D_L3_req, D_L3_edge_req);

        double L2WriteBandWidthPerCU = hw_consts.L2WriteArbEff * 128 * 16 / WGs_per_tile_XCD_full; //58% eff
        double L3BandWidthPerCU      = hw_consts.L3BandWidth / WGs_per_tile_full;
        double HBMBandWidthPerCU     = hw_consts.hbmBandWidth / WGs_per_tile_full;
        double D_L1_clk              = D_L1_req * 64 / hw_consts.L1WriteBusWidthPerCU;
        double D_L2_clk = D_L2_req * 64 / std::min(hw_consts.L2WriteBusWidthPerCU, L2WriteBandWidthPerCU);
        double D_L3_clk = D_L3_req * 64 / L3BandWidthPerCU;
        // TODO: D_hbm_clk use D_L3_req.
        double D_hbm_clk     = 0 * 64 / HBMBandWidthPerCU;
        double D_L1_clk_edge = D_L1_edge_req * 64 / hw_consts.L1WriteBusWidthPerCU;
        double D_L2_clk_edge
            = D_L2_edge_req * 64 / std::min(hw_consts.L2WriteBusWidthPerCU, L2WriteBandWidthPerCU);
        double D_L3_clk_edge  = D_L3_edge_req * 64 / L3BandWidthPerCU;
        double D_hbm_clk_edge = 0 * 64 / HBMBandWidthPerCU;

        double D_L1_clk_total = total_store_req1 * 64 / hw_consts.L1WriteBusWidthPerCU;
        double D_L2_clk_total
            = total_store_req2 * 64 / std::min(hw_consts.L2WriteBusWidthPerCU, L2WriteBandWidthPerCU);
        double D_L3_clk_total  = total_store_req3 * 64 / L3BandWidthPerCU;
        double D_hbm_clk_total = 0 * 64 / HBMBandWidthPerCU;

        double store_edge_overall = ((D_L1_clk_edge + D_L2_clk_edge) / hw_consts.math_frequency)
                                    + ((D_L3_clk_edge + D_hbm_clk_edge) / hw_consts.mem_frequency);
        double store_non_edge_overall
            = ((D_L1_clk + D_L2_clk) / hw_consts.math_frequency) + ((D_L3_clk + D_hbm_clk) / hw_consts.mem_frequency);
        double store_total = ((D_L1_clk_total + D_L2_clk_total) / hw_consts.math_frequency)
                             + ((D_L3_clk_total + D_hbm_clk_total) / hw_consts.mem_frequency);
        // Use the max of edge/non-edge store
        store      = store_non_edge_overall;
        store_edge = store_edge_overall;
        store_total = (GWVWD==1) ? store_total*2: store_total;
        store_total = (GWVWD==2) ? store_total*1.5: store_total;

        store = (GWVWD==1) ? store*2: store;
        store = (GWVWD==2) ? store*1.5: store;

        store_edge = (GWVWD==1) ? store_edge*2: store_edge;
        store_edge = (GWVWD==2) ? store_edge*1.5: store_edge;

        return store_total / std::fmin(hw_consts.NumCUs, WGs_per_tile_full);
    }

    Formocast::L1CacheHitRate
    Formocast::computeL1CacheHitRate(const HardwareConstants& hw,
                                          double MT0, double MT1, uint32_t depthU, uint32_t bpeA, uint32_t bpeB,
                                          int NTA, int NTB, uint32_t GRVWA, uint32_t GRVWB,
                                          bool DTVA, bool DTVB, bool isSwizzleA, bool isSwizzleB,
                                          uint32_t VWA, uint32_t VWB, bool transA, bool transB,
                                          double lda, double ldb, int NLCA, int NLCB,
                                          uint32_t threadnum, uint32_t NumWave0, uint32_t NumWave1) const
    {
        auto hr = simulator::computeL1CacheHitRate(
            hw.L1CacheCapacity, hw.L1CacheLineSize, hw.L1BusWidthPerCU,
            MT0, MT1, depthU, bpeA, bpeB, NTA, NTB, GRVWA, GRVWB, DTVA, DTVB,
            isSwizzleA, isSwizzleB, VWA, VWB, transA, transB, lda, ldb,
            NLCA, NLCB, threadnum, NumWave0, NumWave1, hw.architecture == hardware_t::architecture_t::gfx942);

        L1CacheHitRate result;
        result.tile0HitRate = hr.tile0HitRate;
        result.tile1HitRate = hr.tile1HitRate;
        return result;
    }

    Formocast::L3CacheHitRate
    Formocast::computeL3CacheHitRate(double M, double N, double K, const HardwareConstants& hw,
                                          uint32_t bpeA, uint32_t bpeB, int NTA, int NTB,
                                          int N_WGs_total, int M_WGs_total, int N_WGs_per_tile, int M_WGs_per_tile) const
    {
        auto hr = simulator::computeL3CacheHitRate(
            M, N, K, hw.L3CacheCapacity, hw.NumCUs, bpeA, bpeB, NTA, NTB,
            N_WGs_total, M_WGs_total, N_WGs_per_tile, M_WGs_per_tile);

        L3CacheHitRate result;
        result.totalHitRate = hr.totalHitRate;
        result.tile0HitRate = hr.tile0HitRate;
        result.tile1HitRate = hr.tile1HitRate;
        return result;
    }

    double Formocast::calculateGlobalSplitUOverhead(double M, double N, double K,
                                                        double NumBatches, double GlobalSplitU,
                                                        uint32_t gsuMethod, ProblemInfo problem,
                                                        const HardwareConstants& hw_consts,
                                                        uint32_t num_tiles, uint32_t CUOccupancy,
                                                        uint32_t WGs_per_tile_full, uint32_t WGs_per_tile_XCD_full,
                                                        double MT0, double MT1, uint32_t WGs_per_gsu_XCD, double vgprCheck,
                                                        double storeGSU) const
    {
        double gsu_overall = 0.0;

        if(gsuMethod == 2 && GlobalSplitU > 1) //MB
        {
            gsu_overall = simulator::getMultipleBufferOverhead(
                M, N, GlobalSplitU, NumBatches,
                problem.bpeCompute, problem.bpeD, hw_consts.hbmBandWidth,
                hw_consts.L1CacheLineSize, hw_consts.NumCUs, num_tiles, CUOccupancy, hw_consts.boost_frequency,
                hw_consts.mem_frequency, hw_consts.L2WriteArbEff, hw_consts.L2ReadArbEff,
                hw_consts.L3BandWidth, hw_consts.L1BusWidthPerCU, hw_consts.L2BusWidthPerCU,
                hw_consts.L1WriteBusWidthPerCU, hw_consts.L2WriteBusWidthPerCU
            );
        }
        else if(gsuMethod == 3 && GlobalSplitU > 1) //MBSK
        {
            gsu_overall = simulator::getMultipleBufferSingleKernelOverhead(
                GlobalSplitU, MT0, MT1, problem.bpeCompute,
                hw_consts.NumCUs, WGs_per_gsu_XCD, num_tiles, CUOccupancy, hw_consts.boost_frequency,
                hw_consts.L2ReadArbEff, hw_consts.L1BusWidthPerCU, hw_consts.L2BusWidthPerCU,
                storeGSU
            );
        }

        return gsu_overall;
    }

    double Formocast::calculateLocalSplitUOverhead(double MT0, double MT1, double lsu,
                                                     uint32_t svw, uint32_t numThreads,
                                                     ProblemInfo problem,
                                                     const HardwareConstants& hw_consts) const
    {
        return simulator::getLocalSplitKOverhead(MT0, MT1, lsu, svw, numThreads,
                                         problem.bpeCompute, hw_consts.math_frequency);
    }

    static void calculateTilesMemory_req_time(
        Formocast::MemoryAccessCosts& mem_costs,
        double A_L1_req, double A_L2_req, double A_L3_req, double A_hbm_req,
        double B_L1_req, double B_L2_req, double B_L3_req, double B_hbm_req,
        bool isSwizzleA, bool isSwizzleB,
        bool transA, bool transB,
        const Formocast::HardwareConstants& hw_consts,
        double M, double N, double MT0, double MT1, uint32_t WGs_per_tile_XCD_full, uint32_t WGs_per_tile_last, uint32_t WGs_per_tile_XCD_last,
        uint32_t num_tiles)
    {
        double A_L1_clk=0, A_L2_clk=0, A_L3_clk=0, A_hbm_clk=0;
        double B_L1_clk=0, B_L2_clk=0, B_L3_clk=0, B_hbm_clk=0;

        double A_L1_mem_req=0, A_L2_mem_req=0, A_L3_mem_req=0, A_hbm_mem_req=0;
        double B_L1_mem_req=0, B_L2_mem_req=0, B_L3_mem_req=0, B_hbm_mem_req=0;

        bool TLUA = !transA, TLUB = transB;
        double scale_edge_A = (TLUA && M >= MT0) ? 1.0 : (M / math::ceiling_math(M, MT0));
        double scale_edge_B = (TLUB && N >= MT1) ? 1.0 : (N / math::ceiling_math(N, MT1));

        auto calcLClk = [&](double num_tiles, double L1BandWidthPerCU, uint32_t WGs_per_tile, uint32_t WGs_per_tile_XCD) {
            double L2BandWidthPerCU = hw_consts.L2ReadArbEff * 128 * 16 / WGs_per_tile_XCD;
            L2BandWidthPerCU = std::min(L2BandWidthPerCU, hw_consts.L2BusWidthPerCU);
            double L3BandWidthPerCU = hw_consts.L3BandWidth / WGs_per_tile;
            double HBMBandWidthPerCU = hw_consts.hbmBandWidth / WGs_per_tile;

            A_L1_clk += A_L1_req * 64 / L1BandWidthPerCU * num_tiles;
            A_L1_mem_req += A_L1_req * num_tiles * WGs_per_tile * scale_edge_A;

            if(isSwizzleA)
                A_L2_clk += A_L2_req * 128 / L2BandWidthPerCU * num_tiles;
            else
                A_L2_clk += A_L2_req * 128 / L2BandWidthPerCU * num_tiles;
            A_L2_mem_req += A_L2_req * num_tiles * WGs_per_tile * scale_edge_A;

            A_L3_clk += A_L3_req * 64 / L3BandWidthPerCU * num_tiles;
            A_L3_mem_req += A_L3_req * num_tiles * WGs_per_tile * scale_edge_A;

            A_hbm_clk += A_hbm_req * 8 / HBMBandWidthPerCU * num_tiles;
            A_hbm_mem_req += A_hbm_req * num_tiles * WGs_per_tile * scale_edge_A;

            B_L1_clk += B_L1_req * 64 / L1BandWidthPerCU * num_tiles;
            B_L1_mem_req += B_L1_req * num_tiles * WGs_per_tile * scale_edge_B;

            if(isSwizzleB)
                B_L2_clk += B_L2_req * 128 / L2BandWidthPerCU * num_tiles;
            else
                B_L2_clk += B_L2_req * 128 / L2BandWidthPerCU * num_tiles;
            B_L2_mem_req += B_L2_req * num_tiles * WGs_per_tile * scale_edge_B;

            B_L3_clk += B_L3_req * 64 / L3BandWidthPerCU * num_tiles;
            B_L3_mem_req += B_L3_req * num_tiles * WGs_per_tile * scale_edge_B;

            B_hbm_clk += B_hbm_req * 8 / HBMBandWidthPerCU * num_tiles;
            B_hbm_mem_req += B_hbm_req * num_tiles * WGs_per_tile * scale_edge_B;
        };

        calcLClk(num_tiles - 1, hw_consts.L1BusWidthPerCU, hw_consts.NumCUs, WGs_per_tile_XCD_full);

        calcLClk(1, hw_consts.L1BusWidthPerCU, WGs_per_tile_last, WGs_per_tile_XCD_last);

        double L1_overall = (A_L1_clk + B_L1_clk) / hw_consts.math_frequency;
        double L2_overall = (A_L2_clk + B_L2_clk) / hw_consts.math_frequency;
        double L3_overall = (A_L3_clk + B_L3_clk) / hw_consts.mem_frequency;
        double hbm_overall = (A_hbm_clk + B_hbm_clk) / hw_consts.mem_frequency;
        double mem_overall = L1_overall + L2_overall + L3_overall + hbm_overall;
        mem_costs.mem_overall = mem_overall;
        mem_costs.A_mem_l1_req = A_L1_mem_req;
        mem_costs.B_mem_l1_req = B_L1_mem_req;
        mem_costs.mem_l1_req = A_L1_mem_req + B_L1_mem_req;
        mem_costs.mem_l2_req = A_L2_mem_req + B_L2_mem_req;
        mem_costs.mem_l3_req = A_L3_mem_req + B_L3_mem_req;
        mem_costs.mem_hbm_req = A_hbm_mem_req + B_hbm_mem_req;
    }

    Formocast::MemoryAccessCosts
    Formocast::calculateMTMemoryAccessCosts(double M, double N, double K_AfterGSU,
                                   double MT0, double MT1,
                                   const HardwareConstants& hw,
                                   uint32_t WGs_per_tile_XCD_full, uint32_t WGs_per_tile_last, uint32_t WGs_per_tile_XCD_last,
                                   bool isSwizzleA, bool isSwizzleB,
                                   uint32_t bpeA, uint32_t bpeB,
                                   uint32_t depthU,
                                   uint32_t GRVWA, uint32_t GRVWB,
                                   bool DTVA, bool DTVB,
                                   uint32_t VWA, uint32_t VWB,
                                   bool transA, bool transB,
                                   int NLCA, int NLCB,
                                   uint32_t NumThreads, uint32_t NumWave0, uint32_t NumWave1,
                                   uint32_t XCC, uint32_t XCCG,
                                   uint32_t GlobalSplitU, int32_t WGM, double NumBatches,
                                   bool isGSUWGMRR,
                                   uint32_t N_WGs_total, uint32_t M_WGs_total,
                                   uint32_t N_WGs_per_tile, uint32_t M_WGs_per_tile, uint32_t num_tiles) const
    {
        MemoryAccessCosts mem;

        // 1. Calculate Cache Hit Rates
        mem.cache_hits.L1_hit = computeL1CacheHitRate(hw,
                                                MT0, MT1, depthU, bpeA, bpeB,
                                                0, 0, GRVWA, GRVWB,
                                                DTVA, DTVB, isSwizzleA, isSwizzleB,
                                                VWA, VWB, transA, transB,
                                                M, N, NLCA, NLCB,
                                                NumThreads, NumWave0, NumWave1);
        mem.cache_hits.L2_hit = computeL2CacheHitRate(M,
                                                N,
                                                K_AfterGSU,
                                                hw,
                                                XCC, XCCG,
                                                GlobalSplitU,
                                                WGM,
                                                NumBatches,
                                                bpeA,
                                                bpeB,
                                                0,
                                                0,
                                                isGSUWGMRR);
        double K = K_AfterGSU * GlobalSplitU; // Reconstruct original K for L3 cache calculation
        mem.cache_hits.L3_hit = computeL3CacheHitRate(M, N, K, hw,
                                                bpeA, bpeB, 0, 0,
                                                N_WGs_total, M_WGs_total, N_WGs_per_tile, M_WGs_per_tile);

        // 2. Calculate load requests
        double tcc_ea0_coalscedA;
        double tcc_ea0_coalscedB;
        mem.MT_A_L1_req = simulator::getL1LoadRequest(std::min(MT0, M), depthU, hw.L1CacheLineSize,
                                         GRVWA, bpeA, DTVA,
                                         transA,
                                         isSwizzleA,
                                         VWA,
                                         hw.L1BusWidthPerCU,
                                         NLCA,
                                         NumWave1,
                                         tcc_ea0_coalscedA);

        mem.MT_B_L1_req = simulator::getL1LoadRequest(std::min(MT1, N), depthU, hw.L1CacheLineSize,
                                         GRVWB, bpeB, DTVB,
                                         !transB,
                                         isSwizzleB,
                                         VWB,
                                         hw.L1BusWidthPerCU,
                                         NLCB,
                                         NumWave0,
                                         tcc_ea0_coalscedB);
        mem.tcc_ea0_coalscedA = tcc_ea0_coalscedA;
        mem.tcc_ea0_coalscedB = tcc_ea0_coalscedB;
        mem.MT_A_L2_req = simulator::getL2LoadRequest(mem.MT_A_L1_req, mem.cache_hits.L1_hit.tile0HitRate, tcc_ea0_coalscedA);
        mem.MT_A_L3_req = simulator::getL3LoadRequest(mem.MT_A_L2_req, mem.cache_hits.L2_hit.tile0HitRate, tcc_ea0_coalscedA);
        mem.MT_A_hbm_req = simulator::getHBMLoadRequest(mem.MT_A_L3_req, mem.cache_hits.L3_hit.tile0HitRate);
        mem.MT_B_L2_req = simulator::getL2LoadRequest(mem.MT_B_L1_req, mem.cache_hits.L1_hit.tile1HitRate, tcc_ea0_coalscedB);
        mem.MT_B_L3_req = simulator::getL3LoadRequest(mem.MT_B_L2_req, mem.cache_hits.L2_hit.tile1HitRate, tcc_ea0_coalscedB);
        mem.MT_B_hbm_req = simulator::getHBMLoadRequest(mem.MT_B_L3_req, mem.cache_hits.L3_hit.tile1HitRate);

        mem.l1_hit = (mem.cache_hits.L1_hit.tile0HitRate * MT0 + mem.cache_hits.L1_hit.tile1HitRate * MT1) / (MT0 + MT1);
        mem.l2_hit = mem.cache_hits.L2_hit.totalHitRate;
        mem.l3_hit = mem.cache_hits.L3_hit.totalHitRate;

        // Calculate memory overall using GetMemoryOverall function
        calculateTilesMemory_req_time(mem,
            mem.MT_A_L1_req, mem.MT_A_L2_req, mem.MT_A_L3_req, mem.MT_A_hbm_req,
            mem.MT_B_L1_req, mem.MT_B_L2_req, mem.MT_B_L3_req, mem.MT_B_hbm_req,
            isSwizzleA, isSwizzleB,
            transA, transB,
            hw_consts,
            M, N, MT0, MT1,
            WGs_per_tile_XCD_full, WGs_per_tile_last, WGs_per_tile_XCD_last,
            num_tiles);

        return mem;
    }

    double Formocast::resolveOccupancy(const HardwareConstants& hw, double perf, double prefetch, double mathCost, double storeCost, uint32_t num_tiles, uint32_t CUOccupancy, uint32_t loopCnt) const
    {
        if ((num_tiles > 1)  && CUOccupancy >= 2)
        {
            uint32_t tiles = safe_ceil_div(num_tiles, CUOccupancy);
            perf = hw.initialCost + (prefetch/tiles + mathCost + storeCost * 4 * num_tiles) + loopCnt * tiles * 0.1;
        }
        else if (CUOccupancy >= 2)
        {
            perf += (CUOccupancy - 1) * 4 * storeCost - storeCost + loopCnt * 0.1;
        }
        else
        {
            perf += loopCnt * 0.1;
        }
        return perf;
    }

    bool Formocast::isBetter(ProblemInfo problem, TieBreakerInfo previousSolution) const
    {
        auto currSol = getTieBreakerInfo();

        // Call standalone tie-breaker function
        return compareConfigTieBreaker(
            problem.M, problem.N, problem.K, problem.NumBatches,
            currSol.mt0, currSol.mt1, currSol.du, currSol.svw,
            previousSolution.mt0, previousSolution.mt1, previousSolution.du, previousSolution.svw
        );
    }

    // NB:
    //  isBetter return
    //   True means current is better than previous
    //   False doesn't means worse, means tie (equal) --> IMPORTANT note since this would be used in std::sort
    bool Formocast::isBetter(TieBreakerInfo previousSolution, TieBreakerInfo currentSolution) const
    {
        // Call standalone tie-breaker function
        return compareConfigTieBreaker(
            problem.M, problem.N, problem.K, problem.NumBatches,
            currentSolution.mt0, currentSolution.mt1, currentSolution.du, currentSolution.svw,
            previousSolution.mt0, previousSolution.mt1, previousSolution.du, previousSolution.svw
        );
    }

    Formocast::TieBreakerInfo Formocast::getTieBreakerInfo() const
    {
        return perfInfo;
    }

    // NB:
    //  isBetter return
    //   True means current is better than previous
    //   False doesn't means worse, means tie (EQUAL) --> IMPORTANT note since this would be used in std::sort
    bool Formocast::isBetter(MinTieBreakerInfo previousSolution, MinTieBreakerInfo currentSolution) const
    {
        // just early return, return false means equal
        if(previousSolution == currentSolution)
            return false;

        // Call standalone tie-breaker function
        return compareConfigTieBreaker(
            problem.M, problem.N, problem.K, problem.NumBatches,
            currentSolution.mt0, currentSolution.mt1, currentSolution.du, currentSolution.svw,
            previousSolution.mt0, previousSolution.mt1, previousSolution.du, previousSolution.svw
        );
    }

    Formocast::MinTieBreakerInfo Formocast::getMinTieBreakerInfo() const
    {
        MinTieBreakerInfo tbInfo;

        tbInfo.mt0 = perfInfo.mt0;
        tbInfo.mt1 = perfInfo.mt1;
        tbInfo.du  = perfInfo.du;
        tbInfo.svw = perfInfo.svw;

        return tbInfo;
    }

    Formocast::PredictedPerformance
    Formocast::predictedPerformance(void) const
    {
        PredictedPerformance pp;

        // 1. Problem Dimension Calculation
        double M = problem.M;
        double N = problem.N;
        double NumBatches = problem.NumBatches;
        double K = problem.K;
        bool transA = problem.transA;
        bool transB = problem.transB;
        uint32_t bpeA    = problem.bpeA;
        uint32_t bpeB    = problem.bpeB;
        uint32_t bpeD    = problem.bpeD;
        // swizzle settings
        bool     isSwizzleA = problem.swizzleTensorA;
        bool     isSwizzleB = problem.swizzleTensorB;

        // 2. Hardware Parameter Extraction
        //HardwareConstants hw_consts = getHardwareConstants();
        //hw_consts.print();

        // 3. Variables directly from sizeMapping

        // Basic tile and workgroup configuration
        double MT0 = sizeMapping.macroTile[0];
        double MT1 = sizeMapping.macroTile[1];
        int      WGM = sizeMapping.workGroupMapping != 0 ? sizeMapping.workGroupMapping : 1;
        int      CUOccupancy = sizeMapping.CUOccupancy;
        uint32_t XCC  = sizeMapping.workGroupMappingXCC;
        uint32_t XCCG = (sizeMapping.workGroupMappingXCCGroup < 0)? hw_consts.NumCUs : sizeMapping.workGroupMappingXCCGroup;
        uint32_t depthU = sizeMapping.depthU;

        // Global split
        bool     isGSUWGMRR = sizeMapping.globalSplitUWorkGroupMappingRoundRobin;
        uint32_t gsuMethod = sizeMapping.globalAccumulation;

        // Prefetch and memory access configuration
        int      PGR = sizeMapping.PrefetchGlobalRead;

        // Wave and global read configuration
        uint32_t GRVWA = sizeMapping.grvwA;
        uint32_t GRVWB = sizeMapping.grvwB;
        uint32_t GWVWD = sizeMapping.gwvwD;
        uint32_t VWA   = sizeMapping.VectorWidthA;
        uint32_t VWB   = sizeMapping.VectorWidthB;
        uint32_t waveNum  = sizeMapping.waveNum;
        uint32_t NumWave0 = sizeMapping.waveGroup[0];
        uint32_t NumWave1 = sizeMapping.waveGroup[1];
        uint32_t NumThreads = hw_consts.wavefrontSize * waveNum;

        // Matrix instruction and VGPR configuration
        int miSize = sizeMapping.matrixInstruction[0];
        bool DTVA = sizeMapping.DirectToVgprA;
        bool DTVB = sizeMapping.DirectToVgprB;

        // NLCA/B is used for non-TN cases to calculate load requests.
        int NLCA = sizeMapping.NumLoadsCoalescedA;
        int NLCB = sizeMapping.NumLoadsCoalescedB;

        //GlobalSplitU
        uint32_t GlobalSplitU = sizeMapping.globalSplitU;
        //LocalSplitU
        int LSU = sizeMapping.LocalSplitU;

        //DirectToLdsA
        bool DirectToLdsA = sizeMapping.DirectToLdsA;
        //DirectToLdsB
        bool DirectToLdsB = sizeMapping.DirectToLdsB;

        // Clock calculation
        // TODO: No need to check minMathClock if we guarantee that MathClocksUnrolledLoop is correct.
        double math_clk = sizeMapping.MathClocksUnrolledLoop;

        // 3.1 Early terminate. FIXME: Can filter most of the solutions with an outside function.
        // FIXME: add an extra function to reject the solutions first.
        if (GlobalSplitU == 0)
        {
            // FIXME: Need to support streamK kernels.
            GlobalSplitU = 1;
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }
        if ((M < 128 && MT0 - M >= 16) || (N < 128 && MT1 - N >= 16))
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }
        if ((M >= 128 && MT0 - M >= 32) || (N >= 128 && MT1 - N >= 32))
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }
        if(problem.dataType == data_type_t::BFloat16 || problem.dataType == data_type_t::Half)
        {
            if(problem.bpeA == 2 && problem.bpeB == 2)
                if (((K >= 64 && depthU <=32) || (K <= 32 && depthU > 32) || (K > 32 && depthU > K)) && NumBatches < hw_consts.NumCUs && sizeMapping.matrixInstruction[2] >= 32)
                {
                    pp.microSeconds = 9999999.9;
                    pp.hitRate = 0;
                    return pp;
                }
        }
        if(DirectToLdsA && M < MT0)
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }
        if(DirectToLdsB && N < MT1)
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }

        // 4. Derived Problem/Workgroup Dimensions
        uint32_t M_WGs_total = safe_ceil_div(static_cast<uint32_t>(M), static_cast<uint32_t>(MT0));
        uint32_t N_WGs_total = safe_ceil_div(static_cast<uint32_t>(N), static_cast<uint32_t>(MT1));
        int N_WGs_per_tile_XCD = std::min((uint32_t)WGM, N_WGs_total);
        int M_WGs_per_tile_XCD
            = std::min(M_WGs_total, static_cast<uint32_t>(safe_ceil_div(int(hw_consts.NumCUs / hw_consts.NumXCDs), N_WGs_per_tile_XCD)));
        int M_WGs_per_tile = std::min(M_WGs_total, static_cast<uint32_t>(safe_ceil_div(int(hw_consts.NumCUs), N_WGs_per_tile_XCD)));
        int N_WGs_per_tile
            = std::min(N_WGs_total, static_cast<uint32_t>(N_WGs_per_tile_XCD * safe_ceil_div(M_WGs_per_tile, M_WGs_total)));
        uint32_t numberWGs = M_WGs_total * N_WGs_total * NumBatches * GlobalSplitU;
        uint32_t num_tiles = safe_ceil_div(numberWGs, uint32_t(hw_consts.NumCUs));
        uint32_t NumCUs = hw_consts.NumCUs;
        uint32_t WGs_per_tile_last = numberWGs % NumCUs == 0 ? NumCUs : numberWGs % NumCUs;
        uint32_t WGs_per_tile_XCD_last = safe_ceil_div(WGs_per_tile_last, hw_consts.NumXCDs);
        uint32_t WGs_per_tile_full = num_tiles == 1 ? WGs_per_tile_last : NumCUs;
        uint32_t WGs_per_tile_XCD_full = safe_ceil_div(WGs_per_tile_full, hw_consts.NumXCDs);
        uint32_t WGs_per_gsu_XCD = safe_ceil_div(safe_ceil_div(std::min(numberWGs, NumCUs), hw_consts.NumXCDs), GlobalSplitU);

        // compute K
        uint32_t total_loop = safe_ceil_div(static_cast<uint32_t>(K), static_cast<uint32_t>(depthU));
        uint32_t loopCnt = safe_ceil_div(total_loop, static_cast<uint32_t>(GlobalSplitU));
        double K_AfterGSU = depthU * loopCnt;
        uint32_t K_tail = static_cast<uint32_t>(K) - static_cast<uint32_t>(int(K / depthU) * depthU);
        PGR = std::ceil((K - K_tail) / depthU / GlobalSplitU);
        if(PGR > 1)
          PGR = sizeMapping.PrefetchGlobalRead;
        int PLR = loopCnt < sizeMapping.LocalSplitU ? 0 : 1;

        if (PLR == 0)
        {
            pp.microSeconds = 9999999.9;
            pp.hitRate = 0;
            return pp;
        }

        // 5.1 Calculate initial costs
        double doinit = calculateInitialCost(num_tiles);

        // 5.6 Calculate Store Performance (D matrix writes)
        double store, store_edge;
        double store_total = calculateStorePerformance(M, N, num_tiles, NumBatches, MT0, MT1, GWVWD, bpeD, hw_consts, WGs_per_tile_full, WGs_per_tile_XCD_full, store, store_edge);

        // 5.7 Calculate GSU Overhead
        double storeGSU = 4 * store_total * GlobalSplitU;
        auto vgprUsageCheck = MT0 * MT1 / miSize / miSize;
        double gsu_overall = calculateGlobalSplitUOverhead(M, N, K, NumBatches, GlobalSplitU, gsuMethod,
                                                  problem, hw_consts, num_tiles, CUOccupancy, WGs_per_tile_full, WGs_per_tile_XCD_full,
                                                  MT0, MT1, WGs_per_gsu_XCD, vgprUsageCheck, storeGSU);
        gsu_overall *= num_tiles;

        // 5.4 Calcupate LSU Overhead
        double lsu_overall = calculateLocalSplitUOverhead(MT0, MT1, LSU, GWVWD, NumThreads, problem, hw_consts);
        lsu_overall *= num_tiles;

        MemoryAccessCosts mem_costs = calculateMTMemoryAccessCosts(M, N, K_AfterGSU,
                                                            MT0, MT1,
                                                            hw_consts,
                                                            WGs_per_tile_XCD_full, WGs_per_tile_last, WGs_per_tile_XCD_last,
                                                            isSwizzleA, isSwizzleB,
                                                            bpeA, bpeB,
                                                            depthU,
                                                            GRVWA, GRVWB,
                                                            DTVA, DTVB,
                                                            VWA, VWB,
                                                            transA, transB,
                                                            NLCA, NLCB,
                                                            NumThreads, NumWave0, NumWave1,
                                                            XCC, XCCG,
                                                            GlobalSplitU, WGM, NumBatches,
                                                            isGSUWGMRR,
                                                            N_WGs_total, M_WGs_total,
                                                            N_WGs_per_tile, M_WGs_per_tile, num_tiles);

        // 5.2 Calculate Prefetch Performance
        int numAccPerWave = MT0 * MT1 / waveNum / hw_consts.wavefrontSize;
        double prefetch      = calculatePrefetchPerformance(PGR, GRVWA,
                                                 GRVWB,
                                                 bpeA,
                                                 bpeB,
                                                 depthU,
                                                 waveNum,
                                                 MT0,
                                                 MT1,
                                                 hw_consts.math_frequency, 0,
                                                 numAccPerWave);
        prefetch *= num_tiles;
        prefetch += mem_costs.mem_overall;

        // 6. Calculate loop Performance
        double math_overall = math_clk / hw_consts.math_frequency; math_overall *= num_tiles;
        bool large = M*K*bpeA > 67108864 || N*K*bpeB > 67108864;
        double loop_overall = getLoop_time(mem_costs, math_overall, loopCnt, PGR, num_tiles, large);

        // 5.5 tail loop Overhead
        double tail_overall = 0.0;
        if (K_tail > 0)
        {
            // FIXME: need to add new opt.
            tail_overall = (mem_costs.mem_overall*K_tail/depthU + math_overall) + prefetch*2;
            tail_overall *= num_tiles;
        }

        // prediction model implementation

        // 7. Aggregate Performance: pre-loop + unrolled-loop + Tail Loop + post-loop
        double perf = doinit +prefetch + loop_overall + tail_overall + store_total;

        // 9. Add LSU Reduction Part might be removed before 8.
        perf += lsu_overall;

        // 8. Apply CU Occupancy
        perf = resolveOccupancy(hw_consts, perf, prefetch, loop_overall + tail_overall, store_total, num_tiles, CUOccupancy, loopCnt);

        // 10. Add GSU Reduction Part might be removed before 8.
        perf += gsu_overall;

        pp.microSeconds = perf;
        pp.hitRate = mem_costs.l2_hit * 100;

        pp.MT0 = MT0;
        pp.MT1 = MT1;
        pp.depthU = depthU;
        pp.NumCUs = hw_consts.NumCUs;
        pp.WorkGroupMapping = WGM;
        pp.CUOccupancy = CUOccupancy;
        pp.GlobalSplitU = GlobalSplitU;
        pp.LocalSplitU = LSU;
        pp.loopCnt = loopCnt;

        pp.math_overall = math_overall;
        pp.mem_overall = mem_costs.mem_overall;

        pp.memCosts = mem_costs;

        pp.init = doinit;
        pp.preloop = prefetch;
        pp.loop = loop_overall;
        pp.tail = tail_overall;
        pp.store = store_total;
        pp.gsu = gsu_overall;
        pp.lsu = lsu_overall;
        pp.num_tiles = num_tiles;
        pp.perf = perf;

        perfInfo.memory = mem_costs;
        perfInfo.math = math_overall;
        perfInfo.svw = GWVWD;
        perfInfo.perf = perf;
        perfInfo.preloop = prefetch;
        perfInfo.loop = loop_overall;
        perfInfo.tail = tail_overall;
        perfInfo.store = store;
        perfInfo.gsu = gsu_overall;
        perfInfo.lsu = lsu_overall;
        perfInfo.mt0 = MT0;
        perfInfo.mt1 = MT1;
        perfInfo.du = depthU;

        return pp;
    }

    Formocast::L2CacheHitRate
        Formocast::computeL2CacheHitRate(uint32_t M,
                                         uint32_t N,
                                         uint32_t K,
                                         const HardwareConstants& hw,
                                         uint32_t XCC, uint32_t XCCG,
                                         uint32_t gsu,
                                         int32_t  wgm,
                                         uint32_t batches,
                                         uint32_t bpeA,
                                         uint32_t bpeB,
                                         int32_t  NTA,
                                         int32_t  NTB,
                                         bool     isGSUWGMRR) const
    {
        uint32_t MT0 = sizeMapping.macroTile[0];
        uint32_t MT1 = sizeMapping.macroTile[1];
        uint32_t depthU = sizeMapping.depthU;

        auto hr = simulator::computeL2CacheHitRate(
            M, N, K, MT0, MT1, depthU, hw.L2CacheCapacity, hw.NumCUs, hw.NumXCDs,
            XCC, XCCG, gsu, wgm, batches, bpeA, bpeB, NTA, NTB, isGSUWGMRR);

        L2CacheHitRate hitRate;
        hitRate.totalHitRate = hr.totalHitRate;
        hitRate.tile0HitRate = hr.tile0HitRate;
        hitRate.tile1HitRate = hr.tile1HitRate;

        return hitRate;
    }

    void Formocast::setProblem(ProblemInfo p)
    {
        problem = p;
        if (problem.M == 0 || problem.N == 0 || problem.K == 0)
            throw std::runtime_error(
                "Problem size is invalid");
    }

    void Formocast::setSolution(SizeMapping sm)
    {
        sizeMapping = sm;
    }

    void Formocast::setHardware(hardware_t::architecture_t hw)
    {
        hw_consts = getHardwareConstants(hw);
    }

    int Formocast::getGlobalReadQueueFullStallCycles(int currentCycle, std::deque<int>& fifo, int bpRead, int numWaves, bool isStall, bool isSgprOffset) const
    {
        return simulator::getGlobalReadQueueFullStallCycles(currentCycle, fifo, bpRead, numWaves, isStall, isSgprOffset);
    }

    int Formocast::getLocalReadCompletionCycle(int currentCycle, std::queue<int>& fifo, int numLR) const
    {
        return simulator::getLocalReadCompletionCycle(currentCycle, fifo, numLR);
    }

    int Formocast::getLocalReadQueueFullStallCycles(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves, bool isStall, double bankConflict) const
    {
        int lrStallLatencyBuffer;
        if (!isStall){
            lrStallLatencyBuffer = 1;
        }
        else if (bpRead == 16) {
            lrStallLatencyBuffer = simulator::getLocalReadLatency(hw_consts.LocalReadBaseLatencyB128, hw_consts.LocalReadConflictMultiplierB128, bankConflict);
        } else if (bpRead == 8) {
            lrStallLatencyBuffer = simulator::getLocalReadLatency(hw_consts.LocalReadBaseLatencyB64, hw_consts.LocalReadConflictMultiplierB64, bankConflict);
        } else {
            lrStallLatencyBuffer = simulator::getLocalReadLatency(hw_consts.LocalReadBaseLatencyB32, hw_consts.LocalReadConflictMultiplierB32, bankConflict);
        }
        return simulator::getLocalReadQueueFullStallCycles(currentCycle, fifo, bpRead, numWaves, lrStallLatencyBuffer);
    }

    int Formocast::getLocalWriteQueueFullStallCycles(int currentCycle, int previousLW, int issueCycles, int bpWrite, int numWaves) const
    {
        int issuePenality = 0;
        if(numWaves == 4)
        {
            // only 2 simds can be issued at the same time. if 4 simds, need 1 extra issue cycle.
            issuePenality = issueCycles;
        }
        if(currentCycle - previousLW >= issueCycles)
        {
            // if the space is enough, no stall.
            return currentCycle + issueCycles;
        }
        return currentCycle + issueCycles + issuePenality;
    }

    void Formocast::pushLocalReadWrite(int currentCycle, std::queue<int>& fifo, int bpr, double bankConflict, bool isLocalRead, int numPreviousLRs)
    {
        int lrMemLatency;
        if (isLocalRead) {
            if (bpr == 16) {
                lrMemLatency = simulator::getLocalReadLatency(hw_consts.LocalReadBaseLatencyB128, hw_consts.LocalReadConflictMultiplierB128, bankConflict);
                if (numPreviousLRs <= 4) {
                    lrMemLatency += 2 * numPreviousLRs;
                }
                else {
                    // Maximum 4 * 2 latency for previous local reads
                    lrMemLatency += 2 * 4;
                }
            } else if (bpr == 8) {
                lrMemLatency = simulator::getLocalReadLatency(hw_consts.LocalReadBaseLatencyB64, hw_consts.LocalReadConflictMultiplierB64, bankConflict);
            } else {
                lrMemLatency = simulator::getLocalReadLatency(hw_consts.LocalReadBaseLatencyB32, hw_consts.LocalReadConflictMultiplierB32, bankConflict);
            }
        }
        else {
            // Local write latency
            if (bpr == 16) {
                lrMemLatency = simulator::getLocalWriteLatency(hw_consts.LocalWriteBaseLatencyB128, hw_consts.LocalWriteConflictMultiplierB128, bankConflict);
            } else if (bpr == 8) {
                lrMemLatency = simulator::getLocalWriteLatency(hw_consts.LocalWriteBaseLatencyB64, hw_consts.LocalWriteConflictMultiplierB64, bankConflict);
            } else {
                lrMemLatency = simulator::getLocalWriteLatency(hw_consts.LocalWriteBaseLatencyB32, hw_consts.LocalWriteConflictMultiplierB32, bankConflict);
            }
        }
        fifo.push(currentCycle + lrMemLatency);
    }

    // Standalone tie-breaker function implementation
    bool compareConfigTieBreaker(
        double M, double N, double K, size_t batch,
        double mt0_a, double mt1_a, double du_a, int svw_a,
        double mt0_b, double mt1_b, double du_b, int svw_b
    )
    {
        // Skinny N case: N <= 32 && M >= 1024 && K >= 1024
        // Prefer configurations with larger "skinny ratio" when mt1 matches N
        if (N <= 32 && M >= 1024 && K >= 1024) {
            auto skinny_a = mt0_a * du_a / M;
            auto skinny_b = mt0_b * du_b / M;
            if (mt1_a == N && mt1_b == N && skinny_a != skinny_b) {
                return skinny_a > skinny_b;
            }
        }

        // Skinny M case: M <= 32 && N >= 1024 && K >= 1024
        // Prefer configurations with larger "skinny ratio" when mt0 matches M
        if (M <= 32 && N >= 1024 && K >= 1024) {
            auto skinny_a = mt1_a * du_a / N;
            auto skinny_b = mt1_b * du_b / N;
            if (mt0_a == M && mt0_b == M && skinny_a != skinny_b) {
                return skinny_a > skinny_b;
            }
        }

        // Wide & short K case: batch == 1 && K <= 512 && M >= 1024 && N >= 1024
        // Prefer larger store vector width for better memory efficiency
        if (batch == 1 && K <= 512 && M >= 1024 && N >= 1024) {
            if (svw_a != svw_b) {
                return svw_a > svw_b;
            }
        }

        // No preference - configurations are considered equal
        return false;
    }

    // Helper function to analyze bank conflicts from VGPR states
    Formocast::BankConflictResult Formocast::analyzeBankConflictsFromVGPR(
        const std::vector<std::unordered_map<std::string, int64_t>>& vgprState,
        const std::string& vgprLocalReadAddrA,
        const std::string& vgprLocalReadAddrB,
        int LocalReadBytesA,
        int LocalReadBytesB)
    {
        BankConflictResult result;
        int NUM_THREADS_TO_SIMULATE = hw_consts.wavefrontSize;
        int NUM_BANKS = 32;
        int BANK_WIDTH = 4;
        result.ratioA = simulator::analyzeBankConflictsFromVGPR(vgprState, vgprLocalReadAddrA, NUM_THREADS_TO_SIMULATE, NUM_BANKS, BANK_WIDTH, LocalReadBytesA);
        result.ratioB = simulator::analyzeBankConflictsFromVGPR(vgprState, vgprLocalReadAddrB, NUM_THREADS_TO_SIMULATE, NUM_BANKS, BANK_WIDTH, LocalReadBytesB);
        return result;
    }
} // namespace origami
