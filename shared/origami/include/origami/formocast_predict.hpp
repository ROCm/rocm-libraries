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

#include <cstdint>
#include <tuple>
#include <origami/hardware.hpp>

namespace origami
{
    // Hardware constants structure
    struct HardwareConstants
    {
        double L1CacheCapacity;
        double L2CacheCapacity;
        double L3CacheCapacity;
        double L1CacheLineSize;
        double L2CacheLineSize;
        double L1BusWidthPerCU;
        double L2BusWidthPerCU;
        double L1WriteBusWidthPerCU;
        double L2WriteBusWidthPerCU;
        double maxBandWidthHBM;
        double mem_frequency;
        double hbmBandWidth;
        double L3BandWidth;
        double math_frequency;
        double boost_frequency;
        double initialCost;
        double initialCostHit;
        double flopsPerClk;
        double NumCUs;
        double wavefrontSize;
        double L2ReadArbEff;
        double L2WriteArbEff;
        uint32_t NumXCDs;
        uint32_t LocalReadBaseLatencyB128;
        uint32_t LocalReadBaseLatencyB64;
        uint32_t LocalReadBaseLatencyB32;
        uint32_t LocalReadConflictMultiplierB128;
        uint32_t LocalReadConflictMultiplierB64;
        uint32_t LocalReadConflictMultiplierB32;
        hardware_t::architecture_t architecture;
    };

    // Memory access costs structure
    struct MemoryAccessCosts
    {
        double mem_l1;
        double mem_l2;
        double mem_l3;
        double mem_hbm;
        double l1_hit;
        double l2_hit;
        double l3_hit;
        double mem_overall;
        //for debug
        double A_L1_req;
        double B_L1_req;
        double A_L2_req;
        double B_L2_req;

        // for == compare
        bool operator==(MemoryAccessCosts const &rhs) const
        {
            return std::tie(mem_l1, mem_l2, mem_l3, mem_hbm, l1_hit, l2_hit, l3_hit, mem_overall, A_L1_req, B_L1_req, A_L2_req, B_L2_req) ==
                   std::tie(rhs.mem_l1, rhs.mem_l2, rhs.mem_l3, rhs.mem_hbm, rhs.l1_hit, rhs.l2_hit, rhs.l3_hit, rhs.mem_overall, rhs.A_L1_req, rhs.B_L1_req, rhs.A_L2_req, rhs.B_L2_req);
        };
    };

    // Tie-breaker info structure
    struct TieBreakerInfo
    {
        MemoryAccessCosts memory;
        double perf;
        double preloop;
        double loop;
        double tail;
        double store;
        double gsu;
        double lsu;
        double math;
        double mt0;
        double mt1;
        uint32_t du;
        int    svw;

        // for == compare
        bool operator==(TieBreakerInfo const &rhs) const
        {
            return std::tie(memory, perf, preloop, loop, tail, store, gsu, lsu, math, mt0, mt1, du, svw) ==
                   std::tie(rhs.memory, rhs.perf, rhs.preloop, rhs.loop, rhs.tail, rhs.store, rhs.gsu, rhs.lsu, rhs.math, rhs.mt0, rhs.mt1, rhs.du, rhs.svw);
        };
    };

    // Problem information structure
    struct ProblemInfo
    {
        double M;
        double N;
        double NumBatches;
        double K;
        uint32_t bpeA;
        uint32_t bpeB;
        uint32_t bpeD;
        uint32_t bpeCompute;
        bool transA;
        bool transB;
        bool swizzleTensorA;
        bool swizzleTensorB;
        data_type_t dataType;
    };

    // Simplified ConfigMapping structure (only fields used by calculateFinalPerformance)
    struct ConfigMapping
    {
        struct MacroTile {
            double x;
            double y;
        } macroTile;
        
        int workGroupMapping;
        int CUOccupancy;
        uint32_t depthU;
        uint32_t globalSplitU;
        uint32_t gwvwD;
        int PrefetchGlobalRead;
    };

    // Intermediate performance metrics structure
    struct IntermediatePerformanceMetrics
    {
        // Cache hit rates for tile 0 (A) and tile 1 (B)
        double tile0_L1_hit;
        double tile1_L1_hit;
        double tile0_L2_hit;
        double tile1_L2_hit;
        double tile0_L3_hit;
        double tile1_L3_hit;
        double totalL2HitRate;
        double totalL3HitRate;
        
        // Output write performance
        double output_write_cost;
        double output_write_cost_edge;
        
        // Overall overheads
        double split_accumulation_overhead;
        double compute_cycles;
        double local_split_overhead;
        
        // Memory request counts per cache level for tile 0 (A)
        double tile0_l1_request;
        double tile0_l2_request;
        double tile0_l3_request;
        double tile0_mem_request;
        
        // Memory request counts per cache level for tile 1 (B)
        double tile1_l1_request;
        double tile1_l2_request;
        double tile1_l3_request;
        double tile1_mem_request;
        
        // Prefetch and startup cost
        double prefetch_cost;
        double startup_cost;
    };

    // Predicted performance structure
    struct PredictedPerformance
    {
        double   microSeconds = 0.0;
        double   hitRate      = 0.0;
    };

    // Function declarations
    
    /**
     * @brief Calculate overall loop performance
     * @param mem Memory access costs
     * @param math Math computation time
     * @param loopCnt Number of loop iterations
     * @param pgr Prefetch global read value
     * @return Overall loop time
     */
    double getLoopOverall(const MemoryAccessCosts& mem, double math, uint32_t loopCnt, double pgr);

    /**
     * @brief Calculate memory access costs across cache hierarchy
     * @param MT0 Macro tile dimension 0
     * @param MT1 Macro tile dimension 1
     * @param hw Hardware constants
     * @param tile0_L1_hit Tile 0 (A) L1 cache hit rate
     * @param tile1_L1_hit Tile 1 (B) L1 cache hit rate
     * @param totalL2HitRate Total L2 cache hit rate
     * @param totalL3HitRate Total L3 cache hit rate
     * @param L2BandWidthPerCU L2 bandwidth per CU
     * @param L3BandWidthPerCU L3 bandwidth per CU
     * @param HBMBandWidthPerCU HBM bandwidth per CU
     * @param isSwizzleA Whether tensor A is swizzled
     * @param isSwizzleB Whether tensor B is swizzled
     * @param A_L1_req A matrix L1 requests
     * @param B_L1_req B matrix L1 requests
     * @param A_L2_req A matrix L2 requests
     * @param A_L3_req A matrix L3 requests
     * @param A_hbm_req A matrix HBM requests
     * @param B_L2_req B matrix L2 requests
     * @param B_L3_req B matrix L3 requests
     * @param B_hbm_req B matrix HBM requests
     * @return Memory access costs structure
     */
    MemoryAccessCosts calculateMemoryAccessCosts(
        double MT0, double MT1,
        const HardwareConstants& hw,
        double tile0_L1_hit, double tile1_L1_hit,
        double totalL2HitRate, double totalL3HitRate,
        double L2BandWidthPerCU, double L3BandWidthPerCU, double HBMBandWidthPerCU,
        bool isSwizzleA, bool isSwizzleB,
        double A_L1_req, double B_L1_req,
        double A_L2_req, double A_L3_req, double A_hbm_req,
        double B_L2_req, double B_L3_req, double B_hbm_req);

    /**
     * @brief Resolve performance based on CU occupancy
     * @param hw Hardware constants
     * @param perf Current performance estimate
     * @param prefetch Prefetch cost
     * @param mathCost Math computation cost
     * @param storeCost Store cost
     * @param num_tiles Number of tiles
     * @param CUOccupancy CU occupancy level
     * @return Adjusted performance
     */
    double resolveOccupancy(
        const HardwareConstants& hw, 
        double perf, 
        double prefetch, 
        double mathCost, 
        double storeCost, 
        uint32_t num_tiles, 
        uint32_t CUOccupancy);

    /**
     * @brief Calculate final predicted performance from intermediate metrics
     * @param metrics Intermediate performance metrics
     * @param problem Problem information
     * @param configMapping Config mapping configuration
     * @param arch Hardware architecture
     * @param perfInfo Tie-breaker info (output parameter)
     * @return Predicted performance (microseconds and hit rate)
     */
    PredictedPerformance derivePerformanceProjection(
        const IntermediatePerformanceMetrics& metrics,
        const ProblemInfo& problem,
        const ConfigMapping& configMapping,
        hardware_t::architecture_t arch,
        TieBreakerInfo& perfInfo);

} // namespace origami

