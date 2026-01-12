// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <queue>
#include <iostream>
#include <unordered_map>
#include <formocast.hpp>
#include "origami/types.hpp"
#include "origami/hardware.hpp"

namespace origami
{
    class Formocast
    {
    public:
        struct SizeMapping
        {
            size_t waveNum;

            std::array<int, 3> macroTile;
            std::array<int, 4> matrixInstruction;
            size_t             grvwA = 1;
            size_t             grvwB = 1;
            size_t             gwvwC = 1;
            size_t             gwvwD = 1;

            size_t  depthU             = 0;
            int16_t globalSplitU       = 0;

            int     workGroupMapping   = 0;
            int     globalAccumulation = 0;

            int  workGroupMappingXCC                    = 0;
            int  workGroupMappingXCCGroup               = 0;
            bool globalSplitUCoalesced                  = false;
            bool globalSplitUWorkGroupMappingRoundRobin = false;

            int CUOccupancy            = 0;
            int PrefetchGlobalRead     = 2;
            int MathClocksUnrolledLoop = 0;

            bool DirectToVgprA = false;
            bool DirectToVgprB = false;
            int NumLoadsCoalescedA = 0;
            int NumLoadsCoalescedB = 0;
            int VectorWidthA = 1;
            int VectorWidthB = 1;
            int LocalSplitU = 1;

            std::array<int, 2> waveGroup;

            bool DirectToLdsA = false;
            bool DirectToLdsB = false;
        };

        struct PredictedPerformance
        {
            double   microSeconds = 0.0;
            double   hitRate      = 0.0;
        };

        struct L2CacheHitRate
        {
            double tile0HitRate          = 0.0;
            double tile1HitRate          = 0.0;
            double totalHitRate          = 0.0;
        };

        struct L1CacheHitRate : public L2CacheHitRate{};
        struct L3CacheHitRate : public L2CacheHitRate{};

        struct BankConflictResult {
            double ratioA = 1.0;
            double ratioB = 1.0;
        };

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

            void print() const {
                std::cout << "HardwareConstants:" << std::endl;
                std::cout << "  architecture:         " << (architecture == hardware_t::architecture_t::gfx950 ? "gfx950" : architecture == hardware_t::architecture_t::gfx942 ? "gfx942" : architecture == hardware_t::architecture_t::gfx1201 ? "gfx1201" : "Unknown") << std::endl;
                std::cout << "  L1CacheCapacity:      " << L1CacheCapacity << std::endl;
                std::cout << "  L2CacheCapacity:      " << L2CacheCapacity << std::endl;
                std::cout << "  L3CacheCapacity:      " << L3CacheCapacity << std::endl;
                std::cout << "  L1CacheLineSize:      " << L1CacheLineSize << std::endl;
                std::cout << "  L2CacheLineSize:      " << L2CacheLineSize << std::endl;
                std::cout << "  L1BusWidthPerCU:      " << L1BusWidthPerCU << std::endl;
                std::cout << "  L2BusWidthPerCU:      " << L2BusWidthPerCU << std::endl;
                std::cout << "  L1WriteBusWidthPerCU: " << L1WriteBusWidthPerCU << std::endl;
                std::cout << "  L2WriteBusWidthPerCU: " << L2WriteBusWidthPerCU << std::endl;
                std::cout << "  maxBandWidthHBM:      " << maxBandWidthHBM << std::endl;
                std::cout << "  mem_frequency:        " << mem_frequency << std::endl;
                std::cout << "  hbmBandWidth:         " << hbmBandWidth << std::endl;
                std::cout << "  L3BandWidth:          " << L3BandWidth << std::endl;
                std::cout << "  math_frequency:       " << math_frequency << std::endl;
                std::cout << "  boost_frequency:      " << boost_frequency << std::endl;
                std::cout << "  initialCost:          " << initialCost << std::endl;
                std::cout << "  initialCostHit:       " << initialCostHit << std::endl;
                std::cout << "  flopsPerClk:          " << flopsPerClk << std::endl;
                std::cout << "  NumCUs:               " << NumCUs << std::endl;
                std::cout << "  wavefrontSize:        " << wavefrontSize << std::endl;
                std::cout << "  L2ReadArbEff:         " << L2ReadArbEff << std::endl;
                std::cout << "  L2WriteArbEff:        " << L2WriteArbEff << std::endl;
                std::cout << "  NumXCDs:              " << NumXCDs << std::endl;
                std::cout << "  LocalReadBaseLatencyB128: " << LocalReadBaseLatencyB128 << std::endl;
                std::cout << "  LocalReadBaseLatencyB64: " << LocalReadBaseLatencyB64 << std::endl;
                std::cout << "  LocalReadBaseLatencyB32: " << LocalReadBaseLatencyB32 << std::endl;
                std::cout << "  LocalReadConflictMultiplierB128: " << LocalReadConflictMultiplierB128 << std::endl;
                std::cout << "  LocalReadConflictMultiplierB64: " << LocalReadConflictMultiplierB64 << std::endl;
                std::cout << "  LocalReadConflictMultiplierB32: " << LocalReadConflictMultiplierB32 << std::endl;
            };
        };

        struct CacheHitRates
        {
            double A_L1_hit;
            double B_L1_hit;
            double A_L2_hit;
            double B_L2_hit;
            double A_L3_hit;
            double B_L3_hit;
            double totalL2HitRate;
            double totalL3HitRate; // Added for clarity
        };

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

            // for == compare, can remove this if we are using MinTieBreakerInfo
            bool operator==(MemoryAccessCosts const &rhs) const
            {
                return std::tie(mem_l1, mem_l2, mem_l3, mem_hbm, l1_hit, l2_hit, l3_hit, mem_overall, A_L1_req, B_L1_req, A_L2_req, B_L2_req) ==
                       std::tie(rhs.mem_l1, rhs.mem_l2, rhs.mem_l3, rhs.mem_hbm, rhs.l1_hit, rhs.l2_hit, rhs.l3_hit, rhs.mem_overall, rhs.A_L1_req, rhs.B_L1_req, rhs.A_L2_req, rhs.B_L2_req);
            };
        };

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

            // for == compare, can remove this if we are using MinTieBreakerInfo
            bool operator==(TieBreakerInfo const &rhs) const
            {
                return std::tie(memory, perf, preloop, loop, tail, store, gsu, lsu, math, mt0, mt1, du, svw) ==
                       std::tie(rhs.memory, rhs.perf, rhs.preloop, rhs.loop, rhs.tail, rhs.store, rhs.gsu, rhs.lsu, rhs.math, rhs.mt0, rhs.mt1, rhs.du, rhs.svw);
            };
        };

        // Use this "un-mutable" version for std::sort
        struct MinTieBreakerInfo
        {
            double mt0;
            double mt1;
            uint32_t du;
            int    svw;

            bool operator==(MinTieBreakerInfo const &rhs) const
            {
                return std::tie(mt0, mt1, du, svw) == std::tie(rhs.mt0, rhs.mt1, rhs.du, rhs.svw);
            };
        };

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

        // Structure to hold intermediate calculation results
        struct IntermediatePerformanceMetrics
        {
            // Cache hit rates (contains operand0/1 hit rates for L1/L2/L3)
            CacheHitRates cache_hits;
            
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

        void setProblem(ProblemInfo p);
        void setSolution(SizeMapping sm);
        void setHardware(hardware_t::architecture_t arch);
        HardwareConstants getHardwareConstants(const hardware_t::architecture_t arch) const;
        void calculateStorePerformance(double M,
                                       double N,
                                       double NumBatches,
                                       double MT0,
                                       double MT1,
                                       uint32_t GWVWD,
                                       uint32_t bpeD,
                                       const HardwareConstants& hw_consts,
                                       uint32_t WGs_per_tile,
                                       uint32_t WGs_per_tile_XCD,
                                       double &store,
                                       double &store_edge) const;
        double calculateGSUOverhead(double M, double N, double K,
                                    double NumBatches, double GlobalSplitU,
                                    uint32_t gsuMethod, ProblemInfo problem,
                                    const HardwareConstants& hw_consts,
                                    uint32_t WGs_per_tile, uint32_t WGs_per_tile_XCD,
                                    double MT0, double MT1, uint32_t numWGs, double vgprCheck,
                                    double storeGSU) const;
        double calculateLSUOverhead(double MT0, double MT1, double lsu,
                                    uint32_t svw, uint32_t numThreads,
                                    ProblemInfo problem,
                                    const HardwareConstants& hw_consts) const;
        MemoryAccessCosts
        calculateMemoryAccessCosts(double MT0, double MT1,
                                   const HardwareConstants& hw,
                                   const CacheHitRates& hr,
                                   double L2BandWidthPerCU, double L3BandWidthPerCU, double HBMBandWidthPerCU,
                                   bool isSwizzleA, bool isSwizzleB,
                                   double A_L1_req, double B_L1_req,
                                   double A_L2_req, double A_L3_req, double A_hbm_req,
                                   double B_L2_req, double B_L3_req, double B_hbm_req) const;
        double getLoopOverall(const MemoryAccessCosts& mem, double math, uint32_t loopCnt, double pgr) const;
        PredictedPerformance predictedPerformance() const;
        
        // New refactored functions
        IntermediatePerformanceMetrics calculateIntermediateMetrics() const;
        PredictedPerformance calculateFinalPerformance(const IntermediatePerformanceMetrics& metrics) const;
        L1CacheHitRate
        computeL1CacheHitRate(const HardwareConstants& hw,
                            double MT0, double MT1, uint32_t bpeA, uint32_t bpeB,
                            int NTA, int NTB, uint32_t GRVWA, uint32_t GRVWB,
                            bool DTVA, bool DTVB, bool isSwizzleA, bool isSwizzleB,
                            uint32_t VWA, uint32_t VWB, bool transA, bool transB,
                            double lda, double ldb, int NLCA, int NLCB,
                            uint32_t threadnum, uint32_t NumWave0, uint32_t NumWave1) const;
        L2CacheHitRate computeL2CacheHitRate(uint32_t M,
                                             uint32_t N,
                                             uint32_t K,
                                             const HardwareConstants& hw,
                                             uint32_t gsu,
                                             int32_t  wgm,
                                             uint32_t batches,
                                             uint32_t bpeA,
                                             uint32_t bpeB,
                                             int32_t  NTA,
                                             int32_t  NTB,
                                             bool     isGSUWGMRR) const;
        L3CacheHitRate
        computeL3CacheHitRate(double M, double N, double K, const HardwareConstants& hw,
                                          uint32_t bpeA, uint32_t bpeB, int NTA, int NTB,
                                          int N_WGs_total, int M_WGs_total, int N_WGs_per_tile, int M_WGs_per_tile) const;
        double resolveOccupancy(const HardwareConstants& hw, double perf, double prefetch, double mathCost, double storeCost, uint32_t num_tiles, uint32_t CUOccupancy) const;
        bool isBetter(ProblemInfo problem, TieBreakerInfo previousSolution) const;
        bool isBetter(TieBreakerInfo previousSolution, TieBreakerInfo currentSolution) const;
        TieBreakerInfo getTieBreakerInfo() const;

        bool isBetter(MinTieBreakerInfo previousSolution, MinTieBreakerInfo currentSolution) const;
        MinTieBreakerInfo getMinTieBreakerInfo() const;

        int checkLocalReadFIFOFull(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves, bool isStall, double bankConflict) const;
        int checkLocalReadFinished(int currentCycle, std::queue<int>& fifo, int numLR) const;
        int checkGlobalReadFIFOFull(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves, bool isStall) const;
        void pushLocalReadWrite(int currentCycle, std::queue<int>& fifo, int bpr, double bankConflict);
        void pushLocalRead(int currentCycle, std::queue<int>& fifo, int bpr, bool isGfx950);
        // Helper function to analyze bank conflicts from VGPR states
        BankConflictResult analyzeBankConflictsFromVGPR(
            const std::vector<std::unordered_map<std::string, int64_t>>& vgprState,
            const std::string& vgprLocalReadAddrA,
            const std::string& vgprLocalReadAddrB,
            int LocalReadBytesA,
            int LocalReadBytesB);
    public:
        SizeMapping sizeMapping;
        ProblemInfo problem;
        HardwareConstants hw_consts;
        // TODO- need this mutable member data? or we can just create a new one to return (i.e. MinTieBreakerInfo)
        //  (std::sort with this data casues seg.fault. Using the un-mutable version "MinTieBreakerInfo" works fine)
        mutable TieBreakerInfo perfInfo;
    };

    // Standalone tie-breaker comparison function for configuration selection
    // Returns true if config_a is better than config_b based on problem-specific heuristics
    // This function is used when prediction_mode is set to "accurate"
    /**
     * @brief Compare two configurations using Formocast tie-breaking heuristics
     * @param M Problem dimension M
     * @param N Problem dimension N
     * @param K Problem dimension K
     * @param batch Number of batches
     * @param mt0_a Config A's macro tile M dimension
     * @param mt1_a Config A's macro tile N dimension
     * @param du_a Config A's depth U (K dimension)
     * @param svw_a Config A's store vector width
     * @param mt0_b Config B's macro tile M dimension
     * @param mt1_b Config B's macro tile N dimension
     * @param du_b Config B's depth U (K dimension)
     * @param svw_b Config B's store vector width
     * @return true if config A is better than config B, false otherwise
     */
    bool compareConfigTieBreaker(
        double M, double N, double K, size_t batch,
        double mt0_a, double mt1_a, double du_a, int svw_a,
        double mt0_b, double mt1_b, double du_b, int svw_b
    );

} // namespace origami
