// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <queue>
#include <tuple>
#include <iostream>
#include <unordered_map>
#include <origami/simulator/tensilelite/formocast.hpp>
#include "origami/types.hpp"
#include "origami/hardware.hpp"

namespace origami
{
    /**
     * @brief Formocast performance prediction simulator for GPU GEMM operations
     *
     * This class provides detailed performance prediction for GPU matrix multiplication
     * operations by simulating cache behavior, memory access patterns, and compute
     * throughput. It models various optimization techniques including Global Split K,
     * Local Split K, and different workgroup mapping strategies.
     *
     * This simulator is specifically designed for kernels generated from TensileLite,
     * capturing the performance characteristics and optimization strategies used in
     * TensileLite-generated GEMM kernels.
     */
    class Formocast
    {
    public:
        /**
         * @brief Configuration parameters for matrix tile sizes and memory access patterns
         *
         * Contains all the tuning parameters that define how a GEMM kernel will execute,
         * including tile dimensions, vector widths, and various optimization flags.
         */
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

            int  workGroupMappingXCC                    = 1;
            int  workGroupMappingXCCGroup               = -1;
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

        /**
         * @brief L2 cache hit rates for matrix operands
         */
         struct L2CacheHitRate
         {
             double tile0HitRate          = 0.0;
             double tile1HitRate          = 0.0;
             double totalHitRate          = 0.0;
         };

         /** @brief L1 cache hit rates, inherits structure from L2CacheHitRate */
         struct L1CacheHitRate : public L2CacheHitRate{};

         /** @brief L3 cache hit rates, inherits structure from L2CacheHitRate */
         struct L3CacheHitRate : public L2CacheHitRate{};

         /**
          * @brief Bank conflict analysis results for LDS (Local Data Share) accesses
          */
         struct BankConflictResult {
             double ratioA = 1.0;  ///< Bank conflict ratio for matrix A (1.0 = no conflicts)
             double ratioB = 1.0;  ///< Bank conflict ratio for matrix B (1.0 = no conflicts)
         };

        /**
         * @brief Cache hit rates for all cache levels and both matrix operands
         */
         struct CacheHitRates
         {
             L1CacheHitRate L1_hit;
             L2CacheHitRate L2_hit;
             L3CacheHitRate L3_hit;
         };

        /**
         * @brief Memory access costs at different cache levels
         *
         * Stores the calculated cost (in cycles) for memory accesses at each level
         * of the memory hierarchy, used for overall performance prediction.
         */
         struct MemoryAccessCosts
         {
             double mem_l1_req;
             double mem_l2_req;
             double mem_l3_req;
             double mem_hbm_req;
             double mem_loop_l1_req;
             double mem_loop_l2_req;
             double mem_loop_l3_req;
             double mem_loop_hbm_req;
             double l1_hit;
             double l2_hit;
             double l3_hit;
             double mem_overall;
             //for debug
             double A_mem_l1_req;
             double B_mem_l1_req;
             double tcc_ea0_coalscedA;
             double tcc_ea0_coalscedB;
             double MT_A_L1_req;
             double MT_B_L1_req;
             double MT_A_L2_req;
             double MT_B_L2_req;
             double MT_A_L3_req;
             double MT_B_L3_req;
             double MT_A_hbm_req;
             double MT_B_hbm_req;

             CacheHitRates cache_hits;

             // for == compare, can remove this if we are using MinTieBreakerInfo
             bool operator==(MemoryAccessCosts const &rhs) const
             {
                 return std::tie(mem_l1_req, mem_l2_req, mem_l3_req, mem_hbm_req, l1_hit, l2_hit, l3_hit, mem_overall, MT_A_L1_req, MT_B_L1_req, MT_A_L2_req, MT_B_L2_req) ==
                        std::tie(rhs.mem_l1_req, rhs.mem_l2_req, rhs.mem_l3_req, rhs.mem_hbm_req, rhs.l1_hit, rhs.l2_hit, rhs.l3_hit, rhs.mem_overall, rhs.MT_A_L1_req, rhs.MT_B_L1_req, rhs.MT_A_L2_req, rhs.MT_B_L2_req);
             };
         };

        /**
         * @brief Predicted performance results for a GEMM configuration
         */
        struct PredictedPerformance
        {
            double   microSeconds = 0.0;  ///< Predicted execution time in microseconds
            double   hitRate      = 0.0;  ///< Overall L2 cache hit rate
            double   MT0               = 0.0;
            double   MT1               = 0.0;
            double   PGR               = 0.0;
            double   depthU            = 0.0;
            double   NumCUs            = 0.0;
            double   WorkGroupMapping  = 0.0;
            double   CUOccupancy       = 0.0;
            double   GlobalSplitU      = 0.0;
            double   LocalSplitU       = 0.0;
            double   loopCnt           = 0.0;

            double   init         = 0.0;
            double   preloop      = 0.0;
            double   loop         = 0.0;
            double   math_overall = 0.0;
            double   mem_overall  = 0.0;
            MemoryAccessCosts memCosts;
            double   tail         = 0.0;
            double   store        = 0.0;
            double   gsu          = 0.0;
            double   lsu          = 0.0;
            double   num_tiles    = 0.0;
            double   perf         = 0.0;
        };

        /**
         * @brief Hardware-specific constants and capabilities
         *
         * Contains architecture-specific parameters for cache sizes, bandwidths,
         * frequencies, and latencies used in performance modeling.
         */
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
            uint32_t LocalWriteBaseLatencyB128;
            uint32_t LocalWriteBaseLatencyB64;
            uint32_t LocalWriteBaseLatencyB32;
            uint32_t LocalWriteConflictMultiplierB128;
            uint32_t LocalWriteConflictMultiplierB64;
            uint32_t LocalWriteConflictMultiplierB32;
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

        /**
         * @brief Detailed performance breakdown for tie-breaking between configurations
         *
         * Contains comprehensive performance metrics including memory costs, compute costs,
         * and tile dimensions. Used when multiple configurations have similar predicted
         * performance and need fine-grained comparison.
         */
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

        /**
         * @brief Minimal tie-breaker information for sorting configurations
         *
         * Immutable version of TieBreakerInfo containing only essential parameters.
         * Used for std::sort operations to avoid issues with mutable member variables.
         */
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

        /**
         * @brief Problem specification for matrix multiplication
         *
         * Defines the matrix dimensions, data types, and layout properties
         * for the GEMM operation being analyzed.
         */
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

        /**
         * @brief Set the problem specification for simulation
         * @param p Problem information containing matrix dimensions and data types
         */
        void setProblem(ProblemInfo p);

        /**
         * @brief Set the solution configuration for simulation
         * @param sm Size mapping containing tile sizes and optimization parameters
         */
        void setSolution(SizeMapping sm);

        /**
         * @brief Set the hardware architecture for simulation
         * @param arch GPU architecture identifier (e.g., gfx942, gfx950)
         */
        void setHardware(hardware_t::architecture_t arch);

        /**
         * @brief Get hardware constants for a specific architecture
         * @param arch GPU architecture identifier
         * @return HardwareConstants structure with architecture-specific parameters
         */
        HardwareConstants getHardwareConstants(const hardware_t::architecture_t arch) const;

        /**
         * @brief Calculate store (write-back) performance for matrix output
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param num_tiles Number of tiles
         * @param NumBatches Number of batches
         * @param MT0 Macro tile dimension 0 (M dimension)
         * @param MT1 Macro tile dimension 1 (N dimension)
         * @param GWVWD Global write vector width for matrix D
         * @param bpeD Bytes per element for output matrix D
         * @param hw_consts Hardware constants
         * @param WGs_per_tile Workgroups per tile
         * @param WGs_per_tile_XCD Workgroups per tile per XCD (chiplet)
         * @param store Output parameter for store cost
         * @param store_edge Output parameter for edge case store cost
         */
         double calculateStorePerformance(double M,
                                       double N,
                                       double num_tiles,
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

        /**
         * @brief Calculate overhead for Global Split U (split along K dimension across workgroups)
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param K Matrix dimension K
         * @param NumBatches Number of batches
         * @param GlobalSplitU Global split-K factor
         * @param gsuMethod Global Split U method (2=MultiBuffer, 3=MultiBufferSingleKernel)
         * @param problem Problem specification
         * @param hw_consts Hardware constants
         * @param num_tiles Number of tiles
         * @param CUOccupancy Target CU occupancy
         * @param WGs_per_tile Workgroups per tile
         * @param WGs_per_tile_XCD Workgroups per tile per XCD
         * @param MT0 Macro tile dimension 0
         * @param MT1 Macro tile dimension 1
         * @param numWGs Total number of workgroups
         * @param vgprCheck VGPR usage check value
         * @param storeGSU Store overhead for GSU accumulation
         * @return Calculated Global Split U overhead in cycles
         */
        double calculateGlobalSplitUOverhead(double M, double N, double K,
                                    double NumBatches, double GlobalSplitU,
                                    uint32_t gsuMethod, ProblemInfo problem,
                                    const HardwareConstants& hw_consts,
                                    uint32_t num_tiles, uint32_t CUOccupancy,
                                    uint32_t WGs_per_tile, uint32_t WGs_per_tile_XCD,
                                    double MT0, double MT1, uint32_t numWGs, double vgprCheck,
                                    double storeGSU) const;

        /**
         * @brief Calculate overhead for Local Split U (split along K dimension within workgroup)
         * @param MT0 Macro tile dimension 0 (M dimension)
         * @param MT1 Macro tile dimension 1 (N dimension)
         * @param lsu Local Split U factor
         * @param svw Store vector width
         * @param numThreads Number of threads per workgroup
         * @param problem Problem specification
         * @param hw_consts Hardware constants
         * @return Calculated Local Split U overhead in cycles
         */
        double calculateLocalSplitUOverhead(double MT0, double MT1, double lsu,
                                    uint32_t svw, uint32_t numThreads,
                                    ProblemInfo problem,
                                    const HardwareConstants& hw_consts) const;

        /**
         * @brief Calculate memory access costs at all cache levels
         * @param MT0 Macro tile dimension 0
         * @param MT1 Macro tile dimension 1
         * @param hw Hardware constants
         * @param hr Cache hit rates
         * @param L2BandWidthPerCU L2 bandwidth per compute unit
         * @param L3BandWidthPerCU L3 bandwidth per compute unit
         * @param HBMBandWidthPerCU HBM bandwidth per compute unit
         * @param isSwizzleA Whether matrix A uses swizzled layout
         * @param isSwizzleB Whether matrix B uses swizzled layout
         * @param A_L1_req Matrix A L1 request count
         * @param B_L1_req Matrix B L1 request count
         * @param A_L2_req Matrix A L2 request count
         * @param A_L3_req Matrix A L3 request count
         * @param A_hbm_req Matrix A HBM request count
         * @param B_L2_req Matrix B L2 request count
         * @param B_L3_req Matrix B L3 request count
         * @param B_hbm_req Matrix B HBM request count
         * @return MemoryAccessCosts structure with costs at each cache level
         */
        MemoryAccessCosts
        calculateMTMemoryAccessCosts(double M, double N, double K_AfterGSU,
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
                                   uint32_t N_WGs_per_tile, uint32_t M_WGs_per_tile, uint32_t num_tiles) const;

        /**
         * @brief Calculate overall loop performance combining memory and compute costs
         * @param mem Memory access costs structure
         * @param math Math computation cost
         * @param loopCnt Number of loop iterations
         * @param pgr Prefetch global read parameter
         * @return Overall loop performance cost in cycles
         */
        double getLoop_time(MemoryAccessCosts& mem, double math, uint32_t loopCnt, double pgr, uint32_t num_tiles, bool large) const;

        double calculateInitialCost(double num_tiles) const;

        /**
         * @brief Predict overall performance for the configured problem and solution
         *
         * @return PredictedPerformance structure with execution time and cache hit rates
         */
        PredictedPerformance predictedPerformance() const;

        /**
         * @brief Compute L1 cache hit rates for both matrix operands
         * @param hw Hardware constants
         * @param MT0 Macro tile dimension 0
         * @param MT1 Macro tile dimension 1
         * @param bpeA Bytes per element for matrix A
         * @param bpeB Bytes per element for matrix B
         * @param NTA Non-temporal hint for matrix A
         * @param NTB Non-temporal hint for matrix B
         * @param GRVWA Global read vector width for matrix A
         * @param GRVWB Global read vector width for matrix B
         * @param DTVA DirectToVGPR flag for matrix A
         * @param DTVB DirectToVGPR flag for matrix B
         * @param isSwizzleA Whether matrix A uses swizzled layout
         * @param isSwizzleB Whether matrix B uses swizzled layout
         * @param VWA Vector width for matrix A
         * @param VWB Vector width for matrix B
         * @param transA Whether matrix A is transposed
         * @param transB Whether matrix B is transposed
         * @param lda Leading dimension of matrix A
         * @param ldb Leading dimension of matrix B
         * @param NLCA Number of loads in MT0 dimension
         * @param NLCB Number of loads in MT1 dimension
         * @param threadnum Number of threads
         * @param NumWave0 Number of waves in dimension 0
         * @param NumWave1 Number of waves in dimension 1
         * @return L1CacheHitRate structure with hit rates for both matrices
         */
        L1CacheHitRate
        computeL1CacheHitRate(const HardwareConstants& hw,
                            double MT0, double MT1, uint32_t depthU, uint32_t bpeA, uint32_t bpeB,
                            int NTA, int NTB, uint32_t GRVWA, uint32_t GRVWB,
                            bool DTVA, bool DTVB, bool isSwizzleA, bool isSwizzleB,
                            uint32_t VWA, uint32_t VWB, bool transA, bool transB,
                            double lda, double ldb, int NLCA, int NLCB,
                            uint32_t threadnum, uint32_t NumWave0, uint32_t NumWave1) const;

        /**
         * @brief Compute L2 cache hit rates considering workgroup distribution
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param K Matrix dimension K
         * @param hw Hardware constants
         * @param workgroup mapping xcc
         * @param workgroup mapping xcc group
         * @param gsu Global Split U factor
         * @param wgm Workgroup mapping strategy
         * @param batches Number of batches
         * @param bpeA Bytes per element for matrix A
         * @param bpeB Bytes per element for matrix B
         * @param NTA Non-temporal hint for matrix A
         * @param NTB Non-temporal hint for matrix B
         * @param isGSUWGMRR Whether using GSU workgroup mapping round-robin
         * @return L2CacheHitRate structure with hit rates
         */
        L2CacheHitRate computeL2CacheHitRate(uint32_t M,
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
                                             bool     isGSUWGMRR) const;

        /**
         * @brief Compute L3 cache hit rates based on tile reuse patterns
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param K Matrix dimension K
         * @param hw Hardware constants
         * @param bpeA Bytes per element for matrix A
         * @param bpeB Bytes per element for matrix B
         * @param NTA Non-temporal hint for matrix A
         * @param NTB Non-temporal hint for matrix B
         * @param N_WGs_total Total workgroups in N dimension
         * @param M_WGs_total Total workgroups in M dimension
         * @param N_WGs_per_tile Workgroups per tile in N dimension
         * @param M_WGs_per_tile Workgroups per tile in M dimension
         * @return L3CacheHitRate structure with hit rates
         */
        L3CacheHitRate
        computeL3CacheHitRate(double M, double N, double K, const HardwareConstants& hw,
                                          uint32_t bpeA, uint32_t bpeB, int NTA, int NTB,
                                          int N_WGs_total, int M_WGs_total, int N_WGs_per_tile, int M_WGs_per_tile) const;

        /**
         * @brief Resolve compute unit occupancy based on performance and resource constraints
         * @param hw Hardware constants
         * @param perf Predicted performance
         * @param prefetch Prefetch cost
         * @param mathCost Math computation cost
         * @param storeCost Store operation cost
         * @param num_tiles Number of tiles
         * @param CUOccupancy Target CU occupancy
         * @return Resolved occupancy value
         */
        double resolveOccupancy(const HardwareConstants& hw, double perf, double prefetch, double mathCost, double storeCost, uint32_t num_tiles, uint32_t CUOccupancy, uint32_t loopCnt) const;

        /**
         * @brief Compare if current configuration is better than previous solution
         * @param problem Problem specification
         * @param previousSolution Previous solution's performance info
         * @return true if current is better, false otherwise
         */
        bool isBetter(ProblemInfo problem, TieBreakerInfo previousSolution) const;

        /**
         * @brief Compare two configurations using tie-breaker metrics
         * @param previousSolution Previous configuration metrics
         * @param currentSolution Current configuration metrics
         * @return true if current is better, false otherwise
         */
        bool isBetter(TieBreakerInfo previousSolution, TieBreakerInfo currentSolution) const;

        /**
         * @brief Get tie-breaker information for current configuration
         * @return TieBreakerInfo structure with detailed performance metrics
         */
        TieBreakerInfo getTieBreakerInfo() const;

        /**
         * @brief Compare two minimal configurations (immutable version for sorting)
         * @param previousSolution Previous minimal configuration
         * @param currentSolution Current minimal configuration
         * @return true if current is better, false otherwise
         */
        bool isBetter(MinTieBreakerInfo previousSolution, MinTieBreakerInfo currentSolution) const;

        /**
         * @brief Get minimal tie-breaker information (immutable version)
         * @return MinTieBreakerInfo structure with essential parameters
         */
        MinTieBreakerInfo getMinTieBreakerInfo() const;

        /**
         * @brief Check if local read FIFO is full considering bank conflicts
         * @param currentCycle Current simulation cycle
         * @param fifo FIFO queue to check
         * @param bpRead Bytes per read operation
         * @param numWaves Number of waves
         * @param isStall Whether pipeline is stalled
         * @param bankConflict Bank conflict rate
         * @return Stall cycles if FIFO is full, 0 otherwise
         */
        int getLocalReadQueueFullStallCycles(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves, bool isStall, double bankConflict) const;

        /**
         * @brief Check if local write FIFO is full
         * @param currentCycle Current simulation cycle
         * @param issueCycles Issue cycles for the write operation
         * @param bpWrite Bytes per write operation
         * @param numWaves Number of waves
         * @return The Cycle this instruction can be issued.
         */
        int getLocalWriteQueueFullStallCycles(int currentCycle, int previousLW, int issueCycles, int bpWrite, int numWaves) const;

        /**
         * @brief Check if local read operations have finished
         * @param currentCycle Current simulation cycle
         * @param fifo FIFO queue to check
         * @param numLR Number of local reads
         * @return Cycle when operations finish
         */
        int getLocalReadCompletionCycle(int currentCycle, std::queue<int>& fifo, int numLR) const;

        /**
         * @brief Check if global read FIFO is full
         * @param currentCycle Current simulation cycle
         * @param fifo FIFO queue to check
         * @param bpRead Bytes per read operation
         * @param numWaves Number of waves
         * @param isStall Whether pipeline is stalled
         * @param isSgprOffset Whether the SGPR offset is used
         * @return Stall cycles if FIFO is full, 0 otherwise
         */
        int getGlobalReadQueueFullStallCycles(int currentCycle, std::deque<int>& fifo, int bpRead, int numWaves, bool isStall, bool isSgprOffset) const;

        /**
         * @brief Push a local read-write operation into FIFO
         * @param currentCycle Current simulation cycle
         * @param fifo FIFO queue
         * @param bpr Bytes per read operation
         * @param bankConflict Bank conflict rate
         * @param isLocalRead Whether this is a local read operation
         * @param numPreviousLRs Number of previous local reads
         */
        void pushLocalReadWrite(int currentCycle, std::queue<int>& fifo, int bpr, double bankConflict, bool isLocalRead, int numPreviousLRs);

        /**
         * @brief Analyze bank conflicts from VGPR states for both matrix operands
         * @param vgprState Vector of VGPR state maps for threads
         * @param vgprLocalReadAddrA VGPR register for local read address A
         * @param vgprLocalReadAddrB VGPR register for local read address B
         * @param LocalReadBytesA Bytes to read for matrix A
         * @param LocalReadBytesB Bytes to read for matrix B
         * @return BankConflictResult with conflict ratios for both matrices
         */
        BankConflictResult analyzeBankConflictsFromVGPR(
            const std::vector<std::unordered_map<std::string, int64_t>>& vgprState,
            const std::string& vgprLocalReadAddrA,
            const std::string& vgprLocalReadAddrB,
            int LocalReadBytesA,
            int LocalReadBytesB);

    public:
        SizeMapping sizeMapping;           ///< Current kernel configuration
        ProblemInfo problem;               ///< Current problem specification
        HardwareConstants hw_consts;       ///< Hardware constants for current architecture

        /// Performance information for tie-breaking (mutable for caching)
        /// Note: Using MinTieBreakerInfo is preferred for std::sort to avoid segmentation faults
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
