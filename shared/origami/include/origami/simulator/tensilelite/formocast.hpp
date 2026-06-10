// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace origami
{
    // Forward declarations
    class Formocast;

    namespace simulator
    {
        // Load request calculation functions

        /**
         * @brief Calculate the L1 cache load request for a matrix tile
         * @param MTX Matrix tile size in the X dimension
         * @param DU Depth unroll factor
         * @param L1CacheLineSize L1 cache line size in bytes
         * @param grvw Global read vector width
         * @param bpe Bytes per element
         * @param dtv DirectToVGPR flag (as integer)
         * @param isTransposed Whether the matrix is transposed
         * @param isSwizzled Whether the memory layout is swizzled
         * @param VW Vector width
         * @param L1BusWidthPerCU L1 bus width per compute unit
         * @param NumLoadsCoalesced Number of loads that can be coalesced
         * @param numWaveX Number of waves in X dimension
         * @param tcc_ea0_coalesced Output parameter for TCC EA0 coalesced value
         * @return The calculated load request value
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
                             double&  tcc_ea0_coalesced);

        inline double getL2LoadRequest(double   L1_req,
                                       double   L1_hit,
                                       double   tcc_ea0_coalsced) { return L1_req * (1 - L1_hit) / (tcc_ea0_coalsced >= 1 ? 1 : 2); }

        inline double getL3LoadRequest(double   L2_req,
                                       double   L2_hit,
                                       double   tcc_ea0_coalsced) { return L2_req * (1 - L2_hit) / std::ceil(tcc_ea0_coalsced); }

        inline double getHBMLoadRequest(double   L3_req,
                             double   L3_hit) { return L3_req * (1 - L3_hit); }

        // GSU overhead calculation functions

        /**
         * @brief Calculate memory-bound overhead for matrix operations with Split K for multiple buffers
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param GlobalSplitU Global split-K factor
         * @param NumBatches Number of batches
         * @param bpeCompute Bytes per element for computation
         * @param bpeD Bytes per element for output matrix D
         * @param hbmBandWidth HBM (High Bandwidth Memory) bandwidth
         * @param L1CacheLineSize L1 cache line size
         * @param NumCUs Number of compute units
         * @param boost_frequency Boost frequency
         * @param mem_frequency Memory frequency
         * @param L2WriteArbEff L2 write arbitration efficiency
         * @param L2ReadArbEff L2 read arbitration efficiency
         * @param L3BandWidth L3 cache bandwidth
         * @param L1BusWidthPerCU L1 bus width per CU
         * @param L2BusWidthPerCU L2 bus width per CU
         * @param L1WriteBusWidthPerCU L1 write bus width per CU
         * @param L2WriteBusWidthPerCU L2 write bus width per CU
         * @return The calculated memory-bound overhead of Split K for multiple buffers
         */
        double getMultipleBufferOverhead(double M, double N, double GlobalSplitU, double NumBatches,
                            uint32_t bpeCompute, uint32_t bpeD, double hbmBandWidth,
                            double L1CacheLineSize, double NumCUs, uint32_t num_tiles, uint32_t CUOccupancy, double boost_frequency,
                            double mem_frequency, double L2WriteArbEff, double L2ReadArbEff,
                            double L3BandWidth, double L1BusWidthPerCU, double L2BusWidthPerCU,
                            double L1WriteBusWidthPerCU, double L2WriteBusWidthPerCU);

        /**
         * @brief Calculate overhead for multiple buffer single kernel (StreamK) Global Split K approach
         * @param GlobalSplitU Global split-K factor
         * @param MT0 Macro tile dimension 0 (M dimension)
         * @param MT1 Macro tile dimension 1 (N dimension)
         * @param bpeCompute Bytes per element for computation
         * @param NumCUs Number of compute units
         * @param numWGs Total number of workgroups
         * @param num_tiles Number of tiles
         * @param CUOccupancy Target CU occupancy
         * @param boost_frequency Boost frequency in MHz
         * @param L2ReadArbEff L2 read arbitration efficiency
         * @param L1BusWidthPerCU L1 bus width per compute unit
         * @param L2BusWidthPerCU L2 bus width per compute unit
         * @param storeGSU Store overhead for Global Split K accumulation
         * @return The calculated overhead for MBSK (Multiple Buffer Single Kernel) approach
         */
        double getMultipleBufferSingleKernelOverhead(double GlobalSplitU, double MT0, double MT1, uint32_t bpeCompute,
                              double NumCUs, uint32_t numWGs, uint32_t num_tiles, uint32_t CUOccupancy, double boost_frequency,
                              double L2ReadArbEff, double L1BusWidthPerCU, double L2BusWidthPerCU,
                              double storeGSU);

        /**
         * @brief Calculate overhead for Local Split K operation
         * @param MT0 Macro tile dimension 0 (M dimension)
         * @param MT1 Macro tile dimension 1 (N dimension)
         * @param lsu Local Split K factor (split along K dimension within workgroup)
         * @param svw Store vector width
         * @param numThreads Number of threads per workgroup
         * @param bpeCompute Bytes per element for computation
         * @param math_frequency Math frequency in MHz
         * @return The calculated overhead for Local Split K accumulation and reduction
         */
        double getLocalSplitKOverhead(double MT0, double MT1, double lsu, uint32_t svw,
                             uint32_t numThreads, uint32_t bpeCompute, double math_frequency);

        // Cache hit rate calculation functions

        /** @brief Structure to hold L1 cache hit rates for both tiles */
        struct L1CacheHitRate {
            double tile0HitRate;  ///< Hit rate for tile 0
            double tile1HitRate;  ///< Hit rate for tile 1
        };

        /** @brief Structure to hold L2 cache hit rates including total and per-tile rates */
        struct L2CacheHitRate {
            double totalHitRate;  ///< Total hit rate
            double tile0HitRate;  ///< Hit rate for tile 0
            double tile1HitRate;  ///< Hit rate for tile 1
        };

        /** @brief Structure to hold L3 cache hit rates including total and per-tile rates */
        struct L3CacheHitRate {
            double totalHitRate;  ///< Total hit rate
            double tile0HitRate;  ///< Hit rate for tile 0
            double tile1HitRate;  ///< Hit rate for tile 1
        };

        /**
         * @brief Compute L1 cache hit rate for matrix operations
         * @param L1CacheCapacity L1 cache capacity in bytes
         * @param L1CacheLineSize L1 cache line size in bytes
         * @param L1BusWidthPerCU L1 bus width per compute unit
         * @param MT0 Macro tile dimension 0
         * @param MT1 Macro tile dimension 1
         * @param bpeA Bytes per element for matrix A
         * @param bpeB Bytes per element for matrix B
         * @param NTA Non-temporal access hint for matrix A
         * @param NTB Non-temporal access hint for matrix B
         * @param GRVWA Global read vector width for matrix A
         * @param GRVWB Global read vector width for matrix B
         * @param DTVA DirectToVGPR for matrix A
         * @param DTVB DirectToVGPR for matrix B
         * @param isSwizzleA Whether matrix A uses swizzled layout
         * @param isSwizzleB Whether matrix B uses swizzled layout
         * @param VWA Vector width for matrix A
         * @param VWB Vector width for matrix B
         * @param transA Whether matrix A is transposed
         * @param transB Whether matrix B is transposed
         * @param lda Leading dimension of matrix A
         * @param ldb Leading dimension of matrix B
         * @param NLCA Number of load cycles for matrix A
         * @param NLCB Number of load cycles for matrix B
         * @param threadnum Number of threads
         * @param NumWave0 Number of waves for dimension 0
         * @param NumWave1 Number of waves for dimension 1
         * @param isL1FourBank Whether L1 cache has four banks
         * @return L1CacheHitRate structure containing hit rates for both tiles
         */
        L1CacheHitRate computeL1CacheHitRate(double L1CacheCapacity, double L1CacheLineSize,
                                             double L1BusWidthPerCU, double MT0, double MT1, uint32_t depthU,
                                             uint32_t bpeA, uint32_t bpeB, int NTA, int NTB,
                                             uint32_t GRVWA, uint32_t GRVWB, bool DTVA, bool DTVB,
                                             bool isSwizzleA, bool isSwizzleB, uint32_t VWA, uint32_t VWB,
                                             bool transA, bool transB, double lda, double ldb,
                                             int NLCA, int NLCB, uint32_t threadnum,
                                             uint32_t NumWave0, uint32_t NumWave1, bool isL1FourBank);

        /**
         * @brief Compute L3 cache hit rate for matrix operations
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param K Matrix dimension K
         * @param L3CacheCapacity L3 cache capacity in bytes
         * @param NumCUs Number of compute units
         * @param bpeA Bytes per element for matrix A
         * @param bpeB Bytes per element for matrix B
         * @param NTA Non-temporal access hint for matrix A
         * @param NTB Non-temporal access hint for matrix B
         * @param N_WGs_total Total number of workgroups in N dimension
         * @param M_WGs_total Total number of workgroups in M dimension
         * @param N_WGs_per_tile Number of workgroups per tile in N dimension
         * @param M_WGs_per_tile Number of workgroups per tile in M dimension
         * @return L3CacheHitRate structure containing total and per-tile hit rates
         */
        L3CacheHitRate computeL3CacheHitRate(double M, double N, double K, double L3CacheCapacity,
                                             double NumCUs, uint32_t bpeA, uint32_t bpeB,
                                             int NTA, int NTB, int N_WGs_total, int M_WGs_total,
                                             int N_WGs_per_tile, int M_WGs_per_tile);

        /**
         * @brief Compute L2 cache hit rate for matrix operations
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param K Matrix dimension K
         * @param MT0 Macro tile dimension 0
         * @param MT1 Macro tile dimension 1
         * @param depthU Depth unroll factor
         * @param L2CacheCapacity L2 cache capacity in bytes
         * @param NumCUs Number of compute units
         * @param NumXCDs Number of XCDs (Extended Compute Dies)
         * @param XCC WorkGroupMapping-XCC factor
         * @param XCCG WorkGroupMapping-XCCG factor
         * @param gsu Global split-U factor
         * @param wgm Workgroup mapping strategy
         * @param batches Number of batches
         * @param bpeA Bytes per element for matrix A
         * @param bpeB Bytes per element for matrix B
         * @param NTA Non-temporal access hint for matrix A
         * @param NTB Non-temporal access hint for matrix B
         * @param isGSUWGMRR Whether using GSU WGM round-robin scheduling
         * @return L2CacheHitRate structure containing total and per-tile hit rates
         */
        L2CacheHitRate computeL2CacheHitRate(uint32_t M, uint32_t N, uint32_t K,
                                             uint32_t MT0, uint32_t MT1, uint32_t depthU,
                                             uint32_t L2CacheCapacity, uint32_t NumCUs, uint32_t NumXCDs,
                                             int XCC, int XCCG, uint32_t gsu, int32_t wgm,
                                             uint32_t batches, uint32_t bpeA, uint32_t bpeB, int32_t NTA,
                                             int32_t NTB, bool isGSUWGMRR);

        // Store request calculation functions

        /**
         * @brief Calculate L3 store request for matrix output
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param MT0 Macro tile dimension 0
         * @param MT1 Macro tile dimension 1
         * @param non_edge_req Output parameter for non-edge store requests
         * @param edge_req Output parameter for edge store requests
         * @return Total L3 store request value
         */
        double calculateStoreL3Request(double M, double N, double MT0, double MT1,
                                       double& non_edge_req, double& edge_req);

        /**
         * @brief Calculate L2 store request for matrix output
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param MT0 Macro tile dimension 0
         * @param MT1 Macro tile dimension 1
         * @param SVW Store vector width
         * @param non_edge_req Output parameter for non-edge store requests
         * @param edge_req Output parameter for edge store requests
         * @return Total L2 store request value
         */
        double calculateStoreL2Request(double M, double N, double MT0, double MT1, double SVW,
                                       double& non_edge_req, double& edge_req);

        /**
         * @brief Calculate L1 store request for matrix output
         * @param M Matrix dimension M
         * @param N Matrix dimension N
         * @param MT0 Macro tile dimension 0
         * @param MT1 Macro tile dimension 1
         * @param SVW Store vector width
         * @param non_edge_req Output parameter for non-edge store requests
         * @param edge_req Output parameter for edge store requests
         * @return Total L1 store request value
         */
        double calculateStoreL1Request(double M, double N, double MT0, double MT1, double SVW,
                                       double& non_edge_req, double& edge_req);

        // FIFO and queue simulation functions

        /**
         * @brief Get stall cycles when global read queue is full
         * @param currentCycle Current simulation cycle
         * @param fifo FIFO queue to check
         * @param bpRead Bytes per read operation
         * @param numWaves Number of waves
         * @param isStall Whether the pipeline is stalled
         * @param isSgprOffset Whether the SGPR offset is used
         * @return Stall cycles if FIFO is full, currentCycle otherwise
         */
        int getGlobalReadQueueFullStallCycles(int currentCycle, std::deque<int>& fifo, int bpRead, int numWaves, bool isStall, bool isSgprOffset);

        /**
         * @brief Get the cycle when local read operations complete
         * @param currentCycle Current simulation cycle
         * @param fifo FIFO queue to check
         * @param numLR Number of local reads
         * @return Cycle number if local reads not completed, currentCycle otherwise
         */
        int getLocalReadCompletionCycle(int currentCycle, std::queue<int>& fifo, int numLR);

        /**
         * @brief Get stall cycles when local read queue is full
         * @param currentCycle Current simulation cycle
         * @param fifo FIFO queue to check
         * @param bpRead Bytes per read operation
         * @param numWaves Number of waves
         * @param lrStallLatencyBuffer Local read stall latency buffer
         * @return Stall cycles if FIFO is full, currentCycle otherwise
         */
        int getLocalReadQueueFullStallCycles(int currentCycle, std::queue<int>& fifo, int bpRead, int numWaves, int lrStallLatencyBuffer);

        /**
         * @brief Calculate local read latency considering bank conflicts
         * @param baseLatency Base latency without conflicts
         * @param conflictMultiplier Multiplier for bank conflict penalty
         * @param bankConflict Bank conflict rate
         * @return Total latency including bank conflict penalty
         */
        int getLocalReadLatency(int baseLatency, int conflictMultiplier, double bankConflict);

        /**
         * @brief Calculate local write latency considering bank conflicts
         * @param baseLatency Base latency without conflicts
         * @param conflictMultiplier Multiplier for bank conflict penalty
         * @param bankConflict Bank conflict rate
         * @return Total latency including bank conflict penalty
         */
         int getLocalWriteLatency(int baseLatency, int conflictMultiplier, double bankConflict);

        /**
         * @brief Analyze bank conflicts from VGPR state
         * @param vgprState Vector of VGPR state maps for each thread
         * @param vgprLocalReadAddrA VGPR register name for local read address A
         * @param NUM_THREADS_TO_SIMULATE Number of threads to simulate
         * @param NUM_BANKS Number of memory banks
         * @param BANK_WIDTH Width of each memory bank in bytes
         * @param LocalReadBytesA Number of bytes to read for matrix A
         * @return Bank conflict rate (0.0 = no conflicts, 1.0 = maximum conflicts)
         */
        double analyzeBankConflictsFromVGPR(
            const std::vector<std::unordered_map<std::string, int64_t>>& vgprState,
            const std::string& vgprLocalReadAddrA,
            int NUM_THREADS_TO_SIMULATE,
            int NUM_BANKS,
            int BANK_WIDTH,
            int LocalReadBytesA);

    } // namespace simulator
} // namespace origami

