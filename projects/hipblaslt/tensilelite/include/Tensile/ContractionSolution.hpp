/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Tensile/Tensile.hpp>

#include <Tensile/Activation.hpp>
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/Predicates.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/Utils.hpp>

#include "origami/origami.hpp"
#include "origami/streamk.hpp"

#include <tensilelitehost/export.h>

#define TENSILE_COMMON_KERNEL_ARGS_SIZE 16

namespace TensileLite
{
    // Elements in one GSU (MBSK) reduction region. Usage there is
    // synchronizerSizePerWG * numTiles * batch, which runs into the tens of
    // thousands, so this keeps the size it has always had and a solution whose
    // usage exceeds it is not selected (SynchronizerSizeCheck, both in
    // ContractionSolution and in the predicate of the same name).
    //
    // Must stay in sync with _rocblaslt_handle::c_syncGsuSlotElements.
    constexpr uint32_t GsuSynchronizerElements = 409600;

    // Problems a grouped GEMM can be given private regions for. A wider group
    // cannot be isolated per problem, so no solution that uses these flags may
    // be selected for it.
    //
    // Must stay in sync with _rocblaslt_handle::c_syncGsuSlots.
    constexpr uint32_t SynchronizerGroupedSlots = 16;

    // Elements in one Stream-K flag region. Stream-K indexes its flags by
    // workgroup id, so this is the largest skGrid that fits; gfx950 picks 224.
    // A grid past this would write beyond its own region, so getSKGrid clamps
    // against it.
    //
    // Must stay in sync with _rocblaslt_handle::c_syncSkSlotElements.
    constexpr uint32_t StreamKFlagElements = 2048;

    template <typename TAct>
    struct DeviceUserArguments
    {
        uint32_t m;
        uint32_t n;
        uint32_t batch;
        uint32_t k;
        void*    d;
        void*    c;
        void*    a;
        void*    b;
        uint32_t strideD1;
        uint32_t strideD2;
        uint32_t strideC1;
        uint32_t strideC2;
        uint32_t strideA1;
        uint32_t strideA2;
        uint32_t strideB1;
        uint32_t strideB2;
        int8_t   alpha[16];
        int8_t   beta[16];
        void*    scaleA;
        void*    scaleB;
        void*    scaleC;
        void*    scaleD;
        void*    scaleAlphaVec;
        void*    bias;
        int      biasType;
        uint32_t reserved;
        void*    e;
        uint32_t strideE1;
        uint32_t strideE2;
        TAct     act0;
        TAct     act1;
        int      activationType;
    } __attribute__((packed));

    struct PerfModel
    {
        double clock            = std::numeric_limits<double>::quiet_NaN();
        double memClock         = std::numeric_limits<double>::quiet_NaN();
        double peakGFlops       = std::numeric_limits<double>::quiet_NaN();
        double memBandwidthMBps = std::numeric_limits<double>::quiet_NaN();
        double l2ReadBwMul      = std::numeric_limits<double>::quiet_NaN();
        double gFlops           = std::numeric_limits<double>::quiet_NaN();
        double readEff          = 0.0;
        double l2ReadHitRate    = 0.0;
        double l2WriteHitRate   = 0.0;
        int    CUs              = 0;
    };

    extern TENSILELITEHOST_EXPORT PerfModel perf;

    struct BufferLoadCheckPacket
    {
        size_t shiftPtrElemA;
        size_t shiftPtrElemB;
        size_t depthUorMT0;
        size_t depthUorMT1;
    };

    struct SizeMapping
    {
        size_t waveNum;

        dim3 clusterDim{1, 1, 1};

        dim3 workGroupSize;
        dim3 threadTile;
        dim3 macroTile;

        std::array<int, 4> matrixInstruction;
        size_t             grvwA = 1;
        size_t             grvwB = 1;
        size_t             gwvwC = 1;
        size_t             gwvwD = 1;

        size_t  staggerU           = 0;
        size_t  staggerUMapping    = 0;
        size_t  depthU             = 0;
        size_t  globalSplitUPGR    = 0;
        int16_t globalSplitU       = 0;
        size_t  staggerStrideShift = 0;
        int     workGroupMapping   = 0;

        size_t packBatchDims              = 0;
        int    packSummationDims          = 0;
        int    magicDivAlg                = 1;
        int    streamK                    = 0;
        int    streamKForceDPOnly         = 0;
        int    streamKAtomic              = 0;
        int    prefetchAcrossPersistent   = 0;
        int    persistentKernel           = 0;
        bool   persistentKernelAlongBatch = false;

        bool sourceKernel = false;

        int    globalAccumulation       = 0;
        int    adaptiveGemmGSUA         = 0;
        size_t workspaceSizePerElemC    = 0;
        size_t workspaceSizePerElemBias = 0;

        bool activationFused = true;

        std::string customKernelName;

        int  workGroupMappingXCC                    = 0;
        int  workGroupMappingXCCGroup               = 0;
        bool globalSplitUCoalesced                  = false;
        bool globalSplitUWorkGroupMappingRoundRobin = false;

        int CUOccupancy            = 0;
        int PrefetchGlobalRead     = 2;
        int MathClocksUnrolledLoop = 0;

        size_t synchronizerSizePerWG = 0;

        int nonTemporalA = 0;
        int nonTemporalB = 0;

        int adaptiveGemmNTAB = 0;

        int customMainLoopScheduling = 0;

        // Whether the kernel uses the subtile implementation (UseSubtileImpl).
        // Plumbed into the Origami config so heuristics can reason about subtile
        // kernels (e.g. rejecting them for small K).
        bool useSubtileImpl = false;

        int NonTemporalD = 0;
        int WaveSeparateGlobalReadA = 0;
        int WaveSeparateGlobalReadB = 0;
        int UnrollLoopSwapGlobalReadOrder = 0;
        bool DirectToVgprA = false;
        bool DirectToVgprB = false;
        int NumLoadsCoalescedA = 0;
        int NumLoadsCoalescedB = 0;
        int VectorWidthA = 1;
        int VectorWidthB = 1;
        int LocalSplitU = 1;
        bool DirectToLdsA = false;
        bool DirectToLdsB = false;

        int expertSchedulingMode = 0;

        std::array<int, 2> waveGroup;
    };

    struct CustomKernel
    {
        std::string name;
        bool        generated = false;
    };

    struct StreamKSettings
    {
        origami::reduction_t reduction = origami::reduction_t::tree;
        size_t               grid      = 0;
        // StreamK=5 tri-state (0=OFF default/SK3, 1=ON/SK4, 2=AUTO); see
        // hipblasLtStreamKTileSchedulingMode_t. Ignored when streamK != 5.
        int                  streamKTileSchedulingMode = 0;
        int                  smCountTarget = 0; // 0 = use all device CUs; >0 engages origami heuristic when mode is OFF
    };

    struct GSUSettings
    {
        size_t globalAccumulation = 0;
    };

    /**
     * The three numbers the static two-tile StreamK ABI packs, plus the
     * leftover the kernel recomputes from them.
     *
     * StreamK=3 and the static sub-mode of StreamK=5 both pack SKItersPerWG,
     * skGrid and skTiles, and the device derives everything else from those:
     * a flat iteration space of tiles*itersPerTile, a data-parallel prefix of
     * (tiles - skTiles) whole tiles, and a StreamK region cut into skGrid
     * chunks. extraIters is the leftover skTiles*itersPerTile -
     * SKItersPerWG*skGrid under the historical global first-E mapping;
     * kernels with InternalArgsSupport::perTileExtraIters may redistribute
     * those extras within each tile when skGrid % skTiles == 0
     * (StreamK.py skAssignIters).
     */
    struct StreamKStaticSplit
    {
        uint32_t skTiles      = 0;
        uint32_t skItersPerWG = 0;
        uint32_t extraIters   = 0;
    };

    /**
     * Compute the static two-tile StreamK split.
     *
     * Single source of truth for arithmetic that used to be written out twice
     * in ContractionSolution.cpp (the StreamK=3 packer and the StreamK=5 static
     * packer) and is now also read by checkUniformSummationOrder(), which
     * proves a property of exactly this split. A third copy would let the gate
     * silently start proving a property of a split the kernel does not perform.
     *
     * @param tiles        Batch-inclusive tile count, getNumTiles(sizeMapping, 1).
     * @param itersPerTile getItersPerTile(sizeMapping), clamped to at least 1.
     * @param skGrid       Resolved StreamK grid. 0 yields an all-zero split.
     * @param skFullTiles  AMDGPU::skFullTiles (TENSILE_STREAMK_FULL_TILES).
     * @param forceDPOnly  SizeMapping::streamKForceDPOnly != 0.
     */
    TENSILELITEHOST_EXPORT StreamKStaticSplit streamKStaticSplit(
        size_t tiles, size_t itersPerTile, size_t skGrid, int skFullTiles, bool forceDPOnly);

    /**
     * Whether every output tile of a static two-tile StreamK split is folded
     * from the same ordered list of chunk lengths, and so is bitwise equal to
     * every other tile fed identical inputs.
     *
     * This is the row-uniformity condition
     * ContractionSolution::checkUniformSummationOrder() enforces for StreamK=3
     * and StreamK=5-static. It is a property of the packed split alone, so it
     * is insensitive to how tile indices map to (m-tile, n-tile) -- WGM,
     * SpaceFillingAlgo and XCC swizzling do not enter into it.
     *
     * @param split             The split streamKStaticSplit() produced.
     * @param tiles             The same batch-inclusive tile count fed to it.
     * @param itersPerTile      The same clamped iterations per tile fed to it.
     * @param skGrid            The resolved StreamK grid packed into the split.
     * @param perTileExtraIters True when the selected kernel redistributes
     *                          Stream-K extras within each tile
     *                          (InternalArgsSupport::perTileExtraIters).
     */
    TENSILELITEHOST_EXPORT bool streamKStaticSplitRowUniform(StreamKStaticSplit const& split,
                                                            size_t                    tiles,
                                                            size_t                    itersPerTile,
                                                            size_t                    skGrid            = 0,
                                                            bool                      perTileExtraIters = false);

    /**
     * Whether a resolved Stream-K launch may use parallel reduction under
     * uniform summation order.
     *
     * True when reduction is parallel, streamKAtomic is 0, the ABI is static
     * two-tile packing (StreamK=3, or StreamK=5 with dynamic sub-mode off),
     * and the grid is an exact multiple of the tile count with split factor
     * F = grid/tiles >= 2. Callers must already have cleared the static
     * Stream-K obstacles (the launch gate checks those first). Does not
     * consult StaggerU: the device clears it for F >= 2.
     *
     * Parallel extras are per PartialIdx and tile-symmetric, so I % F == 0 is
     * not required (unlike the tree all-partial model without per-tile extras).
     */
    TENSILELITEHOST_EXPORT bool streamKParallelReductionRowUniform(StreamKSettings const& sk,
                                                                   int  streamKAtomic,
                                                                   bool staticTwoTilePacking,
                                                                   size_t tiles);

    /**
     * Iteration range [start, end) assigned to workgroup w under the static
     * two-tile StreamK mapping. When perTileExtraIters is true and
     * skGrid % tiles == 0, extras are distributed within each tile;
     * otherwise the historical global first-E mapping is used.
     */
    struct StreamKWorkgroupIterRange
    {
        size_t start = 0;
        size_t end   = 0;
    };

    TENSILELITEHOST_EXPORT StreamKWorkgroupIterRange streamKWorkgroupIterRange(
        size_t w,
        size_t tiles,
        size_t itersPerTile,
        size_t skGrid,
        bool   perTileExtraIters);

    /**
     * Thrown when a launch requests uniform summation order but the resolved
     * kernel configuration is not row-uniform. A distinct type so the rocblaslt
     * host layer can map it to rocblaslt_status_invalid_value; a generic
     * exception would be swallowed and reported as an internal error.
     */
    class UniformSummationOrderError : public std::runtime_error
    {
    public:
        explicit UniformSummationOrderError(const std::string& what)
            : std::runtime_error(what)
        {
        }
    };

    // Snapshot of the StreamK launch-parameter DECISIONS for one
    // (solution, problem, hardware) triple. Populated by
    // ContractionSolution::computeStreamKDecisions() and consumed by solve().
    // Each field's provenance is annotated inline as one of:
    //   available  = read from the value the launch uses, or from the same real
    //                helper/param solve() uses, so it reflects the launch exactly.
    //   recomputed = re-derived here rather than read back from a launched value:
    //                either by mirroring the kernel-arg packing in makeArgs()
    //                (skTiles/skSplit/totalItems) or by deriving from other fields.
    //                Mirrors can drift silently if makeArgs() changes; each such
    //                field names what it mirrors.
    // This snapshot is not purely observational: reduction, skGrid and isDynamic are
    // launch INPUTS -- solve() copies them into StreamKSettings and uses isDynamic as
    // its dynamic-queue predicate. Only the PRINTING is optional; see
    // Debug::printStreamKLaunchSummary() (TENSILE_DB bit 0x200000).
    struct StreamKDecisions
    {
        // --- Mode ---
        // available: sizeMapping.streamK, the mode solve() uses (0 = not StreamK, else 3/4/5).
        int  streamKMode      = 0;
        // available: streamK5EffectiveDynamic(), the same helper solve()'s grid path uses
        // (SK5 resolved to the dynamic SK4 sub-path). Only meaningful for SK5: it stays
        // false for SK4 even though SK4 is unconditionally dynamic, so ask isDynamic
        // (below) -- not this -- whether a launch takes the dynamic path.
        bool effectiveDynamic = false;
        // recomputed: derived from streamKMode + effectiveDynamic (SK4, or SK5 resolved dynamic).
        // Launch-relevant rather than merely reported: solve() consumes this as the
        // dynamicQueuePath predicate guarding the work-stealing rejection.
        bool isDynamic        = false;

        // --- Reduction ---
        // available: solve() wires this exact value into sk.reduction -- it is the final
        // (post workspace-DP fallback) reduction the launch uses.
        origami::reduction_t reduction = origami::reduction_t::tree;

        // --- Grid / tiles / split ---
        // available: problem.getNumTiles(sizeMapping, 1), the same call solve() makes.
        size_t tiles            = 0;
        // available: getSKGridImpl out-param -- grid selected by its config/CU/override
        // logic BEFORE the "reset to tiles" clamps (tree-fixup bounds / ForceDPOnly
        // cluster multicast / workspace-DP).
        size_t selectedGrid       = 0;
        // available: getSKGridImpl return -- AFTER the tree-fixup-bounds fallback and
        // the ForceDPOnly cluster-multicast clamp, but BEFORE the workspace-DP
        // fallback in computeStreamKDecisions(). Diagnostic intermediate.
        size_t skGridPreFallback  = 0;
        // available: wired from sk.grid, the FINAL grid solve() launches with (after all
        // fallbacks, in the order they are applied: fixedGrid, treeBounds,
        // clusterDPMulticast, workspaceDP -- the same four names the report's
        // "changedBy" ladder uses, which lists them latest-clamp-wins).
        size_t finalGrid          = 0;
        // available: the grid solve() initialises sk.grid from. finalGrid and skGrid
        // always hold the same value -- computeStreamKDecisions() sets both to its final
        // grid, and solve() writes the launched sk.grid back into both. Both exist
        // because they are read for different reasons: skGrid is the launch INPUT
        // solve() consumes, finalGrid is the reported outcome printed against
        // selectedGrid.
        size_t skGrid             = 0;
        // The next three mirror the kernel-arg packing in makeArgs() (search for
        // "SKTiles"/"skTiles"), evaluated on the FINAL post-fallback grid and
        // reduction. They are hand-written mirrors: any change to the packing in
        // makeArgs() must be reflected in computeStreamKDecisions() or the report
        // silently drifts from the launch.
        //
        // recomputed: dynamic path  -> the SKTiles arg (debug override, else 0);
        //             parallel path -> the split factor, which is what makeArgs packs
        //                              into the skTiles slot on that path;
        //             static SK3    -> streamKStaticSplit().skTiles, the stream-k
        //                              (partial) tile count.
        size_t skTiles          = 0;
        // recomputed: dynamic path -> the SKSplit arg after the CeilDivide round-trip;
        //             parallel path -> grid / tiles;
        //             static SK3   -> 1 (no k-split; SK3 packs SKItersPerWG instead).
        size_t skSplit          = 0;
        // recomputed: work-item count. Dynamic path mirrors the packed TotalItems arg,
        //             (tiles - skTiles) + skTiles*skSplit. The SK3 ABI has no TotalItems
        //             arg, so on the parallel and static paths this is reported as
        //             'tiles' (one work item per output tile) rather than mirroring a
        //             packed value.
        size_t totalItems       = 0;

        // --- DP-only ---
        // The three flags below distinguish the source of a data-parallel-only launch:
        //   forceDPOnly              -> sizeMapping.streamKForceDPOnly compile-time param
        //   streamKDP                -> TENSILE_STREAMK_DATA_PARALLEL debug override
        //   workspaceDPFallbackFired -> runtime workspace-insufficient (below)
        // recomputed: OR of the three DP triggers above.
        bool dpOnly      = false;
        // available: sizeMapping.streamKForceDPOnly param.
        bool forceDPOnly = false;
        // available: Debug::useStreamKDataParrallel() (TENSILE_STREAMK_DATA_PARALLEL).
        bool streamKDP   = false;

        // --- Workspace / partials ---
        // recomputed: skTiles > 0.
        bool   partialsPresent        = false;
        // recomputed: requiredWorkspaceBytes > 0.
        bool   workspaceAllocated     = false;
        // recomputed: partials(+work-queue) bytes this launch reserves; 0 when no
        // partials are needed or the workspace-DP fallback fired.
        //
        // WARNING: this is NOT ContractionSolution::requiredWorkspaceSize()'s return
        // value, even though the names are close. requiredWorkspaceSize() is the
        // separate, caller-facing implementation of the reserve-or-not rule -- it is
        // what the allocator sizes the workspace buffer from -- and it computes the
        // answer differently: it always asks getSKReduction(), and for parallel
        // reduction it sizes with requiredWorkspaceSizeGsu(problem, hardware,
        // grid / tiles) instead of partialTileSize(grid).
        //
        // The two can therefore disagree about WHETHER a workspace is needed, not
        // just about how many bytes. The case where they do is parallel reduction
        // whose selected grid comes back equal to tiles: the k-split factor
        // grid / tiles is then 1, and requiredWorkspaceSizeGsu() short-circuits to 0
        // for a split of 1, while this field still reserves partialTileSize(tiles)
        // because parallel reduction always needs a partials region. Concretely, a
        // 256x4096x4096 SK3 launch with a 128x128x64 macro tile on a 256-CU device,
        // handed a workspace between 4 MiB and 16 MiB, reports 4 MiB here and 0 from
        // requiredWorkspaceSize().
        //
        // That divergence is an over-reservation of a buffer the caller already
        // supplied, never an under-allocation, so it is benign in the real
        // allocate-then-launch flow: the allocator sizes from
        // requiredWorkspaceSize(), that size is what problem.workspaceSize() reports
        // on the subsequent launch, and re-deriving this snapshot against it reaches
        // a self-consistent fixed point (0 bytes -> workspace-DP fallback -> 0 bytes
        // reserved). It only becomes visible when a caller hands in a workspace it
        // sized some other way. Both implementations still encode the same intended
        // rule -- change one, check the other.
        size_t requiredWorkspaceBytes = 0;
        // recomputed: partials(+work-queue) bytes wanted, before the fit check against
        // givenWorkspaceBytes. Non-zero even when the fallback fires, which is what
        // makes the "wanted vs given" comparison legible in the report.
        size_t idealWorkspaceBytes    = 0;
        // available: problem.workspaceSize().
        size_t givenWorkspaceBytes    = 0;
        // recomputed: skTiles*skSplit slot count for the dynamic path (informational; does
        // NOT feed the allocation guard, which is sized by grid).
        size_t dynamicPartialsSlots   = 0;
        // available: streamKBakedQueueCount() -- baked per-XCD work-queue count (NUM_XCD),
        // 0 if unknown.
        size_t numQueues              = 0;

        // --- Fallbacks that fired (each can change selectedGrid -> finalGrid) ---
        // recomputed: the workspace-insufficient fallback (idealWorkspaceBytes >
        // givenWorkspaceBytes -> reduction=tree, grid=tiles). computeStreamKDecisions()
        // is the sole implementation of this fallback; solve() consumes the result.
        // Keep the reserve-or-not condition in sync with
        // ContractionSolution::requiredWorkspaceSize().
        bool workspaceDPFallbackFired = false;
        // available: getSKGridImpl out-param (24-bit tree-fixup bounds -> grid=tiles).
        bool treeBoundsFallbackFired  = false;
        // available: getSKGridImpl out-param -- the StreamKForceDPOnly cluster-multicast
        // clamp (SK3 + streamKForceDPOnly + clusterDim.x*clusterDim.y > 1 -> grid=tiles,
        // one workgroup per output tile). Applied after the tree-bounds fallback.
        bool clusterDPGridClamped     = false;
        // available: getSKGridImpl out-param (AMDGPU skFixedGrid override applied).
        bool fixedGridUsed            = false;
    };

    /**
     * Selection-time diagnostics for uniform summation order.
     *
     * Under uniform summation order the library filters out every solution it
     * cannot prove row-uniform, so a lookup can legitimately return nothing.
     * That is visible to the caller only as "no solution found", which does not
     * say whether uniform summation order was responsible or which of its
     * clauses did the work. These two functions close that gap: the filter tags
     * each rejection with a short stable token, and the caller that observes an
     * empty result asks for the tally back.
     *
     * The tally is thread-local and covers the candidates examined since the
     * last reset, which a caller performs immediately before a lookup.
     *
     * Everything here is inert unless TENSILE_DB bit 0x200000 is set: recording
     * is a branch on a cached flag, so no counting, formatting or allocation
     * happens on a normal run. It is a dedicated bit rather than a log level so
     * that enabling it does not also switch on per-call tracing.
     */
    TENSILELITEHOST_EXPORT void uniformSummationOrderSelectionTallyReset();

    /**
     * Renders the tally as `<token>:<count>` pairs, most frequent first, joined
     * by commas; "none" when no candidate reached the filter. The count of
     * candidates that reached the filter is returned through examined, so a
     * caller can tell "uniform summation order emptied the set" from "the set
     * was already empty when the filter ran".
     */
    TENSILELITEHOST_EXPORT std::string
        uniformSummationOrderSelectionTallyReport(size_t& examined, size_t& refused);

    /**
     * Represents a single kernel or set of kernels that can perform a single
     * tensor contraction.
     *
     * Can generate `KernelInvocation` objects to solve a particular problem
     * given a set of `ContractionInputs`.
     */
    class TENSILELITEHOST_EXPORT ContractionSolution : public Solution
    {
    public:
        using Problem             = ContractionProblemGemm;
        using Inputs              = ContractionInputs;
        using GroupedInputs       = ContractionGroupedInputs;
        using WGMParamsCache      = CacheMap<std::tuple<int32_t, size_t, size_t, size_t>, Problem>;
        using StaggerUParamsCache = CacheMap<std::tuple<size_t, size_t, size_t>, Problem>;

        /**
         * Indicate a solution's matching type
         */
        enum class MatchingTag
        {
            Equal, // EqualityMatching
            Range, // RangeMatching
            FreeSize, // FreeSizeMatching
            GridBased, // GridBasedMatching
            Prediction, // PredictionMatching
            Experimental, // ExperimentalStreamK or ExperimentalMLP
            Others, // Default
        };

        static std::string Type()
        {
            return "Contraction";
        }
        virtual std::string type() const
        {
            return Type();
        }

        virtual std::string KernelName() const
        {
            return kernelName;
        }

        virtual std::string name() const
        {
            return solutionName;
        }
        virtual std::string description() const
        {
            return kernelName;
        }

        virtual bool isFallbackForHW(Hardware const&) const;

        bool isStreamK() const
        {
            return sizeMapping.streamK > 0;
        }

        /**
         * @brief Returns the string representation of the solution's matching type.
         *
         * This tag is used to identify or categorize the solution for matching purposes.
         * @return A string representing the matching type of the solution.
         */
        virtual std::string matchingTag() const;

        //! Estimates based on problem size, solution tile, and  machine hardware
        //! charz:
        struct StaticPerformanceModel
        {
            size_t memReadBytesA   = 0.0; //! Estimated memory reads A
            size_t memReadBytesB   = 0.0; //! Estimated memory reads B
            size_t memReadBytesC   = 0.0; //! Estimated memory reads C
            size_t memWriteBytesD  = 0.0; //! Estimated memory writes D
            size_t memReadBytes    = 0.0;
            size_t memGlobalReads  = 0;
            size_t memGlobalWrites = 0;
        };

        struct Granularities
        {
            double numTiles0  = 0.0; //! number of tiles in 0 dimension
            double numTiles1  = 0.0; //! number of tiles in 1 dimension
            double totalTiles = 0.0;
            double tilesPerCu = 0.0;

            //! Granularity is measured 0..1 with 1.0 meaning no granularity loss
            double tile0Granularity          = 0.0; // loss due to tile0
            double tile1Granularity          = 0.0;
            double cuGranularity             = 0.0;
            double waveGranularity           = 0.0;
            double totalGranularity          = 0.0;
            double totalTileAwareGranularity = 0.0;
            double natCuGranularity          = 0.0;
            double natTilesPerCu             = 0.0;
            double suTilesPerCu              = 0.0;
            double suCuGranularity           = 0.0;
            double waves                     = 0.0;
            double suWavesPerSimdx2          = 0.0;
            double suWaveGranularity         = 0.0;

            int CUs = 0;

            double MT0;
            double MT1;
            double GSU;
            double LSU;
        };

        struct ProjectedPerformance
        {
            Granularities granularities;

            double speedGFlops = 0.0; //! final gflops projection
            int    CUs         = 0;

            StaticPerformanceModel staticModel;
        };

        // Result of host-side AdaptiveGemmNTAB dispatch.
        // nta/ntb are either 0 (cached) or 4 (non-temporal).
        // Packed into internalArg0 bits 12/13 when InternalArgsSupport.version >= 3.
        struct AdaptiveGemmNTAB
        {
            uint32_t nta = 0;
            uint32_t ntb = 0;
        };

        bool checkInternalArgumentsSupport(ContractionProblem const& problem,
                                           std::ostream&             stream,
                                           bool                      debug = false) const;

        /**
         * Calculate required workspace size.
         */
        size_t requiredWorkspaceSize(Problem const& problem, Hardware const& hardware) const;
        size_t requiredWorkspaceSizeGsu(Problem const&  problem,
                                        Hardware const& hardware,
                                        size_t          gsu) const;
        size_t requiredWorkspaceSizeGroupedGemm(std::vector<Problem> const& problems,
                                                Hardware const&             hardware) const;
        size_t requiredHostSizeGroupedGemmSingle(Problem const&  problem,
                                                 Hardware const& hardware) const;

        size_t requiredSynchronizerSize(Problem const& problem, Hardware const& hardware) const;

        void                 calculateGrid(dim3&                               workGroupSize,
                                           dim3&                               numWorkGroups,
                                           ContractionSolution::Problem const& problem) const;
        origami::reduction_t getSKReduction(Problem const& problem, Hardware const& hardware) const;
        size_t               getSKGrid(Problem const&       problem,
                                       Hardware const&      hardware,
                                       size_t               tiles,
                                       origami::reduction_t reductionStrat) const;
        // Resolve the effective StreamK=5 hybrid sub-mode for a launch: returns
        // true for the dynamic (SK4) path, false for the static (SK3) path.
        // Precedence (highest first): the TENSILE_STREAMK5_FORCE_MODE debug env
        // override (0=force static, 1=force dynamic), then the problem tri-state
        // streamKTileSchedulingMode (0=OFF/static unless smCountTarget()>0,
        // 1=ON/dynamic), then AUTO (2) via the origami hybrid-mode heuristic.
        // Only meaningful when
        // sizeMapping.streamK == 5. This is the single source of truth shared by
        // grid sizing (getSKGrid) and kernel-arg packing (generateSingleCall) so
        // the launch grid and the packed args can never disagree.
        bool                 streamK5EffectiveDynamic(Problem const&  problem,
                                                      Hardware const& hardware) const;
        // Selection-time predicate for the StreamK dynamic-queue / work-stealing
        // path. The SK4 and dynamic sub-path of SK5 kernels hardcode a
        // power-of-two per-XCD queue count and mask indices with (Q-1); that fast
        // masking is only valid when the device exposes a power-of-two number of
        // XCDs. Returns false (and warns once) when this solution would take the
        // dynamic-queue path but the hardware's NUM_XCD is not a power of two
        // (e.g. MI300A = 6), so the solution is EXCLUDED from selection rather
        // than silently degraded to tree reduction. All other solutions return
        // true. Wired into softwarePredicate() (SolutionLibrary.hpp).
        bool                 streamKDynamicQueueSupported(Problem const&  problem,
                                                          Hardware const& hardware) const;
        // Selection-time filter for uniform summation order. Resolves StreamK /
        // GSU / StaggerU the same way solve() does and admits only launches
        // checkUniformSummationOrder() would accept, except the StreamK
        // Synchronizer pointer (not allocated yet). Wired into softwarePredicate().
        bool                 uniformSummationOrderSupported(Problem const&  problem,
                                                            Hardware const& hardware) const;
        size_t               partialTileSize(size_t skGrid) const;

        // Compute the StreamK launch-parameter DECISIONS for this solution on the
        // given problem/hardware. solve() consumes the reduction strategy, grid,
        // isDynamic predicate, and workspace/DP fallback from here to populate
        // StreamKSettings -- this is the only place that logic lives -- and it is
        // also directly callable from unit tests. Existing helpers are reused where
        // possible (streamK5EffectiveDynamic, getSKReduction, getSKGridImpl,
        // partialTileSize); the makeArgs packing quantities are re-derived. Each
        // StreamKDecisions field documents its own provenance (available vs
        // recomputed). Has no effect on the launch beyond producing these values, and
        // never mutates solution or problem state. For a non-StreamK solution
        // (sizeMapping.streamK <= 0) it returns immediately with a default-initialised
        // snapshot whose streamKMode is sizeMapping.streamK.
        StreamKDecisions computeStreamKDecisions(Problem const&  problem,
                                                 Hardware const& hardware) const;

        // Print a one-line StreamK launch summary of the given decisions. Called
        // from solve() only when Debug::printStreamKLaunchSummary() is set;
        // exposed for tests.
        void printStreamKLaunchSummary(std::ostream&           os,
                                       Problem const&          problem,
                                       StreamKDecisions const& decisions) const;

        static float computeGranularity(float x);

        Granularities computeGranularities(Hardware const& hardware,
                                           double          M,
                                           double          N,
                                           double          K,
                                           double          NumBatches,
                                           uint32_t        autoGsuVal) const;

        StaticPerformanceModel staticPerformanceModel(double M,
                                                      double N,
                                                      double K,
                                                      double NumBatches,
                                                      double MT0,
                                                      double MT1,
                                                      double NumCUs,
                                                      double totalGranularity,
                                                      int    globalSplitU) const;

        /**
         * Calculate the projected performance based on granularity loss.
         */
        ProjectedPerformance projectedPerformance(Problem const&  problem,
                                                  Hardware const& hardware) const;

        /**
         * Generate a set of kernel calls to solve a particular problem.
         */
        virtual std::vector<KernelInvocation> solve(ContractionProblem const& problem,
                                                    ProblemInputs const&      inputs,
                                                    Hardware const&           hardware,
                                                    void*                     hipHostMemory,
                                                    size_t                    hipHostMemorySize,
                                                    hipStream_t               stream) const;

        virtual std::vector<KernelInvocation>
            solve(Problem const& problem, Inputs const& inputs, Hardware const& hardware) const;

        virtual std::vector<KernelInvocation> solveGroupedGemm(std::vector<Problem> const& problems,
                                                               GroupedInputs const&        inputs,
                                                               Hardware const&             hardware,
                                                               void*       hipHostMemory,
                                                               size_t      hipHostMemorySize,
                                                               hipStream_t stream) const;

        // The problems and inputs are passed by device memory
        virtual std::vector<KernelInvocation>
            solveGroupedGemmGPU(std::vector<Problem> const& problems,
                                GroupedInputs const&        inputs,
                                Hardware const&             hardware,
                                const void*                 dUA,
                                const void*                 workspace,
                                hipStream_t                 stream) const;

        // For Tensile debugging, will allocate and initialize DeviceUserArguments with the problems and inputs.
        virtual std::vector<KernelInvocation> solveTensileGPU(ContractionProblem const& problem,
                                                              ProblemInputs const&      inputs,
                                                              Hardware const&           hardware,
                                                              void**                    dUA,
                                                              void**                    dUAHost,
                                                              void*       hipHostMemory,
                                                              size_t      hipHostMemorySize,
                                                              hipStream_t stream) const;

        // For Tensile debugging, will allocate and initialize DeviceUserArguments with the problems and inputs.
        virtual std::vector<KernelInvocation>
            solveTensileGroupedGemmGPU(std::vector<Problem> const& problems,
                                       GroupedInputs const&        inputs,
                                       Hardware const&             hardware,
                                       void**                      dUA,
                                       void**                      dUAHost,
                                       void*                       hipHostMemory,
                                       size_t                      hipHostMemorySize,
                                       hipStream_t                 stream) const;

        virtual void relaseDeviceUserArgs(void* dUA, void* dUAHost);

        template <bool T_Debug, bool insertKernelArgs, typename KA>
        void singleCallArgs(Problem const&           problem,
                            ContractionInputs const& inputs,
                            uint32_t const&          workspaceOffsetInByte,
                            Hardware const*          hardware,
                            dim3 const&              problemNumGroupTiles,
                            dim3 const&              numWorkGroups,
                            KA&                      args,
                            StreamKSettings const&   sk) const;

        // Common kernel related arguments (e.g. gemm_count, arg type, MT, GSU...)
        template <bool T_Debug, bool Legacy, typename KA>
        void kernelArgs(uint32_t                            gemmCount,
                        uint32_t                            argType,
                        KA&                                 args,
                        uint32_t                            numWorkGroups,
                        Hardware const*                     hardware,
                        const ContractionProblemParameters& param,
                        int32_t                             autoWGM,
                        size_t                              autoWGMXCC,
                        size_t                              autoWGMXCCCHUNK,
                        size_t                              autoWGMXCCSPLITK,
                        size_t                              autoStaggerUMapping,
                        size_t                              autoStaggerU,
                        size_t                              autoStaggerUStrideShift,
                        uint32_t                            autoGsuVal,
                        AdaptiveGemmNTAB                    ntab) const;

        template <typename KA>
        inline void calculateSingleCallWorkGroupItems(std::vector<Problem> const& problems,
                                                      const TensileLite::dim3&    workGroupSize,
                                                      TensileLite::dim3&          numWorkGroups,
                                                      TensileLite::dim3&          numWorkItems,
                                                      KA&                         h_args,
                                                      uint32_t                    gsu) const;

        template <bool T_Debug>
        KernelInvocation generateSingleCall(Problem const&           problem,
                                            ContractionInputs const& inputs,
                                            Hardware const&          hardware,
                                            StreamKSettings const&   sk,
                                            GSUSettings const&       gsuSettings) const;

        template <bool T_Debug, typename KA>
        KernelInvocation generateSingleCallGroupedGemm(std::vector<Problem> const& problems,
                                                       GroupedInputs const&        inputs,
                                                       Hardware const&             hardware,
                                                       KA&                         h_args,
                                                       void const* userArgs = nullptr) const;

        template <bool T_Debug>
        KernelInvocation generateBetaOnlyCall(Problem const&           problem,
                                              ContractionInputs const& inputs) const;

        template <bool T_Debug>
        KernelInvocation generateBetaOnlyCallGroupedGemm(std::vector<Problem> const& problems,
                                                         GroupedInputs const&        inputs) const;

        std::string betaOnlyKernelName(Problem const& problem) const;

        template <bool T_Debug, typename KA>
        void outputConversionCallArgs(Problem const&           problem,
                                      ContractionInputs const& inputs,
                                      uint32_t const&          workspaceOffsetInByte,
                                      KA&                      args,
                                      StreamKSettings const&   sk,
                                      uint32_t                 autoGsuVal,
                                      uint32_t                 additionalPaddingPerBatchGeneralBatch=0) const;

        template <typename KA>
        inline void calculateConversionCallWorkGroupItems(
            std::vector<ContractionSolution::Problem> const& problems,
            size_t&                                          vw,
            const TensileLite::dim3&                         workGroupSize,
            TensileLite::dim3&                               numWorkGroups,
            TensileLite::dim3&                               numWorkItems,
            KA&                                              args) const;

        template <bool T_Debug>
        KernelInvocation generateOutputConversionCall(Problem const&           problem,
                                                      ContractionInputs const& inputs,
                                                      StreamKSettings const&   sk,
                                                      uint32_t                 autoGsuVal) const;

        template <bool T_Debug, typename KA>
        KernelInvocation
            generateOutputConversionCallGroupedGemm(std::vector<Problem> const& problems,
                                                    GroupedInputs const&        inputs,
                                                    Hardware const&             hardware,
                                                    KA&                         h_args) const;

        template <bool T_Debug>
        KernelInvocation updateUserArgsOutputConversionCallGroupedGemm(
            std::vector<ContractionSolution::Problem> const& problems,
            const void*                                      userArgs,
            const void*                                      workspace) const;

        std::string outputConversionKernelName(Problem const&           problem,
                                               ContractionInputs const& inputs,
                                               size_t                   vw,
                                               size_t                   gsu) const;

        template <bool T_Debug>
        KernelInvocation generateReductionCall(Problem const&           problem,
                                               ContractionInputs const& inputs) const;

        std::string outputReductionKernelName(Problem const&           problem,
                                              ContractionInputs const& inputs,
                                              size_t                   mt0,
                                              size_t                   mt1,
                                              size_t                   vw) const;

        struct InternalArgsSupport
        {
            int  version            = 0;
            bool gsu                = true;
            bool wgm                = true;
            bool staggerU           = true;
            // Kernel distributes Stream-K extra iters within each tile when
            // skGrid % skTiles == 0. Older/custom kernels leave this false;
            // newly generated StreamK 3 / SK5 set it true. Uniform-summation-order
            // grid steering consults the same bit.
            bool perTileExtraIters  = false;
            bool useUniversalArgs   = true;
            bool useSFC             = false;
        };

        struct ProblemType
        {
            std::string      operationIdentifier;
            bool             transA                    = false;
            bool             transB                    = false;
            rocisa::DataType aType                     = rocisa::DataType::Float;
            rocisa::DataType bType                     = rocisa::DataType::Float;
            rocisa::DataType cType                     = rocisa::DataType::Float;
            rocisa::DataType dType                     = rocisa::DataType::Float;
            rocisa::DataType eType                     = rocisa::DataType::Float;
            rocisa::DataType computeInputTypeA         = rocisa::DataType::Float;
            rocisa::DataType computeInputTypeB         = rocisa::DataType::Float;
            rocisa::DataType computeType               = rocisa::DataType::Float;
            rocisa::DataType f32XdlMathOp              = rocisa::DataType::Float;
            rocisa::DataType activationComputeDataType = rocisa::DataType::Float;
            bool             highPrecisionAccumulate   = false;
            bool             useBeta                   = true;
            bool             useGradient               = false;
            int              useBias                   = 0;
            bool             useE                      = false;
            bool             useGateResidual           = false;
            std::string      useScaleAB                = "";
            bool             useScaleCD                = false;
            int              useScaleAlphaVec          = 0;
            bool             useInitialStridesAB       = false;
            bool             useInitialStridesCD       = false;
            bool             stridedBatched            = true;
            bool             outputAmaxD               = false;
            bool             groupedGemm               = false;
            ActivationType   activationType            = ActivationType::None;
            int              activationArgLength       = 0;
            bool             activationNoGuard         = false;

            std::vector<int>              biasSrcWhiteList;
            std::vector<rocisa::DataType> biasDataTypeWhiteList;
            std::vector<rocisa::DataType> gateResidualDataTypeWhiteList;

            int  sparse                     = 0;
            bool stochasticRounding         = false;
            bool supportDeviceUserArguments = false;
            bool swizzleTensorA             = false;
            bool swizzleTensorB             = false;
            bool fusedGemmA2A               = false;
            int  metadataLayout             = 0;
            int  mxBlockA                   = 0;
            int  mxBlockB                   = 0;
            rocisa::DataType mxTypeA        = rocisa::DataType::E8;
            rocisa::DataType mxTypeB        = rocisa::DataType::E8;

            // In-device MX scale layout expected by the kernel. Mirrors the
            // MXScaleFormat solution parameter (see Tensile/Common/ValidParameters.py).
            // Encoded as a small int so it round-trips through msgpack and YAML
            // logic files without an explicit enum schema:
            //   0 = NoSwizzle       (default; canonical row/column layout)
            //   1 = HostPreSwizzle  (gfx950 subtile host-preswizzled layout)
            //   2 = InMemorySwizzle (gfx1250 TDM-populated swizzled layout)
            // The host (DataInitialization) consults this to decide whether to
            // apply the K-dimension swizzle on the MX scale tensor before upload.
            int mxScaleFormat = 0;
        };

        struct LinearModel
        {
            double slope     = 1.0;
            double intercept = 0.0;
            double max       = 1000.0;
        };

        int                          index = 0;
        std::string                  kernelName;
        std::string                  solutionName;
        ThreadSafeValue<std::string> codeObjectFilename;
        bool                         debugKernel     = false;
        bool                         kernelArgsLog   = false;
        mutable int                  isFallbackCUSol = -1; // -1:unset, 0:false, 1:true
        mutable WGMParamsCache       wgmParamsCache
            = WGMParamsCache(std::make_tuple(INT32_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX));
        mutable StaggerUParamsCache staggerUParamsCache
            = StaggerUParamsCache(std::make_tuple(SIZE_MAX, SIZE_MAX, SIZE_MAX));

        std::shared_ptr<Predicates::Predicate<Task>> taskPredicate
            = std::make_shared<Predicates::True<Task>>();
        std::shared_ptr<Predicates::Predicate<Problem>> problemPredicate
            = std::make_shared<Predicates::True<Problem>>();
        std::shared_ptr<Predicates::Predicate<Hardware>> hardwarePredicate
            = std::make_shared<Predicates::True<Hardware>>();

        SizeMapping  sizeMapping;
        CustomKernel customKernel;

        InternalArgsSupport internalArgsSupport;

        ProblemType problemType;

        // This will be calculated when getSolution is called
        size_t requiredHostWorkspaceSizePerProblem = static_cast<size_t>(-1);

        /// Debugging purposes.  Shouldn't contain any vital information that isn't
        /// somewhere else.
        int32_t               libraryLogicIndex = -1;
        std::map<int, double> ideals;
        LinearModel           linearModel;
        MatchingTag           tag{MatchingTag::Others};

        uint32_t magicNumberAlg1(uint32_t x, uint32_t* magicShift) const;
        uint32_t magicNumberAlg2(uint32_t x, uint32_t* magicShift) const;
        uint32_t magicNumber(int magicDivAlg, uint32_t x, uint32_t* magicShift) const;
        uint32_t smallMagicNumber(uint32_t x) const;

        std::tuple<int32_t, size_t, size_t, size_t> calculateAutoWGM(Problem const&  problem,
                                                                     Hardware const* hardware,
                                                                     uint32_t        skgrid) const;
        std::tuple<size_t, size_t, size_t>  calculateAutoStaggerU(Problem const&  problem,
                                                                  Hardware const* hardware,
                                                                  uint32_t        skgrid,
                                                                  int32_t         autoWGM) const;
        uint32_t calculateAutoGSU(Problem const& problem, Hardware const* hardware) const;

        double calculateDimensionM(Problem const&  problem) const;
        double calculateDimensionN(Problem const&  problem) const;
        double calculateNumBatches(Problem const&  problem) const;
        SizeMapping getSizeMapping(void) const {return sizeMapping;};
        origami::data_type_t getOrigamiDatatype(Problem const&  problem) const;
        AdaptiveGemmNTAB calculateAdaptiveGemmNTAB(Problem const&  problem,
                                                   Hardware const* hardware) const;

    private:
        bool handwrittenCustomKernel() const;

        // Same StreamK grid / reduction solve() packs, including the
        // insufficient-workspace fall back to tree + grid==tiles.
        StreamKSettings resolveStreamKSettings(Problem const&  problem,
                                               Hardware const& hardware) const;

        // Reasons checkUniformSummationOrder() would refuse this launch.
        // Empty means the launch is row-uniform. requireSynchronizer is the
        // one clause that exists only at solve() (the pointer is not allocated
        // at heuristic time); selection passes false.
        //
        // obstacleToken, when non-null, receives a short stable identifier for
        // whichever clause objected, written by that clause itself rather than
        // recovered from the prose. It points at a string literal with static
        // storage duration and is left untouched when the return value is
        // empty. It exists so a diagnostic can name the reason without parsing
        // the human-readable text, which is free to change.
        std::string uniformSummationOrderLaunchObstacle(
            Problem const&         problem,
            Hardware const&        hardware,
            StreamKSettings const& sk,
            size_t                 resolvedGlobalAccumulation,
            uint32_t               gsu,
            void const*            synchronizer,
            bool                   requireSynchronizer,
            char const**           obstacleToken = nullptr) const;

        // Launch gate. Call once sk and resolvedGlobalAccumulation are final.
        void checkUniformSummationOrder(Problem const&         problem,
                                        Hardware const&        hardware,
                                        StreamKSettings const& sk,
                                        size_t                 resolvedGlobalAccumulation,
                                        uint32_t               gsu,
                                        void const*            synchronizer) const;
    };

    template <typename TAct>
    TENSILELITEHOST_EXPORT void setDeviceUserArgs(std::vector<ContractionSolution::Problem> const& problems,
                           ContractionSolution::GroupedInputs const&        inputs,
                           DeviceUserArguments<TAct>*                       args);

    TENSILELITEHOST_EXPORT std::ostream& operator<<(std::ostream&                                      stream,
                             ContractionSolution::StaticPerformanceModel const& spm);
    TENSILELITEHOST_EXPORT std::ostream& operator<<(std::ostream&                                    stream,
                             ContractionSolution::ProjectedPerformance const& spm);
    TENSILELITEHOST_EXPORT std::ostream& operator<<(std::ostream& stream, BufferLoadCheckPacket const& st);
} // namespace TensileLite

