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
#include <string>
#include <variant>

#include <origami/types.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    struct StreamK
    {
        /**
         *   High bits of slot2
         */
        struct Slot2Bits
        {
            static constexpr uint32_t MagicAddBit              = 1u << 31; // magicNumberAlg2's "add" indicator
            static constexpr uint32_t HybridModeBit            = 1u << 30; // set: dynamic work queue; clear: static
            static constexpr uint32_t UniformSummationOrderBit = 1u << 29;

            static constexpr uint32_t TileReservedMask
                = MagicAddBit | HybridModeBit | UniformSummationOrderBit;
            static constexpr uint32_t ShiftReservedMask = HybridModeBit | UniformSummationOrderBit;
        };

        struct Static
        {
            struct KernelArguments
            {
                uint32_t magicNumber = 0;
                uint32_t magicShift  = 0;
                uint32_t itersPerWG  = 0;
                uint32_t grid        = 0;
                uint32_t skTiles     = 0;
            };

            struct Split
            {
                KernelArguments args;
                uint32_t        extraIters = 0;
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
            TENSILELITEHOST_EXPORT static Split
                computeSplit(size_t tiles,
                             size_t itersPerTile,
                             size_t skGrid,
                             int    skFullTiles,
                             bool   forceDPOnly);

            /**
             * Compute the static two-tile kernel-argument words (StreamK=3, and
             * StreamK=5 with the dynamic sub-path off).
             *
             * Under parallel reduction the split is grid / numTiles partials per tile;
             * otherwise it is the two-tile split from computeSplit(). The magic words
             * come from magic division algorithm 2 on itersPerTile.
             *
             * @param numTiles      Tile count the grid is resolved against.
             * @param itersPerTile  Clamped iterations per tile.
             * @param grid          Resolved StreamK grid.
             * @param reduction     Resolved reduction strategy.
             * @param skFullTiles   AMDGPU::skFullTiles (TENSILE_STREAMK_FULL_TILES).
             * @param uniformSummationOrderActive
             *                      True when the kernel supports per-tile extras AND the
             *                      problem requests uniform summation order. Sets
             *                      Slot2Bits::UniformSummationOrderBit in magicShift.
             */
            [[nodiscard]] static KernelArguments
                calculateKernelArgs(size_t               numTiles,
                                    uint32_t             itersPerTile,
                                    size_t               grid,
                                    origami::reduction_t reduction,
                                    int                  skFullTiles,
                                    bool                 uniformSummationOrderActive);

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
             * @param split             The split StreamK::Static::computeSplit() produced.
             * @param tiles             The same batch-inclusive tile count fed to it.
             * @param itersPerTile      The same clamped iterations per tile fed to it.
             * @param skGrid            The resolved StreamK grid packed into the split.
             * @param perTileCapable    True when the selected kernel CAN redistribute
             *                          Stream-K extras within each tile
             *                          (InternalArgsSupport::perTileExtraIters). Capability
             *                          only; the kernel branches on it at runtime.
             * @param uniformSummationOrder
             *                          ContractionProblemParameters::uniformSummationOrder(),
             *                          the host-side bit the packer forwards to the device in
             *                          MagicShiftItersPerTile bit 29. The per-tile mapping runs
             *                          only when this AND perTileCapable hold; otherwise the
             *                          kernel runs the historical global first-E mapping.
             */
            TENSILELITEHOST_EXPORT static bool
                isRowUniform(Split const& split,
                             size_t       tiles,
                             size_t       itersPerTile,
                             size_t       skGrid                = 0,
                             bool         perTileCapable        = false,
                             bool         uniformSummationOrder = false);

            /**
             * Iteration range [start, end) assigned to workgroup w under the static
             * two-tile StreamK mapping. Extras are distributed within each tile when
             * skGrid % tiles == 0 and both perTileCapable (InternalArgsSupport::perTileExtraIters)
             * and uniformSummationOrder hold -- the pair the kernel branches on at runtime;
             * otherwise the historical global first-E mapping is used.
             */
            struct IterRange
            {
                size_t start = 0;
                size_t end   = 0;
            };

            TENSILELITEHOST_EXPORT static IterRange
                workgroupIterRange(size_t w,
                                   size_t tiles,
                                   size_t itersPerTile,
                                   size_t skGrid,
                                   bool   perTileCapable,
                                   bool   uniformSummationOrder);
        };

        struct Dynamic
        {
            struct KernelArguments
            {
                uint32_t totalItems = 0;
                uint32_t skTiles    = 0;
                uint32_t skSplit    = 0;
                uint32_t itersPerWI = 0;
                uint32_t grid       = 0;
            };

            /**
             * Compute the dynamic work-queue kernel-argument words (StreamK=4, and
             * StreamK=5 resolved to the dynamic sub-path).
             *
             * @param numTiles       Tile count the work queue covers.
             * @param itersPerTile   Clamped iterations per tile.
             * @param grid           Resolved StreamK grid.
             * @param skSplitOverride  AMDGPU::skSplit; negative selects the default split of 2.
             * @param skTilesOverride  AMDGPU::skTiles; negative selects zero Stream-K tiles.
             */
            [[nodiscard]] static KernelArguments
                calculateKernelArgs(size_t   numTiles,
                                    uint32_t itersPerTile,
                                    size_t   grid,
                                    int      skSplitOverride,
                                    int      skTilesOverride);
        };

        // Scheduling ABI 2: a logical worker is a hardware cluster and a scheduling
        // tile is a spatial block. Partial/flag slot q*C+peer retains the peer identity.
        // This resolved value is shared by workspace queries, argument packing, launch
        // geometry and diagnostics; legacy scheduling ABIs keep their original units.
        struct ClusterSchedule
        {
            bool enabled = false;
            size_t clusterSize = 1;
            size_t blocksM = 0;
            size_t blocksN = 0;
            size_t blocks = 0;
            size_t physicalGrid = 0;
            size_t gridBeforeFallback = 0;
            size_t partialSlots = 0;
            size_t flagEntries = 0;
            size_t idealWorkspaceBytes = 0;
            uint32_t itersPerTile = 1;
            uint32_t parallelSplits = 1;
            Static::Split split;
            bool wholeBlocksOnly = false;
            bool workspaceFallback = false;
            bool treeBoundsFallback = false;
            bool fixedGridUsed = false;
            std::string fallbackReason;
        };

        struct ArgumentSlots
        {
            std::variant<Static::KernelArguments, Dynamic::KernelArguments> words
                = Static::KernelArguments{};
        };

        // Snapshot of the StreamK launch-parameter DECISIONS for one
        // (solution, problem, hardware) triple. Populated by
        // ContractionSolution::computeStreamKDecisions() and consumed by solve().
        // Each field's provenance is annotated inline as one of:
        //   available  = read from the value the launch uses, or from the same real
        //                helper/param solve() uses, so it reflects the launch exactly.
        //   recomputed = re-derived here rather than read back from a launched value:
        //                either by mirroring the kernel-arg packing in makeArgs() (the
        //                slots member) or by deriving from other fields. Mirrors can
        //                drift silently if makeArgs() changes.
        // This snapshot is not purely observational: reduction, grid() and isDynamic are
        // launch INPUTS -- solve() copies them into StreamKSettings and uses isDynamic as
        // its dynamic-queue predicate. Only the PRINTING is optional; see
        // Debug::printStreamKLaunchSummary() (TENSILE_DB bit 0x200000).
        struct Decisions
        {
            // Scheduling ABI 2. Stays disabled for legacy scheduling ABIs.
            ClusterSchedule clusterSchedule;

            // --- Mode ---
            // available: streamK5EffectiveDynamic(), the same helper solve()'s grid path uses
            // (SK5 resolved to the dynamic SK4 sub-path). Only meaningful for SK5: it stays
            // false for SK4 even though SK4 is unconditionally dynamic, so ask isDynamic
            // (below) -- not this -- whether a launch takes the dynamic path.
            bool effectiveDynamic = false;
            // recomputed: DynamicWorkQueue assignment, or Hybrid resolved dynamic.
            // Launch-relevant rather than merely reported: solve() consumes this as the
            // dynamicQueuePath predicate guarding the work-stealing rejection.
            bool isDynamic        = false;

            // --- Reduction ---
            // available: solve() wires this exact value into sk.reduction -- it is the final
            // (post workspace-DP fallback) reduction the launch uses.
            origami::reduction_t reduction = origami::reduction_t::none;

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

            //Streamk specific kernel arguments
            ArgumentSlots slots;

            //Accessors to handle streamk3  and streamk4 argument differences
            size_t grid() const
            {
                return std::visit([](auto const& w) -> size_t { return w.grid; }, slots.words);
            }
            void setGrid(size_t g)
            {
                std::visit([g](auto& w) { w.grid = static_cast<uint32_t>(g); }, slots.words);
            }

            size_t skTiles() const
            {
                return std::visit([](auto const& w) -> size_t { return w.skTiles; }, slots.words);
            }

            // --- DP-only ---
            // The three flags below distinguish the source of a data-parallel-only launch:
            //   forceDPOnly              -> sizeMapping.isPersistentDataParallel() compile-time param
            //   streamKDP                -> TENSILE_STREAMK_DATA_PARALLEL debug override
            //   workspaceDPFallbackFired -> runtime workspace-insufficient (below)
            // recomputed: OR of the three DP triggers above.
            bool dpOnly      = false;
            // available: sizeMapping.isPersistentDataParallel() param.
            bool forceDPOnly = false;
            // available: Debug::useStreamKDataParrallel() (TENSILE_STREAMK_DATA_PARALLEL).
            bool streamKDP   = false;

            // --- Workspace / partials ---
            // recomputed: skTiles() > 0.
            bool   partialsPresent        = false;
            // recomputed: requiredWorkspaceBytes > 0.
            bool   workspaceAllocated     = false;
            // recomputed: partials(+work-queue) bytes this launch reserves; 0 when no
            // partials are needed or the workspace-DP fallback fired.
            //
            // WARNING: this is NOT ContractionSolution::requiredWorkspaceSize()'s return
            // value, even though the names are close. requiredWorkspaceSize() is the
            // separate, caller-facing implementation of the reserve-or-not rule -- it is
            // what the allocator sizes the workspace buffer from. For generated
            // kernels, both paths size parallel partial-result storage from the grid,
            // after streamKReconcileReduction() demotes parallel to tree for split
            // factors below two. At grid == tiles, the tree path needs no partials
            // workspace. The parallel query additionally preserves the split-reduction
            // sizing rules for bias-gradient and amaxD; this snapshot reports the
            // partials buffer only.
            //
            // Reduction selection still differs: this snapshot and
            // resolveStreamKSettings() force tree for SK4 and SK5-dynamic, while
            // requiredWorkspaceSize() always asks getSKReduction(). A fixed-grid
            // override can preserve that difference; see requiredWorkspaceSize().
            // Sharing the byte formula does not imply identical decisions in every
            // mode. Changes to any of these three paths must be checked against the
            // other two, including the allocate-then-launch flow where the queried
            // size becomes problem.workspaceSize().
            size_t requiredWorkspaceBytes = 0;
            // recomputed: partials(+work-queue) bytes wanted, before the fit check against
            // givenWorkspaceBytes. Non-zero even when the fallback fires, which is what
            // makes the "wanted vs given" comparison legible in the report.
            size_t idealWorkspaceBytes    = 0;
            // available: problem.workspaceSize().
            size_t givenWorkspaceBytes    = 0;
            // recomputed: StreamK4 skTiles*skSplit slot count for the dynamic path (informational; does
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
            // DataParallel cluster prefetch clamp: PrefetchAcrossPersistent uses one
            // cluster per padded tile block so prefetch cannot overwrite a peer's LDS.
            // Ordinary persistent clusters keep the selected grid in whole clusters.
            bool clusterDPGridClamped     = false;
            // available: getSKGridImpl out-param (AMDGPU persistentFixedGrid override applied).
            bool fixedGridUsed            = false;
        };

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
        TENSILELITEHOST_EXPORT static bool
            isParallelReductionRowUniform(origami::reduction_t reduction,
                                          size_t               grid,
                                          int                  streamKAtomic,
                                          bool                 staticTwoTilePacking,
                                          size_t               tiles);
    };
}
