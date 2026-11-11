// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include <origami/types.hpp>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/Tensile_fwd.hpp>
#include <Tensile/Utils.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    struct PersistentLaunchSettings;

    struct StreamK
    {
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
             * @param kGranularity Iterations per indivisible K unit,
             *                     ContractionSolution::skKGranularity(). 1 leaves
             *                     the historical flat space unchanged.
             */
            TENSILELITEHOST_EXPORT static Split
                computeSplit(size_t tiles,
                             size_t itersPerTile,
                             size_t skGrid,
                             int    skFullTiles,
                             bool   forceDPOnly,
                             size_t kGranularity = 1);

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
                             bool         uniformSummationOrder = false,
                             size_t       kGranularity          = 1);

            /**
             * Iteration range [start, end) assigned to workgroup w under the static
             * two-tile StreamK mapping. Extras are distributed within each tile when
             * skGrid % tiles == 0 and both perTileCapable (InternalArgsSupport::perTileExtraIters)
             * and uniformSummationOrder hold -- the pair the kernel branches on at runtime;
             * otherwise the historical global first-E mapping is used.
             *
             * kGranularity is ContractionSolution::skKGranularity(): both mappings hand
             * out their leftover in units of that many iterations, so every range starts
             * on a multiple of it.
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
                                   bool   uniformSummationOrder,
                                   size_t kGranularity = 1);
        };

        struct Dynamic
        {
            /**
             * Kernel arguments of a dynamic StreamK launch: whole tiles plus `split`
             * parts per split tile. Tiles split only under the debug overrides
             * (TENSILE_STREAMK_TILES / _SPLIT).
             */
            struct KernelArguments
            {
                uint32_t totalItems = 0;
                uint32_t tiles      = 0;
                uint32_t split      = 0;
                uint32_t itersPerWI = 0;
                uint32_t grid       = 0;
            };

            /**
             * Compute the work items of a dynamic launch. Every field but `grid`
             * is filled in; `grid` is left at 0 for the caller.
             *
             * @param hardware      Source of the AMDGPU skTiles / skSplit overrides.
             * @param tiles         Tile count the work queue covers.
             * @param itersPerTile  Iterations per tile; clamped to at least 1.
             * @param kGranularity  Iterations per indivisible K unit. Chunk
             *                      boundaries are rounded onto that unit.
             */
            TENSILELITEHOST_EXPORT static KernelArguments items(Hardware const& hardware,
                                                                size_t          tiles,
                                                                size_t          itersPerTile,
                                                                size_t          kGranularity = 1);

            /**
             * Work item i sits in queue i % numQueues, and a workgroup pops only its
             * home queue (rank % numQueues) or, with stealing, the next one. A queue
             * without a home workgroup is never drained, and the counters' auto-reset
             * bound assumes every queue has one. Grid selection raises the grid to
             * cover the queues; this guards grids that bypass it.
             */
            static void assertGridCoversQueues(Hardware const& hardware,
                                               size_t          grid,
                                               size_t          totalItems)
            {
                const size_t numQueues = bakedQueueCount(hardware);
                TENSILE_ASSERT_EXC(grid >= std::min(totalItems, numQueues)
                                   && "dynamic StreamK grid leaves a non-empty work queue "
                                      "without a home workgroup");
            }

            // Baked per-XCD work-queue count of the dynamic-queue kernels, from origami's
            // per-arch XCD count. 0 when the architecture is unknown (treated as unsupported).
            TENSILELITEHOST_EXPORT static size_t bakedQueueCount(Hardware const& hardware);

            // Bytes the per-XCD work-queue counters occupy: one cache line each, so
            // no two share a line. The kernel places them at the base of the
            // AddressFlags buffer and starts the ready flags after them (StreamK.py
            // _wsFlagsBaseOffset is this same product). 0 when the architecture
            // cannot be determined.
            static size_t queueRegionBytes(Hardware const& hardware)
            {
                return perQueueStrideBytes(hardware) * bakedQueueCount(hardware);
            }

            // True when the launch takes the dynamic work-queue path and so carries
            // that region: SK4, and the SK4 sub-mode of SK5. `effectiveDynamic` is
            // the resolved SK5 sub-mode and is ignored for every other value.
            TENSILELITEHOST_EXPORT static bool usesQueue(SizeMapping const& sizeMapping,
                                                         bool               effectiveDynamic);

            // True (UNSUPPORTED) when work stealing is not valid on this device: unknown
            // hardware, or NUM_XCD is 0, not a power of two, or not equal to bakedQueueCount().
            TENSILELITEHOST_EXPORT static bool queueUnsupported(Hardware const& hardware);

            /**
             * One-shot warning that a dynamic-queue solution was excluded from
             * selection because queueUnsupported() holds.
             */
            TENSILELITEHOST_EXPORT static void warnQueueUnsupportedOnce(Hardware const& hardware);

            /**
             * One-shot warning that the grid was raised so every work queue has a
             * workgroup.
             */
            TENSILELITEHOST_EXPORT static void warnGridRaiseOnce(size_t requested, size_t raised);

        private:
            // Cache-line stride of one per-XCD counter. Used only by queueRegionBytes().
            TENSILELITEHOST_EXPORT static size_t perQueueStrideBytes(Hardware const& hardware);
        };

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
                return std::visit(
                    [](auto const& w) -> size_t {
                        if constexpr(std::is_same_v<std::decay_t<decltype(w)>,
                                                    Dynamic::KernelArguments>)
                            return w.tiles;
                        else
                            return w.skTiles;
                    },
                    slots.words);
            }

            // Work split and work-item count the launch summary reports on every
            // path. The dynamic path reads them from its kernel arguments; the
            // static SK3 ABI has no such words, so they are derived:
            //   parallel -> skSplit = grid / tiles, totalItems = tiles,
            //   static   -> skSplit = 1,            totalItems = tiles,
            //   cluster  -> parallelSplits,         spatial blocks.
            size_t skSplit() const
            {
                if(auto const* dynamicArgs = std::get_if<Dynamic::KernelArguments>(&slots.words))
                    return dynamicArgs->split;
                if(clusterSchedule.enabled)
                    return clusterSchedule.parallelSplits;
                if(reduction == origami::reduction_t::parallel && tiles > 0)
                    return grid() / tiles;
                return 1;
            }

            size_t totalItems() const
            {
                if(auto const* dynamicArgs = std::get_if<Dynamic::KernelArguments>(&slots.words))
                    return dynamicArgs->totalItems;
                if(clusterSchedule.enabled)
                    return clusterSchedule.blocks;
                return tiles;
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
            // after StreamK::reconcileReduction() demotes parallel to tree for split
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
         * The tile's iteration count rounded up to a whole number of K units.
         *
         * Stream-K cuts K out of a flat space of tiles*itersPerTile iterations, so
         * tile t begins at flat index t*itersPerTile. With a unit wider than one
         * iteration that start is only on a unit boundary when itersPerTile is a
         * multiple of the unit; otherwise a cut that is unit-aligned in flat space
         * still lands mid-unit inside the odd tiles, which is what the unit
         * forbids. Quantising the stride alone does not fix it, and dropping the
         * unit back to one iteration makes every cut free to land anywhere, which
         * on the MX scale path silently applies the wrong scales.
         *
         * Measuring the flat space in padded tiles restores the property for every
         * tile at once: tile starts and the cuts between them are all multiples of
         * the unit, so every workgroup begins on a unit boundary whatever
         * itersPerTile is. The padding is not work -- skTileIndex() clamps each
         * workgroup's local range to the real itersPerTile, so the up-to-(unit-1)
         * padded iterations at the end of a tile belong to no workgroup and are
         * never run. The matching device-side rounding is skPaddedItersPerTile()
         * in Components/StreamK.py.
         *
         * Row uniformity is preserved: every tile pads to the same length, so the
         * ordered list of chunk lengths folded into one tile is the list folded
         * into every other.
         *
         * A unit of one pads nothing, so every non-MX kernel keeps the exact flat
         * space, split and argument values it had before the unit existed.
         */
        static inline size_t paddedItersPerTile(size_t itersPerTile, size_t kGranularity)
        {
            if(kGranularity <= 1 || itersPerTile == 0)
                return itersPerTile;
            return ((itersPerTile + kGranularity - 1) / kGranularity) * kGranularity;
        }

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

        /**
         * Warn that the clustered StreamK worker budget was rounded down to the
         * cluster width. Keeps only the last notice per thread.
         */
        TENSILELITEHOST_EXPORT static void warnClusterGridAdjustment(std::string const& kernelName,
                                                                     size_t             requested,
                                                                     size_t             aligned,
                                                                     size_t             clusterWidth);

        /**
         * One-shot warning that uniform-summation-order grid steering displaced a
         * developer grid override or a production CU knob.
         */
        TENSILELITEHOST_EXPORT static void warnUniformityGridSnapOnce(size_t g0, size_t gStar);

        // Reconcile the reduction strategy with the grid that was finally
        // chosen. The strategy is picked BEFORE the grid -- getSKReduction()
        // feeds getPersistentGridImpl() as an input -- so the grid can still land on a
        // splitting factor F = grid / tiles of 1, which parallel reduction
        // cannot express: it splits each output tile across F workgroups and
        // sums the partials in a second kernel, so F < 2 leaves it nothing to
        // reduce (solve() rejects it outright). Ways the grid gets there:
        //
        //   * the uniform-summation-order F-star snap in getPersistentGridImpl() finds
        //     no admissible F >= 2 (workspace too small for 2 * tiles partial
        //     tiles, or ItersPerTile / F below MinItersPerCU) and falls back to
        //     the all-full grid == tiles;
        //   * the same snap sees g0 < tiles and snaps up to tiles;
        //   * persistentFixedGrid / persistentMaxCUs / persistentGridMultiplier / the analytical grid
        //     land anywhere in [1, 2 * tiles).
        //
        // Tree reduction is always expressible at those grids, so fall back to
        // it rather than failing the launch. Applied AFTER the grid is final
        // and never fed back into grid selection, so it cannot perturb the
        // grid.
        //
        // Both requiredWorkspaceSize() -- the workspace query the caller sizes
        // its allocation from -- and resolveStreamKSettings() -- what solve()
        // launches -- call this on the same (reduction, grid, tiles) triple
        // immediately after getPersistentGridImpl(). For SK3 and SK5-static, where the
        // two start from the same getSKReduction() answer, that keeps query and
        // launch agreed on the reduction and so on the workspace. SK4 and
        // SK5-dynamic are pinned to tree at the launch sites only; see
        // requiredWorkspaceSize() for why the resulting divergence is bounded.
        //
        // Unconditional, not uniform-summation-order enforcement: F < 2 is
        // unlaunchable for parallel whatever the mode, and solve() still throws
        // on the triple. Demoting here keeps that throw unreachable.
        static inline origami::reduction_t reconcileReduction(
            origami::reduction_t reductionStrat, size_t skGrid, size_t tiles)
        {
            if(reductionStrat == origami::reduction_t::parallel
               && (tiles == 0 || (skGrid / tiles) < 2))
                return origami::reduction_t::tree;
            return reductionStrat;
        }

        // Whether a Stream-K workgroup can stop part-way through a tile, which
        // is the only thing the partial-tile workspace ever holds.
        //
        // itersPerTile must be the same value makeArgs() packs into the
        // ItersPerTile kernarg -- getItersPerTile() clamped to at least 1 --
        // or host and device disagree about which branch the kernel takes.
        //
        // This covers tile straddling only. Parallel reduction writes partials
        // unconditionally through its own path at any itersPerTile, so every
        // caller keeps that as a separate check.
        static inline bool straddlesTiles(size_t itersPerTile, size_t skGrid, size_t tiles)
        {
            return skGrid > 0 && itersPerTile > 1 && (tiles % skGrid) != 0;
        }

        /**
         * Whether the per-tile StreamK split applies to this launch: the extras are
         * redistributed within each tile rather than in one global first-E pass.
         * Requires a kernel with InternalArgsSupport::perTileExtraIters.
         *
         * @param tiles         Batch-inclusive tile count.
         * @param itersPerTile  Clamped iterations per tile.
         */
        TENSILELITEHOST_EXPORT static bool usePerTileSplit(ContractionSolution const&    solution,
                                                           ContractionProblemGemm const& problem,
                                                           PersistentLaunchSettings const& launch,
                                                           size_t                        tiles,
                                                           size_t                        itersPerTile);
    };
}
