// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/UtilsOrigami.hpp>

#include <origami/streamk.hpp>

namespace TensileLite
{
    // The pieces of the persistent launch that resolveDefaultPersistentLaunch()
    // and ContractionSolution's other persistent paths share.
    namespace PersistentLaunch
    {
        // The dynamic-queue StreamK kernels (SK4 and the SK4 sub-path of SK5)
        // bake a fixed power-of-two per-XCD queue count for fast index masking.
        // Codegen derives it from the arch's XCD count (StreamK.py
        // _wsQueueConstants / archCaps["NumXCD"], mirroring origami
        // get_default_num_xcds); the host reads the SAME origami value here so
        // codegen and the runtime guard stay in lockstep. Returns 0 when the
        // architecture cannot be determined (guard treats that as unsupported).
        inline size_t streamKBakedQueueCount(origami::hardware_t const* analytical)
        {
            if(analytical == nullptr)
                return 0;
            try
            {
                return origami::hardware_t::get_default_num_xcds(analytical->arch);
            }
            catch(std::exception const&)
            {
                // origami throws for architectures without a hardcoded default
                // XCD count; treat that as "cannot determine" (0 == unsupported)
                // rather than propagating the exception through solution
                // selection.
                return 0;
            }
        }

        // Per-XCD counter stride (bytes) for the dynamic-queue work-queue
        // region. Set equal to the hardware L2 cache-line size so each per-XCD
        // atomic counter occupies its own line (no false sharing). Sourced from
        // origami (hardware_t::get_default_cache_line_bytes) -- the SAME value
        // the codegen mirrors via rocisa archCaps["CacheLineBytes"]
        // (StreamK.py _wsQueueConstants). Host (origami) and codegen (archCaps)
        // strides are two mirrors of the one origami cache-line size, so the
        // flag region the host sizes matches the layout the kernel addresses.
        // Returns 0 when the architecture cannot be determined.
        inline size_t streamKPerQueueStrideBytes(origami::hardware_t const* analytical)
        {
            if(analytical == nullptr)
                return 0;
            return origami::hardware_t::get_default_cache_line_bytes(analytical->arch);
        }

        // Bytes the per-XCD work-queue counters occupy: one cache line each, so
        // no two share a line. The kernel places them at the base of the
        // AddressFlags buffer and starts the ready flags after them (StreamK.py
        // _wsFlagsBaseOffset is this same product). 0 when the architecture
        // cannot be determined.
        inline size_t streamKQueueRegionBytes(origami::hardware_t const* analytical)
        {
            return streamKPerQueueStrideBytes(analytical) * streamKBakedQueueCount(analytical);
        }

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
        inline origami::reduction_t streamKReconcileReduction(origami::reduction_t reductionStrat,
                                                              size_t               skGrid,
                                                              size_t               tiles)
        {
            if(reductionStrat == origami::reduction_t::parallel
               && (tiles == 0 || (skGrid / tiles) < 2))
                return origami::reduction_t::tree;
            return reductionStrat;
        }

        // streamK5EffectiveDynamic() over what it reads: `shape` and `kernel`
        // (macro tile and occupancy) feed origami's hybrid-mode heuristic.
        __attribute__((always_inline)) inline bool
            streamKHybridDynamic(origami::problem_t const&  shape,
                                 origami::config_t const&   kernel,
                                 origami::hardware_t const* analyticalHardware,
                                 int                        requestedMode,
                                 int                        smCountTarget,
                                 char const*                kernelName,
                                 int                        forceMode,
                                 bool                       printModeSelection)
        {
            bool        effectiveDynamic = false;
            const char* reasonStr        = "default";

            if(forceMode == 0)
            {
                effectiveDynamic = false;
                reasonStr        = "force-env";
            }
            else if(forceMode == 1)
            {
                effectiveDynamic = true;
                reasonStr        = "force-env";
            }
            else
            {
                // -1 (or any non 0/1) -> respect the API attribute and run the
                // original requested-mode logic.
                bool runHeuristic = false;
                switch(requestedMode)
                {
                case 1: // ON -> dynamic (SK4) path
                    effectiveDynamic = true;
                    reasonStr        = "api-on";
                    break;
                case 0: // OFF -> static (SK3) unless sm_count_target engages heuristic
                    if(smCountTarget <= 0)
                    {
                        effectiveDynamic = false;
                        reasonStr        = "api-off-static";
                    }
                    else
                    {
                        runHeuristic = true;
                        reasonStr    = "api-off-heuristic";
                    }
                    break;
                case 2: // AUTO -> origami hybrid-mode heuristic
                    runHeuristic = true;
                    reasonStr    = "api-auto-heuristic";
                    break;
                default:
                    effectiveDynamic = false;
                    reasonStr        = "default";
                    break;
                }

                if(runHeuristic)
                {
                    TENSILE_ASSERT_EXC(analyticalHardware != nullptr);
                    const auto autoMode = origami::streamk::select_hybrid_mode(
                        shape, *analyticalHardware, kernel, static_cast<size_t>(smCountTarget));
                    effectiveDynamic = autoMode == origami::hybrid_mode_t::dynamic;
                }
            }

            if(printModeSelection)
            {
                const char* reqStr = (requestedMode == 0   ? "OFF"
                                      : requestedMode == 1 ? "ON"
                                      : requestedMode == 2 ? "AUTO"
                                                           : "?");
                std::cerr << "TensileLite::DEBUG: SK5 hybrid mode for kernel '" << kernelName
                          << "': requested=" << reqStr << " forceEnv=" << forceMode
                          << " smCountTarget=" << smCountTarget << " reason=" << reasonStr
                          << " -> effective="
                          << (effectiveDynamic ? "dynamic(SK4/work-queue)"
                                               : "static(SK3/static-tile)")
                          << "\n";
            }

            return effectiveDynamic;
        }

        inline size_t dynamicQueueGrid(size_t tiles, size_t cuCount, int occupancy, int maxCUs)
        {
            // Limit workgroups per CU to 3
            // TODO Verify this limit is best
            const size_t kernelOccupancy
                = std::min(static_cast<size_t>(std::max(occupancy, 1)), size_t{3});
            auto maxGrid = cuCount * kernelOccupancy;
            if(maxCUs > 0)
            {
                maxGrid = std::min(maxGrid, static_cast<size_t>(maxCUs));
            }
            // TODO Calculate total work items when dynamic queue works with stream-k
            // For now, all work items are full tiles
            auto workItems = tiles;
            // Select grid to use all CUs, unless number of work items is less
            return std::min(workItems, maxGrid);
        }

        // Fold both CU budgets into origami_problem.num_cus (the single
        // source of truth select_grid_size derives its budget from).
        // smCountTarget and persistentMaxCUs each use 0 to mean "no cap"; take the
        // tighter (minimum) positive cap so the analytical path honors both.
        inline size_t origamiCuBudget(int smt, int maxCUs)
        {
            size_t budget = 0; // 0 = use all CUs
            if(smt > 0)
                budget = static_cast<size_t>(smt);
            if(maxCUs > 0)
                budget = (budget == 0) ? static_cast<size_t>(maxCUs)
                                       : std::min(budget, static_cast<size_t>(maxCUs));
            return budget;
        }

        // `shape` as origami::streamk::select_grid_size() takes it.
        inline origami::problem_t origamiGridProblem(origami::problem_t shape,
                                                     bool               transA,
                                                     bool               transB,
                                                     rocisa::DataType   aType,
                                                     rocisa::DataType   bType,
                                                     rocisa::DataType   computeInputType)
        {
            shape.a_transpose = transA ? origami::transpose_t::T : origami::transpose_t::N;
            shape.b_transpose = transB ? origami::transpose_t::T : origami::transpose_t::N;
            shape.a_dtype     = datatypeToAnalyticalDatatype(aType);
            shape.b_dtype     = datatypeToAnalyticalDatatype(bType);
            shape.mi_dtype    = datatypeToAnalyticalDatatype(computeInputType);
            return shape;
        }

        inline void notePersistentOccupancy(char const* kernelName,
                                            int         occupancy,
                                            bool        printPropertyEvaluation)
        {
            if(printPropertyEvaluation && occupancy <= 0)
            {
                std::cerr << "TensileLite::DEBUG: sizeMapping.CUOccupancy=" << occupancy
                          << " (<=0) for kernel '" << kernelName
                          << "'; clamping to 1 for origami grid selection.\n";
            }
        }

        // hipExtModuleLaunchKernel takes uint32 work-item counts.
        // Whole-tile scheduling still needs this launch bound even
        // though it does not use StreamK's tree-fixup arithmetic.
        inline size_t
            dataParallelLaunchGrid(size_t grid, size_t threads, size_t cuCount, int occupancy)
        {
            const size_t maxGrid
                = std::numeric_limits<uint32_t>::max() / std::max(threads, size_t{1});
            if(grid > maxGrid)
                grid = std::min(cuCount * std::max(occupancy, 1), maxGrid);
            return grid;
        }

        // Tree-fixup uses scalarUInt24DivideAndRemainder (dividend < 2^24, divisor < 2^16).
        // If we exceed those bounds, fall back to DP.
        inline size_t streamKTreeFixupGrid(size_t grid,
                                           size_t tiles,
                                           size_t itersPerTile,
                                           size_t threadsPerWorkGroup,
                                           size_t cuCount,
                                           int    occupancy,
                                           bool*  outTreeBoundsFallback)
        {
            size_t itersPerWG = tiles * itersPerTile / grid;
            // numWorkItems = grid * threadsPerWorkGroup
            // the largest grid this fallback can express is
            // UINT32_MAX / threadsPerWorkGroup -- 2^24 tiles at 256 threads
            // but 2^25 at 128 and 2^22 at 1024.
            const size_t maxTiles = (size_t{std::numeric_limits<uint32_t>::max()} + 1)
                                    / std::max(threadsPerWorkGroup, size_t{1});

            if(itersPerTile >= 65536 || itersPerWG >= 65536 || (tiles * itersPerTile) >= maxTiles)
            {
                grid = (tiles >= maxTiles) ? cuCount * std::max(occupancy, 1) : tiles;
                if(outTreeBoundsFallback)
                    *outTreeBoundsFallback = true;
            }
            return grid;
        }

        // The flag region holds one int per Stream-K workgroup. The static
        // paths index it by CTA id ("flag offset based on CTA index" in
        // StreamK.py); SK4 and the SK4 sub-path of SK5 index it by partial
        // index instead ("flag offset based on partial index", StreamK.py
        // ~1742 and ~4033) off a base that already skips the work-queue
        // counters. Either way a grid wider than what that indexing reaches
        // would have workgroups writing past the end of their own block.
        // Every caller reaches grid through here and this is the last
        // write, so it is the one place the bound has to hold.
        //
        // Only the launches that actually reach the flags are bounded:
        //
        //   - atomic and ForceDPOnly kernels never take the pointer at all
        //     (see the Flags kernarg in singleCallArgs)
        //   - parallel reduction is passed Flags == nullptr, and the kernel
        //     branches on that to skip the flag protocol
        //   - tiles % grid == 0 spreads the tiles evenly over the
        //     workgroups, so no partial tiles exist to fix up. It is what
        //     excludes every data-parallel path: they all arrive at grid ==
        //     tiles, whether from K == 0, from the tree-fixup bound above,
        //     or from the data-parallel debug knob. The workspace sizing
        //     additionally skips itersPerTile == 1 (streamKStraddlesTiles),
        //     which this clamp does not: clamping a grid whose flags are
        //     never written is only conservative, and widening the grid
        //     here would be a grid decision, not an allocation one.
        //
        // Clamping outside those cases would shrink a grid that has no flag
        // region to overrun, and against the data-parallel fallbacks it
        // would do real harm: the tree-fixup bound sets grid = tiles
        // precisely to leave Stream-K behind, and cutting that back would
        // return the launch to the K-split it was escaping. grid defaults
        // to the CU count, well inside the bound on a 256-CU gfx950, but
        // TENSILE_STREAMK_GRID_MULTIPLIER scales it with no cap of its own.
        //
        // `usesDynamicQueue` is asked only when the launch reaches the flags.
        template <typename UsesDynamicQueue>
        __attribute__((always_inline)) inline size_t
            streamKFlagRegionGrid(size_t                     grid,
                                  size_t                     tiles,
                                  bool                       atomic,
                                  bool                       dataParallel,
                                  origami::reduction_t       reductionStrat,
                                  UsesDynamicQueue&&         usesDynamicQueue,
                                  origami::hardware_t const* analyticalHardware,
                                  char const*                kernelName,
                                  bool                       printPropertyEvaluation)
        {
            const bool usesFlagRegion = !atomic && !dataParallel
                                        && reductionStrat != origami::reduction_t::parallel
                                        && grid > 0 && (tiles % grid) != 0;

            // SK4 and the SK4 sub-path of SK5 start their flags after the
            // per-XCD work-queue counters, so they index fewer entries than the
            // region holds. The rest index from 0 and keep all of them;
            // tightening every path would cost them grid they are entitled to.
            size_t flagEntries = StreamKFlagElements;
            // The && short-circuits, so a launch that never reaches the flags
            // does not pay for resolving the SK5 sub-mode.
            if(usesFlagRegion && usesDynamicQueue())
            {
                const size_t prefixEntries
                    = streamKQueueRegionBytes(analyticalHardware) / sizeof(int);
                // Subtracting is only safe while the prefix leaves entries
                // behind. Neither way in is reachable today: an unknown arch
                // reports a 0-byte prefix and is rejected before this by
                // streamKDynamicQueueUnsupported, and no shipping part has a
                // cache line * XCD product anywhere near StreamKFlagElements
                // ints. Should one ever get here the full bound is kept, which
                // is the unsafe direction -- it would hand back grid past the
                // flags the kernel indexes -- so this guard is a placeholder
                // for a real decision, not a safe fallback.
                if(prefixEntries < flagEntries)
                    flagEntries -= prefixEntries;
            }

            if(usesFlagRegion && grid > flagEntries)
            {
                if(printPropertyEvaluation)
                {
                    std::cerr << "TensileLite::DEBUG: kernel '" << kernelName << "' StreamK grid "
                              << grid << " exceeds the " << flagEntries << " usable entries of the "
                              << StreamKFlagElements << "-entry flag region (tiles=" << tiles
                              << "); clamping the grid to " << flagEntries << ".\n";
                }
                grid = flagEntries;
            }
            return grid;
        }
    } // namespace PersistentLaunch

    // The launch ContractionSolution::resolvePersistentSettings() resolves in
    // the case DefaultPersistentLaunchInputs describes, before its workspace
    // check.
    __attribute__((always_inline)) inline DefaultPersistentLaunch
        resolveDefaultPersistentLaunch(DefaultPersistentLaunchInputs const& in)
    {
        using namespace PersistentLaunch;

        TENSILE_ASSERT_EXC(in.kernel != nullptr && in.hardware != nullptr);
        constexpr auto gridSelection = origami::grid_selection_t::k_split_aware;
        // CU budget hint; 0 = use all CUs.
        origami::problem_t const shape = {.size    = {in.m, in.n, in.k},
                                          .batch   = in.batch,
                                          .num_cus = static_cast<size_t>(in.smCountTarget)};

        bool const effectiveDynamic
            = in.workAssignment == WorkAssignment::Hybrid
              && (in.effectiveDynamic != nullptr
                      ? *in.effectiveDynamic
                      : streamKHybridDynamic(shape,
                                             *in.kernel,
                                             in.hardware,
                                             in.schedulingMode,
                                             in.smCountTarget,
                                             in.kernelName,
                                             in.debug.streamK5ForceMode,
                                             in.debug.printStreamKModeSelection));
        bool const dynamicQueue
            = in.workAssignment == WorkAssignment::DynamicWorkQueue || effectiveDynamic;

        DefaultPersistentLaunch out;
        if(in.streamK)
        {
            out.effectiveWorkAssignment
                = dynamicQueue ? WorkAssignment::DynamicWorkQueue : WorkAssignment::StaticGrid;
            out.reduction = dynamicQueue ? origami::reduction_t::tree
                                         : origami::streamk::select_reduction(
                                               shape, *in.hardware, *in.kernel, gridSelection);
        }
        else
            out.effectiveWorkAssignment = in.workAssignment;

        size_t grid = 0;
        if(dynamicQueue)
            grid = dynamicQueueGrid(in.tiles, in.computeUnits, in.occupancy, 0);
        else
        {
            origami::problem_t gridShape     = shape;
            gridShape.num_cus                = origamiCuBudget(in.smCountTarget, 0);
            origami::problem_t const problem = origamiGridProblem(
                gridShape, in.transA, in.transB, in.aType, in.bType, in.computeInputType);
            notePersistentOccupancy(in.kernelName, in.occupancy, in.debug.printPropertyEvaluation);
            origami::config_t config  = *in.kernel;
            config.workspace_size     = in.workspaceSize;
            config.reduction_strategy = in.streamK ? out.reduction : origami::reduction_t::tree;
            grid = origami::streamk::select_grid_size(problem, *in.hardware, config, gridSelection);
        }
        out.selectedGrid = grid;

        if(!in.streamK)
        {
            out.grid = dataParallelLaunchGrid(grid, in.threads, in.computeUnits, in.occupancy);
            return out;
        }

        if(out.reduction == origami::reduction_t::tree)
            grid = streamKTreeFixupGrid(grid,
                                        in.tiles,
                                        in.itersPerTile,
                                        in.threads,
                                        in.computeUnits,
                                        in.occupancy,
                                        nullptr);
        out.grid = streamKFlagRegionGrid(
            grid,
            in.tiles,
            in.atomic,
            false,
            out.reduction,
            [&] { return dynamicQueue; },
            in.hardware,
            in.kernelName,
            in.debug.printPropertyEvaluation);
        out.reduction = streamKReconcileReduction(out.reduction, out.grid, in.tiles);
        return out;
    }
} // namespace TensileLite
