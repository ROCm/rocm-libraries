// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/StreamK.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include <origami/streamk.hpp>

#include <algorithm>
#include <iostream>
#include <mutex>

namespace TensileLite
{
    StreamK::Static::Split StreamK::Static::computeSplit(size_t tiles,
                                                         size_t itersPerTile,
                                                         size_t skGrid,
                                                         int    skFullTiles,
                                                         bool   forceDPOnly,
                                                         size_t kGranularity)
    {
        Split split;
        if(skGrid == 0)
            return split;

        // Two-tile algorithm: each workgroup runs an even number of Stream-K
        // iterations followed by an even number of data-parallel tiles. When
        // the grid divides the tile count no Stream-K tiles are needed at all
        // and every tile is data-parallel. Force-DP-only is a persistent
        // DP-only use of StreamK=3: skTiles of zero keeps every output tile in
        // the DP region.
        const bool bigEnough = tiles > skGrid;
        uint32_t   skTiles   = forceDPOnly ? 0u : static_cast<uint32_t>(skGrid);
        if(!forceDPOnly && tiles % skGrid != 0)
        {
            skTiles = bigEnough ? static_cast<uint32_t>(skGrid * skFullTiles + tiles % skGrid)
                                : static_cast<uint32_t>(tiles);
            // Cap Stream-K tiles at total number of tiles in case of large multiplier
            skTiles = std::min(skTiles, static_cast<uint32_t>(tiles));
        }

        split.args.skTiles = skTiles;
        // Rounded down to a whole number of K units. A kernel whose K unit spans
        // several iterations cannot start a workgroup partway through one, and
        // the device hands the leftover out a whole unit at a time, so both the
        // stride and the remainder have to be multiples of the unit. The flat
        // space counts padded tiles (StreamK::paddedItersPerTile), which is what puts
        // the tile starts the cuts fall between on unit boundaries as well.
        const size_t g           = std::max(kGranularity, size_t{1});
        const size_t paddedIters = StreamK::paddedItersPerTile(itersPerTile, g);
        const size_t skIters     = static_cast<size_t>(skTiles) * paddedIters;
        split.args.itersPerWG    = static_cast<uint32_t>(g * (skIters / g / skGrid));
        // Global leftover under the historical first-E mapping. The device may
        // redistribute these extras within each tile when
        // InternalArgsSupport::perTileExtraIters is set and skGrid % skTiles == 0;
        // the packed value itself is unchanged.
        split.extraIters = static_cast<uint32_t>(
            skIters - static_cast<size_t>(split.args.itersPerWG) * skGrid);
        return split;
    }

    bool StreamK::Static::isRowUniform(Split const& split,
                                       size_t       tiles,
                                       size_t       itersPerTile,
                                       size_t       skGrid,
                                       bool         perTileCapable,
                                       bool         uniformSummationOrder,
                                       size_t       kGranularity)
    {
        // Chunk lengths are cut from the padded flat space, so the divisibility
        // this checks is against the padded tile length, not the real one.
        const size_t paddedIters = StreamK::paddedItersPerTile(itersPerTile, kGranularity);
        // The kernel branches at runtime on the uniform-summation-order bit the
        // host packs into MagicShiftItersPerTile: with the bit clear it runs the
        // historical global first-E mapping even when it supports per-tile
        // extras. Model both terms, or host and device disagree about which tile
        // a workgroup owns.
        const bool perTileActive = perTileCapable && uniformSummationOrder;
        // A tile's fold signature is the ordered list of chunk lengths whose
        // partials are summed to produce it, and two tiles are bitwise equal
        // for identical inputs exactly when their signatures match. Every tile
        // in the launch must therefore share one signature. Ways that happens:
        //
        //  skTiles == 0        force-DP-only, so every tile is whole.
        //  skTiles == tiles && I % skItersPerWG == 0 && extraIters == 0
        //                      no data-parallel region. Includes GridEqualsTiles
        //                      (skItersPerWG == I) and all-partial equal chunks
        //                      (tiles | grid and (grid/tiles) | I).
        //                      Mixed GridDividesTiles (skTiles == grid < tiles,
        //                      skItersPerWG == I) is two-tile DP-first then SK.
        //                      gfx950 skips tree-partials workspace when
        //                      tiles % grid == 0, and the SK half of D is never
        //                      stored (uso-row-sweep C1: 992/1024 rows stay
        //                      poison). Refuse that split until the device path
        //                      writes every tile.
        //  per-tile extras: skTiles == tiles && grid % tiles == 0 with a kernel
        //                      that redistributes extras within each tile.
        //                      extraIters may be nonzero; fold signatures still
        //                      match because each tile gets the same intra-tile
        //                      remainder pattern.
        //
        // Without perTileActive, extraIters != 0 means chunks come in two
        // lengths under the global first-E mapping, so the chunk lattice has no
        // single period. skItersPerWG != 0 is not implied by the rest: tiles == 0
        // is reachable from the grouped-GEMM callers and would otherwise leave
        // I % skItersPerWG undefined.
        if(split.args.skTiles == 0)
            return true;

        if(split.args.itersPerWG != 0 && split.extraIters == 0 && split.args.skTiles == tiles
           && paddedIters % split.args.itersPerWG == 0)
            return true;

        if(perTileActive && split.args.skTiles == tiles && tiles != 0 && skGrid % tiles == 0
           && split.args.itersPerWG != 0)
            return true;

        return false;
    }

    StreamK::Static::IterRange StreamK::Static::workgroupIterRange(size_t w,
                                                                   size_t tiles,
                                                                   size_t itersPerTile,
                                                                   size_t skGrid,
                                                                   bool   perTileCapable,
                                                                   bool   uniformSummationOrder,
                                                                   size_t kGranularity)
    {
        IterRange range;
        if(skGrid == 0 || tiles == 0 || kGranularity == 0)
            return range;

        const size_t g           = kGranularity;
        const size_t paddedIters = StreamK::paddedItersPerTile(itersPerTile, g);
        const size_t totalIters  = tiles * paddedIters;
        // Both the stride and the leftover are whole K units: the device adds
        // g, not 1, to the long chunks, and E/g workgroups get one. The flat
        // space counts padded tiles, so a multiple of g in it is a unit
        // boundary inside whichever tile it lands in.
        const size_t W = g * (totalIters / g / skGrid);
        const size_t E = (totalIters - W * skGrid) / g;

        // The device takes the per-tile branch only when the kernel supports it
        // AND the packed uniform-summation-order bit is set. Mirror both terms,
        // or the host attributes iterations to the wrong workgroup and the fixup
        // reads the wrong partials.
        const bool perTileActive = perTileCapable && uniformSummationOrder;

        if(perTileActive && skGrid % tiles == 0)
        {
            const size_t F    = skGrid / tiles;
            const size_t q    = w / F;
            const size_t s    = w % F;
            const size_t base = g * (paddedIters / g / F);
            const size_t remI = (paddedIters - base * F) / g;
            range.start       = q * paddedIters + s * base + g * std::min(s, remI);
            range.end         = range.start + base + (s < remI ? g : size_t{0});
            return range;
        }

        // Historical global first-E mapping.
        if(w < E)
        {
            range.start = w * (W + g);
            range.end   = range.start + (W + g);
        }
        else
        {
            range.start = E * (W + g) + (w - E) * W;
            range.end   = range.start + W;
        }
        return range;
    }

    // The dynamic-queue StreamK kernels (SK4 and the SK4 sub-path of SK5)
    // bake a fixed power-of-two per-XCD queue count for fast index masking.
    // Codegen derives it from the arch's XCD count (StreamK.py
    // _wsQueueConstants / archCaps["NumXCD"], mirroring origami
    // get_default_num_xcds); the host reads the SAME origami value here so
    // codegen and the runtime guard stay in lockstep. Returns 0 when the
    // architecture cannot be determined (guard treats that as unsupported).
    size_t StreamK::Dynamic::bakedQueueCount(Hardware const& hardware)
    {
        auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
        if(hipAMDGPU == nullptr || hipAMDGPU->analyticalHardware == nullptr)
            return 0;
        try
        {
            return origami::hardware_t::get_default_num_xcds(
                hipAMDGPU->analyticalHardware->arch);
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
    size_t StreamK::Dynamic::perQueueStrideBytes(Hardware const& hardware)
    {
        auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
        if(hipAMDGPU == nullptr || hipAMDGPU->analyticalHardware == nullptr)
            return 0;
        return origami::hardware_t::get_default_cache_line_bytes(
            hipAMDGPU->analyticalHardware->arch);
    }

    // The dynamic-queue fetch / work stealing is only correct when the
    // device's runtime NUM_XCD is a power of two AND equals the baked
    // per-XCD queue count. Returns true (UNSUPPORTED) when the hardware is
    // unknown (not a HipAMDGPU, missing analytical hardware, or no baked
    // per-XCD queue count), when NUM_XCD is 0, not a power of two, or
    // NUM_XCD != baked (e.g. MI300A's 6 XCDs, or a 4-XCD partition of an
    // 8-XCD gfx942). Unknown hardware is treated as UNSUPPORTED: the
    // dynamic-queue solution is then excluded from selection and a
    // non-dynamic-queue solution serves the GEMM. That exclusion is what
    // keeps the flag-region clamp in getSKGrid honest -- it subtracts the
    // work-queue prefix from the grid bound, and on unknown hardware that
    // prefix comes back 0, so the clamp would silently fall through to the
    // full StreamKFlagElements bound and hand the launch grid past the
    // flags its kernel actually indexes. Kept isolated here so it stays
    // trivially unit-testable (see CuCount_test.cpp).
    bool StreamK::Dynamic::queueUnsupported(Hardware const& hardware)
    {
        auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
        if(hipAMDGPU == nullptr || hipAMDGPU->analyticalHardware == nullptr)
            return true;
        size_t baked  = bakedQueueCount(hardware);
        size_t numXCD = hipAMDGPU->analyticalHardware->NUM_XCD;
        return baked == 0 || numXCD == 0 || (numXCD & (numXCD - 1)) != 0
               || numXCD != baked;
    }

    // Emit a single, user-visible warning (not once-per-call spam) when a
    // StreamK dynamic-queue / work-stealing solution is excluded from
    // selection because the device's XCD count does not match the compiled
    // per-XCD queue count. This is what surfaces the reject to the user
    // instead of silently degrading to tree reduction.
    void StreamK::Dynamic::warnQueueUnsupportedOnce(Hardware const& hardware)
    {
        static std::once_flag warnedFlag;
        std::call_once(warnedFlag, [&]() {
            size_t      numXCD    = 0;
            auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            if(hipAMDGPU != nullptr && hipAMDGPU->analyticalHardware != nullptr)
                numXCD = hipAMDGPU->analyticalHardware->NUM_XCD;
            size_t baked = bakedQueueCount(hardware);
            std::cerr << "hipBLASLt Warning: StreamK dynamic-queue (work-stealing) solutions "
                         "require the device's XCD count to be a power of two and to equal the "
                         "compiled per-XCD queue count; this device reports NUM_XCD="
                      << numXCD << " with a compiled per-XCD queue count of " << baked
                      << ", so those solutions are excluded from selection and a "
                         "non-work-stealing solution will be used instead.\n";
        });
    }

    // Workspace queries and launch preparation can resolve the same grid
    // repeatedly. Keep only the last notice per thread so distinct adjustments
    // remain visible without retaining an unbounded set of kernel names.
    void StreamK::warnClusterGridAdjustment(std::string const& kernelName,
                                          size_t requested, size_t aligned, size_t clusterWidth)
    {
        static thread_local std::tuple<std::string, size_t, size_t, size_t> lastNotice;
        auto notice = std::tie(kernelName, requested, aligned, clusterWidth);
        if(lastNotice == notice)
            return;
        lastNotice = notice;
        std::cerr << "hipBLASLt Warning: rounding StreamK worker budget down from " << requested
                  << " to " << aligned << " for ClusterDim=[" << clusterWidth
                  << ",1] (kernel '" << kernelName << "').\n";
    }

    // One-shot notice when uniform-summation-order grid steering displaces a developer
    // override (persistentFixedGrid) or production CU knobs (persistentMaxCUs / persistentGridMultiplier).
    void StreamK::warnUniformityGridSnapOnce(size_t g0, size_t gStar)
    {
        static std::once_flag warnedFlag;
        std::call_once(warnedFlag, [g0, gStar]() {
            std::cerr << "hipBLASLt Warning: uniformSummationOrder steered the Stream-K grid "
                         "from "
                      << g0 << " to " << gStar
                      << " (never upward) so the launch stays row-uniform; "
                         "persistentFixedGrid / persistentMaxCUs / persistentGridMultiplier act as hints under "
                         "uniform summation order.\n";
        });
    }

    // One-shot notice when the queue grid raise overrides a grid or CU limit.
    void StreamK::Dynamic::warnGridRaiseOnce(size_t requested, size_t raised)
    {
        static std::once_flag warnedFlag;
        std::call_once(warnedFlag, [requested, raised]() {
            std::cerr << "hipBLASLt Warning: raised the dynamic Stream-K grid from "
                      << requested << " to " << raised
                      << " so every work queue has a workgroup; persistentFixedGrid / "
                         "persistentMaxCUs below the queue count act as hints for dynamic "
                         "Stream-K.\n";
        });
    }

    StreamK::Dynamic::KernelArguments StreamK::Dynamic::items(Hardware const& hardware,
                                                              size_t          tiles,
                                                              size_t          itersPerTile,
                                                              size_t          kGranularity)
    {
        AMDGPU const*   pAMDGPU       = dynamic_cast<AMDGPU const*>(&hardware);
        const int       overrideTiles = pAMDGPU ? pAMDGPU->skTiles : -1;
        const int       overrideSplit = pAMDGPU ? pAMDGPU->skSplit : -1;
        KernelArguments items;
        items.tiles = overrideTiles > -1 ? static_cast<uint32_t>(overrideTiles) : 0;
        items.split = overrideSplit > -1 ? static_cast<uint32_t>(overrideSplit) : 2;
        // Rounded up to a whole number of K units so that every chunk
        // boundary p*itersPerWI lands on one (see skKGranularity()). The
        // tile is measured padded to a whole unit, so the boundaries stay on
        // units even when its real K is not; the device clamps the last
        // chunk back to ItersPerTile. A unit of one changes nothing.
        const uint32_t kGran = static_cast<uint32_t>(std::max(kGranularity, size_t{1}));
        const uint32_t iters = static_cast<uint32_t>(
            StreamK::paddedItersPerTile(std::max(size_t{1}, itersPerTile), kGran));
        // The real split once the iterations are divided up.
        items.itersPerWI = kGran * CeilDivide(CeilDivide(iters, items.split), kGran);
        items.split      = CeilDivide(iters, items.itersPerWI);
        items.totalItems = (tiles - items.tiles) + items.tiles * items.split;
        return items;
    }

    bool StreamK::usePerTileSplit(ContractionSolution const&      solution,
                                  ContractionProblemGemm const&   problem,
                                  PersistentLaunchSettings const& launch,
                                  size_t                          tiles,
                                  size_t                          itersPerTile)
    {
        if(!solution.internalArgsSupport.perTileExtraIters)
            return false;
        if(problem.getParams().uniformSummationOrder())
            return true;
        // With an integral number of workers per tile, distribute the
        // remainder within each tile. Global first-E distribution otherwise
        // makes workers straddle tile boundaries and creates extra fixups.
        // A single tile, exact K division and whole tiles need no change.
        if(!solution.sizeMapping.hasStaticAssignment()
           || (!solution.customKernel.name.empty() && !solution.customKernel.generated)
           || solution.sizeMapping.useSubtileImpl || solution.problemType.useGradient
           || solution.problemType.sparse || problem.groupedGemm()
           || launch.reduction != origami::reduction_t::tree
           || tiles <= 1 || launch.grid <= tiles || launch.grid % tiles != 0)
            return false;
        const size_t factor = launch.grid / tiles;
        return factor <= itersPerTile && itersPerTile % factor != 0
               && (!solution.usesStreamKSpatialCluster()
                   || launch.clusterSchedule.split.args.skTiles == tiles);
    }

    bool StreamK::isParallelReductionRowUniform(origami::reduction_t reduction,
                                                size_t               grid,
                                                int                  streamKAtomic,
                                                bool                 staticTwoTilePacking,
                                                size_t               tiles)
    {
        // Parallel Stream-K under static two-tile packing maps each workgroup
        // to TileIdx = StreamKIdx // F and PartialIdx = StreamKIdx % F, with
        // F = grid/tiles. Every tile sees the same PartialIdx set and the same
        // per-partial K ranges (extras go by PartialIdx), so identical A rows
        // produce identical D rows when F >= 2 and grid is an exact multiple
        // of tiles. Atomic fixup and non-static ABIs are out of scope.
        if(reduction != origami::reduction_t::parallel)
            return false;
        if(streamKAtomic != 0)
            return false;
        if(!staticTwoTilePacking)
            return false;
        if(grid == 0 || tiles == 0)
            return false;
        if(grid % tiles != 0)
            return false;
        if((grid / tiles) < 2)
            return false;
        return true;
    }

    bool StreamK::Dynamic::usesQueue(SizeMapping const& sizeMapping, bool effectiveDynamic)
    {
        return sizeMapping.hasDynamicAssignment()
               || (sizeMapping.hasHybridAssignment() && effectiveDynamic);
    }
}
