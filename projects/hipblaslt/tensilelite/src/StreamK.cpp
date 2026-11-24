// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/StreamK.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include <origami/streamk.hpp>

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>
#include <tuple>

namespace TensileLite
{
    // The arithmetic of computeSplit, isRowUniform, workgroupIterRange, Dynamic::items and
    // isParallelReductionRowUniform lives in origami::streamk (streamk.hpp), which Origami's own
    // launch model shares.
    StreamK::Static::Split StreamK::Static::computeSplit(size_t tiles,
                                                         size_t itersPerTile,
                                                         size_t skGrid,
                                                         int    skFullTiles,
                                                         bool   forceDPOnly,
                                                         size_t kGranularity)
    {
        const auto split = origami::streamk::static_split(
            tiles, itersPerTile, skGrid, skFullTiles, forceDPOnly, kGranularity);
        Split out;
        out.args.skTiles    = split.sk_tiles;
        out.args.itersPerWG = split.sk_iters_per_wg;
        out.extraIters      = split.extra_iters;
        return out;
    }

    bool StreamK::Static::isRowUniform(Split const& split,
                                       size_t       tiles,
                                       size_t       itersPerTile,
                                       size_t       skGrid,
                                       bool         perTileCapable,
                                       bool         uniformSummationOrder,
                                       size_t       kGranularity)
    {
        return origami::streamk::static_split_row_uniform(
            {split.args.skTiles, split.args.itersPerWG, split.extraIters},
            tiles,
            itersPerTile,
            skGrid,
            perTileCapable,
            uniformSummationOrder,
            kGranularity);
    }

    StreamK::Static::IterRange StreamK::Static::workgroupIterRange(size_t w,
                                                                   size_t tiles,
                                                                   size_t itersPerTile,
                                                                   size_t skGrid,
                                                                   bool   perTileCapable,
                                                                   bool   uniformSummationOrder,
                                                                   size_t kGranularity)
    {
        const auto range = origami::streamk::workgroup_iter_range(
            w, tiles, itersPerTile, skGrid, perTileCapable, uniformSummationOrder, kGranularity);
        return {range.start, range.end};
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
                                            size_t             requested,
                                            size_t             aligned,
                                            size_t             clusterWidth)
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
        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
        const auto    d       = origami::streamk::dynamic_items(tiles,
                                                         itersPerTile,
                                                         kGranularity,
                                                         pAMDGPU ? pAMDGPU->skTiles : -1,
                                                         pAMDGPU ? pAMDGPU->skSplit : -1);
        KernelArguments items;
        items.tiles      = d.sk_tiles;
        items.split      = d.sk_split;
        items.itersPerWI = d.sk_iters_per_wi;
        items.totalItems = d.total_items;
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
        return origami::streamk::parallel_reduction_row_uniform(
            reduction, grid, tiles, streamKAtomic, staticTwoTilePacking);
    }

    bool StreamK::Dynamic::usesQueue(SizeMapping const& sizeMapping, bool effectiveDynamic)
    {
        return sizeMapping.hasDynamicAssignment()
               || (sizeMapping.hasHybridAssignment() && effectiveDynamic);
    }
}
