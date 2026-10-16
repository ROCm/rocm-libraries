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

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/RocRollerWgmKernargs.hpp>

#include <Tensile/hip/HipUtils.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/UtilsOrigami.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include <Tensile/UtilsOrigami.hpp>
#include <iostream>
#include <optional>
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
#include <mxDataGenerator/PreSwizzle.hpp>
#endif
#include <origami/streamk.hpp>
#include <roc/hostnumerics/amd_gpu_layout/mx.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <random>
#include <unordered_map>

#include <sstream>

#ifdef ENABLE_ROCTX
#include <roctracer/roctx.h>
#endif

#define TENSILELITE_TO_STR(x) #x
#define TENSILELITE_ENUMSTR(x) x, TENSILELITE_TO_STR(x)

#define HIP_VERSION_WITH_EXPERT_SCHEDULING_MODE 70353390

namespace TensileLite
{
    namespace
    {
        size_t checkedAdd(size_t a, size_t b)
        {
            if(b > std::numeric_limits<size_t>::max() - a)
                throw std::overflow_error("Spatial cluster workspace exceeds size_t");
            return a + b;
        }

        size_t checkedMultiply(size_t a, size_t b)
        {
            if(b != 0 && a > std::numeric_limits<size_t>::max() / b)
                throw std::overflow_error("Spatial cluster product exceeds size_t");
            return a * b;
        }

        size_t divideUp(size_t a, size_t b)
        {
            if(b == 0)
                throw std::runtime_error("Spatial cluster has a zero tile dimension");
            return a / b + (a % b != 0);
        }

        void checkPersistentArgument(size_t value)
        {
            if(value > std::numeric_limits<uint32_t>::max())
                throw std::overflow_error("StreamK cluster schedule exceeds its 32-bit ABI");
        }

        // Batch stride for pre-swizzled gfx950 MX scales.
        size_t preSwizzledScaleBatchStride(TensorDescriptor const& tensor, size_t boundIndex)
        {
            using namespace roc::hostnumerics::amd_gpu_layout;
            return planMxScaleStorage({tensor.sizes()[1 - boundIndex], tensor.sizes()[boundIndex]},
                                      0,
                                      MxScaleStorageLayout::Gfx950)
                .physicalByteCount;
        }
    }

    std::string toString(CustomArgSemantic arg)
    {
        static const std::array<std::string, static_cast<int>(CustomArgSemantic::COUNT)>
            CustomArgSemanticStrings = {
#define X_MACRO(name) TENSILELITE_TO_STR(name),
                CustomArgSemantic_MACRO
#undef X_MACRO
            };
        return CustomArgSemanticStrings[static_cast<int>(arg)];
    }

    CustomArgSemantic fromStringCustomArgSemantic(std::string& str)
    {
        static const std::map<std::string, CustomArgSemantic> CustomArgSemanticMap = {
#define X_MACRO(name) {TENSILELITE_TO_STR(name), CustomArgSemantic::name},
            CustomArgSemantic_MACRO
#undef X_MACRO
        };

        auto it = CustomArgSemanticMap.find(str);
        if(it == CustomArgSemanticMap.end())
            throw std::runtime_error(concatenate("Invalid CustomArgSemantic value: ", str));
        return it->second;
    }

    std::ostream& operator<<(std::ostream& stream, const CustomArgSemantic& t)
    {
        return stream << toString(t);
    }

    std::istream& operator>>(std::istream& stream, CustomArgSemantic& t)
    {
        std::string strValue;
        stream >> strValue;
        t = fromStringCustomArgSemantic(strValue);
        return stream;
    }

    std::string toString(CustomArgDefinition arg)
    {
        auto s = toString(arg.type) + " " + toString(arg.semantic);
        if(arg.padding > 0)
            s += " padding:" + std::to_string(arg.padding);
        return s;
    }

    std::ostream& operator<<(std::ostream& stream, const CustomArgDefinition& t)
    {
        return stream << toString(t);
    }

    std::istream& operator>>(std::istream& stream, CustomArgDefinition& t)
    {
        std::string strType;
        std::string strSemantic;
        stream >> strType >> strSemantic;
        t.type     = fromStringCustomArgType(strType);
        t.semantic = fromStringCustomArgSemantic(strSemantic);
        return stream;
    }

    std::string toString(CustomGridSize size)
    {
        switch(size)
        {
        case CustomGridSize::One:
            return "One";
        case CustomGridSize::TilesX:
            return "TilesX";
        case CustomGridSize::TilesY:
            return "TilesY";
        case CustomGridSize::Batch:
            return "Batch";
        case CustomGridSize::TilesXY:
            return "TilesXY";
        case CustomGridSize::TilesXYBatch:
            return "TilesXYBatch";
        case CustomGridSize::StreamKWithBatch:
            return "StreamKWithBatch";
        case CustomGridSize::StreamKNoBatch:
            return "StreamKNoBatch";
        case CustomGridSize::PersistentGrid:
            return "PersistentGrid";
        case CustomGridSize::PersistentWithBatch:
            return "PersistentWithBatch";
        case CustomGridSize::PersistentNoBatch:
            return "PersistentNoBatch";
        case CustomGridSize::TilesXYBatchGSU:
            return "TilesXYBatchGSU";
        case CustomGridSize::TilesYGSU:
            return "TilesYGSU";
        case CustomGridSize::ComputeUnits:
            return "ComputeUnits";
        case CustomGridSize::CustomGridSize_Count:
            break;
        }

        throw std::runtime_error(
            concatenate("Invalid CustomGridSize value: ", static_cast<int>(size)));
    }

    CustomGridSize fromStringCustomGridSize(std::string& str)
    {
        static const std::unordered_map<std::string, CustomGridSize> lookup = {
            {"One", CustomGridSize::One},
            {"TilesX", CustomGridSize::TilesX},
            {"TilesY", CustomGridSize::TilesY},
            {"Batch", CustomGridSize::Batch},
            {"TilesXY", CustomGridSize::TilesXY},
            {"TilesXYBatch", CustomGridSize::TilesXYBatch},
            {"StreamKWithBatch", CustomGridSize::StreamKWithBatch},
            {"StreamKNoBatch", CustomGridSize::StreamKNoBatch},
            {"PersistentGrid", CustomGridSize::PersistentGrid},
            {"PersistentWithBatch", CustomGridSize::PersistentWithBatch},
            {"PersistentNoBatch", CustomGridSize::PersistentNoBatch},
            {"TilesXYBatchGSU", CustomGridSize::TilesXYBatchGSU},
            {"TilesYGSU", CustomGridSize::TilesYGSU},
            {"ComputeUnits", CustomGridSize::ComputeUnits},
        };

        auto it = lookup.find(str);
        if(it != lookup.end())
            return it->second;

        throw std::runtime_error(concatenate("Invalid CustomGridSize value: ", str));
    }

    std::ostream& operator<<(std::ostream& stream, const CustomGridSize& t)
    {
        return stream << toString(t);
    }

    std::istream& operator>>(std::istream& stream, CustomGridSize& t)
    {
        std::string strValue;
        stream >> strValue;
        t = fromStringCustomGridSize(strValue);
        return stream;
    }

    std::string toString(CustomWorkspaceType type)
    {
        switch(type)
        {
        case CustomWorkspaceType::None:
            return "None";
        case CustomWorkspaceType::SplitK:
            return "SplitK";
        case CustomWorkspaceType::StreamK:
            return "StreamK";
        case CustomWorkspaceType::StreamKWithReduction:
            return "StreamKWithReduction";
        case CustomWorkspaceType::CustomWorkspaceType_Count:
            break;
        }

        throw std::runtime_error(
            concatenate("Invalid CustomWorkspaceType value: ", static_cast<int>(type)));
    }

    CustomWorkspaceType fromStringCustomWorkspaceType(std::string& str)
    {
        if(str == toString(CustomWorkspaceType::None))
            return CustomWorkspaceType::None;
        else if(str == toString(CustomWorkspaceType::SplitK))
            return CustomWorkspaceType::SplitK;
        else if(str == toString(CustomWorkspaceType::StreamK))
            return CustomWorkspaceType::StreamK;
        else if(str == toString(CustomWorkspaceType::StreamKWithReduction))
            return CustomWorkspaceType::StreamKWithReduction;
        else
            throw std::runtime_error(concatenate("Invalid CustomWorkspaceType value: ", str));
        return CustomWorkspaceType::CustomWorkspaceType_Count;
    }

    std::ostream& operator<<(std::ostream& stream, const CustomWorkspaceType& t)
    {
        return stream << toString(t);
    }

    std::istream& operator>>(std::istream& stream, CustomWorkspaceType& t)
    {
        std::string strValue;
        stream >> strValue;
        t = fromStringCustomWorkspaceType(strValue);
        return stream;
    }

    namespace
    {
        // The dynamic-queue StreamK kernels (SK4 and the SK4 sub-path of SK5)
        // bake a fixed power-of-two per-XCD queue count for fast index masking.
        // Codegen derives it from the arch's XCD count (StreamK.py
        // _wsQueueConstants / archCaps["NumXCD"], mirroring origami
        // get_default_num_xcds); the host reads the SAME origami value here so
        // codegen and the runtime guard stay in lockstep. Returns 0 when the
        // architecture cannot be determined (guard treats that as unsupported).
        inline size_t streamKBakedQueueCount(Hardware const& hardware)
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
        inline size_t streamKPerQueueStrideBytes(Hardware const& hardware)
        {
            auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            if(hipAMDGPU == nullptr || hipAMDGPU->analyticalHardware == nullptr)
                return 0;
            return origami::hardware_t::get_default_cache_line_bytes(
                hipAMDGPU->analyticalHardware->arch);
        }

        // Bytes the per-XCD work-queue counters occupy: one cache line each, so
        // no two share a line. The kernel places them at the base of the
        // AddressFlags buffer and starts the ready flags after them (StreamK.py
        // _wsFlagsBaseOffset is this same product). 0 when the architecture
        // cannot be determined.
        inline size_t streamKQueueRegionBytes(Hardware const& hardware)
        {
            return streamKPerQueueStrideBytes(hardware) * streamKBakedQueueCount(hardware);
        }

        // True when the launch takes the dynamic work-queue path and so carries
        // that region: SK4, and the SK4 sub-mode of SK5. `effectiveDynamic` is
        // the resolved SK5 sub-mode and is ignored for every other value.
        inline bool streamKUsesDynamicQueue(SizeMapping const& sizeMapping, bool effectiveDynamic)
        {
            return sizeMapping.hasDynamicAssignment()
                   || (sizeMapping.hasHybridAssignment() && effectiveDynamic);
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
        inline bool streamKDynamicQueueUnsupported(Hardware const& hardware)
        {
            auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            if(hipAMDGPU == nullptr || hipAMDGPU->analyticalHardware == nullptr)
                return true;
            size_t baked  = streamKBakedQueueCount(hardware);
            size_t numXCD = hipAMDGPU->analyticalHardware->NUM_XCD;
            return baked == 0 || numXCD == 0 || (numXCD & (numXCD - 1)) != 0 || numXCD != baked;
        }

        // Emit a single, user-visible warning (not once-per-call spam) when a
        // StreamK dynamic-queue / work-stealing solution is excluded from
        // selection because the device's XCD count does not match the compiled
        // per-XCD queue count. This is what surfaces the reject to the user
        // instead of silently degrading to tree reduction.
        void warnStreamKDynamicQueueUnsupportedOnce(Hardware const& hardware)
        {
            static std::once_flag warnedFlag;
            std::call_once(warnedFlag, [&]() {
                size_t      numXCD    = 0;
                auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
                if(hipAMDGPU != nullptr && hipAMDGPU->analyticalHardware != nullptr)
                    numXCD = hipAMDGPU->analyticalHardware->NUM_XCD;
                size_t baked = streamKBakedQueueCount(hardware);
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
        void warnStreamKClusterGridAdjustment(std::string const& kernelName,
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
                      << " to " << aligned << " for ClusterDim=[" << clusterWidth << ",1] (kernel '"
                      << kernelName << "').\n";
        }

        // One-shot notice when uniform-summation-order grid steering displaces a developer
        // override (persistentFixedGrid) or production CU knobs (persistentMaxCUs / persistentGridMultiplier).
        void warnStreamKUniformityGridSnapOnce(size_t g0, size_t gStar)
        {
            static std::once_flag warnedFlag;
            std::call_once(warnedFlag, [g0, gStar]() {
                std::cerr << "hipBLASLt Warning: uniformSummationOrder steered the Stream-K grid "
                             "from "
                          << g0 << " to " << gStar
                          << " (never upward) so the launch stays row-uniform; "
                             "persistentFixedGrid / persistentMaxCUs / persistentGridMultiplier "
                             "act as hints under "
                             "uniform summation order.\n";
            });
        }
    }

    StreamKStaticSplit streamKStaticSplit(
        size_t tiles, size_t itersPerTile, size_t skGrid, int skFullTiles, bool forceDPOnly)
    {
        StreamKStaticSplit split;
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

        split.skTiles      = skTiles;
        split.skItersPerWG = static_cast<uint32_t>(skTiles * itersPerTile / skGrid);
        // Global leftover under the historical first-E mapping. The device may
        // redistribute these extras within each tile when
        // InternalArgsSupport::perTileExtraIters is set and skGrid % skTiles == 0;
        // the packed value itself is unchanged.
        split.extraIters
            = static_cast<uint32_t>(static_cast<size_t>(skTiles) * itersPerTile
                                    - static_cast<size_t>(split.skItersPerWG) * skGrid);
        return split;
    }

    bool streamKStaticSplitRowUniform(StreamKStaticSplit const& split,
                                      size_t                    tiles,
                                      size_t                    itersPerTile,
                                      size_t                    skGrid,
                                      bool                      perTileCapable,
                                      bool                      uniformSummationOrder)
    {
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
        if(split.skTiles == 0)
            return true;

        if(split.skItersPerWG != 0 && split.extraIters == 0 && split.skTiles == tiles
           && itersPerTile % split.skItersPerWG == 0)
            return true;

        if(perTileActive && split.skTiles == tiles && tiles != 0 && skGrid % tiles == 0
           && split.skItersPerWG != 0)
            return true;

        return false;
    }

    bool streamKParallelReductionRowUniform(PersistentLaunchSettings const& sk,
                                            int                             streamKAtomic,
                                            bool                            staticTwoTilePacking,
                                            size_t                          tiles)
    {
        // Parallel Stream-K under static two-tile packing maps each workgroup
        // to TileIdx = StreamKIdx // F and PartialIdx = StreamKIdx % F, with
        // F = grid/tiles. Every tile sees the same PartialIdx set and the same
        // per-partial K ranges (extras go by PartialIdx), so identical A rows
        // produce identical D rows when F >= 2 and grid is an exact multiple
        // of tiles. Atomic fixup and non-static ABIs are out of scope.
        if(sk.reduction != origami::reduction_t::parallel)
            return false;
        if(streamKAtomic != 0)
            return false;
        if(!staticTwoTilePacking)
            return false;
        if(sk.grid == 0 || tiles == 0)
            return false;
        if(sk.grid % tiles != 0)
            return false;
        if((sk.grid / tiles) < 2)
            return false;
        return true;
    }

    StreamKWorkgroupIterRange streamKWorkgroupIterRange(size_t w,
                                                        size_t tiles,
                                                        size_t itersPerTile,
                                                        size_t skGrid,
                                                        bool   perTileCapable,
                                                        bool   uniformSummationOrder)
    {
        StreamKWorkgroupIterRange range;
        if(skGrid == 0 || tiles == 0)
            return range;

        const size_t totalIters = tiles * itersPerTile;
        const size_t W          = totalIters / skGrid;
        const size_t E          = totalIters - W * skGrid;

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
            const size_t remI = itersPerTile % F;
            const size_t base = itersPerTile / F;
            range.start       = q * itersPerTile + s * base + std::min(s, remI);
            range.end         = range.start + base + (s < remI ? size_t{1} : size_t{0});
            return range;
        }

        // Historical global first-E mapping.
        if(w < E)
        {
            range.start = w * (W + 1);
            range.end   = range.start + (W + 1);
        }
        else
        {
            range.start = E * (W + 1) + (w - E) * W;
            range.end   = range.start + W;
        }
        return range;
    }

    enum class KERNELARGTYPE
    {
        NORMAL   = 0,
        HBM      = 1,
        USERARGS = 2
    };

    void setVariantToBuffer(ConstantVariant const& value,
                            void*                  buffer,
                            size_t                 bufferLength,
                            rocisa::DataType       type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
        {
            float* f_buffer = (float*)buffer;
            *f_buffer       = *std::get_if<float>(&value);
        }
        break;
        case rocisa::DataType::Double:
        {
            double* d_buffer = (double*)buffer;
            *d_buffer        = *std::get_if<double>(&value);
        }
        break;
        case rocisa::DataType::Half:
        {
            Half* fp16_buffer = (Half*)buffer;
            *fp16_buffer      = *std::get_if<Half>(&value);
        }
        break;
        case rocisa::DataType::Int32:
        {
            int32_t* i32_buffer = (int32_t*)buffer;
            *i32_buffer         = *std::get_if<int32_t>(&value);
        }
        break;
        case rocisa::DataType::BFloat16:
        {
            BFloat16* bf16_buffer = (BFloat16*)buffer;
            *bf16_buffer          = *std::get_if<BFloat16>(&value);
        }
        break;
        case rocisa::DataType::Int8:
        {
            int8_t* i8_buffer = (int8_t*)buffer;
            *i8_buffer        = *std::get_if<int8_t>(&value);
        }
        break;
        default:
        {
            if(bufferLength >= 16) // For complex
            {
                if(type == rocisa::DataType::ComplexFloat)
                {
                    std::complex<float>* c_buffer = (std::complex<float>*)buffer;
                    *c_buffer                     = *std::get_if<std::complex<float>>(&value);
                    return;
                }
                else if(type == rocisa::DataType::ComplexDouble)
                {
                    std::complex<double>* z_buffer = (std::complex<double>*)buffer;
                    *z_buffer                      = *std::get_if<std::complex<double>>(&value);
                    return;
                }
            }
            throw std::runtime_error("Unsupported ConstantVariant append type.");
        }
        }
    }

    class PrintBufferValueClass
    {
    public:
        explicit PrintBufferValueClass(void* buffer, size_t bufferLength, rocisa::DataType type)
            : m_buffer(buffer)
            , m_bufferLength(bufferLength)
            , m_type(type)
        {
        }

        friend std::ostream& operator<<(std::ostream& os, const PrintBufferValueClass& buf)
        {
            buf.printBufferValue(os);
            return os;
        }

    private:
        void printBufferValue(std::ostream& os) const
        {
            switch(m_type)
            {
            case rocisa::DataType::Float:
            {
                float* f_buffer = (float*)m_buffer;
                os << *f_buffer;
            }
            break;
            case rocisa::DataType::Double:
            {
                double* d_buffer = (double*)m_buffer;
                os << *d_buffer;
            }
            break;
            case rocisa::DataType::Half:
            {
                Half* fp16_buffer = (Half*)m_buffer;
                os << *fp16_buffer;
            }
            break;
            case rocisa::DataType::Int32:
            {
                int32_t* i32_buffer = (int32_t*)m_buffer;
                os << *i32_buffer;
            }
            break;
            case rocisa::DataType::BFloat16:
            {
                BFloat16* bf16_buffer = (BFloat16*)m_buffer;
                os << *bf16_buffer;
            }
            break;
            case rocisa::DataType::Int8:
            {
                int8_t* i8_buffer = (int8_t*)m_buffer;
                os << *i8_buffer;
            }
            break;
            default:
            {
                if(m_bufferLength >= 16) // For complex
                {
                    if(m_type == rocisa::DataType::ComplexFloat)
                    {
                        std::complex<float>* c_buffer = (std::complex<float>*)m_buffer;
                        os << *c_buffer;
                    }
                    else if(m_type == rocisa::DataType::ComplexDouble)
                    {
                        std::complex<double>* z_buffer = (std::complex<double>*)m_buffer;
                        os << *z_buffer;
                    }
                }
                throw std::runtime_error("Unsupported ConstantVariant append type.");
            }
            }
        }
        void*            m_buffer;
        size_t           m_bufferLength;
        rocisa::DataType m_type;
    };

    template <typename TAct>
    void setDeviceUserArgs(std::vector<ContractionSolution::Problem> const& problems,
                           ContractionSolution::GroupedInputs const&        inputs,
                           DeviceUserArguments<TAct>*                       args)
    {
        for(int i = 0; i < problems.size(); i++)
        {
            const TensorDescriptor& e = problems[i].tensor(ContractionProblemGemm::TENSOR::E);
            const TensorDescriptor& d = problems[i].d();
            const TensorDescriptor& c = problems[i].c();
            const TensorDescriptor& b = problems[i].b();
            const TensorDescriptor& a = problems[i].a();

            size_t startStrideCD = 1; // FIXME: Magic number
            size_t startStrideAB = 1; // FIXME: Magic number

            auto& arg    = args[i];
            arg.m        = problems[i].problemSizes()[0];
            arg.n        = problems[i].problemSizes()[1];
            arg.batch    = problems[i].problemSizes()[2];
            arg.k        = problems[i].problemSizes()[3];
            arg.d        = const_cast<void*>(inputs.grouped[i].d);
            arg.c        = const_cast<void*>(inputs.grouped[i].c);
            arg.b        = const_cast<void*>(inputs.grouped[i].b);
            arg.a        = const_cast<void*>(inputs.grouped[i].a);
            arg.strideD1 = d.strides()[startStrideCD];
            arg.strideD2 = d.strides()[startStrideCD + 1];
            arg.strideC1 = c.strides()[startStrideCD];
            arg.strideC2 = c.strides()[startStrideCD + 1];
            arg.strideA1 = a.strides()[startStrideAB];
            arg.strideA2 = a.strides()[startStrideAB + 1];
            arg.strideB1 = b.strides()[startStrideAB];
            arg.strideB2 = b.strides()[startStrideAB + 1];
            setVariantToBuffer(
                inputs.grouped[i].alpha, arg.alpha, sizeof(arg.alpha), problems[i].alphaType());
            setVariantToBuffer(
                inputs.grouped[i].beta, arg.beta, sizeof(arg.beta), problems[i].betaType());
            arg.scaleA        = const_cast<void*>(inputs.grouped[i].scaleA);
            arg.scaleB        = const_cast<void*>(inputs.grouped[i].scaleB);
            arg.scaleC        = const_cast<void*>(inputs.grouped[i].scaleC);
            arg.scaleD        = const_cast<void*>(inputs.grouped[i].scaleD);
            arg.bias          = const_cast<void*>(inputs.grouped[i].bias);
            arg.scaleAlphaVec = const_cast<void*>(inputs.grouped[i].scaleAlphaVec);
            arg.e             = const_cast<void*>(inputs.grouped[i].e);
            arg.biasType      = (uint32_t)problems[i].bias().dataType();
            if(problems[i].useE())
            {
                arg.strideE1 = e.strides()[startStrideCD];
                arg.strideE2 = e.strides()[startStrideCD + 1];
            }
            else
            {
                arg.strideE1 = 0;
                arg.strideE2 = 0;
            }
            arg.act0           = (*std::get_if<TAct>(&inputs.grouped[i].activationArgs[0]));
            arg.act1           = (*std::get_if<TAct>(&inputs.grouped[i].activationArgs[1]));
            arg.activationType = (uint32_t)problems[i].getParams().activationEnum();
        }

        bool debug = Debug::Instance().printKernelArguments();
        if(debug)
        {
            std::cout << "Grouped gemm argsPtr kernels: " << std::endl;
            for(size_t i = 0; i < problems.size(); i++)
            {
                PrintBufferValueClass alphaPrint(
                    (void*)args[i].alpha, sizeof(args[i].alpha), problems[i].alphaType());
                PrintBufferValueClass betaPrint(
                    (void*)args[i].beta, sizeof(args[i].beta), problems[i].betaType());
                std::cout << "Gemm " << i << ":" << std::endl;
                std::cout << "   "
                          << "m: " << args[i].m << std::endl;
                std::cout << "   "
                          << "n: " << args[i].n << std::endl;
                std::cout << "   "
                          << "batch: " << args[i].batch << std::endl;
                std::cout << "   "
                          << "k: " << args[i].k << std::endl;
                std::cout << "   "
                          << "D: " << args[i].d << std::endl;
                std::cout << "   "
                          << "C: " << args[i].c << std::endl;
                std::cout << "   "
                          << "A: " << args[i].a << std::endl;
                std::cout << "   "
                          << "B: " << args[i].b << std::endl;
                std::cout << "   "
                          << "strideD1: " << args[i].strideD1 << std::endl;
                std::cout << "   "
                          << "strideD2: " << args[i].strideD2 << std::endl;
                std::cout << "   "
                          << "strideC1: " << args[i].strideC1 << std::endl;
                std::cout << "   "
                          << "strideC2: " << args[i].strideC2 << std::endl;
                std::cout << "   "
                          << "strideA1: " << args[i].strideA1 << std::endl;
                std::cout << "   "
                          << "strideA2: " << args[i].strideA2 << std::endl;
                std::cout << "   "
                          << "strideB1: " << args[i].strideB1 << std::endl;
                std::cout << "   "
                          << "strideB2: " << args[i].strideB2 << std::endl;
                std::cout << "   "
                          << "Alpha: " << alphaPrint << std::endl;
                std::cout << "   "
                          << "Beta: " << betaPrint << std::endl;
                std::cout << "   "
                          << "scaleAlphaVec: " << args[i].scaleAlphaVec << std::endl;
                std::cout << "   "
                          << "bias: " << args[i].bias << std::endl;
                std::cout << "   "
                          << "e: " << args[i].e << std::endl;
                std::cout << "   "
                          << "strideE1: " << args[i].strideE1 << std::endl;
                std::cout << "   "
                          << "strideE2: " << args[i].strideE2 << std::endl;
                std::cout << "   "
                          << "act0: " << args[i].act0 << std::endl;
                std::cout << "   "
                          << "act1: " << args[i].act1 << std::endl;
                std::cout << "   "
                          << "activationType: " << args[i].activationType << std::endl;
            }
        }
    }

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(default)
#endif
    template void
        setDeviceUserArgs<float>(std::vector<ContractionSolution::Problem> const& problems,
                                 ContractionSolution::GroupedInputs const&        inputs,
                                 DeviceUserArguments<float>*                      args);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif

    PerfModel perf;

    static const std::map<ContractionSolution::MatchingTag, const char*>& MatchingTag2StringMap()
    {
        static const std::map<ContractionSolution::MatchingTag, const char*> MatchingTag2String
            = {{TENSILELITE_ENUMSTR(ContractionSolution::MatchingTag::Equal)},
               {TENSILELITE_ENUMSTR(ContractionSolution::MatchingTag::GridBased)},
               {TENSILELITE_ENUMSTR(ContractionSolution::MatchingTag::Range)},
               {TENSILELITE_ENUMSTR(ContractionSolution::MatchingTag::FreeSize)},
               {TENSILELITE_ENUMSTR(ContractionSolution::MatchingTag::Prediction)},
               {TENSILELITE_ENUMSTR(ContractionSolution::MatchingTag::Experimental)},
               {TENSILELITE_ENUMSTR(ContractionSolution::MatchingTag::Others)}};
        return MatchingTag2String;
    }

    std::string ContractionSolution::matchingTag() const
    {
        return MatchingTag2StringMap().at(tag);
    }

    // check if this solution is a CU-Fallback solution for current hardware
    bool ContractionSolution::isFallbackForHW(Hardware const& hardware) const
    {
        using std::static_pointer_cast;

        // return the result if we already tested it.
        if(isFallbackCUSol != -1)
            return (isFallbackCUSol == 1);

        auto hw_pred
            = static_pointer_cast<Predicates::IsSubclass<Hardware, AMDGPU>>(hardwarePredicate);
        auto amdGPU = static_cast<AMDGPU const*>(&hardware);
        // if solution is from a standard cu lib, but current HW is not, then this is a Fallback sol.
        isFallbackCUSol
            = (hw_pred->value->type() == "Processor" && !(amdGPU->isStandardCU())) ? 1 : 0;

        return (isFallbackCUSol == 1);
    }

    // Return magic number.  If magicShift is 0, compute and return it.
    uint32_t ContractionSolution::magicNumberAlg1(uint32_t x, uint32_t* magicShift) const
    {
        uint64_t magicNum;
        *magicShift = 33;
        magicNum    = (1L << *magicShift) / x + 1;
        if((magicNum >> 32) != 0)
        {
            *magicShift = 31;
            magicNum    = (1L << *magicShift) / x + 1;
        }

        assert(magicNum >> 32 == 0); // ensure magic number fits

        return static_cast<uint32_t>(magicNum);
    }

    uint32_t ContractionSolution::magicNumberAlg2(uint32_t d, uint32_t* magicShift) const
    {
        struct mu
        {
            unsigned M; // Magic number,
            int      a; // "add" indicator,
            int      s;
        }; // and shift amount.

        struct mu magu;
        if(d == 0)
        {
            // Make dividend of 0 return 0
            magu.M = 0;
            magu.a = 0;
            magu.s = 0;
        }
        else
        {
            // Must have 1 <= d <= 2**32-1.
            int      p;
            unsigned nc, delta, q1, r1, q2, r2;
            magu.a = 0; // Initialize "add" indicator.
            nc     = -1 - (-d) % d; // Unsigned arithmetic here.
            p      = 31; // Init. p.
            q1     = 0x80000000 / nc; // Init. q1 = 2**p/nc.
            r1     = 0x80000000 - q1 * nc; // Init. r1 = rem(2**p, nc).
            q2     = 0x7FFFFFFF / d; // Init. q2 = (2**p - 1)/d.
            r2     = 0x7FFFFFFF - q2 * d; // Init. r2 = rem(2**p - 1, d).
            do
            {
                p = p + 1;
                if(r1 >= nc - r1)
                {
                    q1 = 2 * q1 + 1; // Update q1.
                    r1 = 2 * r1 - nc;
                } // Update r1.
                else
                {
                    q1 = 2 * q1;
                    r1 = 2 * r1;
                }
                if(r2 + 1 >= d - r2)
                {
                    if(q2 >= 0x7FFFFFFF)
                        magu.a = 1;
                    q2 = 2 * q2 + 1; // Update q2.
                    r2 = 2 * r2 + 1 - d;
                } // Update r2.
                else
                {
                    if(q2 >= 0x80000000)
                        magu.a = 1;
                    q2 = 2 * q2;
                    r2 = 2 * r2 + 1;
                }
                delta = d - 1 - r2;
            } while(p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));

            magu.M = q2 + 1; // Magic number
            magu.s = p - 32; // and shift amount to return
        }

        *magicShift         = magu.s;
        const uint32_t abit = 0x80000000;
        if(magu.a)
            *magicShift |= abit;

        // std::cout << " d=" << d << " M=" << magu.M << " a=" << magu.a << " s=" <<
        // magu.s << "\n";

        return magu.M;
    }

    uint32_t
        ContractionSolution::magicNumber(int magicDivAlg, uint32_t x, uint32_t* magicShift) const
    {
        if(magicDivAlg == 1)
            return magicNumberAlg1(x, magicShift);
        else if(magicDivAlg == 2)
            return magicNumberAlg2(x, magicShift);
        else
            throw std::runtime_error("bad magicDivAlg");
    }

    uint32_t ContractionSolution::smallMagicNumber(uint32_t x) const
    {
        uint64_t  magicNum;
        const int smallMagicShift = 31;
        magicNum                  = (1L << smallMagicShift) / x + 1;
        assert(magicNum >> 32 == 0); // ensure magic number fits
        return static_cast<uint32_t>(magicNum);
    }

    std::vector<size_t> generatePackedIndicesA(ContractionSolution::Problem const& problem,
                                               size_t                              packBatchDims)
    {
        std::vector<size_t> packedIndices;

        // TODO -move packedIndices calc to problem decode.
        for(auto idx = 0; idx < problem.a().dimensions(); idx++)
        {
            bool isSum = problem.boundIndices().end()
                         != std::find_if(problem.boundIndices().begin(),
                                         problem.boundIndices().end(),
                                         [idx](const ContractionProblemGemm::BoundIndex& bi) {
                                             return bi.a == idx;
                                         });

            bool nonPackableBatch = false;
            // TODO - base this check on if the batch is SetConstStrideA=0 - if so,
            // don't pack
            if(!(packBatchDims & 0x1))
            {
                nonPackableBatch
                    = problem.batchIndices().end()
                      != std::find_if(problem.batchIndices().begin(),
                                      problem.batchIndices().end(),
                                      [idx](const ContractionProblemGemm::BatchIndex& bi) {
                                          return bi.a == idx;
                                      });
            }

            if(!isSum && !nonPackableBatch)
                packedIndices.push_back(idx);
        }

        return packedIndices;
    }

    std::vector<size_t> generatePackedIndicesB(ContractionSolution::Problem const& problem,
                                               size_t                              packBatchDims)
    {
        std::vector<size_t> packedIndices;

        // Pack in all non-summation indices, except don't need magic number for the
        // last one
        for(auto idx = 0; idx < problem.b().dimensions(); idx++)
        {
            bool isSum = problem.boundIndices().end()
                         != std::find_if(problem.boundIndices().begin(),
                                         problem.boundIndices().end(),
                                         [idx](const ContractionProblemGemm::BoundIndex& bi) {
                                             return bi.b == idx;
                                         });

            bool nonPackableBatch = false;
            // TODO - base this check on if the batch is SetConstStrideB=0 - if so,
            // don't pack
            if(!(packBatchDims & 0x2))
            {
                nonPackableBatch
                    = problem.batchIndices().end()
                      != std::find_if(problem.batchIndices().begin(),
                                      problem.batchIndices().end(),
                                      [idx](const ContractionProblemGemm::BatchIndex& bi) {
                                          return bi.b == idx;
                                      });
            }

            if(!isSum && !nonPackableBatch)
                packedIndices.push_back(idx);
        }

        return packedIndices;
    }

    namespace
    {
        bool usePerTileStreamKSplit(ContractionSolution const& solution,
                                   ContractionProblemGemm const& problem,
                                   PersistentLaunchSettings const& launch,
                                   size_t tiles, size_t itersPerTile)
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
                       || launch.clusterSchedule.split.skTiles == tiles);
        }

        template <typename KA>
        void appendLegacyDataParallelSchedulingArgs(ContractionSolution const&          solution,
                                                    ContractionSolution::Problem const& problem,
                                                    KA&                                 args,
                                                    uint32_t itersPerTile,
                                                    uint32_t grid)
        {
            // Prebuilt DP layout v0: six u32 slots, no workspace/flag pointer.
            // DataParallel argument layout v1 never uses these scheduling fields.
            uint32_t       magicShift = 0;
            const uint32_t magic      = solution.magicNumber(2, itersPerTile, &magicShift);
            if(solution.internalArgsSupport.perTileExtraIters
               && problem.getParams().uniformSummationOrder() != 0)
                magicShift |= 0x20000000u;
            args.template append<uint32_t>("itersPerTile", itersPerTile);
            args.template append<uint32_t>("magicNumberItersPerTile", magic);
            args.template append<uint32_t>("magicShiftItersPerTile", magicShift);
            args.template append<uint32_t>("SKItersPerWG", 0);
            args.template append<uint32_t>("skGrid", grid);
            args.template append<uint32_t>("skTiles", 0);
        }
    }

    template <bool T_Debug, bool insertKernelArgs, typename KA>
    void ContractionSolution::singleCallArgs(ContractionSolution::Problem const& problem,
                                             ContractionInputs const&            inputs,
                                             uint32_t const&                 workspaceOffsetInByte,
                                             Hardware const*                 hardware,
                                             dim3 const&                     problemNumGroupTiles,
                                             dim3 const&                     numWorkGroups,
                                             KA&                             args,
                                             PersistentLaunchSettings const& launch,
                                             size_t resolvedGlobalAccumulation) const
    {
        validatePersistentLoopArgsVersion();
        if(usesStreamKSpatialCluster())
        {
            if(!launch.clusterSchedule.enabled || launch.argsVersion != 2
               || (launch.reduction == origami::reduction_t::parallel
                   && (handwrittenCustomKernel() || problemType.sparse || problemType.useGradient
                       || sizeMapping.useSubtileImpl)))
                throw std::runtime_error("Invalid resolved StreamK cluster schedule");
            // In-kernel fixup and whole-block fallbacks require a flag buffer.
            if(launch.reduction == origami::reduction_t::tree && inputs.Synchronizer == nullptr)
                throw std::runtime_error(
                    "StreamK cluster multicast requires a non-null flag buffer");
            if(launch.workspaceBytes != 0 && inputs.ws == nullptr)
                throw std::runtime_error(
                    "StreamK cluster multicast requires its resolved partial workspace");
        }
        if(debugKernel)
        {
            args.template appendUnbound<unsigned int*>("debugBuffer");
        }

        TensorDescriptor const& a          = problem.a();
        TensorDescriptor const& mxsa       = problem.mxsa();
        TensorDescriptor const& b          = problem.b();
        TensorDescriptor const& mxsb       = problem.mxsb();
        TensorDescriptor const& c          = problem.c();
        TensorDescriptor const& d          = problem.d();
        TensorDescriptor const& e          = problem.tensor(ContractionProblemGemm::TENSOR::E);
        TensorDescriptor const& bias       = problem.tensor(ContractionProblemGemm::TENSOR::BIAS);
        TensorDescriptor const& compressed = problem.compressed();
        TensorDescriptor const& metadata   = problem.metadata();
        bool const              pointerArrayBatch
            = problem.batchMode() == ContractionProblemGemm::BATCHMODE::POINTER_ARRAY;

        auto [autoWGM, autoWGMXCC, autoWGMXCCCHUNK, autoWGMXCCSPLITK]
            = calculateAutoWGM(problem, hardware, launch.grid);
        auto [autoStaggerUMapping, autoStaggerU, autoStaggerUStrideShift]
            = calculateAutoStaggerU(problem, hardware, launch.grid, autoWGM);
        uint32_t autoGsuVal = calculateAutoGSU(problem, hardware);
        uint32_t gsu = problem.getParams().gsu() > 0 ? problem.getParams().gsu() : autoGsuVal;
        AdaptiveGemmNTAB ntab = calculateAdaptiveGemmNTAB(problem, hardware);

        {
            int idx = 0;
            for(auto size : problem.problemSizes())
            {
                args.template append<uint32_t>(concatenate_if<T_Debug>("size_", idx), size);
                idx++;
            }
        }

        if(internalArgsSupport.version < 3)
        {
            bool singleWSD = false;
            if(resolvedGlobalAccumulation == 1
               && (problemType.computeType != problemType.dType
                   || problemType.activationType != ActivationType::None))
                singleWSD = true;
            // Additional check for General Batched GEMM until GSU and StreamK are supported
            // in General Batched GEMM
            if(gsu > 1 && !sizeMapping.isStreamK()
               && ((singleWSD || resolvedGlobalAccumulation == 2)
                   || (resolvedGlobalAccumulation == 3)))
            {
                args.template append<void const*>("ws_d",
                                                  (uint8_t*)inputs.ws + workspaceOffsetInByte);
                if(resolvedGlobalAccumulation == 3)
                {
                    args.template append<void const*>("c", inputs.c);
                }
                else
                {
                    args.template append<void const*>("ws_c",
                                                      (uint8_t*)inputs.ws + workspaceOffsetInByte);
                }
            }
            else if(problemType.stridedBatched)
            {
                if(sizeMapping.isStreamK() && launch.reduction == origami::reduction_t::parallel)
                {
                    args.template append<void const*>("ws_d",
                                                      (uint8_t*)inputs.ws + workspaceOffsetInByte);
                    args.template append<void const*>("ws_c",
                                                      (uint8_t*)inputs.ws + workspaceOffsetInByte);
                }
                else
                {
                    args.template append<void const*>("d",
                                                      pointerArrayBatch && inputs.batchD
                                                          ? static_cast<void const*>(inputs.batchD)
                                                          : inputs.d);
                    args.template append<void const*>("c",
                                                      pointerArrayBatch && inputs.batchC
                                                          ? static_cast<void const*>(inputs.batchC)
                                                          : inputs.c);
                }
            }
            else
            {
                args.template append<void const* const*>("batchD", inputs.batchD);
                args.template append<void const* const*>("batchC", inputs.batchC);
            }
        }

        if(problemType.stridedBatched)
        {
            args.template append<void const*>("a",
                                              pointerArrayBatch && inputs.batchA
                                                  ? static_cast<void const*>(inputs.batchA)
                                              : problemType.sparse == 1 ? inputs.compressed
                                                                        : inputs.a);
            if(problemType.mxBlockA)
                args.template append<void const*>("mxsa", inputs.mxsa);
            args.template append<void const*>("b",
                                              pointerArrayBatch && inputs.batchB
                                                  ? static_cast<void const*>(inputs.batchB)
                                              : problemType.sparse == 2 ? inputs.compressed
                                                                        : inputs.b);
            if(problemType.mxBlockB)
                args.template append<void const*>("mxsb", inputs.mxsb);
        }
        else
        {
            args.template append<void const* const*>("batchA", inputs.batchA);
            args.template append<void const* const*>("batchB", inputs.batchB);
        }

        if(internalArgsSupport.version < 3)
        {
            if(problemType.sparse)
                args.template append<unsigned char const*>("metadata", inputs.metadata);

            // Additional check for General Batched GEMM until GSU and StreamK are supported
            // in General Batched GEMM
            //
            // Only non-atomic StreamK uses partial workspace and flags.
            // DataParallel writes final tiles without either pointer.
            if(sizeMapping.isStreamK() && sizeMapping.streamKAtomic == 0)
            {
                // Assert hardware is not null
                // For now grouped gemm is not supported and passes nullptr
                TENSILE_ASSERT_EXC(hardware != nullptr);

                // StreamK workspace + flags. Synchronizer has already been pointed
                // at the per-stream Stream-K region by the host for this solution,
                // which is what keeps two concurrent Stream-K kernels from clearing
                // each other's flags.
                args.template append<void const*>("ws", inputs.ws);
                if(launch.reduction == origami::reduction_t::parallel)
                    args.template append<void*>("Flags", nullptr);
                else
                    args.template append<void*>("Flags", inputs.Synchronizer);
            }
        }

        size_t startStrideAB = problemType.useInitialStridesAB ? 0 : 1;
        size_t startStrideCD = problemType.useInitialStridesCD ? 0 : 1;

        if(internalArgsSupport.version < 3)
        {
            // Pass wsStride if it's not in MBSK mode
            bool gsuWSStride
                = gsu > 1 && resolvedGlobalAccumulation != 3 && !sizeMapping.isStreamK();
            bool skWSStride
                = sizeMapping.isStreamK() && launch.reduction == origami::reduction_t::parallel;
            // Additional check for General Batched GEMM until GSU and StreamK are supported
            // in General Batched GEMM
            if(gsuWSStride || skWSStride)
            {
                size_t wsStride = startStrideCD ? d.sizes()[0] : 1;
                for(size_t i = startStrideCD; i < d.dimensions(); i++)
                {
                    args.template append<uint32_t>(concatenate_if<T_Debug>("strideW_D", i),
                                                   wsStride);
                    wsStride *= d.sizes()[i];
                }

                wsStride = startStrideCD ? d.sizes()[0] : 1;
                for(size_t i = startStrideCD; i < c.dimensions(); i++)
                {
                    args.template append<uint32_t>(concatenate_if<T_Debug>("strideW_C", i),
                                                   wsStride);
                    wsStride *= d.sizes()[i];
                }
            }
            else
            {
                for(size_t i = startStrideCD; i < d.dimensions(); i++)
                    args.template append<uint32_t>(concatenate_if<T_Debug>("strideD", i),
                                                   d.strides()[i]);

                for(size_t i = startStrideCD; i < c.dimensions(); i++)
                    args.template append<uint32_t>(concatenate_if<T_Debug>("strideC", i),
                                                   c.strides()[i]);
            }
        }

        for(size_t i = startStrideAB; i < a.dimensions(); i++)
        {
            auto stride_a = problemType.sparse == 1 ? compressed.strides()[i] : a.strides()[i];
            args.template append<uint32_t>(concatenate_if<T_Debug>("strideA", i), stride_a);
        }

        if(problemType.mxBlockA)
            for(size_t i = startStrideAB; i < mxsa.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideMXSA", i),
                                               mxsa.strides()[i]);

        for(size_t i = startStrideAB; i < b.dimensions(); i++)
        {
            auto stride_b = problemType.sparse == 2 ? compressed.strides()[i] : b.strides()[i];
            args.template append<uint32_t>(concatenate_if<T_Debug>("strideB", i), stride_b);
        }

        if(problemType.mxBlockB)
            for(size_t i = startStrideAB; i < mxsb.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideMXSB", i),
                                               mxsb.strides()[i]);

        if(problemType.sparse)
        {
            for(size_t i = startStrideAB; i < a.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideMetadata", i),
                                               metadata.strides()[i]);
        }

        if(internalArgsSupport.version >= 3)
        {
            if(problemType.sparse)
                args.template append<unsigned char const*>("metadata", inputs.metadata);

            // In outer version 3 only Flags stays here; StreamK's workspace
            // pointer is appended after alpha/beta. DataParallel has neither.
            if(sizeMapping.isStreamK() && sizeMapping.streamKAtomic == 0)
            {
                // Assert hardware is not null
                // For now grouped gemm is not supported and passes nullptr
                TENSILE_ASSERT_EXC(hardware != nullptr);

                if(launch.reduction == origami::reduction_t::parallel)
                    args.template append<void*>("Flags", nullptr);
                else
                    args.template append<void*>("Flags", inputs.Synchronizer);
            }
        }

        if(internalArgsSupport.version < 3)
        {
            args.append("alpha", inputs.alpha, problem.alphaType());

            if(problem.alphaType() == rocisa::DataType::Half)
                args.append("alpha_2", inputs.alpha, problem.alphaType());

            if(problemType.useBeta)
            {
                args.append("beta", inputs.beta, problem.betaType());

                if(problem.betaType() == rocisa::DataType::Half)
                    args.append("beta_2", inputs.beta, problem.betaType());
            }

            if(sizeMapping.expertSchedulingMode > 0)
            {
                hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(hardware);
                if(hipAMDGPU
                   && (hipAMDGPU->processor == AMDGPU::Processor::gfx1200
                       || hipAMDGPU->processor == AMDGPU::Processor::gfx1201))
                {
                    int32_t esmRuntimeSupported = 0;
#if HIP_VERSION >= HIP_VERSION_WITH_EXPERT_SCHEDULING_MODE
                    HIP_CHECK_EXC(hipDeviceGetAttribute(&esmRuntimeSupported,
                                                        hipDeviceAttributeExpertSchedMode,
                                                        hipAMDGPU->deviceId));
#endif
                    args.template append<int32_t>("ESMRuntimeSupported", esmRuntimeSupported);
                }
            }
        }

        if(sizeMapping.isPersistentDataParallel())
        {
            const uint32_t itersPerTile = std::max(size_t{1}, problem.getItersPerTile(sizeMapping));
            if(internalArgsSupport.persistentLoopArgsVersion == 1)
            {
                args.template append<uint32_t>("ItersPerTile", itersPerTile);
                args.template append<uint32_t>("PersistentGrid", launch.grid);
            }
            else
                appendLegacyDataParallelSchedulingArgs(
                    *this, problem, args, itersPerTile, launch.grid);
        }
        else if(sizeMapping.isStreamK())
        {
            if(!sizeMapping.hasStaticAssignment() && !sizeMapping.hasDynamicAssignment()
               && !sizeMapping.hasHybridAssignment())
            {
                throw std::runtime_error("TileProcessingStrategy=StreamK requires "
                                         "WorkAssignment=StaticGrid, DynamicWorkQueue, or Hybrid");
            }

            if(gsu > 1)
            {
                std::cerr << "Warning: Stream-K Data Parallel does not support GSU > 1, "
                          << "setting GSU to 1." << std::endl;
                gsu = 1;
            }

            // Dynamic Stream-K uses a different kernel argument layout from Stream-K 3.
            if(sizeMapping.hasDynamicAssignment())
            {
                AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(hardware);
                assert(pAMDGPU != nullptr);
                int overrideTiles = pAMDGPU->skTiles;
                int overrideSplit = pAMDGPU->skSplit;

                auto itersPerTile = std::max(size_t{1}, problem.getItersPerTile(sizeMapping));
                auto tiles        = problem.getNumTiles(sizeMapping, 1);
                // Determine number of stream-k tiles and splitting factor
                uint32_t skTiles = 0;
                uint32_t skSplit = 2;
                // Check for debug overrides
                if(overrideTiles > -1)
                    skTiles = overrideTiles;
                if(overrideSplit > -1)
                    skSplit = overrideSplit;
                // Calculate number of stream-k iterations per workitem
                uint32_t skItersPerWI = CeilDivide(static_cast<uint32_t>(itersPerTile), skSplit);
                // Calculate real splitting factor in case iterations don't divide evenly
                skSplit             = CeilDivide(static_cast<uint32_t>(itersPerTile), skItersPerWI);
                uint32_t totalItems = (tiles - skTiles) + skTiles * skSplit;

                args.template append<uint32_t>("ItersPerTile", itersPerTile);
                args.template append<uint32_t>("TotalItems", totalItems);
                args.template append<uint32_t>("SKTiles", skTiles);
                args.template append<uint32_t>("SKSplit", skSplit);
                args.template append<uint32_t>("SKItersPerWI", skItersPerWI);
                args.template append<uint32_t>("SKGrid", launch.grid);
            }
            else if(sizeMapping.hasHybridAssignment())
            {
                // SK5 hybrid: pack 6 args for the active sub-mode (SK3/SK4 RegSet-alias
                // the same SGPR slots). Mode bit is bit 30 of slot 2 — not bit 31,
                // which magicNumberAlg2 uses as the magic-division "add" indicator.
                // Bit 29 of the same slot carries uniform summation order on the
                // static sub-path; see the wire-format note in the SK3 packer.

                AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(hardware);
                assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);

                auto sk3_tiles        = problem.getNumTiles(sizeMapping, 1);
                auto sk3_itersPerTile = std::max(size_t{1}, problem.getItersPerTile(sizeMapping));

                const bool effectiveDynamic = streamK5EffectiveDynamic(problem, *hardware);

                if(effectiveDynamic)
                {
                    int      overrideTiles = pAMDGPU->skTiles;
                    int      overrideSplit = pAMDGPU->skSplit;
                    uint32_t sk4_skTiles   = 0;
                    uint32_t sk4_skSplit   = 2;
                    if(overrideTiles > -1)
                        sk4_skTiles = overrideTiles;
                    if(overrideSplit > -1)
                        sk4_skSplit = overrideSplit;
                    uint32_t sk4_skItersPerWI
                        = CeilDivide(static_cast<uint32_t>(sk3_itersPerTile), sk4_skSplit);
                    sk4_skSplit
                        = CeilDivide(static_cast<uint32_t>(sk3_itersPerTile), sk4_skItersPerWI);
                    uint32_t sk4_totalItems = (sk3_tiles - sk4_skTiles) + sk4_skTiles * sk4_skSplit;

                    // Slot 2 aliases MagicShiftItersPerTile on the static
                    // sub-path, whose top three bits are abit(31), the SK5 mode
                    // bit(30) and the uniform-summation-order bit(29). The
                    // dynamic sub-path never sets bit 29 -- the device reads 0
                    // and takes the global mapping, which is all it has -- but
                    // skTiles must still not collide with it. Free in practice:
                    // sk4_skTiles <= tiles and a 2^29 tile count is unreachable.
                    TENSILE_ASSERT_EXC((sk4_skTiles & 0xE0000000u) == 0u
                                       && "SK5 SK4 skTiles collides with mode/magic bits");
                    uint32_t packedSkTiles = sk4_skTiles | 0x40000000u;

                    args.template append<uint32_t>("ItersPerTile", sk3_itersPerTile);
                    args.template append<uint32_t>("TotalItems", sk4_totalItems);
                    args.template append<uint32_t>("SKTiles|ModeBit", packedSkTiles);
                    args.template append<uint32_t>("SKSplit", sk4_skSplit);
                    args.template append<uint32_t>("SKItersPerWI", sk4_skItersPerWI);
                    args.template append<uint32_t>("SKGrid", launch.grid);
                }
                else
                {
                    // SK5-off mirrors standalone SK3 arg packing.
                    uint32_t magicNumberItersPerTile;
                    uint32_t magicShiftItersPerTile;
                    magicNumberItersPerTile
                        = magicNumber(2, sk3_itersPerTile, &magicShiftItersPerTile);
                    // Bit 30 = SK5 mode bit, bit 29 = uniform summation order.
                    // Both are stolen from the always-zero 5..30 window of
                    // magicNumberAlg2's shift field; bit 31 stays the "add"
                    // indicator.
                    assert((magicShiftItersPerTile & 0x60000000u) == 0u);
                    // Same host rule as the standalone SK3 packer: capability
                    // AND mode, so a kernel without the runtime gate never sees
                    // the bit.
                    if(internalArgsSupport.perTileExtraIters
                       && problem.getParams().uniformSummationOrder() != 0)
                        magicShiftItersPerTile |= 0x20000000u;

                    uint32_t sk3_skItersPerWG;
                    uint32_t sk3_skTiles;
                    if(launch.reduction == origami::reduction_t::parallel)
                    {
                        uint32_t skSplit = static_cast<uint32_t>(launch.grid / sk3_tiles);
                        sk3_skItersPerWG = static_cast<uint32_t>(sk3_itersPerTile) / skSplit;
                        sk3_skTiles      = skSplit;
                    }
                    else
                    {
                        const StreamKStaticSplit sk3_split = streamKStaticSplit(
                            sk3_tiles, sk3_itersPerTile, launch.grid, pAMDGPU->skFullTiles, false);
                        sk3_skTiles      = sk3_split.skTiles;
                        sk3_skItersPerWG = sk3_split.skItersPerWG;
                    }

                    args.template append<uint32_t>("ItersPerTile", sk3_itersPerTile);
                    args.template append<uint32_t>("MagicNumberItersPerTile",
                                                   magicNumberItersPerTile);
                    args.template append<uint32_t>("MagicShiftItersPerTile",
                                                   magicShiftItersPerTile);
                    args.template append<uint32_t>("SKItersPerWG", sk3_skItersPerWG);
                    args.template append<uint32_t>("skGrid", static_cast<uint32_t>(launch.grid));
                    args.template append<uint32_t>("skTiles", sk3_skTiles);
                }
            }
            else
            {
                auto tiles = usesStreamKSpatialCluster() ? launch.clusterSchedule.blocks
                                                         : problem.getNumTiles(sizeMapping, 1);

                // Clamp minimum iters per tile to 1 to allow stream-k index calculation to work in case K==0
                // In this case no actual iterations will be run, but workgroups will be mapped correctly for beta*C
                auto itersPerTile = std::max(size_t{1}, problem.getItersPerTile(sizeMapping));
                auto totalIters   = tiles * itersPerTile;

                uint32_t magicNumberItersPerTile;
                uint32_t magicShiftItersPerTile;
                magicNumberItersPerTile = magicNumber(2, itersPerTile, &magicShiftItersPerTile);

                // MagicShiftItersPerTile wire format:
                //   31    = magicNumberAlg2's "add" indicator (abit)
                //   30    = SK5 hybrid mode bit (SK5 path only)
                //   29    = distribute extra K iterations within each tile
                //   28..6 = zero
                //   5..0  = shift amount
                // magicNumberAlg2 returns p - 32 with p <= 64 (the loop's post
                // test permits exit at p == 64), so the shift never exceeds 32
                // and bits 6..30 are always clear here.
                //
                // Set iff the kernel has the runtime gate AND the mode is on:
                // without the capability term, a solution whose assembly
                // predates the gate (custom kernels, older logic) would receive
                // a bit it misreads.
                assert((magicShiftItersPerTile & 0x60000000u) == 0u);
                if(usePerTileStreamKSplit(*this, problem, launch, tiles, itersPerTile))
                    magicShiftItersPerTile |= 0x20000000u;

                args.template append<uint32_t>("itersPerTile", itersPerTile);
                args.template append<uint32_t>("magicNumberItersPerTile", magicNumberItersPerTile);
                args.template append<uint32_t>("magicShiftItersPerTile", magicShiftItersPerTile);

                // Hand-written custom kernels still use totalIters; Tensile-generated
                // kernels follow the legacy non-custom layout.
                if(!customKernel.name.empty() && !customKernel.generated)
                {
                    args.template append<uint32_t>("totalIters", totalIters);
                }

                // Stream-K 3 uses the two-tile ABI.
                if(launch.reduction == origami::reduction_t::parallel)
                {
                    uint32_t skSplit
                        = launch.grid / tiles; // skTiles is skSplit in parallel reduction path
                    uint32_t skItersPerWG = itersPerTile / skSplit;

                    args.template append<uint32_t>("SKItersPerWG", skItersPerWG);
                    args.template append<uint32_t>("skGrid", launch.grid);
                    args.template append<uint32_t>("skTiles", skSplit);
                }
                else
                {
                    AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(hardware);
                    assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);

                    const StreamKStaticSplit split
                        = usesStreamKSpatialCluster()
                              ? launch.clusterSchedule.split
                              : streamKStaticSplit(
                                  tiles, itersPerTile, launch.grid, pAMDGPU->skFullTiles, false);

                    args.template append<uint32_t>("SKItersPerWG", split.skItersPerWG);
                    args.template append<uint32_t>("skGrid", launch.grid);
                    args.template append<uint32_t>("skTiles", split.skTiles);
                }
            }
        }

        if(internalArgsSupport.version >= 3)
        {
            args.append("alpha", inputs.alpha, problem.alphaType());

            if(problem.alphaType() == rocisa::DataType::Half)
                args.append("alpha_2", inputs.alpha, problem.alphaType());

            // The beta slot is always emitted so the layout does not depend on UseBeta;
            // the kernel gates the beta math separately. Keep the "beta" name: append()
            // promotes small types to 32-bit only for names "alpha"/"beta", so a
            // differently named pad would be written at the wrong size.
            if(problemType.useBeta)
            {
                args.append("beta", inputs.beta, problem.betaType());

                if(problem.betaType() == rocisa::DataType::Half)
                    args.append("beta_2", inputs.beta, problem.betaType());
            }
            else
            {
                args.append("beta", 0.0f, problem.betaType());

                if(problem.betaType() == rocisa::DataType::Half)
                    args.append("beta_2", 0.0f, problem.betaType());
            }

            // ver3 places AddressWS after alpha/beta, see the StreamK block above.
            if(sizeMapping.isStreamK() && sizeMapping.streamKAtomic == 0)
            {
                args.template append<void const*>("ws", inputs.ws);
            }

            bool singleWSD = false;
            if(resolvedGlobalAccumulation == 1
               && (problemType.computeType != problemType.dType
                   || problemType.activationType != ActivationType::None))
                singleWSD = true;
            // Additional check for General Batched GEMM until GSU and StreamK are supported
            // in General Batched GEMM
            if(gsu > 1 && !sizeMapping.isStreamK()
               && ((singleWSD || resolvedGlobalAccumulation == 2)
                   || (resolvedGlobalAccumulation == 3)))
            {
                args.template append<void const*>("ws_d",
                                                  (uint8_t*)inputs.ws + workspaceOffsetInByte);
                if(resolvedGlobalAccumulation == 3)
                {
                    args.template append<void const*>("c", inputs.c);
                }
                else
                {
                    args.template append<void const*>("ws_c",
                                                      (uint8_t*)inputs.ws + workspaceOffsetInByte);
                }
            }
            else if(problemType.stridedBatched)
            {
                if(sizeMapping.isStreamK() && launch.reduction == origami::reduction_t::parallel)
                {
                    args.template append<void const*>("ws_d",
                                                      (uint8_t*)inputs.ws + workspaceOffsetInByte);
                    args.template append<void const*>("ws_c",
                                                      (uint8_t*)inputs.ws + workspaceOffsetInByte);
                }
                else
                {
                    args.template append<void const*>("d",
                                                      pointerArrayBatch && inputs.batchD
                                                          ? static_cast<void const*>(inputs.batchD)
                                                          : inputs.d);
                    args.template append<void const*>("c",
                                                      pointerArrayBatch && inputs.batchC
                                                          ? static_cast<void const*>(inputs.batchC)
                                                          : inputs.c);
                }
            }
            else
            {
                args.template append<void const* const*>("batchD", inputs.batchD);
                args.template append<void const* const*>("batchC", inputs.batchC);
            }

            // Pass wsStride if it's not in MBSK mode
            bool gsuWSStride
                = gsu > 1 && resolvedGlobalAccumulation != 3 && !sizeMapping.isStreamK();
            bool skWSStride
                = sizeMapping.isStreamK() && launch.reduction == origami::reduction_t::parallel;
            // Additional check for General Batched GEMM until GSU and StreamK are supported
            // in General Batched GEMM
            if(gsuWSStride || skWSStride)
            {
                size_t wsStride = startStrideCD ? d.sizes()[0] : 1;
                for(size_t i = startStrideCD; i < d.dimensions(); i++)
                {
                    args.template append<uint32_t>(concatenate_if<T_Debug>("strideW_D", i),
                                                   wsStride);
                    wsStride *= d.sizes()[i];
                }

                wsStride = startStrideCD ? d.sizes()[0] : 1;
                for(size_t i = startStrideCD; i < c.dimensions(); i++)
                {
                    args.template append<uint32_t>(concatenate_if<T_Debug>("strideW_C", i),
                                                   wsStride);
                    wsStride *= d.sizes()[i];
                }
            }
            else
            {
                for(size_t i = startStrideCD; i < d.dimensions(); i++)
                    args.template append<uint32_t>(concatenate_if<T_Debug>("strideD", i),
                                                   d.strides()[i]);

                for(size_t i = startStrideCD; i < c.dimensions(); i++)
                    args.template append<uint32_t>(concatenate_if<T_Debug>("strideC", i),
                                                   c.strides()[i]);
            }

            if(sizeMapping.expertSchedulingMode > 0)
            {
                hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(hardware);
                if(hipAMDGPU
                   && (hipAMDGPU->processor == AMDGPU::Processor::gfx1200
                       || hipAMDGPU->processor == AMDGPU::Processor::gfx1201))
                {
                    int32_t esmRuntimeSupported = 0;
#if HIP_VERSION >= HIP_VERSION_WITH_EXPERT_SCHEDULING_MODE
                    HIP_CHECK_EXC(hipDeviceGetAttribute(&esmRuntimeSupported,
                                                        hipDeviceAttributeExpertSchedMode,
                                                        hipAMDGPU->deviceId));
#endif
                    args.template append<int32_t>("ESMRuntimeSupported", esmRuntimeSupported);
                }
            }
        }

        if constexpr(insertKernelArgs)
            if(!internalArgsSupport.useUniversalArgs)
                kernelArgs<T_Debug, true>(0,
                                          (uint32_t)KERNELARGTYPE::NORMAL,
                                          args,
                                          0,
                                          hardware,
                                          problem.getParams(),
                                          autoWGM,
                                          autoWGMXCC,
                                          autoWGMXCCCHUNK,
                                          autoWGMXCCSPLITK,
                                          autoStaggerUMapping,
                                          autoStaggerU,
                                          autoStaggerUStrideShift,
                                          autoGsuVal,
                                          ntab);

        // NOTE: an assumption here is A & B must be both MX data types or non-MX data types.
        //       Mixing is not supported.
        if(!problemType.useScaleAB.empty())
        {
            args.template append<void const*>("scaleA", inputs.scaleA);
            args.template append<void const*>("scaleB", inputs.scaleB);
        }
        if(problemType.useScaleCD) //kernel input data
        {
            args.template append<void const*>("scaleC", inputs.scaleC);
            args.template append<void const*>("scaleD", inputs.scaleD);
        }

        if(problemType.useScaleAlphaVec) //kernel input data
        {
            args.template append<void const*>("scaleAlphaVec", inputs.scaleAlphaVec);
        }

        bool runActivation = false;
        if((problemType.activationType != ActivationType::None) && sizeMapping.activationFused)
            runActivation = true;
        if(problemType.useBias)
        {
            // We save the bias data in ws_d
            if(problemType.useGradient && problem.biasSrc() == ContractionProblemGemm::TENSOR::D
               && inputs.bias != nullptr)
                args.template append<void const*>("ws_bias",
                                                  (uint8_t*)inputs.ws + workspaceOffsetInByte);
            else
            {
                if(problemType.stridedBatched)
                {
                    args.template append<void const*>("bias", inputs.bias);
                }
                else
                {
                    args.template append<void const* const*>("batchBias", inputs.batchBias);
                }
            }

            if(!problemType.useGradient
               || (problemType.useGradient
                   && (problem.biasSrc() == ContractionProblemGemm::TENSOR::A
                       || problem.biasSrc() == ContractionProblemGemm::TENSOR::B)))
            {
                args.template append<uint32_t>("bias_type",
                                               static_cast<uint32_t>(problem.bias().dataType()));
                if(problemType.useBias)
                {
                    args.template append<uint32_t>(
                        "strideBias",
                        static_cast<uint32_t>(problem.useBias() && bias.dimensions()
                                                  ? bias.strides()[bias.dimensions() - 1]
                                                  : 0)); // reserved
                }
            }
        }

        if(problemType.useGateResidual)
        {
            if(problemType.stridedBatched)
                args.template append<void const*>("gateResidual", inputs.gateResidual);
            else
                args.template append<void const* const*>("batchGateResidual",
                                                         inputs.batchGateResidual);
            bool hasGate = problem.useGateResidual();
            args.template append<uint32_t>(
                "gate_type",
                static_cast<uint32_t>(
                    hasGate
                        ? problem.tensor(ContractionProblemGemm::TENSOR::GATE_RESIDUAL).dataType()
                        : problemType.gateResidualDataTypeWhiteList.at(0)));

            TensorDescriptor const& gate
                = problem.tensor(ContractionProblemGemm::TENSOR::GATE_RESIDUAL);
            for(size_t i = startStrideCD; i < d.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideGate", i),
                                               hasGate ? gate.strides()[i] : 0);
        }

        if(problemType.useScaleAlphaVec == 3 || problemType.useBias == 3)
        {
            args.template append<uint32_t>("factorDim",
                                           static_cast<uint32_t>(problem.getParams().factorDim()));
        }

        if(problemType.useE)
        {
            args.template append<void*>("e", inputs.e);
            for(size_t i = startStrideCD; i < e.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideE", i),
                                               e.strides()[i]);
        }

        if(runActivation)
        {
            for(int i = 0; i < problemType.activationArgLength; i++)
            {
                std::string name = "activation_" + std::to_string(i);
                if(inputs.activationArgs.size() < problemType.activationArgLength)
                {
                    if(problemType.activationComputeDataType == rocisa::DataType::BFloat16)
                    {
                        args.template append<float>(name.c_str(), 0.f);
                    }
                    else
                    {
                        args.append(name.c_str(), 0, problemType.activationComputeDataType);
                    }
                }
                else
                {
                    if(problemType.activationComputeDataType == rocisa::DataType::BFloat16)
                    {
                        args.template append<float>(name.c_str(),
                                                    static_cast<float>((*std::get_if<BFloat16>(
                                                        &inputs.activationArgs[i]))));
                    }
                    else
                    {
                        args.append(name.c_str(),
                                    inputs.activationArgs[i],
                                    problemType.activationComputeDataType);
                    }
                }
            }
            if(problemType.activationType == ActivationType::All
               || problemType.activationType == ActivationType::Hipblaslt_all)
            {
                args.template append<uint32_t>(
                    "activationType", static_cast<uint32_t>(problem.getParams().activationEnum()));
            }
        }

        if(problemType.outputAmaxD)
        {
            args.template append<const void*>("AddrAmaxOut", inputs.amaxD);
            args.template append<const void*>("AmaxWS",
                                              (uint8_t*)inputs.ws + workspaceOffsetInByte);
            args.template append<const void*>("AmaxSync", inputs.Synchronizer);
        }
    }

    inline uint32_t getNumWorkGroups(const KernelInvocation& rv)
    {
        return rv.numWorkItems.x / rv.workGroupSize.x / rv.workGroupSize.y / rv.workGroupSize.z;
    }

    inline uint32_t getNumWorkGroups(ContractionSolution::Problem const& problem,
                                     const SizeMapping&                  sizeMapping)
    {
        size_t numWorkGroupsX = 1;
        size_t numWorkGroupsY = 1;
        size_t numWorkGroupsZ = 1;

        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
        {
            numWorkGroupsX *= problem.freeSizeA(i);
        }
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
        {
            numWorkGroupsY *= problem.freeSizeB(i);
        }

        for(size_t i = 0; i < problem.batchIndices().size(); i++)
        {
            if(sizeMapping.packBatchDims & 0x1)
                numWorkGroupsX *= problem.batchSize(i);
            if(sizeMapping.packBatchDims & 0x2)
                numWorkGroupsY *= problem.batchSize(i);
            if(!sizeMapping.packBatchDims)
                numWorkGroupsZ *= problem.batchSize(i);
        }

        if(problem.transposeC01())
            std::swap(numWorkGroupsX, numWorkGroupsY);

        numWorkGroupsX = CeilDivide(numWorkGroupsX, sizeMapping.macroTile.x);
        numWorkGroupsY = CeilDivide(numWorkGroupsY, sizeMapping.macroTile.y);

        return numWorkGroupsX * numWorkGroupsY * numWorkGroupsZ;
    }

    std::tuple<int32_t, size_t, size_t, size_t> ContractionSolution::calculateAutoWGM(
        Problem const& problem, Hardware const* hardware, uint32_t const skgrid) const
    {
        if(usesStreamKSpatialCluster())
            return std::make_tuple(1, size_t{1}, size_t{0}, size_t{0});
        // Hardware
        AMDGPU const*         pAMDGPU   = dynamic_cast<AMDGPU const*>(hardware);
        hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(hardware);

        // Default WGM
        int32_t  defaultWGM          = 1;
        uint32_t defaultWGMXCC       = 1;
        uint32_t defaultWGMXCCCHUNK  = 0;
        uint32_t defaultWGMXCCSPLITK = 0;

        // Dynamically pick the values
        if(sizeMapping.isPersistent() && skgrid != 0 && sizeMapping.workGroupMapping == 0
           && sizeMapping.workGroupMappingXCC == -1)
        {
            auto sizes = problem.problemSizes();
            // Try to find cached WGM, WGMXCC, WGMXCCCHUNK, WGMXCCSPLITK
            auto cachedWGMParams = wgmParamsCache.find(problem);

            if(cachedWGMParams == std::make_tuple(INT32_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX))
            {
                if(sizes.size() >= 4)
                {
                    origami::problem_t origami_problem = {
                        .size  = {sizes[0], sizes[1], sizes[3]},
                        .batch = sizes[2],
                        // CU budget hint; 0 = use all CUs.
                        .num_cus = static_cast<size_t>(problem.getParams().smCountTarget()),
                        .a_dtype = datatypeToAnalyticalDatatype(problem.a().dataType()),
                        .b_dtype = datatypeToAnalyticalDatatype(problem.b().dataType()),
                    };
                    origami::config_t origami_config = {
                        .mt            = {static_cast<size_t>(sizeMapping.macroTile.x),
                               static_cast<size_t>(sizeMapping.macroTile.y),
                               static_cast<size_t>(sizeMapping.depthU)},
                        .cache_hints_a = sizeMapping.cacheHintA(),
                        .cache_hints_b = sizeMapping.cacheHintB(),
                    };

                    origami::workgroup_mapping_t prediction_results
                        = origami::select_workgroup_mapping(origami_problem,
                                                            *(hipAMDGPU->analyticalHardware),
                                                            origami_config,
                                                            skgrid);

                    defaultWGM          = prediction_results.wgm;
                    defaultWGMXCC       = prediction_results.wgmxcc;
                    defaultWGMXCCCHUNK  = prediction_results.wgmxccchunk;
                    defaultWGMXCCSPLITK = prediction_results.wgmxccsplitk;

                    // Add to cache only if dynamically calculated.
                    wgmParamsCache.add(
                        std::make_tuple(
                            defaultWGM, defaultWGMXCC, defaultWGMXCCCHUNK, defaultWGMXCCSPLITK),
                        problem);
                    if(Debug::Instance().printPropertyEvaluation())
                        std::cout << "AutoWGM - WGM: " << defaultWGM
                                  << ", WGMXCC: " << defaultWGMXCC
                                  << ", WGMXCCCHUNK: " << defaultWGMXCCCHUNK
                                  << ", WGMXCCSPLITK: " << defaultWGMXCCSPLITK << std::endl;
                }
            }
            else
            {
                std::tie(defaultWGM, defaultWGMXCC, defaultWGMXCCCHUNK, defaultWGMXCCSPLITK)
                    = cachedWGMParams;
            }
        }
        else
        {
            // Default WGM
            if(sizeMapping.workGroupMapping == 0)
            {
                auto numCU  = hipAMDGPU->analyticalHardware->N_CU;
                auto numXCD = hipAMDGPU->analyticalHardware->NUM_XCD;

                defaultWGM = std::ceil(std::sqrt(numCU / numXCD));
            }
            else
                defaultWGM = sizeMapping.workGroupMapping;

            // Default WGMXCC
            if(sizeMapping.workGroupMappingXCC == -1)
                defaultWGMXCC = hipAMDGPU->analyticalHardware->NUM_XCD;
            else
                defaultWGMXCC = sizeMapping.workGroupMappingXCC;

            // Default WGMXCCCHUNK
            defaultWGMXCCCHUNK = 0;

            // Default WGMXCCSPLITK
            defaultWGMXCCSPLITK = 0;
        }

        // If values are explicitly specified at runtime, they override predictions and default values
        if(pAMDGPU->fixedWGM != std::numeric_limits<int>::max())
            defaultWGM = pAMDGPU->fixedWGM;
        if(pAMDGPU->fixedWGMXCC != std::numeric_limits<size_t>::max())
            defaultWGMXCC = pAMDGPU->fixedWGMXCC;
        if(pAMDGPU->fixedWGMXCCCHUNK != std::numeric_limits<size_t>::max())
            defaultWGMXCCCHUNK = pAMDGPU->fixedWGMXCCCHUNK;
        if(pAMDGPU->fixedWGMXCCSPLITK != std::numeric_limits<size_t>::max())
            defaultWGMXCCSPLITK = pAMDGPU->fixedWGMXCCSPLITK;

        // These range assertions only apply when SpaceFillingCurve (SFC) is not used.
        // When SFC is enabled, workGroupMapping contains a packed 32-bit encoding of
        // grid dimensions (SFCWGM) which can exceed the normal WGM range.
        if(!internalArgsSupport.useSFC)
        {
            if(sizeMapping.workGroupMappingXCC == -1)
            {
                // New bit layout: 10 K + 8 chunk + 4 XCC + 10 WGM
                assert(std::fabs(defaultWGM) < 512); // 10-bit signed
                assert(defaultWGMXCC < 16); // 4 bits
                assert(defaultWGMXCCCHUNK < 256); // 8 bits
                assert(defaultWGMXCCSPLITK < 1024); // 10 bits
            }
            else
            {
                // Old bit layout (used when WorkGroupMappingXCC != -1)
                assert(std::fabs(defaultWGM) < 1024);
                assert(defaultWGMXCC >= 0 && defaultWGMXCC < 64);
                assert(defaultWGMXCCCHUNK >= 0 && defaultWGMXCCCHUNK < 1024);
            }
        }

        return std::make_tuple(defaultWGM, defaultWGMXCC, defaultWGMXCCCHUNK, defaultWGMXCCSPLITK);
    }

    std::tuple<size_t, size_t, size_t> ContractionSolution::calculateAutoStaggerU(
        Problem const& problem, Hardware const* hardware, uint32_t skgrid, int32_t autoWGM) const
    {
        if(usesStreamKSpatialCluster())
            return std::make_tuple(size_t{0}, size_t{0}, size_t{0});
        // Hardware
        AMDGPU const*         pAMDGPU   = dynamic_cast<AMDGPU const*>(hardware);
        hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(hardware);

        // Default StaggerU
        size_t defaultStaggerUMapping     = 0;
        size_t defaultStaggerU            = 0;
        size_t defaultStaggerUStrideShift = 0;

        // Dynamically pick the values
        if(sizeMapping.isPersistent() && skgrid != 0 && sizeMapping.workGroupMapping == 0
           && sizeMapping.workGroupMappingXCC == -1)
        {
            auto sizes = problem.problemSizes();
            // Try to find cached StaggerUMapping, StaggerU and StaggerUStrideShift
            auto cachedStaggerUParams = staggerUParamsCache.find(problem);

            if(cachedStaggerUParams == std::make_tuple(SIZE_MAX, SIZE_MAX, SIZE_MAX))
            {
                if(sizes.size() >= 4)
                {
                    origami::problem_t origami_problem = {
                        .size  = {sizes[0], sizes[1], sizes[3]},
                        .batch = sizes[2],
                        // CU budget hint; 0 = use all CUs.
                        .num_cus = static_cast<size_t>(problem.getParams().smCountTarget()),
                        .a_dtype = datatypeToAnalyticalDatatype(problem.a().dataType()),
                        .b_dtype = datatypeToAnalyticalDatatype(problem.b().dataType()),
                    };
                    origami::config_t origami_config = {
                        .mt            = {static_cast<size_t>(sizeMapping.macroTile.x),
                               static_cast<size_t>(sizeMapping.macroTile.y),
                               static_cast<size_t>(sizeMapping.depthU)},
                        .cache_hints_a = sizeMapping.cacheHintA(),
                        .cache_hints_b = sizeMapping.cacheHintB(),
                    };

                    origami::staggerU_t prediction_results
                        = origami::select_staggerU(origami_problem,
                                                   *(hipAMDGPU->analyticalHardware),
                                                   origami_config,
                                                   skgrid,
                                                   autoWGM);

                    defaultStaggerUMapping     = prediction_results.staggerUMapping;
                    defaultStaggerU            = prediction_results.staggerU;
                    defaultStaggerUStrideShift = prediction_results.staggerUStrideShift;

                    // Add to cache only if dynamically calculated.
                    staggerUParamsCache.add(std::make_tuple(defaultStaggerUMapping,
                                                            defaultStaggerU,
                                                            defaultStaggerUStrideShift),
                                            problem);
                    if(Debug::Instance().printPropertyEvaluation())
                        std::cout << "AutoStaggerU - Mapping: " << defaultStaggerUMapping
                                  << ", StaggerU: " << defaultStaggerU
                                  << ", StaggerUStrideShift: " << defaultStaggerUStrideShift
                                  << std::endl;
                }
            }
            else
            {
                std::tie(defaultStaggerUMapping, defaultStaggerU, defaultStaggerUStrideShift)
                    = cachedStaggerUParams;
            }
        }
        else
        {
            defaultStaggerUMapping     = sizeMapping.staggerUMapping;
            defaultStaggerU            = sizeMapping.staggerU;
            defaultStaggerUStrideShift = sizeMapping.staggerStrideShift;
        }

        // If values are explicitly specified at runtime, they override predictions and default values
        if(pAMDGPU->fixedStaggerUMapping != std::numeric_limits<size_t>::max())
            defaultStaggerUMapping = pAMDGPU->fixedStaggerUMapping;
        if(pAMDGPU->fixedStaggerU != std::numeric_limits<size_t>::max())
            defaultStaggerU = pAMDGPU->fixedStaggerU;
        if(pAMDGPU->fixedStaggerUStrideShift != std::numeric_limits<size_t>::max())
            defaultStaggerUStrideShift = pAMDGPU->fixedStaggerUStrideShift;

        // Uniform summation order can keep mapping 1 only when the kernel can
        // apply row-uniform staggering. Persistent kernels also require the
        // per-tile iteration capability; otherwise disable staggering.
        //
        // The placement of this clamp is load-bearing. It must sit AFTER the
        // TENSILE_FIXED_STAGGERU* overrides directly above, or an env override
        // could reintroduce a stagger, and BELOW the staggerUParamsCache write,
        // so the cache keeps holding the real origami prediction rather than
        // this mode's zeros for concurrent callers who did not request it.
        if(problem.getParams().uniformSummationOrder() != 0)
        {
            const bool rowUniformStaggerCapable
                = internalArgsSupport.staggerU
                  && (!sizeMapping.isPersistent() || internalArgsSupport.perTileExtraIters);
            if(!(rowUniformStaggerCapable && defaultStaggerUMapping == 1))
            {
                defaultStaggerUMapping     = 0;
                defaultStaggerU            = 0;
                defaultStaggerUStrideShift = 0;
            }
        }

        // Mapping should be in this range: [0, 1, 2, 3, 4]
        assert(defaultStaggerUMapping < 5);
        // StaggerU should be power of 2 and less than 65: [0, 2, 4, 8, 16, 32, 64]
        assert((defaultStaggerU & (defaultStaggerU - 1)) == 0 && defaultStaggerU < 65);
        // StaggerUStrideShift is packed in 5-bit field (bits [12:8]), valid range [0, 31]
        assert(defaultStaggerUStrideShift <= 31);

        return std::make_tuple(defaultStaggerUMapping, defaultStaggerU, defaultStaggerUStrideShift);
    }

    uint32_t ContractionSolution::calculateAutoGSU(Problem const&  problem,
                                                   Hardware const* hardware) const
    {
        // if original GSU is not -1
        if(sizeMapping.globalSplitU != -1)
        {
            // std::cout<<"Returning the sizeMapping.globalsplitU value as autoGSU: "<<sizeMapping.globalSplitU<<"\n";
            return sizeMapping.globalSplitU;
        }

        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(hardware);
        assert(pAMDGPU);
        uint32_t numCUs = pAMDGPU->computeUnitCount;
        uint32_t numWGs = getNumWorkGroups(problem, sizeMapping);
        // avoid zero division
        if(numWGs == 0)
        {
            return 1;
        }
        uint32_t MT0       = sizeMapping.macroTile.x;
        uint32_t MT1       = sizeMapping.macroTile.y;
        uint32_t MT2       = sizeMapping.depthU;
        uint32_t M         = problem.freeSizeA(0);
        uint32_t N         = problem.freeSizeB(0);
        uint32_t B         = problem.batchSize(0);
        uint32_t K         = problem.boundSize(0);
        uint32_t GSULimit1 = std::max(1u, (uint32_t)std::floor(numCUs / numWGs));
        uint32_t GSULimit2 = std::max(1u, (uint32_t)std::floor((float)K / (float)MT2 / 3.0));
        uint32_t gsuVal    = std::min(GSULimit2, std::max(1u, GSULimit1));

        // WorkgroupNumberCheck
#define MAX_WORKGROUP_NUMBER 16777216
        if(gsuVal > 1)
            gsuVal = std::min(
                gsuVal,
                static_cast<uint32_t>(MAX_WORKGROUP_NUMBER / std::ceil(static_cast<float>(M) / MT0)
                                      / std::ceil(static_cast<float>(N) / MT1) / B));

        // GlobalSplitUCheckMinK
        if(gsuVal > 1)
            gsuVal
                = std::min(gsuVal, static_cast<uint32_t>(std::ceil(static_cast<float>(K) / MT2)));

        // SynchronizerSizeCheck
        //
        // MBSK owns the GSU region: a grouped GEMM is bounded by the one slot at
        // its problem index and by the slot count, a non-grouped one by the whole
        // buffer (see GsuSynchronizerElements). Bounding usage is what keeps a
        // solution from running past the end of its region.
        if(gsuVal > 1 && sizeMapping.globalAccumulation == 3) // MBSK
        {
            uint32_t synchronizerUsage
                = sizeMapping.synchronizerSizePerWG * problem.getNumTiles(sizeMapping, 1) * B;

            bool fits
                = problem.groupedGemm()
                      ? (synchronizerUsage <= GsuSynchronizerElements
                         && problem.groupedGemmCount() <= SynchronizerGroupedSlots)
                      : (synchronizerUsage <= GsuSynchronizerElements * SynchronizerGroupedSlots);

            gsuVal = fits ? gsuVal : 1;
        }

        // Avoid selecting a gsu value that would make launch grid over the limit
        uint32_t tiles0        = CeilDivide(M, MT0);
        uint32_t tiles1        = CeilDivide(N, MT1);
        uint32_t tiles         = tiles0 * tiles1 * B;
        uint32_t workGroupSize = sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y
                                 * sizeMapping.workGroupSize.z;
        uint32_t maxGsuValue = (std::numeric_limits<uint32_t>::max() / workGroupSize) / tiles;
        gsuVal               = std::min(gsuVal, maxGsuValue);

        // avoid gsu < 1
        gsuVal = std::max(gsuVal, 1u);

        static const char* envStr = std::getenv("TENSILE_AUTO_GSU_ALGO");
        if(envStr != NULL)
            std::cout << "autoGSU is calculated: " << gsuVal << std::endl;

        return gsuVal;
    }

    ContractionSolution::AdaptiveGemmNTAB
        ContractionSolution::calculateAdaptiveGemmNTAB(Problem const&  problem,
                                                       Hardware const* hardware) const
    {
        // Hardware is currently unused: the heuristic below is purely
        // shape/alignment/iter-count based and has been validated.
        // Kept in the signature so a future ISA-specific fallback (e.g. a
        // different L1 policy on a new arch) can plug in without touching
        // every call site.
        (void)hardware;

        AdaptiveGemmNTAB result{};
        if(sizeMapping.adaptiveGemmNTAB == 0)
            return result;

        const uint32_t MT0    = sizeMapping.macroTile.x; // MT_M
        const uint32_t MT1    = sizeMapping.macroTile.y; // MT_N
        const uint32_t depthU = static_cast<uint32_t>(sizeMapping.depthU); // MT_K

        const uint32_t bpeA    = static_cast<uint32_t>(problem.a().elementBytes());
        const uint32_t bpeB    = static_cast<uint32_t>(problem.b().elementBytes());
        const uint32_t minLenA = 128 / bpeA; // elements that fit one 128 B cache line
        const uint32_t minLenB = 128 / bpeB;

        const uint32_t ldA = problem.a().strides()[1];
        const uint32_t ldB = problem.b().strides()[1];

        const uint32_t M = problem.freeSizeA(0);
        const uint32_t N = problem.freeSizeB(0);
        const uint32_t K = problem.boundSize(0);

        // Stride-1 (contiguous) macro-tile dim per transpose. A and B are
        // both col-major tensors, so stride-1 is whichever index is the
        // leading one in the Tensile tensor name:
        //   transA()==false  (Ailk): stride-1 = i (=M) -> contigA = MT_M (=MT0)
        //   transA()==true   (Alik): stride-1 = l (=K) -> contigA = MT_K (=depthU)
        //   transB()==false  (Bljk): stride-1 = l (=K) -> contigB = MT_K (=depthU)
        //   transB()==true   (Bjlk): stride-1 = j (=N) -> contigB = MT_N (=MT1)
        // Note the A/B asymmetry: Tensile's transA()==true shifts A's contig
        // from M to K, but transB()==true shifts B's contig from K to N.
        const bool     transA  = problem.transA();
        const bool     transB  = problem.transB();
        const uint32_t contigA = transA ? depthU : MT0;
        const uint32_t contigB = transB ? MT1 : depthU;

        // Shape gate. NT=4 (cache bypass) is only profitable when the OTHER
        // dim fits in one macro-tile: then the dominant tensor has no
        // cross-WG reuse in L1 (every WG streams its slab once), so bypassing
        // L1 is pure win and frees L1 for the short side. Dense (non-skinny)
        // GEMMs still reuse each line across WGs; skip NT=4 there.
        const bool skinnyA = (N <= MT1);
        const bool skinnyB = (M <= MT0);
        const bool longA   = (static_cast<uint64_t>(M) >= static_cast<uint64_t>(MT0) * 256);
        const bool longB   = (static_cast<uint64_t>(N) >= static_cast<uint64_t>(MT1) * 128);

        // Alignment gate. NT=4 has to consume whole 128 B cache lines;
        // otherwise each straddling load pulls two lines from HBM and the
        // bypass savings evaporate. Both the contiguous macro-tile dim AND
        // the leading stride must be cache-line multiples.
        const bool alignedA = (contigA % minLenA == 0) && (ldA % minLenA == 0);
        const bool alignedB = (contigB % minLenB == 0) && (ldB % minLenB == 0);

        // Main-loop iteration floor. The NTA/NTB kernArg bits only drive the
        // main unroll body; PGR prefetches and the tail loop run under the
        // static NonTemporalA/B kernel parameter. loopIters > 2 guarantees
        // at least one main-body iter under PGR=2 so the bypass actually
        // executes; at or below 2 iters everything lives in PGR/tail and the
        // runtime choice is a no-op (pure overhead).
        const uint32_t loopIters = (depthU > 0) ? (K / depthU) : 0;
        const bool     enough    = loopIters > 2;

        // These constraints are intentionally conservative: keep the gate off
        // small problems where bypassing L1 has no streaming benefit.
        if(skinnyA && longA && alignedA && enough)
            result.nta = 4;
        if(skinnyB && longB && alignedB && enough)
            result.ntb = 4;

        static const char* envStr = std::getenv("TENSILE_ADAPTIVE_GEMM_NTAB_ALGO");
        if(envStr != nullptr)
        {
            std::cout << "AdaptiveGemmNTAB: nta=" << result.nta << " ntb=" << result.ntb
                      << " (M=" << M << " N=" << N << " K=" << K << " MT0=" << MT0 << " MT1=" << MT1
                      << " DU=" << depthU << " transA=" << transA << " transB=" << transB
                      << " contigA=" << contigA << " contigB=" << contigB << " minLenA=" << minLenA
                      << " minLenB=" << minLenB << " ldA=" << ldA << " ldB=" << ldB
                      << " loopIters=" << loopIters << " longA=" << longA << " longB=" << longB
                      << ")"
                      << "\n";
        }

        return result;
    }

    template <bool T_Debug>
    void ContractionSolution::calculateInternalArgs(uint32_t&       internalArg0,
                                                    uint32_t&       internalArg1,
                                                    Hardware const* hardware,
                                                    const ContractionProblemParameters& param,
                                                    int32_t                             autoWGM,
                                                    size_t                              autoWGMXCC,
                                                    size_t           autoWGMXCCCHUNK,
                                                    size_t           autoWGMXCCSPLITK,
                                                    size_t           autoStaggerUMapping,
                                                    size_t           autoStaggerU,
                                                    size_t           autoStaggerUStrideShift,
                                                    uint32_t         autoGsuVal,
                                                    AdaptiveGemmNTAB ntab) const
    {
        uint32_t       gsu                 = param.gsu() > 0 ? param.gsu() : autoGsuVal;
        bool           gsuc                = false; // initialized false
        bool           gsuwgmrr            = false; // initialized false
        int32_t        wgm                 = param.wgm() != 0 ? param.wgm() : autoWGM;
        size_t         wgmxcc              = param.wgmxcc() != 0 ? param.wgmxcc() : autoWGMXCC;
        size_t         wgmxccchunk         = autoWGMXCCCHUNK;
        size_t         wgmxccsplitk        = autoWGMXCCSPLITK;
        int32_t        wgmxccg             = -1; // initialized -1
        size_t         staggerUMapping     = autoStaggerUMapping;
        size_t         staggerU            = autoStaggerU;
        size_t         staggerUStrideShift = autoStaggerUStrideShift;
        const uint32_t mask16              = 0xFFFF;
        const uint32_t mask14              = 0x3FFF;
        const uint32_t mask8               = 0xFF;

        internalArg0 = 0;
        internalArg1 = 0;

        // GSU bit-width depends on InternalArgsSupport.version:
        //   v <  3: GSU occupies bits 0..13 (mask 0x3FFF, max 16383)
        //   v >= 3: GSU narrowed to bits 0..11 (mask 0x0FFF, max 4095);
        //           bits 12/13 carry NTA / NTB for AdaptiveGemmNTAB.
        constexpr uint32_t kGsuMaskV3  = 0x0FFF;
        constexpr uint32_t kNtaBitPos  = 12;
        constexpr uint32_t kNtbBitPos  = 13;
        const bool         useNtabBits = (internalArgsSupport.version >= 3);
        const uint32_t     gsuMask     = useNtabBits ? kGsuMaskV3 : mask14;
        // Belt-and-suspenders: if a v<3 solution somehow has AGNTAB!=0, that's a codegen bug.
        assert((useNtabBits || sizeMapping.adaptiveGemmNTAB == 0)
               && "AdaptiveGemmNTAB requires InternalArgsSupport.version >= 3");
        // Only forward NT bits if both the kernel knows the new layout AND the
        // solution actually opted in to AdaptiveGemmNTAB. Otherwise leave bits clear
        // so a v<3 kernel reading 0x3FFF doesn't see them folded into GSU.
        const uint32_t ntaBit
            = (useNtabBits && sizeMapping.adaptiveGemmNTAB != 0 && ntab.nta == 4) ? 1 : 0;
        const uint32_t ntbBit
            = (useNtabBits && sizeMapping.adaptiveGemmNTAB != 0 && ntab.ntb == 4) ? 1 : 0;

        if(internalArgsSupport.wgm && internalArgsSupport.version == 0)
        {
            if(wgm > 255)
                wgm = 255;
            if(gsu > 255)
                gsu = 255;
            uint32_t wgShift8 = (mask8 & (uint32_t)wgm) << 8;
            internalArg0      = internalArg0 | wgShift8;
        }

        if(internalArgsSupport.wgm && internalArgsSupport.version >= 1)
        {
            if(internalArgsSupport.version == 1)
            {
                internalArg1 = wgm;
            }
            else if(internalArgsSupport.version >= 2 && !internalArgsSupport.useSFC)
            {
                // NB: get value from param= set in runtime / vs value from sizeMapping: from logic yaml.
                //     param: default values: [xcc = 0, xccg = 0]. So when we never set xcc/xccg in runtime: we always get from sizeMapping.
                //     From sizeMapping = from logic yaml. If not set in Config-Yaml, use default value [1, -1]
                wgmxccg
                    = param.wgmxccg() != 0 ? param.wgmxccg() : sizeMapping.workGroupMappingXCCGroup;
                if(wgmxcc >= 1 && wgmxccg == -1)
                {
                    AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(hardware);
                    assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);
                    wgmxccg = pAMDGPU->computeUnitCount;
                }
                // if using WGMXCCn1, wgmxccg is not used. Repurpose it for wgmxccchunk
                if(sizeMapping.workGroupMappingXCC == -1)
                {
                    // New bit layout: K(31:22) | chunk(21:14) | xcc(13:10) | wgm(9:0)
                    internalArg1 = internalArg1 | ((wgmxccsplitk & 0x3FF) << 22)
                                   | ((wgmxccchunk & 0xFF) << 14) | ((wgmxcc & 0xF) << 10)
                                   | (wgm & 0x3FF);
                }
                else
                {
                    // Old bit layout: wgmxccg(31:22) | wgmxcc(21:16) | wgm(15:0)
                    internalArg1 = internalArg1 | (wgmxccg << 22) | (wgmxcc << 16) | (mask16 & wgm);
                }
            }
            else if(internalArgsSupport.version >= 2 && internalArgsSupport.useSFC)
            {
                internalArg1 = wgm;
            }
        }

        // support gsuc and gsuwgmrr after version 2
        if(internalArgsSupport.version >= 2)
        {
            gsuc     = param.gsuc() > 0 ? param.gsuc() : sizeMapping.globalSplitUCoalesced;
            gsuwgmrr = param.gsuwgmrr() > 0 ? param.gsuwgmrr()
                                            : sizeMapping.globalSplitUWorkGroupMappingRoundRobin;
        }

        // Runtime sanity: GSU must fit in the available bits (12 or 14, depending on version).
        if(((uint32_t)gsu & ~gsuMask) != 0)
        {
            std::stringstream gsuMaskHex;
            gsuMaskHex << "0x" << std::hex << gsuMask;
            std::string msg
                = std::string("GSU value ") + std::to_string((uint32_t)gsu)
                  + " exceeds the GSU bit-field in internalArg0 (max allowed="
                  + std::to_string(gsuMask) + ", gsuMask=" + gsuMaskHex.str()
                  + ", InternalArgsSupport.version=" + std::to_string(internalArgsSupport.version)
                  + ", AdaptiveGemmNTAB=" + std::to_string(sizeMapping.adaptiveGemmNTAB)
                  + "). When AdaptiveGemmNTAB is enabled (version>=3), GSU is narrowed"
                  + " from bits 0..13 (max 16383) to bits 0..11 (max 4095) because"
                  + " bits 12/13 carry the NTA/NTB selector.";
            throw std::runtime_error(msg.c_str());
        }
        internalArg0 = internalArg0 | ((uint32_t)gsuc << 15) | ((uint32_t)gsuwgmrr << 14)
                       | (ntbBit << kNtbBitPos) | (ntaBit << kNtaBitPos)
                       | (gsuMask & (uint32_t)gsu);

        // StaggerU
        if(internalArgsSupport.staggerU)
        {
            constexpr size_t staggerMask1 = 0x1F00;
            // Mapping owns the 3-bit field [15:13]. The range assert in
            // calculateAutoStaggerU() compiles out under NDEBUG, so mask it or
            // an out-of-range value spills into neighbouring bits.
            size_t sum = (staggerUMapping & 0x7) << 13;
            size_t sus = staggerMask1 & (staggerUStrideShift << 8);
            size_t su  = mask8 & staggerU;
            if(Debug::Instance().disableStaggerU())
                su = 0;
            su           = su | sus;
            su           = su | sum;
            internalArg0 = internalArg0 | (su << 16);
        }
        else if(T_Debug && Debug::Instance().disableStaggerU())
            std::cout << "solution doesn't support configurable staggerU" << std::endl;
    }

    template <bool T_Debug, bool Legacy, typename KA>
    void ContractionSolution::kernelArgs(uint32_t                            gemmCount,
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
                                         size_t           autoStaggerUStrideShift,
                                         uint32_t         autoGsuVal,
                                         AdaptiveGemmNTAB ntab) const
    {
        if constexpr(!Legacy)
        {
            gemmCount = gemmCount & 0x3FFFFFFF;
            // Currently 0 for kernel args, 1 for args located in HBM. This is a temporary slot.
            gemmCount = gemmCount | (argType << 30);
            args.template append<uint32_t>("gemm_count", gemmCount);
        }

        uint32_t internalArg0;
        uint32_t internalArg1;
        calculateInternalArgs<T_Debug>(internalArg0,
                                       internalArg1,
                                       hardware,
                                       param,
                                       autoWGM,
                                       autoWGMXCC,
                                       autoWGMXCCCHUNK,
                                       autoWGMXCCSPLITK,
                                       autoStaggerUMapping,
                                       autoStaggerU,
                                       autoStaggerUStrideShift,
                                       autoGsuVal,
                                       ntab);

        args.template append<uint32_t>("internalArgs", internalArg0);

        if(internalArgsSupport.version >= 1)
        {
            args.template append<int32_t>("internalArgs1", internalArg1);
            args.template append<uint32_t>("numWorkGroups", numWorkGroups);
        }
    }

    void ContractionSolution::calculateTiles(dim3&                               tiles,
                                             ContractionSolution::Problem const& problem) const
    {
        tiles.x = 1;
        tiles.y = 1;

        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
        {
            tiles.x *= problem.freeSizeA(i);
        }
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
        {
            tiles.y *= problem.freeSizeB(i);
        }

        tiles.z = 1;
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
        {
            // if(sizeMapping.packBatchDims & 0x1)
            //     numWorkGroups.x *= problem.batchSize(i);
            // if(sizeMapping.packBatchDims & 0x2)
            //     numWorkGroups.y *= problem.batchSize(i);
            if(!sizeMapping.packBatchDims)
                tiles.z *= problem.batchSize(i);
        }

        if(problem.transposeC01())
            std::swap(tiles.x, tiles.y);

        tiles.x = CeilDivide(tiles.x, customKernel.macrotile.x);
        tiles.y = CeilDivide(tiles.y, customKernel.macrotile.y);
    }

    void ContractionSolution::calculateGrid(dim3&                               workGroupSize,
                                            dim3&                               numWorkGroups,
                                            ContractionSolution::Problem const& problem) const
    {
        workGroupSize.x = checkedMultiply(
            checkedMultiply(sizeMapping.workGroupSize.x, sizeMapping.workGroupSize.y),
            sizeMapping.workGroupSize.z);
        workGroupSize.y = 1;
        workGroupSize.z = 1;

        numWorkGroups.x = 1;
        numWorkGroups.y = 1;

        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
        {
            numWorkGroups.x = checkedMultiply(numWorkGroups.x, problem.freeSizeA(i));
        }
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
        {
            numWorkGroups.y = checkedMultiply(numWorkGroups.y, problem.freeSizeB(i));
        }

        numWorkGroups.z = 1;
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
        {
            if(sizeMapping.packBatchDims & 0x1)
                numWorkGroups.x = checkedMultiply(numWorkGroups.x, problem.batchSize(i));
            if(sizeMapping.packBatchDims & 0x2)
                numWorkGroups.y = checkedMultiply(numWorkGroups.y, problem.batchSize(i));
            if(!sizeMapping.packBatchDims)
                numWorkGroups.z = checkedMultiply(numWorkGroups.z, problem.batchSize(i));
        }

        if(problem.transposeC01())
            std::swap(numWorkGroups.x, numWorkGroups.y);

        if(usesStreamKSpatialCluster())
        {
            checkPersistentArgument(numWorkGroups.x);
            checkPersistentArgument(numWorkGroups.y);
            checkPersistentArgument(numWorkGroups.z);
        }
        numWorkGroups.x = divideUp(numWorkGroups.x, sizeMapping.macroTile.x);
        numWorkGroups.y = divideUp(numWorkGroups.y, sizeMapping.macroTile.y);
    }

    namespace
    {
        struct SpatialClusterGeometry
        {
            dim3   tiles, blocks;
            size_t totalTiles, totalBlocks, clusterSize;
        };

        SpatialClusterGeometry spatialClusterGeometry(ContractionSolution const&    solution,
                                                      ContractionProblemGemm const& problem)
        {
            dim3 workGroupSize, tiles;
            solution.calculateGrid(workGroupSize, tiles, problem);
            const auto& c = solution.sizeMapping.clusterDim;
            const dim3  blocks{divideUp(tiles.x, c.x), divideUp(tiles.y, c.y), tiles.z};
            return {tiles,
                    blocks,
                    checkedMultiply(checkedMultiply(tiles.x, tiles.y), tiles.z),
                    checkedMultiply(checkedMultiply(blocks.x, blocks.y), blocks.z),
                    checkedMultiply(c.x, c.y)};
        }

        // The caller converts its scheduling grid to logical clusters first:
        // DataParallel stores physical peers; StreamK ABI2 stores clusters.
        dim3 spatialClusterLaunch(size_t clusters, dim3 const& clusterDim)
        {
            return {checkedMultiply(clusters, clusterDim.x), clusterDim.y, 1};
        }
    }

    // Temporary: restored from develop. Builds the kernarg buffer via the
    // hand-written per-feature path (singleCallArgs) rather than the generic
    // CustomKernel.args iteration in generateCustomCall. Used for all
    // Tensile-generated kernels (see gating in solve()) so their launch
    // arguments are byte-identical to develop while generateCustomCall is
    // validated for newer features.
    template <bool T_Debug>
    KernelInvocation
        ContractionSolution::generateSingleCall(ContractionSolution::Problem const& problem,
                                                ContractionInputs const&            inputs,
                                                Hardware const&                     hardware,
                                                PersistentLaunchSettings const&     launch,
                                                GSUSettings const& gsuSettings) const
    {
        KernelInvocation rv;

        rv.isSingleCall = true;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(1024, 128);

        rv.kernelName = kernelName;

        calculateGrid(rv.workGroupSize, rv.numWorkGroups, problem);

        dim3 problemNumGroupTiles = rv.numWorkGroups;

        uint32_t autoGsuVal = calculateAutoGSU(problem, &hardware);
        uint32_t gsu = problem.getParams().gsu() > 0 ? problem.getParams().gsu() : autoGsuVal;
        if(gsu > 0)
            rv.numWorkGroups.y *= gsu;

        if(sizeMapping.isPersistent())
        {
            if(usesStreamKSpatialCluster()
               || (sizeMapping.isPersistentDataParallel()
                   && (sizeMapping.clusterDim.x > 1 || sizeMapping.clusterDim.y > 1)))
            {
                if(usesStreamKSpatialCluster() && !launch.clusterSchedule.enabled)
                    throw std::runtime_error("Missing resolved StreamK cluster schedule");
                const size_t clusters = usesStreamKSpatialCluster()
                                            ? launch.grid
                                            : launch.grid
                                                  / checkedMultiply(sizeMapping.clusterDim.x,
                                                                    sizeMapping.clusterDim.y);
                rv.numWorkGroups      = spatialClusterLaunch(clusters, sizeMapping.clusterDim);
            }
            else
            {
                // Linear persistent launch outside the spatial DP cluster path.
                rv.numWorkGroups.x = launch.grid;
                rv.numWorkGroups.y = 1;
                rv.numWorkGroups.z = 1;
                // A whole-tile StreamK fallback keeps grid == totalTiles so it
                // needs no partial workspace. Extra physical workgroups hit the
                // static kernel's no-work exit; do not change its logical grid.
                if(sizeMapping.isStreamK() && sizeMapping.hasStaticAssignment()
                   && sizeMapping.clusterDim.x > 1 && sizeMapping.clusterDim.y == 1
                   && launch.grid == launch.totalTiles)
                    rv.numWorkGroups.x
                        = RoundUpToMultiple(rv.numWorkGroups.x, sizeMapping.clusterDim.x);
            }
        }

        bool enableCluster = (sizeMapping.clusterDim.x > 1 || sizeMapping.clusterDim.y > 1);
        if(!enableCluster)
        {
            if(internalArgsSupport.version >= 1)
            {
                rv.numWorkGroups.x *= (rv.numWorkGroups.y * rv.numWorkGroups.z);
                rv.numWorkGroups.y = 1;
                rv.numWorkGroups.z = 1;
            }
        }

        rv.clusterDim = sizeMapping.clusterDim;

        // The HIP driver rejects a cluster launch whose grid is not divisible by
        // clusterDim, so round the tile-space grid of a non-persistent kernel up.
        // Its padded work-groups early-exit before the cluster barrier and the
        // surviving peers' broadcast masks are trimmed to the present lanes
        // (computeMulticastMaskReduction). The DataParallel cluster grid above is
        // already a whole number of clusters. StreamK selects aligned workers;
        // only its whole-tile fallback can pad the physical launch above.
        if(enableCluster && !sizeMapping.isPersistent())
        {
            rv.numWorkGroups.x = RoundUpToMultiple(rv.numWorkGroups.x, rv.clusterDim.x);
            rv.numWorkGroups.y = RoundUpToMultiple(rv.numWorkGroups.y, rv.clusterDim.y);
        }

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        rv.sharedMemBytes = 0;

        if(internalArgsSupport.useUniversalArgs)
        {
            auto [autoWGM, autoWGMXCC, autoWGMXCCCHUNK, autoWGMXCCSPLITK]
                = calculateAutoWGM(problem, &hardware, launch.grid);
            auto [autoStaggerUMapping, autoStaggerU, autoStaggerUStrideShift]
                = calculateAutoStaggerU(problem, &hardware, launch.grid, autoWGM);
            if(T_Debug)
            {
                std::cout << "OCCUPANCY: " << sizeMapping.CUOccupancy << std::endl;
                std::cout << "WGM: " << autoWGM << ", WGMXCC: " << autoWGMXCC
                          << ", WGMXCCCHUNK: " << autoWGMXCCCHUNK
                          << ", WGMXCCSPLITK: " << autoWGMXCCSPLITK << std::endl;
                std::cout << "StaggerUMapping: " << autoStaggerUMapping
                          << ", StaggerU: " << autoStaggerU
                          << ", StaggerUStrideShift: " << autoStaggerUStrideShift << std::endl;
            }
            AdaptiveGemmNTAB ntab = calculateAdaptiveGemmNTAB(problem, &hardware);
            if(problem.batchMode() == ContractionProblemGemm::BATCHMODE::POINTER_ARRAY)
            {
                kernelArgs<T_Debug, false>(1,
                                           3,
                                           rv.args,
                                           usesStreamKSpatialCluster()
                                               ? launch.clusterSchedule.physicalGrid
                                               : getNumWorkGroups(rv),
                                           &hardware,
                                           problem.getParams(),
                                           autoWGM,
                                           autoWGMXCC,
                                           autoWGMXCCCHUNK,
                                           autoWGMXCCSPLITK,
                                           autoStaggerUMapping,
                                           autoStaggerU,
                                           autoStaggerUStrideShift,
                                           autoGsuVal,
                                           ntab);
            }
            else
            {
                kernelArgs<T_Debug, false>(1,
                                           0,
                                           rv.args,
                                           usesStreamKSpatialCluster()
                                               ? launch.clusterSchedule.physicalGrid
                                               : getNumWorkGroups(rv),
                                           &hardware,
                                           problem.getParams(),
                                           autoWGM,
                                           autoWGMXCC,
                                           autoWGMXCCCHUNK,
                                           autoWGMXCCSPLITK,
                                           autoStaggerUMapping,
                                           autoStaggerU,
                                           autoStaggerUStrideShift,
                                           autoGsuVal,
                                           ntab);
            }
        }
        singleCallArgs<T_Debug, true>(problem,
                                      inputs,
                                      0,
                                      &hardware,
                                      problemNumGroupTiles,
                                      rv.numWorkGroups,
                                      rv.args,
                                      launch,
                                      gsuSettings.globalAccumulation);

        if(gsuSettings.globalAccumulation == 3
           || sizeMapping.adaptiveGemmGSUA == 1) // MBSK or MB with AdaptiveGemmGSUA
        {
            rv.args.append<void const*>("dstD", inputs.d);
            // MBSK: synchronizer address, MB: null address
            rv.args.append<void const*>(
                "Synchronizer", gsuSettings.globalAccumulation == 3 ? inputs.Synchronizer : NULL);
            rv.args.append<uint32_t>("GSUSync", 0);
        }

        // Batch offset support for General Batched GEMM (SupportUserArgs kernels).
        // Appended at the tail, after the dstD/Synchronizer block, to match the
        // kernel signature order (see Signature.py).
        if(!problemType.groupedGemm && (customKernel.name.empty() || customKernel.generated))
        {
            rv.args.append<int64_t>("batchOffsetD", inputs.batchOffsetD);
            rv.args.append<int64_t>("batchOffsetC", inputs.batchOffsetC);
            rv.args.append<int64_t>("batchOffsetA", inputs.batchOffsetA);
            rv.args.append<int64_t>("batchOffsetB", inputs.batchOffsetB);
        }

        // The fused GEMM+A2A segment follows batchOffsets in the kernel signature.
        if(problem.fusedGemmA2A())
            appendFusedSegment(rv.args,
                               inputs.fusedA2APeers,
                               inputs.fusedA2ACounter,
                               inputs.fusedA2AMyRank,
                               problem.fusedA2AWorld(),
                               inputs.fusedA2ADrain,
                               static_cast<uint32_t>(problem.fusedA2AExtent()));

        // CU-occupancy probe tail args (layout: Signature.py addOccupancyProbeArgs).
        // A grid smaller than minGrid is padded with probe-only WGs so the probe
        // covers minGrid CUs. Only a 1-D grid can be padded; otherwise the probe
        // is disabled for this launch. !groupedGemm mirrors the codegen gate.
        if(internalArgsSupport.occupancyProbe && !problemType.groupedGemm)
        {
            auto const&    params    = problem.getParams();
            void*          probeAddr = params.occupancyProbeAddr();
            const uint32_t minGrid   = params.occupancyProbeMinGrid();
            auto&          wg        = rv.numWorkGroups;
            const size_t   realGrid  = static_cast<size_t>(wg.x) * wg.y * wg.z;

            const bool needsPad = probeAddr && realGrid < minGrid;
            const bool canPad   = wg.y == 1 && wg.z == 1;
            if(needsPad && !canPad)
                probeAddr = nullptr;

            rv.occupancyProbe = probeAddr != nullptr;
            rv.args.appendAligned<void*>("ProbeAddr", probeAddr);
            rv.args.append<uint32_t>("ProbeEpoch", params.occupancyProbeEpoch());
            rv.args.append<uint32_t>("ProbeGrid", static_cast<uint32_t>(realGrid));

            // Grow only the dispatch; the kernargs above keep the real grid.
            if(needsPad && canPad)
            {
                wg.x              = minGrid;
                rv.numWorkItems.x = rv.workGroupSize.x * minGrid;
            }
        }

        if(problemType.stochasticRounding)
        {
            // generate seed from random generator
            std::random_device                      rd;
            std::mt19937                            gen(rd());
            std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
            uint32_t                                seed = distribution(gen);
            rv.args.append<uint32_t>("RNDSeed", seed);
        }
        rv.codeObjectFile = codeObjectFilename.load();
        return rv;
    }

    template <bool T_Debug>
    KernelInvocation
        ContractionSolution::generateCustomCall(ContractionSolution::Problem const& problem,
                                                ContractionInputs const&            inputs,
                                                Hardware const&                     hardware,
                                                PersistentLaunchSettings const&     launch) const
    {
        validatePersistentLoopArgsVersion();
        if(usesStreamKSpatialCluster())
            throw std::runtime_error(
                "StreamK scheduling ABI 2 requires generated argument packing");
        KernelInvocation rv;
        rv.isSingleCall = true;
        rv.args         = KernelArguments(T_Debug);
        rv.args.reserve(1024, 128);
        rv.kernelName = kernelName;

        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
        assert(pAMDGPU);

        // A persistent kernel strides by its ComputeUnits kernarg, so it and the
        // ComputeUnits grid must be the same value. Cap it like Tensile's persistent
        // grids, by the caller's SM-count target and persistentMaxCUs.
        int computeUnits = pAMDGPU->computeUnitCount;
        if(problem.getParams().smCountTarget() > 0)
            computeUnits = std::min(computeUnits, problem.getParams().smCountTarget());
        if(pAMDGPU->persistentMaxCUs > 0)
            computeUnits = std::min(computeUnits, pAMDGPU->persistentMaxCUs);
        computeUnits = std::max(1, computeUnits);

        if(customKernel.threads.x == 0 || customKernel.macrotile.x == 0)
            throw std::runtime_error(
                concatenate("Solution ", kernelName, " has uninitialized customKernel metadata"));

        dim3 tiles;
        calculateTiles(tiles, problem);

        uint32_t autoGsuVal = calculateAutoGSU(problem, &hardware);
        uint32_t gsu = problem.getParams().gsu() > 0 ? problem.getParams().gsu() : autoGsuVal;
        // Runtime-effective accumulation mode; AdaptiveGemmGSUA can downgrade this from
        // the solution's compiled sizeMapping.globalAccumulation (e.g. MBSK -> MultipleBuffer
        // for small/low-tile problems).
        size_t adaptiveGlobalAccumulation = problem.getAccumulation(hardware, sizeMapping, gsu);

        const uint32_t itersPerTile
            = launch.grid > 0
                  ? static_cast<uint32_t>(max(size_t(1), problem.getItersPerTile(sizeMapping)))
                  : 0;
        // Legacy/custom scheduling adapter. DataParallel version-1 descriptors use only
        // ItersPerTile and PersistentGrid and never compute iteration splits.
        uint32_t skTotalIters              = 0;
        uint32_t skMagicNumberItersPerTile = 0;
        uint32_t skMagicShiftItersPerTile  = 0;
        uint32_t skItersPerWG              = 0;
        uint32_t skTotalTiles              = 0;
        if(launch.grid > 0 && internalArgsSupport.persistentLoopArgsVersion == 0)
        {
            size_t numTiles           = tiles.x * tiles.y * tiles.z;
            skTotalIters              = static_cast<uint32_t>(numTiles * itersPerTile);
            skMagicNumberItersPerTile = magicNumber(2, itersPerTile, &skMagicShiftItersPerTile);

            bool twoTileStreamK
                = sizeMapping.isStreamK()
                  || customKernel.workspaceType == CustomWorkspaceType::StreamK
                  || customKernel.workspaceType == CustomWorkspaceType::StreamKWithReduction;

            if(twoTileStreamK) // Two-tile K partitioning
            {
                if(launch.reduction == origami::reduction_t::parallel)
                {
                    uint32_t skSplit = static_cast<uint32_t>(launch.grid / numTiles);
                    skItersPerWG     = itersPerTile / max(skSplit, uint32_t(1));
                    skTotalTiles     = skSplit;
                }
                else
                {
                    int  fullTiles = pAMDGPU->skFullTiles;
                    bool bigEnough = numTiles > launch.grid;

                    // skTotalTiles is the number of Stream-K tiles to complete.
                    // Two-tile SK gives each WG an even number of Stream-K iterations,
                    // followed by an even number of data-parallel tiles.
                    skTotalTiles = static_cast<uint32_t>(launch.grid);
                    if(numTiles % launch.grid != 0)
                    {
                        // If tiles are not evenly divisible, determine the number of
                        // Stream-K tiles and cap it at the total tile count.
                        size_t skT   = bigEnough ? launch.grid * fullTiles + numTiles % launch.grid
                                                 : numTiles;
                        skTotalTiles = static_cast<uint32_t>(min(skT, numTiles));
                    }
                    skItersPerWG = skTotalTiles * itersPerTile
                                   / max(uint32_t(1), static_cast<uint32_t>(launch.grid));
                }
            }
        }

        rv.workGroupSize.x = customKernel.threads.x;
        rv.workGroupSize.y = customKernel.threads.y;
        rv.workGroupSize.z = customKernel.threads.z;

        if(T_Debug)
        {
            std::cout << "Wavefront size: " << pAMDGPU->wavefrontSize << std::endl;
            std::cout << "Threads: " << customKernel.threads.x << ", " << customKernel.threads.y
                      << ", " << customKernel.threads.z << std::endl;
            std::cout << "Work group size: " << rv.workGroupSize.x << ", " << rv.workGroupSize.y
                      << ", " << rv.workGroupSize.z << std::endl;
            std::cout << "Macrotile: " << customKernel.macrotile.x << ", "
                      << customKernel.macrotile.y << ", " << customKernel.macrotile.z << std::endl;
        }

        if(T_Debug)
        {
            std::cout << "Tiles: " << tiles.x << ", " << tiles.y << ", " << tiles.z << std::endl;
        }

        auto assignGridSize = [&](size_t& dim, CustomGridSize size) {
            switch(size)
            {
            case CustomGridSize::One:
                dim = 1;
                break;
            case CustomGridSize::TilesX:
                dim = tiles.x;
                break;
            case CustomGridSize::TilesY:
                dim = tiles.y;
                break;
            case CustomGridSize::Batch:
                dim = tiles.z;
                break;
            case CustomGridSize::TilesXY:
                dim = tiles.x * tiles.y;
                break;
            case CustomGridSize::TilesXYBatch:
                dim = tiles.x * tiles.y * tiles.z;
                break;
            case CustomGridSize::TilesXYBatchGSU:
                dim = tiles.x * tiles.y * tiles.z * (gsu > 0 ? gsu : 1);
                break;
            case CustomGridSize::TilesYGSU:
                dim = tiles.y * (gsu > 0 ? gsu : 1);
                break;
            case CustomGridSize::PersistentWithBatch:
            case CustomGridSize::StreamKWithBatch:
                // generateCustomCall is only used for handwritten/external
                // custom kernels; Tensile-generated kernels are routed to
                // generateSingleCall in solve().  For a batched handwritten
                // Stream-K custom kernel, launch.grid is per-batch and must be
                // expanded by the batch tile count.
                dim = launch.grid * tiles.z;
                break;
            case CustomGridSize::PersistentNoBatch:
            case CustomGridSize::PersistentGrid:
            case CustomGridSize::StreamKNoBatch:
                dim = launch.grid;
                break;
            case CustomGridSize::ComputeUnits:
                dim = static_cast<size_t>(computeUnits);
                break;
            default:
                throw std::runtime_error(
                    concatenate("Invalid CustomGridSize value: ", static_cast<int>(size)));
                break;
            }
        };

        assignGridSize(rv.numWorkGroups.x, customKernel.grid.x);
        assignGridSize(rv.numWorkGroups.y, customKernel.grid.y);
        assignGridSize(rv.numWorkGroups.z, customKernel.grid.z);

        // A split-K kernel reduces into D only once every GSU slice of a tile has
        // arrived, so a grid sized from tile counts alone would return success with
        // D unwritten. GSU and persistent grids account for the split themselves.
        auto tileCountOnly = [](CustomGridSize size) {
            switch(size)
            {
            case CustomGridSize::One:
            case CustomGridSize::TilesX:
            case CustomGridSize::TilesY:
            case CustomGridSize::Batch:
            case CustomGridSize::TilesXY:
            case CustomGridSize::TilesXYBatch:
                return true;
            default:
                return false;
            }
        };
        if(gsu > 1 && tileCountOnly(customKernel.grid.x) && tileCountOnly(customKernel.grid.y)
           && tileCountOnly(customKernel.grid.z))
            throw std::runtime_error(concatenate("Solution ",
                                                 kernelName,
                                                 " runs with GSU ",
                                                 gsu,
                                                 " but its custom-kernel grid [",
                                                 customKernel.grid.x,
                                                 ", ",
                                                 customKernel.grid.y,
                                                 ", ",
                                                 customKernel.grid.z,
                                                 "] launches one GSU slice per tile"));

        bool enableCluster = (sizeMapping.clusterDim.x > 1 || sizeMapping.clusterDim.y > 1);
        if(internalArgsSupport.persistentLoopArgsVersion == 1 && enableCluster)
        {
            // DataParallel version-1 cluster kernels fold hardware clusters into
            // cluster ranks. Match the generated-kernel launch.
            const size_t clusters
                = launch.grid / checkedMultiply(sizeMapping.clusterDim.x, sizeMapping.clusterDim.y);
            rv.numWorkGroups = spatialClusterLaunch(clusters, sizeMapping.clusterDim);
        }
        bool hasNumWorkGroupsArg
            = std::any_of(customKernel.args.begin(), customKernel.args.end(), [](auto const& a) {
                  return a.semantic == CustomArgSemantic::NumWorkGroups;
              });
        if(!enableCluster && hasNumWorkGroupsArg && internalArgsSupport.version >= 1)
        {
            rv.numWorkGroups.x *= (rv.numWorkGroups.y * rv.numWorkGroups.z);
            rv.numWorkGroups.y = 1;
            rv.numWorkGroups.z = 1;
        }

        if(T_Debug)
        {
            std::cout << "Num work groups: " << rv.numWorkGroups.x << ", " << rv.numWorkGroups.y
                      << ", " << rv.numWorkGroups.z << std::endl;
        }

        rv.clusterDim = sizeMapping.clusterDim;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        rv.sharedMemBytes = 0;

        auto [autoWGM, autoWGMXCC, autoWGMXCCCHUNK, autoWGMXCCSPLITK]
            = calculateAutoWGM(problem, &hardware, launch.grid);
        auto [autoStaggerUMapping, autoStaggerU, autoStaggerUStrideShift]
            = calculateAutoStaggerU(problem, &hardware, launch.grid, autoWGM);
        AdaptiveGemmNTAB ntab         = calculateAdaptiveGemmNTAB(problem, &hardware);
        uint32_t         internalArg0 = 0;
        uint32_t         internalArg1 = 0;
        calculateInternalArgs<T_Debug>(internalArg0,
                                       internalArg1,
                                       &hardware,
                                       problem.getParams(),
                                       autoWGM,
                                       autoWGMXCC,
                                       autoWGMXCCCHUNK,
                                       autoWGMXCCSPLITK,
                                       autoStaggerUMapping,
                                       autoStaggerU,
                                       autoStaggerUStrideShift,
                                       autoGsuVal,
                                       ntab);

        uint32_t debugPattern         = 0xDB000001;
        bool     appendedInternalArgs = false;

        // For multi-buffer GSU and Stream-K parallel reduction, the kernel
        // writes to the GSU/SK workspace and expects strides derived from
        // d.sizes() (matching the legacy `strideW_D`/`strideW_C` layout used
        // in singleCallArgs on develop).  Outside those modes the regular
        // tensor strides apply.
        bool gsuWSStride
            = gsu > 1 && sizeMapping.globalAccumulation != 3 && !sizeMapping.isStreamK();
        bool skWSStride
            = sizeMapping.isStreamK() && launch.reduction == origami::reduction_t::parallel;
        bool         useWSStride            = gsuWSStride || skWSStride;
        size_t       startStrideCD          = problemType.useInitialStridesCD ? 0 : 1;
        const bool   usesDataParallelArgsV1 = internalArgsSupport.persistentLoopArgsVersion == 1;
        const size_t descriptorStrideAB
            = usesDataParallelArgsV1 && problemType.useInitialStridesAB ? 0 : 1;
        const size_t descriptorStrideCD = usesDataParallelArgsV1 ? startStrideCD : 1;
        bool         sparseA            = problemType.sparse == 1;
        bool         sparseB            = problemType.sparse == 2;
        auto const&  descriptorA        = sparseA ? problem.compressed() : problem.a();
        auto const&  descriptorB        = sparseB ? problem.compressed() : problem.b();
        void const*  aAddress           = sparseA ? inputs.compressed : inputs.a;
        void const*  bAddress           = sparseB ? inputs.compressed : inputs.b;

        const bool pointerArrayBatch
            = problem.batchMode() == ContractionProblemGemm::BATCHMODE::POINTER_ARRAY;

        auto wsStrideForIdx = [&](size_t idx) -> size_t {
            // Mirrors the develop accumulator: stride[startStrideCD] = sizes[0],
            // and each subsequent stride multiplies in the next size.
            size_t s = startStrideCD ? problem.d().sizes()[0] : 1;
            for(size_t i = startStrideCD; i < idx; ++i)
                s *= problem.d().sizes()[i];
            return s;
        };

        if(T_Debug)
            std::cout << "Custom call arguments:" << std::endl;

        // rocRoller workgroup-mapping kernels read M, N, the macrotile, and
        // WGM. hipBLASLt supplies DEFAULT_WGM (2). Computed only if a custom
        // kernel argument asks for one of those semantics.
        std::optional<RocRollerWgmKernargs> wgmKernargs;
        auto                                wgmKernarg = [&]() -> RocRollerWgmKernargs const& {
            if(!wgmKernargs)
            {
                if(problem.problemSizes().size() < 2)
                    throw std::runtime_error("rocRoller WGM kernargs need M and N problem sizes");
                wgmKernargs
                    = evaluateRocRollerWgmKernargs(static_cast<int64_t>(problem.problemSizes()[0]),
                                                   static_cast<int64_t>(problem.problemSizes()[1]),
                                                   customKernel.macrotile.x,
                                                   customKernel.macrotile.y,
                                                   RocRollerDefaultWgm);
            }
            return *wgmKernargs;
        };

        for(auto arg : customKernel.args)
        {
            if(T_Debug)
                std::cout << "Type: " << toString(arg.type)
                          << " Semantic: " << toString(arg.semantic) << std::endl;

            switch(arg.semantic)
            {
            // ---- Core GEMM problem args ----
            case CustomArgSemantic::SizeFree0:
                rv.args.appendCustomType("SizeFree0", problem.problemSizes()[0], arg.type);
                break;
            case CustomArgSemantic::SizeFree1:
                rv.args.appendCustomType("SizeFree1", problem.problemSizes()[1], arg.type);
                break;
            case CustomArgSemantic::SizeFree2:
                rv.args.appendCustomType("SizeFree2", problem.problemSizes()[2], arg.type);
                break;
            case CustomArgSemantic::SizeFree3:
                rv.args.appendCustomType("SizeFree3", problem.problemSizes()[3], arg.type);
                break;
            case CustomArgSemantic::SizeSum:
                rv.args.appendCustomType(
                    "SizeSum", problem.problemSizes()[problem.problemSizes().size() - 1], arg.type);
                break;
            case CustomArgSemantic::SizeSumDiv2:
                rv.args.appendCustomType("SizeSumDiv2",
                                         problem.problemSizes()[problem.problemSizes().size() - 1]
                                             / 2,
                                         arg.type);
                break;
            // size() is unsigned, so a problem with too few dimensions would wrap the
            // index to a huge value and read out of bounds; fall back to 1 (an
            // identity summation extent) instead.
            case CustomArgSemantic::SizeSum1:
                rv.args.appendCustomType(
                    "SizeSum1",
                    problem.problemSizes().size() >= 2
                        ? problem.problemSizes()[problem.problemSizes().size() - 2]
                        : 1,
                    arg.type);
                break;
            case CustomArgSemantic::SizeSum2:
                rv.args.appendCustomType(
                    "SizeSum2",
                    problem.problemSizes().size() >= 3
                        ? problem.problemSizes()[problem.problemSizes().size() - 3]
                        : 1,
                    arg.type);
                break;
            case CustomArgSemantic::StrideA0:
                rv.args.appendCustomType(
                    "StrideA0", descriptorA.strides()[descriptorStrideAB], arg.type);
                break;
            case CustomArgSemantic::StrideA1:
                rv.args.appendCustomType(
                    "StrideA1", descriptorA.strides()[descriptorStrideAB + 1], arg.type);
                break;
            case CustomArgSemantic::StrideA2:
                rv.args.appendCustomType(
                    "StrideA2", descriptorA.strides()[descriptorStrideAB + 2], arg.type);
                break;
            case CustomArgSemantic::StrideB0:
                rv.args.appendCustomType(
                    "StrideB0", descriptorB.strides()[descriptorStrideAB], arg.type);
                break;
            case CustomArgSemantic::StrideB1:
                rv.args.appendCustomType(
                    "StrideB1", descriptorB.strides()[descriptorStrideAB + 1], arg.type);
                break;
            case CustomArgSemantic::StrideB2:
                rv.args.appendCustomType(
                    "StrideB2", descriptorB.strides()[descriptorStrideAB + 2], arg.type);
                break;
            case CustomArgSemantic::StrideC0:
                rv.args.appendCustomType("StrideC0",
                                         useWSStride ? wsStrideForIdx(1)
                                                     : problem.c().strides()[descriptorStrideCD],
                                         arg.type);
                break;
            case CustomArgSemantic::StrideC1:
                rv.args.appendCustomType(
                    "StrideC1",
                    useWSStride ? wsStrideForIdx(2) : problem.c().strides()[descriptorStrideCD + 1],
                    arg.type);
                break;
            case CustomArgSemantic::StrideC2:
                rv.args.appendCustomType(
                    "StrideC2",
                    useWSStride ? wsStrideForIdx(3) : problem.c().strides()[descriptorStrideCD + 2],
                    arg.type);
                break;
            case CustomArgSemantic::StrideD0:
                rv.args.appendCustomType("StrideD0",
                                         useWSStride ? wsStrideForIdx(1)
                                                     : problem.d().strides()[descriptorStrideCD],
                                         arg.type);
                break;
            case CustomArgSemantic::StrideD1:
                rv.args.appendCustomType(
                    "StrideD1",
                    useWSStride ? wsStrideForIdx(2) : problem.d().strides()[descriptorStrideCD + 1],
                    arg.type);
                break;
            case CustomArgSemantic::StrideD2:
                rv.args.appendCustomType(
                    "StrideD2",
                    useWSStride ? wsStrideForIdx(3) : problem.d().strides()[descriptorStrideCD + 2],
                    arg.type);
                break;
            case CustomArgSemantic::StrideE0:
                rv.args.appendCustomType(
                    "StrideE0",
                    problem.tensor(ContractionProblemGemm::TENSOR::E).strides()[descriptorStrideCD],
                    arg.type);
                break;
            case CustomArgSemantic::StrideE1:
                rv.args.appendCustomType("StrideE1",
                                         problem.tensor(ContractionProblemGemm::TENSOR::E)
                                             .strides()[descriptorStrideCD + 1],
                                         arg.type);
                break;
            case CustomArgSemantic::StrideE2:
                rv.args.appendCustomType("StrideE2",
                                         problem.tensor(ContractionProblemGemm::TENSOR::E)
                                             .strides()
                                             .at(descriptorStrideCD + 2),
                                         arg.type);
                break;
            case CustomArgSemantic::StrideScaleA0:
            {
                size_t scaleStride = problem.mxsa().strides()[descriptorStrideAB];
                rv.args.appendCustomType("StrideScaleA0", scaleStride, arg.type);
                break;
            }
            case CustomArgSemantic::StrideScaleA1:
            {
                size_t batchStride = usesDataParallelArgsV1
                                         ? problem.mxsa().strides()[descriptorStrideAB + 1]
                                         : preSwizzledScaleBatchStride(problem.mxsa(),
                                                                       problem.boundIndices()[0].a);
                rv.args.appendCustomType("StrideScaleA1", batchStride, arg.type);
                break;
            }
            case CustomArgSemantic::StrideScaleB0:
            {
                size_t scaleStride = problem.mxsb().strides()[descriptorStrideAB];
                rv.args.appendCustomType("StrideScaleB0", scaleStride, arg.type);
                break;
            }
            case CustomArgSemantic::StrideScaleB1:
            {
                size_t batchStride = usesDataParallelArgsV1
                                         ? problem.mxsb().strides()[descriptorStrideAB + 1]
                                         : preSwizzledScaleBatchStride(problem.mxsb(),
                                                                       problem.boundIndices()[0].b);
                rv.args.appendCustomType("StrideScaleB1", batchStride, arg.type);
                break;
            }
            case CustomArgSemantic::StrideScaleA2:
                rv.args.appendCustomType(
                    "StrideScaleA2", problem.mxsa().strides().at(descriptorStrideAB + 2), arg.type);
                break;
            case CustomArgSemantic::StrideScaleB2:
                rv.args.appendCustomType(
                    "StrideScaleB2", problem.mxsb().strides().at(descriptorStrideAB + 2), arg.type);
                break;
            case CustomArgSemantic::StrideA0Bytes:
                rv.args.appendCustomType("StrideA0Bytes",
                                         descriptorA.strides()[1] * descriptorA.elementBytes(),
                                         arg.type);
                break;
            case CustomArgSemantic::StrideB0Bytes:
                rv.args.appendCustomType("StrideB0Bytes",
                                         descriptorB.strides()[1] * descriptorB.elementBytes(),
                                         arg.type);
                break;
            case CustomArgSemantic::StrideC0Bytes:
                rv.args.appendCustomType("StrideC0Bytes",
                                         problem.c().strides()[1] * problem.c().elementBytes(),
                                         arg.type);
                break;
            case CustomArgSemantic::StrideD0Bytes:
                rv.args.appendCustomType("StrideD0Bytes",
                                         problem.d().strides()[1] * problem.d().elementBytes(),
                                         arg.type);
                break;
            case CustomArgSemantic::StrideMetadata0:
                rv.args.appendCustomType(
                    "StrideMetadata0", problem.metadata().strides()[descriptorStrideAB], arg.type);
                break;
            case CustomArgSemantic::StrideMetadata1:
                rv.args.appendCustomType("StrideMetadata1",
                                         problem.metadata().strides()[descriptorStrideAB + 1],
                                         arg.type);
                break;
            case CustomArgSemantic::StrideMetadata2:
                rv.args.appendCustomType("StrideMetadata2",
                                         problem.metadata().strides().at(descriptorStrideAB + 2),
                                         arg.type);
                break;
            case CustomArgSemantic::StrideCK:
            {
                size_t splitKStride = 0;
                if(gsu > 1)
                    splitKStride = problem.d().sizes()[0] * problem.d().sizes()[1];
                rv.args.appendCustomType("StrideCK", splitKStride, arg.type);
                break;
            }
            // The lowercase names are load-bearing: KernelArguments::append matches
            // on "alpha"/"beta" to promote a sub-32-bit compute type to 32 bits, which
            // is what a custom kernel's declared 32-bit scalar slot expects. Spelling
            // them any other way emits the narrow type instead and shifts every
            // argument that follows.
            case CustomArgSemantic::Alpha:
                rv.args.append("alpha", inputs.alpha, problem.alphaType());
                break;
            case CustomArgSemantic::Beta:
                if(internalArgsSupport.persistentLoopArgsVersion == 1 && !problemType.useBeta)
                    rv.args.append("beta", 0.0f, problem.betaType());
                else
                    rv.args.append("beta", inputs.beta, problem.betaType());
                break;
            case CustomArgSemantic::ComputeUnits:
                rv.args.appendCustomType("ComputeUnits", computeUnits, arg.type);
                break;
            case CustomArgSemantic::SplitK:
            {
                // AITER kernel expects log2(GSU) rather than the raw GSU value.
                // Derive it from the runtime-effective gsu, not sizeMapping's
                // compiled-in value: a user-specified or adaptively-downgraded GSU
                // must agree with the grid/workspace sizing and with StrideCK above.
                uint32_t remaining = gsu;
                uint32_t splitK    = 0;
                if(remaining > 1)
                {
                    while(remaining >>= 1)
                        splitK++;
                }
                rv.args.appendCustomType("SplitK", splitK, arg.type);
                break;
            }
            case CustomArgSemantic::OutputBF16:
            {
                uint32_t isBF16 = (problem.d().dataType() == rocisa::DataType::BFloat16) ? 1 : 0;
                rv.args.appendCustomType("OutputBF16", isBF16, arg.type);
                break;
            }
            case CustomArgSemantic::Padding:
                rv.args.template append<uint32_t>("Padding", 0);
                break;
            case CustomArgSemantic::ConstantZero:
                rv.args.appendCustomType("ConstantZero", 0, arg.type);
                break;
            case CustomArgSemantic::ConstantOne:
                rv.args.appendCustomType("ConstantOne", 1, arg.type);
                break;
            case CustomArgSemantic::DebugPattern:
                rv.args.template append<uint32_t>("DebugPattern", debugPattern);
                ++debugPattern;
                break;

            // ---- Pointer args ----
            case CustomArgSemantic::AddressA:
                rv.args.template append<void const*>(
                    "AddressA",
                    usesDataParallelArgsV1
                        ? (!problemType.stridedBatched || (pointerArrayBatch && inputs.batchA)
                               ? static_cast<void const*>(inputs.batchA)
                               : aAddress)
                        : aAddress);
                break;
            case CustomArgSemantic::AddressB:
                rv.args.template append<void const*>(
                    "AddressB",
                    usesDataParallelArgsV1
                        ? (!problemType.stridedBatched || (pointerArrayBatch && inputs.batchB)
                               ? static_cast<void const*>(inputs.batchB)
                               : bAddress)
                        : bAddress);
                break;
            case CustomArgSemantic::AddressC:
            {
                bool useWsC = false;
                if(gsu > 1 && !sizeMapping.isStreamK())
                {
                    bool singleWSD = (sizeMapping.globalAccumulation == 1
                                      && (problemType.computeType != problemType.dType
                                          || problemType.activationType != ActivationType::None));
                    if((singleWSD || sizeMapping.globalAccumulation == 2)
                       && sizeMapping.globalAccumulation != 3)
                        useWsC = true;
                }
                if(sizeMapping.isStreamK() && launch.reduction == origami::reduction_t::parallel)
                    useWsC = true;
                if(useWsC)
                    rv.args.template append<void const*>("AddressC", inputs.ws);
                else
                    rv.args.template append<void const*>(
                        "AddressC",
                        usesDataParallelArgsV1
                                && (!problemType.stridedBatched
                                    || (pointerArrayBatch && inputs.batchC))
                            ? static_cast<void const*>(inputs.batchC)
                            : inputs.c);
                break;
            }
            case CustomArgSemantic::AddressD:
            {
                bool useWsD = false;
                if(gsu > 1 && !sizeMapping.isStreamK())
                {
                    bool singleWSD = (sizeMapping.globalAccumulation == 1
                                      && (problemType.computeType != problemType.dType
                                          || problemType.activationType != ActivationType::None));
                    if(singleWSD || sizeMapping.globalAccumulation == 2
                       || sizeMapping.globalAccumulation == 3)
                        useWsD = true;
                }
                if(sizeMapping.isStreamK() && launch.reduction == origami::reduction_t::parallel)
                    useWsD = true;
                if(useWsD)
                    rv.args.template append<void const*>("AddressD", inputs.ws);
                else
                    rv.args.template append<void const*>(
                        "AddressD",
                        usesDataParallelArgsV1
                                && (!problemType.stridedBatched
                                    || (pointerArrayBatch && inputs.batchD))
                            ? static_cast<void const*>(inputs.batchD)
                            : inputs.d);
                break;
            }
            case CustomArgSemantic::AddressE:
                rv.args.template append<void const*>("AddressE", inputs.e);
                break;
            case CustomArgSemantic::AddressMetadata:
                rv.args.template append<void const*>("AddressMetadata", inputs.metadata);
                break;
            case CustomArgSemantic::AddressWorkspace:
                rv.args.template append<void const*>(toString(arg.semantic), inputs.ws);
                break;
            case CustomArgSemantic::AddressFlags:
                if(launch.reduction == origami::reduction_t::parallel)
                    rv.args.template append<void*>("AddressFlags", nullptr);
                else
                    rv.args.template append<void*>("AddressFlags", inputs.Synchronizer);
                break;
            case CustomArgSemantic::AddressSynchronizer:
                rv.args.template append<void const*>("AddressSynchronizer", inputs.Synchronizer);
                break;
            case CustomArgSemantic::AddressTD:
                rv.args.template append<void const*>("AddressTD", inputs.d);
                break;
            case CustomArgSemantic::AddressScaleZeroA:
                rv.args.template append<void const*>("AddressScaleZeroA", inputs.scaleZeroA);
                break;
            case CustomArgSemantic::ScaleBlockSizeA:
                rv.args.appendCustomType("groupSize", problem.scaleBlockSizeA(), arg.type);
                break;
            case CustomArgSemantic::AddressScaleA:
                rv.args.template append<void const*>("AddressScaleA", inputs.scaleA);
                break;
            case CustomArgSemantic::AddressScaleB:
                rv.args.template append<void const*>("AddressScaleB", inputs.scaleB);
                break;
            case CustomArgSemantic::AddressScaleC:
                rv.args.template append<void const*>("AddressScaleC", inputs.scaleC);
                break;
            case CustomArgSemantic::AddressScaleD:
                rv.args.template append<void const*>("AddressScaleD", inputs.scaleD);
                break;
            case CustomArgSemantic::AddressMXScaleA:
                rv.args.template append<void const*>("AddressMXScaleA", inputs.mxsa);
                break;
            case CustomArgSemantic::AddressMXScaleB:
                rv.args.template append<void const*>("AddressMXScaleB", inputs.mxsb);
                break;
            case CustomArgSemantic::AddressScaleAlphaVec:
                rv.args.template append<void const*>("AddressScaleAlphaVec", inputs.scaleAlphaVec);
                break;
            case CustomArgSemantic::AddressBias:
                if(problemType.useGradient && problem.biasSrc() == ContractionProblemGemm::TENSOR::D
                   && inputs.bias != nullptr)
                    rv.args.template append<void const*>("AddressBias", inputs.ws);
                else
                    rv.args.template append<void const*>(
                        "AddressBias",
                        usesDataParallelArgsV1 && !problemType.stridedBatched
                            ? static_cast<void const*>(inputs.batchBias)
                            : inputs.bias);
                break;
            case CustomArgSemantic::AddressGateResidual:
                if(problemType.stridedBatched)
                    rv.args.template append<void const*>("gateResidual", inputs.gateResidual);
                else
                    rv.args.template append<void const* const*>("batchGateResidual",
                                                                inputs.batchGateResidual);
                break;
            case CustomArgSemantic::GateResidualType:
                rv.args.template append<uint32_t>(
                    "gate_type",
                    static_cast<uint32_t>(
                        problem.useGateResidual()
                            ? problem.tensor(ContractionProblemGemm::TENSOR::GATE_RESIDUAL)
                                  .dataType()
                            : problemType.gateResidualDataTypeWhiteList.at(0)));
                break;
            case CustomArgSemantic::StrideGate0:
            case CustomArgSemantic::StrideGate1:
            case CustomArgSemantic::StrideGate2:
            {
                auto index = startStrideCD + static_cast<size_t>(arg.semantic)
                             - static_cast<size_t>(CustomArgSemantic::StrideGate0);
                auto const& gate = problem.tensor(ContractionProblemGemm::TENSOR::GATE_RESIDUAL);
                rv.args.template append<uint32_t>(
                    toString(arg.semantic),
                    problem.useGateResidual() ? gate.strides().at(index) : 0);
                break;
            }
            case CustomArgSemantic::AddressAmaxOut:
                rv.args.template append<void const*>("AddressAmaxOut", inputs.amaxD);
                break;
            case CustomArgSemantic::AmaxWS:
                rv.args.template append<void const*>("AmaxWS", inputs.ws);
                break;
            case CustomArgSemantic::AmaxSync:
                rv.args.template append<void const*>(toString(arg.semantic), inputs.Synchronizer);
                break;
            case CustomArgSemantic::Synchronizer:
                // Real buffer only in MBSK mode; otherwise the reduction happens in a
                // separate kernel and the pointer must be null.
                rv.args.template append<void const*>(
                    "Synchronizer",
                    adaptiveGlobalAccumulation == 3 ? inputs.Synchronizer : nullptr);
                break;
            case CustomArgSemantic::DebugBuffer:
                rv.args.template append<void const*>("DebugBuffer", nullptr);
                break;

            // ---- Kernel metadata args ----
            case CustomArgSemantic::GemmInfo:
            {
                uint32_t gemmCount = 1;
                // Encode argType (high 2 bits) so the kernel can distinguish
                // single/strided-batched (0) from general-batched POINTER_ARRAY (3).
                // Matches the legacy generateSingleCall encoding used by the
                // assembly kernel's ArgType branch (cf. KernelWriterAssembly.py).
                uint32_t argType
                    = (problem.batchMode() == ContractionProblemGemm::BATCHMODE::POINTER_ARRAY)
                          ? 3u
                          : static_cast<uint32_t>(KERNELARGTYPE::NORMAL);
                gemmCount = (gemmCount & 0x3FFFFFFF) | (argType << 30);
                rv.args.template append<uint32_t>("gemm_count", gemmCount);
                break;
            }
            case CustomArgSemantic::GemmCount:
                rv.args.appendCustomType("GemmCount", 1, arg.type);
                break;
            case CustomArgSemantic::InternalArgs:
            {
                auto [autoWGM, autoWGMXCC, autoWGMXCCCHUNK, autoWGMXCCSPLITK]
                    = calculateAutoWGM(problem, &hardware, launch.grid);
                auto [autoStaggerUMapping, autoStaggerU, autoStaggerUStrideShift]
                    = calculateAutoStaggerU(problem, &hardware, launch.grid, autoWGM);
                kernelArgs<T_Debug, true>(1,
                                          0,
                                          rv.args,
                                          getNumWorkGroups(rv),
                                          &hardware,
                                          problem.getParams(),
                                          autoWGM,
                                          autoWGMXCC,
                                          autoWGMXCCCHUNK,
                                          autoWGMXCCSPLITK,
                                          autoStaggerUMapping,
                                          autoStaggerU,
                                          autoStaggerUStrideShift,
                                          autoGsuVal,
                                          ntab);
                appendedInternalArgs = true;
                break;
            }
            case CustomArgSemantic::InternalArgs1:
                if(!appendedInternalArgs)
                    rv.args.appendCustomType("InternalArgs1", internalArg1, arg.type);
                break; // handled by InternalArgs via kernelArgs when present
            case CustomArgSemantic::TensileInternalArg0:
                rv.args.appendCustomType("TensileInternalArg0", internalArg0, arg.type);
                break;
            case CustomArgSemantic::TensileInternalArg1:
                rv.args.appendCustomType("TensileInternalArg1", internalArg1, arg.type);
                break;
            case CustomArgSemantic::NumWorkGroups:
                if(!appendedInternalArgs)
                    rv.args.appendCustomType("NumWorkGroups", getNumWorkGroups(rv), arg.type);
                break; // handled by InternalArgs via kernelArgs when present

            // ---- DataParallel scheduling args ----
            case CustomArgSemantic::ItersPerTile:
                rv.args.template append<uint32_t>("ItersPerTile", itersPerTile);
                break;
            case CustomArgSemantic::PersistentGrid:
                rv.args.template append<uint32_t>("PersistentGrid",
                                                  static_cast<uint32_t>(launch.grid));
                break;
            // ---- Legacy/custom scheduling adapter ----
            case CustomArgSemantic::MagicNumberItersPerTile:
                rv.args.template append<uint32_t>("MagicNumberItersPerTile",
                                                  skMagicNumberItersPerTile);
                break;
            case CustomArgSemantic::MagicShiftItersPerTile:
                rv.args.template append<uint32_t>("MagicShiftItersPerTile",
                                                  skMagicShiftItersPerTile);
                break;
            case CustomArgSemantic::TotalIters:
                rv.args.template append<uint32_t>("TotalIters", skTotalIters);
                break;
            case CustomArgSemantic::SKItersPerWG:
                rv.args.template append<uint32_t>("SKItersPerWG", skItersPerWG);
                break;
            case CustomArgSemantic::SKGrid:
                rv.args.template append<uint32_t>("SKGrid", static_cast<uint32_t>(launch.grid));
                break;
            case CustomArgSemantic::SKTilesAndSplit:
                rv.args.appendCustomType(toString(arg.semantic), skTotalTiles, arg.type);
                break;

            // ---- Packed batch dimension divisors ----
            // arg.index comes from the kernel's custom.config, so it is untrusted;
            // fall back to a dimension of 1 rather than indexing out of bounds.
            case CustomArgSemantic::MagicNumberSize:
            {
                uint32_t dimSize = arg.index < problem.problemSizes().size()
                                       ? problem.problemSizes()[arg.index]
                                       : 1;
                uint32_t magicShift;
                uint32_t magicNum = magicNumber(sizeMapping.magicDivAlg, dimSize, &magicShift);
                rv.args.template append<uint32_t>("MagicNumberSize", magicNum);
                break;
            }
            case CustomArgSemantic::MagicShiftSize:
            {
                uint32_t dimSize = arg.index < problem.problemSizes().size()
                                       ? problem.problemSizes()[arg.index]
                                       : 1;
                uint32_t magicShift;
                magicNumber(sizeMapping.magicDivAlg, dimSize, &magicShift);
                rv.args.template append<uint32_t>("MagicShiftSize", magicShift);
                break;
            }

            // ---- Epilogue control args ----
            case CustomArgSemantic::BiasType:
                rv.args.template append<uint32_t>("BiasType",
                                                  static_cast<uint32_t>(problem.bias().dataType()));
                break;
            case CustomArgSemantic::StrideBias:
            {
                TensorDescriptor const& bias = problem.tensor(ContractionProblemGemm::TENSOR::BIAS);
                uint32_t                strideBias = static_cast<uint32_t>(
                    problem.useBias() && bias.dimensions() ? bias.strides()[bias.dimensions() - 1]
                                                                          : 0);
                rv.args.template append<uint32_t>("StrideBias", strideBias);
                break;
            }
            case CustomArgSemantic::FactorDim:
                rv.args.template append<uint32_t>("FactorDim", problem.getParams().factorDim());
                break;
            case CustomArgSemantic::ActivationTypeArg:
                rv.args.template append<uint32_t>(
                    "ActivationType", static_cast<uint32_t>(problem.getParams().activationEnum()));
                break;
            case CustomArgSemantic::ActivationArg:
            {
                if(arg.index < inputs.activationArgs.size())
                    rv.args.appendCustomType(
                        "ActivationArg", inputs.activationArgs[arg.index], arg.type);
                else
                    rv.args.template append<uint32_t>("ActivationArg", 0);
                break;
            }
            case CustomArgSemantic::GSUSync:
                rv.args.template append<uint32_t>("GSUSync", 0);
                break;
            case CustomArgSemantic::ESMRuntimeSupported:
            {
                int32_t esmRuntimeSupported = 0;

                hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
                if(hipAMDGPU
                   && (hipAMDGPU->processor == AMDGPU::Processor::gfx1200
                       || hipAMDGPU->processor == AMDGPU::Processor::gfx1201))
                {
#if HIP_VERSION >= HIP_VERSION_WITH_EXPERT_SCHEDULING_MODE
                    HIP_CHECK_EXC(hipDeviceGetAttribute(&esmRuntimeSupported,
                                                        hipDeviceAttributeExpertSchedMode,
                                                        hipAMDGPU->deviceId));
#endif
                }
                rv.args.template append<int32_t>("ESMRuntimeSupported", esmRuntimeSupported);
                break;
            }
            case CustomArgSemantic::BatchOffsetD:
                rv.args.template append<int64_t>("batchOffsetD", inputs.batchOffsetD);
                break;
            case CustomArgSemantic::BatchOffsetC:
                rv.args.template append<int64_t>("batchOffsetC", inputs.batchOffsetC);
                break;
            case CustomArgSemantic::BatchOffsetA:
                rv.args.template append<int64_t>("batchOffsetA", inputs.batchOffsetA);
                break;
            case CustomArgSemantic::BatchOffsetB:
                rv.args.template append<int64_t>("batchOffsetB", inputs.batchOffsetB);
                break;
            case CustomArgSemantic::FusedA2ASegment:
            {
                appendFusedSegment(rv.args,
                                   inputs.fusedA2APeers,
                                   inputs.fusedA2ACounter,
                                   inputs.fusedA2AMyRank,
                                   problem.fusedA2AWorld(),
                                   inputs.fusedA2ADrain,
                                   static_cast<uint32_t>(problem.fusedA2AExtent()));
                break;
            }
            // ---- Random seed args ----
            case CustomArgSemantic::RNDSeed:
            {
                std::random_device                      rd;
                std::mt19937                            gen(rd());
                std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
                rv.args.template append<uint32_t>("RNDSeed", distribution(gen));
                break;
            }
