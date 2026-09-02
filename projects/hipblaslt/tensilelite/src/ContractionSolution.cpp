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

#include <Tensile/hip/HipUtils.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/UtilsOrigami.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include <Tensile/UtilsOrigami.hpp>
#include <iostream>
#include <origami/streamk.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <random>

#include <sstream>

#ifdef ENABLE_ROCTX
#include <roctracer/roctx.h>
#endif

#define TENSILELITE_TO_STR(x) #x
#define TENSILELITE_ENUMSTR(x) x, TENSILELITE_TO_STR(x)

namespace TensileLite
{
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
        // workspace the host reserves matches the layout the kernel addresses.
        // Returns 0 when the architecture cannot be determined.
        inline size_t streamKPerQueueStrideBytes(Hardware const& hardware)
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
        // non-dynamic-queue solution serves the GEMM, rather than staying
        // selectable while the per-XCD counter workspace is sized with an
        // unknown (0) queue count (which would under-allocate). Kept isolated
        // here so it stays trivially unit-testable (see CuCount_test.cpp).
        inline bool streamKDynamicQueueUnsupported(Hardware const& hardware)
        {
            auto const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            if(hipAMDGPU == nullptr || hipAMDGPU->analyticalHardware == nullptr)
                return true;
            size_t baked  = streamKBakedQueueCount(hardware);
            size_t numXCD = hipAMDGPU->analyticalHardware->NUM_XCD;
            return baked == 0 || numXCD == 0 || (numXCD & (numXCD - 1)) != 0
                   || numXCD != baked;
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

        // One-shot notice when uniform-summation-order grid steering displaces a developer
        // override (skFixedGrid) or production CU knobs (skMaxCUs / skGridMultiplier).
        void warnStreamKUniformityGridSnapOnce(size_t g0, size_t gStar)
        {
            static std::once_flag warnedFlag;
            std::call_once(warnedFlag, [g0, gStar]() {
                std::cerr << "hipBLASLt Warning: uniformSummationOrder steered the Stream-K grid "
                             "from "
                          << g0 << " to " << gStar
                          << " (never upward) so the launch stays row-uniform; "
                             "skFixedGrid / skMaxCUs / skGridMultiplier act as hints under "
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
        split.extraIters   = static_cast<uint32_t>(static_cast<size_t>(skTiles) * itersPerTile
                                                 - static_cast<size_t>(split.skItersPerWG)
                                                       * skGrid);
        return split;
    }

    bool streamKStaticSplitRowUniform(StreamKStaticSplit const& split,
                                      size_t                    tiles,
                                      size_t                    itersPerTile,
                                      size_t                    skGrid,
                                      bool                      perTileExtraIters)
    {
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
        // Without the capability bit, extraIters != 0 means chunks come in two
        // lengths under the global first-E mapping, so the chunk lattice has no
        // single period. skItersPerWG != 0 is not implied by the rest: tiles ==
        // 0 is reachable from the grouped-GEMM callers and would otherwise leave
        // I % skItersPerWG undefined.
        if(split.skTiles == 0)
            return true;

        if(split.skItersPerWG != 0 && split.extraIters == 0 && split.skTiles == tiles
           && itersPerTile % split.skItersPerWG == 0)
            return true;

        if(perTileExtraIters && split.skTiles == tiles && tiles != 0 && skGrid % tiles == 0
           && split.skItersPerWG != 0)
            return true;

        return false;
    }

    bool streamKParallelReductionRowUniform(StreamKSettings const& sk,
                                            int                    streamKAtomic,
                                            bool                   staticTwoTilePacking,
                                            size_t                 tiles)
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

    StreamKWorkgroupIterRange streamKWorkgroupIterRange(
        size_t w, size_t tiles, size_t itersPerTile, size_t skGrid, bool perTileExtraIters)
    {
        StreamKWorkgroupIterRange range;
        if(skGrid == 0 || tiles == 0)
            return range;

        const size_t totalIters = tiles * itersPerTile;
        const size_t W          = totalIters / skGrid;
        const size_t E          = totalIters - W * skGrid;

        if(perTileExtraIters && skGrid % tiles == 0)
        {
            const size_t F    = skGrid / tiles;
            const size_t q    = w / F;
            const size_t s    = w % F;
            const size_t remI = itersPerTile % F;
            const size_t base = itersPerTile / F;
            range.start       = q * itersPerTile + s * base + std::min(s, remI);
            range.end = range.start + base + (s < remI ? size_t{1} : size_t{0});
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
                std::cout << "   " << "m: " << args[i].m << std::endl;
                std::cout << "   " << "n: " << args[i].n << std::endl;
                std::cout << "   " << "batch: " << args[i].batch << std::endl;
                std::cout << "   " << "k: " << args[i].k << std::endl;
                std::cout << "   " << "D: " << args[i].d << std::endl;
                std::cout << "   " << "C: " << args[i].c << std::endl;
                std::cout << "   " << "A: " << args[i].a << std::endl;
                std::cout << "   " << "B: " << args[i].b << std::endl;
                std::cout << "   " << "strideD1: " << args[i].strideD1 << std::endl;
                std::cout << "   " << "strideD2: " << args[i].strideD2 << std::endl;
                std::cout << "   " << "strideC1: " << args[i].strideC1 << std::endl;
                std::cout << "   " << "strideC2: " << args[i].strideC2 << std::endl;
                std::cout << "   " << "strideA1: " << args[i].strideA1 << std::endl;
                std::cout << "   " << "strideA2: " << args[i].strideA2 << std::endl;
                std::cout << "   " << "strideB1: " << args[i].strideB1 << std::endl;
                std::cout << "   " << "strideB2: " << args[i].strideB2 << std::endl;
                std::cout << "   " << "Alpha: " << alphaPrint << std::endl;
                std::cout << "   " << "Beta: " << betaPrint << std::endl;
                std::cout << "   " << "scaleAlphaVec: " << args[i].scaleAlphaVec << std::endl;
                std::cout << "   " << "bias: " << args[i].bias << std::endl;
                std::cout << "   " << "e: " << args[i].e << std::endl;
                std::cout << "   " << "strideE1: " << args[i].strideE1 << std::endl;
                std::cout << "   " << "strideE2: " << args[i].strideE2 << std::endl;
                std::cout << "   " << "act0: " << args[i].act0 << std::endl;
                std::cout << "   " << "act1: " << args[i].act1 << std::endl;
                std::cout << "   " << "activationType: " << args[i].activationType << std::endl;
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

    template <bool T_Debug, bool insertKernelArgs, typename KA>
    void ContractionSolution::singleCallArgs(ContractionSolution::Problem const& problem,
                                             ContractionInputs const&            inputs,
                                             uint32_t const&        workspaceOffsetInByte,
                                             Hardware const*        hardware,
                                             dim3 const&            problemNumGroupTiles,
                                             dim3 const&            numWorkGroups,
                                             KA&                    args,
                                             StreamKSettings const& sk) const
    {
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

        auto [autoWGM, autoWGMXCC, autoWGMXCCCHUNK, autoWGMXCCSPLITK]
            = calculateAutoWGM(problem, hardware, sk.grid);
        auto [autoStaggerUMapping, autoStaggerU, autoStaggerUStrideShift]
            = calculateAutoStaggerU(problem, hardware, sk.grid, autoWGM);
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
        bool singleWSD = false;
        if(sizeMapping.globalAccumulation == 1
           && (problemType.computeType != problemType.dType
               || problemType.activationType != ActivationType::None))
            singleWSD = true;
        // Additional check for General Batched GEMM until GSU and StreamK are supported
        // in General Batched GEMM
        if(gsu > 1 && sizeMapping.streamK == 0
           && ((singleWSD || sizeMapping.globalAccumulation == 2)
               || (sizeMapping.globalAccumulation == 3)))
        {
            args.template append<void const*>("ws_d", (uint8_t*)inputs.ws + workspaceOffsetInByte);
            if(sizeMapping.globalAccumulation == 3)
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
            if(sizeMapping.streamK > 0 && sk.reduction == origami::reduction_t::parallel)
            {
                args.template append<void const*>("ws_d",
                                                  (uint8_t*)inputs.ws + workspaceOffsetInByte);
                args.template append<void const*>("ws_c",
                                                  (uint8_t*)inputs.ws + workspaceOffsetInByte);
            }
            else
            {
                args.template append<void const*>("d", inputs.d);
                args.template append<void const*>("c", inputs.c);
            }
        }
        else
        {
            args.template append<void const* const*>("batchD", inputs.batchD);
            args.template append<void const* const*>("batchC", inputs.batchC);
        }

        if(problemType.stridedBatched)
        {
            args.template append<void const*>(
                "a", problemType.sparse == 1 ? inputs.compressed : inputs.a);
            if(problemType.mxBlockA)
                args.template append<void const*>("mxsa", inputs.mxsa);
            args.template append<void const*>(
                "b", problemType.sparse == 2 ? inputs.compressed : inputs.b);
            if(problemType.mxBlockB)
                args.template append<void const*>("mxsb", inputs.mxsb);
        }
        else
        {
            args.template append<void const* const*>("batchA", inputs.batchA);
            args.template append<void const* const*>("batchB", inputs.batchB);
        }

        if(problemType.sparse)
            args.template append<unsigned char const*>("metadata", inputs.metadata);

        // Additional check for General Batched GEMM until GSU and StreamK are supported
        // in General Batched GEMM
        //
        // StreamKForceDPOnly (SK3 DP-first, gfx1250) always reduces via the tree path
        // (getSKReduction returns tree, Flags == Synchronizer, never parallel) and never
        // touches the workspace partials/fixup path, so AddressWS/AddressFlags are dead.
        // The device kernel drops them from the SGPR define and .kd metadata, so we must
        // not append ws/Flags here or the positional kernarg layout would corrupt the
        // downstream (StridesD/Alpha/...) offsets. Keep appending for every other
        // streamK>0 && atomic==0 kernel (layout unchanged).
        if(sizeMapping.streamK > 0 && sizeMapping.streamKAtomic == 0
           && sizeMapping.streamKForceDPOnly == 0)
        {
            // Assert hardware is not null
            // For now grouped gemm is not supported and passes nullptr
            TENSILE_ASSERT_EXC(hardware != nullptr);

            // StreamK workspace + flags. Synchronizer has already been pointed
            // at the per-stream Stream-K region by the host for this solution,
            // which is what keeps two concurrent Stream-K kernels from clearing
            // each other's flags.
            args.template append<void const*>("ws", inputs.ws);
            if(sk.reduction == origami::reduction_t::parallel)
                args.template append<void*>("Flags", nullptr);
            else
                args.template append<void*>("Flags", inputs.Synchronizer);
        }

        size_t startStrideCD = problemType.useInitialStridesCD ? 0 : 1;
        size_t startStrideAB = problemType.useInitialStridesAB ? 0 : 1;

        // Pass wsStride if it's not in MBSK mode
        bool gsuWSStride
            = gsu > 1 && sizeMapping.globalAccumulation != 3 && sizeMapping.streamK == 0;
        bool skWSStride = sizeMapping.streamK > 0 && sk.reduction == origami::reduction_t::parallel;
        // Additional check for General Batched GEMM until GSU and StreamK are supported
        // in General Batched GEMM
        if(gsuWSStride || skWSStride)
        {
            size_t wsStride = startStrideCD ? d.sizes()[0] : 1;
            for(size_t i = startStrideCD; i < d.dimensions(); i++)
            {
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideW_D", i), wsStride);
                wsStride *= d.sizes()[i];
            }

            wsStride = startStrideCD ? d.sizes()[0] : 1;
            for(size_t i = startStrideCD; i < c.dimensions(); i++)
            {
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideW_C", i), wsStride);
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

        for(size_t i = startStrideAB; i < a.dimensions(); i++)
        {
            auto stride_a = problemType.sparse == 1 ? compressed.strides()[i] : a.strides()[i];
            args.template append<uint32_t>(concatenate_if<T_Debug>("strideA", i), stride_a);
        }

        if(problemType.mxBlockA)
            for(size_t i = startStrideAB; i < mxsa.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideMXSA", i), mxsa.strides()[i]);

        for(size_t i = startStrideAB; i < b.dimensions(); i++)
        {
            auto stride_b = problemType.sparse == 2 ? compressed.strides()[i] : b.strides()[i];
            args.template append<uint32_t>(concatenate_if<T_Debug>("strideB", i), stride_b);
        }

        if(problemType.mxBlockB)
            for(size_t i = startStrideAB; i < mxsb.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideMXSB", i), mxsb.strides()[i]);

        if(problemType.sparse)
        {
            for(size_t i = startStrideAB; i < a.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideMetadata", i),
                                               metadata.strides()[i]);
        }

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
#if HIP_VERSION >= 70353390
                HIP_CHECK_EXC(hipDeviceGetAttribute(&esmRuntimeSupported,
                                                    hipDeviceAttributeExpertSchedMode,
                                                    hipAMDGPU->deviceId));
#endif
                args.template append<int32_t>("ESMRuntimeSupported", esmRuntimeSupported);
            }
        }

        // Additional check for General Batched GEMM until GSU and StreamK are supported
        // in General Batched GEMM
        if(sizeMapping.streamK != 0)
        {
            if(sizeMapping.streamK != 3 && sizeMapping.streamK != 4 && sizeMapping.streamK != 5)
            {
                throw std::runtime_error("Stream-K modes 1 and 2 are no longer supported; "
                                         "use StreamK=3, 4, or 5");
            }

            if(gsu > 1)
            {
                std::cerr << "Warning: Stream-K Data Parallel does not support GSU > 1, "
                          << "setting GSU to 1." << std::endl;
                gsu = 1;
            }

            // Dynamic Stream-K uses a different kernel argument layout from Stream-K 3.
            if(sizeMapping.streamK == 4)
            {
                AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(hardware);
                assert(pAMDGPU != nullptr);
                int overrideTiles = pAMDGPU->skTiles;
                int overrideSplit = pAMDGPU->skSplit;

                auto itersPerTile = std::max(size_t{1}, problem.getItersPerTile(sizeMapping));
                auto tiles = problem.getNumTiles(sizeMapping, 1);
                // Determine number of stream-k tiles and splitting factor
                uint32_t skTiles = 0;
                uint32_t skSplit = 2;
                // Check for debug overrides
                if (overrideTiles > -1)
                    skTiles = overrideTiles;
                if (overrideSplit > -1)
                    skSplit = overrideSplit;
                // Calculate number of stream-k iterations per workitem
                uint32_t skItersPerWI = CeilDivide(static_cast<uint32_t>(itersPerTile), skSplit);
                // Calculate real splitting factor in case iterations don't divide evenly
                skSplit = CeilDivide(static_cast<uint32_t>(itersPerTile), skItersPerWI);
                uint32_t totalItems = (tiles - skTiles) + skTiles * skSplit;

                args.template append<uint32_t>("ItersPerTile", itersPerTile);
                args.template append<uint32_t>("TotalItems", totalItems);
                args.template append<uint32_t>("SKTiles", skTiles);
                args.template append<uint32_t>("SKSplit", skSplit);
                args.template append<uint32_t>("SKItersPerWI", skItersPerWI);
                args.template append<uint32_t>("SKGrid", sk.grid);
            }
            else if(sizeMapping.streamK == 5)
            {
                // SK5 hybrid: pack 6 args for the active sub-mode (SK3/SK4 RegSet-alias
                // the same SGPR slots). Mode bit is bit 30 of slot 2 — not bit 31,
                // which magicNumberAlg2 uses as the magic-division "add" indicator.

                AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(hardware);
                assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);

                auto sk3_tiles = problem.getNumTiles(sizeMapping, 1);
                auto sk3_itersPerTile
                    = std::max(size_t{1}, problem.getItersPerTile(sizeMapping));

                const bool effectiveDynamic
                    = streamK5EffectiveDynamic(problem, *hardware);

                if(effectiveDynamic)
                {
                    int overrideTiles = pAMDGPU->skTiles;
                    int overrideSplit = pAMDGPU->skSplit;
                    uint32_t sk4_skTiles = 0;
                    uint32_t sk4_skSplit = 2;
                    if(overrideTiles > -1)
                        sk4_skTiles = overrideTiles;
                    if(overrideSplit > -1)
                        sk4_skSplit = overrideSplit;
                    uint32_t sk4_skItersPerWI
                        = CeilDivide(static_cast<uint32_t>(sk3_itersPerTile),
                                     sk4_skSplit);
                    sk4_skSplit
                        = CeilDivide(static_cast<uint32_t>(sk3_itersPerTile),
                                     sk4_skItersPerWI);
                    uint32_t sk4_totalItems
                        = (sk3_tiles - sk4_skTiles) + sk4_skTiles * sk4_skSplit;

                    TENSILE_ASSERT_EXC((sk4_skTiles & 0xC0000000u) == 0u
                                       && "SK5 SK4 skTiles collides with mode/magic bits");
                    uint32_t packedSkTiles = sk4_skTiles | 0x40000000u;

                    args.template append<uint32_t>("ItersPerTile",
                                                   sk3_itersPerTile);
                    args.template append<uint32_t>("TotalItems",
                                                   sk4_totalItems);
                    args.template append<uint32_t>("SKTiles|ModeBit",
                                                   packedSkTiles);
                    args.template append<uint32_t>("SKSplit", sk4_skSplit);
                    args.template append<uint32_t>("SKItersPerWI",
                                                   sk4_skItersPerWI);
                    args.template append<uint32_t>("SKGrid", sk.grid);
                }
                else
                {
                    // SK5-off mirrors standalone SK3 arg packing.
                    uint32_t magicNumberItersPerTile;
                    uint32_t magicShiftItersPerTile;
                    magicNumberItersPerTile = magicNumber(
                        2, sk3_itersPerTile, &magicShiftItersPerTile);
                    assert((magicShiftItersPerTile & 0x40000000u) == 0u);

                    uint32_t sk3_skItersPerWG;
                    uint32_t sk3_skTiles;
                    if(sk.reduction == origami::reduction_t::parallel)
                    {
                        uint32_t skSplit
                            = static_cast<uint32_t>(sk.grid / sk3_tiles);
                        sk3_skItersPerWG
                            = static_cast<uint32_t>(sk3_itersPerTile) / skSplit;
                        sk3_skTiles = skSplit;
                    }
                    else
                    {
                        const StreamKStaticSplit sk3_split = streamKStaticSplit(
                            sk3_tiles,
                            sk3_itersPerTile,
                            sk.grid,
                            pAMDGPU->skFullTiles,
                            sizeMapping.streamKForceDPOnly != 0);
                        sk3_skTiles      = sk3_split.skTiles;
                        sk3_skItersPerWG = sk3_split.skItersPerWG;
                    }

                    args.template append<uint32_t>("ItersPerTile",
                                                   sk3_itersPerTile);
                    args.template append<uint32_t>("MagicNumberItersPerTile",
                                                   magicNumberItersPerTile);
                    args.template append<uint32_t>("MagicShiftItersPerTile",
                                                   magicShiftItersPerTile);
                    args.template append<uint32_t>("SKItersPerWG",
                                                   sk3_skItersPerWG);
                    args.template append<uint32_t>("skGrid",
                                                   static_cast<uint32_t>(sk.grid));
                    args.template append<uint32_t>("skTiles", sk3_skTiles);
                }
            }
            else
            {
                auto tiles = problem.getNumTiles(sizeMapping, 1);

                // Clamp minimum iters per tile to 1 to allow stream-k index calculation to work in case K==0
                // In this case no actual iterations will be run, but workgroups will be mapped correctly for beta*C
                auto     itersPerTile = std::max(size_t{1}, problem.getItersPerTile(sizeMapping));
                auto     totalIters   = tiles * itersPerTile;

                uint32_t magicNumberItersPerTile;
                uint32_t magicShiftItersPerTile;
                magicNumberItersPerTile = magicNumber(2, itersPerTile, &magicShiftItersPerTile);

                args.template append<uint32_t>("itersPerTile", itersPerTile);
                args.template append<uint32_t>("magicNumberItersPerTile", magicNumberItersPerTile);
                args.template append<uint32_t>("magicShiftItersPerTile", magicShiftItersPerTile);

                // Custom kernels still use totalIters
                if(!sizeMapping.customKernelName.empty())
                {
                    args.template append<uint32_t>("totalIters", totalIters);
                }

                // Stream-K 3 uses the two-tile ABI.
                if(sk.reduction == origami::reduction_t::parallel)
                {
                    uint32_t skSplit
                        = sk.grid / tiles; // skTiles is skSplit in parallel reduction path
                    uint32_t skItersPerWG = itersPerTile / skSplit;

                    args.template append<uint32_t>("SKItersPerWG", skItersPerWG);
                    args.template append<uint32_t>("skGrid", sk.grid);
                    args.template append<uint32_t>("skTiles", skSplit);
                }
                else
                {
                    AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(hardware);
                    assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);

                    const StreamKStaticSplit split
                        = streamKStaticSplit(tiles,
                                             itersPerTile,
                                             sk.grid,
                                             pAMDGPU->skFullTiles,
                                             sizeMapping.streamKForceDPOnly != 0);

                    args.template append<uint32_t>("SKItersPerWG", split.skItersPerWG);
                    args.template append<uint32_t>("skGrid", sk.grid);
                    args.template append<uint32_t>("skTiles", split.skTiles);
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
                    hasGate ? problem.tensor(ContractionProblemGemm::TENSOR::GATE_RESIDUAL).dataType()
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

    inline double calculateGranularity(
        uint32_t m, uint32_t n, uint32_t mt0, uint32_t mt1, uint32_t gsu, uint32_t cuCount)
    {
        return (double)(std::ceil(m / mt0) * std::ceil(n / mt1) * gsu / cuCount)
               / std::ceil(std::ceil(m / mt0) * std::ceil(n / mt1) * gsu / cuCount);
    }

    std::tuple<int32_t, size_t, size_t, size_t> ContractionSolution::calculateAutoWGM(
        Problem const& problem, Hardware const* hardware, uint32_t const skgrid) const
    {
        // Hardware
        AMDGPU const*         pAMDGPU   = dynamic_cast<AMDGPU const*>(hardware);
        hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(hardware);

        // Default WGM
        int32_t  defaultWGM          = 1;
        uint32_t defaultWGMXCC       = 1;
        uint32_t defaultWGMXCCCHUNK  = 0;
        uint32_t defaultWGMXCCSPLITK = 0;

        // Dynamically pick the values
        if(sizeMapping.streamK != 0 && skgrid != 0 && sizeMapping.workGroupMapping == 0
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
                        .size    = {sizes[0], sizes[1], sizes[3]},
                        .batch   = sizes[2],
                        // CU budget hint; 0 = use all CUs.
                        .num_cus = static_cast<size_t>(problem.getParams().smCountTarget()),
                        .a_dtype = datatypeToAnalyticalDatatype(problem.a().dataType()),
                        .b_dtype = datatypeToAnalyticalDatatype(problem.b().dataType()),
                    };
                    origami::config_t origami_config = {
                        .mt            = {static_cast<size_t>(sizeMapping.macroTile.x),
                                          static_cast<size_t>(sizeMapping.macroTile.y),
                                          static_cast<size_t>(sizeMapping.depthU)},
                        .cache_hints_a = sizeMapping.nonTemporalA,
                        .cache_hints_b = sizeMapping.nonTemporalB,
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
                        std::make_tuple(defaultWGM, defaultWGMXCC, defaultWGMXCCCHUNK, defaultWGMXCCSPLITK),
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
            defaultWGMXCCCHUNK  = 0;

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
                assert(std::fabs(defaultWGM) < 512);   // 10-bit signed
                assert(defaultWGMXCC < 16);             // 4 bits
                assert(defaultWGMXCCCHUNK < 256);       // 8 bits
                assert(defaultWGMXCCSPLITK < 1024);     // 10 bits
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
        // Hardware
        AMDGPU const*         pAMDGPU   = dynamic_cast<AMDGPU const*>(hardware);
        hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(hardware);

        // Default StaggerU
        size_t defaultStaggerUMapping     = 0;
        size_t defaultStaggerU            = 0;
        size_t defaultStaggerUStrideShift = 0;

        // Dynamically pick the values
        if(sizeMapping.streamK != 0 && skgrid != 0 && sizeMapping.workGroupMapping == 0
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
                        .size    = {sizes[0], sizes[1], sizes[3]},
                        .batch   = sizes[2],
                        // CU budget hint; 0 = use all CUs.
                        .num_cus = static_cast<size_t>(problem.getParams().smCountTarget()),
                        .a_dtype = datatypeToAnalyticalDatatype(problem.a().dataType()),
                        .b_dtype = datatypeToAnalyticalDatatype(problem.b().dataType()),
                    };
                    origami::config_t origami_config = {
                        .mt            = {static_cast<size_t>(sizeMapping.macroTile.x),
                                          static_cast<size_t>(sizeMapping.macroTile.y),
                                          static_cast<size_t>(sizeMapping.depthU)},
                        .cache_hints_a = sizeMapping.nonTemporalA,
                        .cache_hints_b = sizeMapping.nonTemporalB,
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

        // Uniform summation order requires StaggerU == 0; clamping the mapping
        // alone is not enough, because a StreamK workgroup that owns more than
        // one tile breaks uniformity under every mapping.
        //
        // The placement of this clamp is load-bearing. It must sit AFTER the
        // TENSILE_FIXED_STAGGERU* overrides directly above, or an env override
        // could reintroduce a stagger, and BELOW the staggerUParamsCache write,
        // so the cache keeps holding the real origami prediction rather than
        // this mode's zeros for concurrent callers who did not request it.
        if(problem.getParams().uniformSummationOrder())
        {
            defaultStaggerUMapping     = 0;
            defaultStaggerU            = 0;
            defaultStaggerUStrideShift = 0;
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
            gsuVal = std::min(gsuVal,
                         static_cast<uint32_t>(MAX_WORKGROUP_NUMBER / std::ceil(static_cast<float>(M) / MT0)
                             / std::ceil(static_cast<float>(N) / MT1) / B));

        // GlobalSplitUCheckMinK
        if(gsuVal > 1)
            gsuVal = std::min(gsuVal, static_cast<uint32_t>(std::ceil(static_cast<float>(K) / MT2)));

        // SynchronizerSizeCheck
        //
        // MBSK owns one GSU region, indexed by problem. Bounding usage by the
        // region size is what makes a solution unable to run past the end of it.
        if(gsuVal > 1 && sizeMapping.globalAccumulation == 3) // MBSK
        {
            uint32_t synchronizerUsage
                = sizeMapping.synchronizerSizePerWG * problem.getNumTiles(sizeMapping, 1) * B;

            gsuVal = synchronizerUsage > GsuSynchronizerElements ? 1 : gsuVal;
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

        const uint32_t MT0    = sizeMapping.macroTile.x;                    // MT_M
        const uint32_t MT1    = sizeMapping.macroTile.y;                    // MT_N
        const uint32_t depthU = static_cast<uint32_t>(sizeMapping.depthU);  // MT_K

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
        const uint32_t contigB = transB ? MT1    : depthU;

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
                      << " (M=" << M << " N=" << N << " K=" << K
                      << " MT0=" << MT0 << " MT1=" << MT1 << " DU=" << depthU
                      << " transA=" << transA << " transB=" << transB
                      << " contigA=" << contigA << " contigB=" << contigB
                      << " minLenA=" << minLenA << " minLenB=" << minLenB
                      << " ldA=" << ldA << " ldB=" << ldB
                      << " loopIters=" << loopIters
                      << " longA=" << longA << " longB=" << longB << ")"
                      << "\n";
        }

        return result;
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
                                         size_t       autoStaggerUStrideShift,
                                         uint32_t     autoGsuVal,
                                         AdaptiveGemmNTAB ntab) const
    {
        if constexpr(!Legacy)
        {
            gemmCount = gemmCount & 0x3FFFFFFF;
            // Currently 0 for kernel args, 1 for args located in HBM. This is a temporary slot.
            gemmCount = gemmCount | (argType << 30);
            args.template append<uint32_t>("gemm_count", gemmCount);
        }

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
        uint32_t       internalArg0        = 0;
        uint32_t       internalArg1        = 0;

        // GSU bit-width depends on InternalArgsSupport.version:
        //   v <  3: GSU occupies bits 0..13 (mask 0x3FFF, max 16383)
        //   v >= 3: GSU narrowed to bits 0..11 (mask 0x0FFF, max 4095);
        //           bits 12/13 carry NTA / NTB for AdaptiveGemmNTAB.
        constexpr uint32_t kGsuMaskV3   = 0x0FFF;
        constexpr uint32_t kNtaBitPos   = 12;
        constexpr uint32_t kNtbBitPos   = 13;
        const bool         useNtabBits  = (internalArgsSupport.version >= 3);
        const uint32_t     gsuMask      = useNtabBits ? kGsuMaskV3 : mask14;
        // Belt-and-suspenders: if a v<3 solution somehow has AGNTAB!=0, that's a codegen bug.
        assert((useNtabBits || sizeMapping.adaptiveGemmNTAB == 0)
               && "AdaptiveGemmNTAB requires InternalArgsSupport.version >= 3");
        // Only forward NT bits if both the kernel knows the new layout AND the
        // solution actually opted in to AdaptiveGemmNTAB. Otherwise leave bits clear
        // so a v<3 kernel reading 0x3FFF doesn't see them folded into GSU.
        const uint32_t ntaBit = (useNtabBits && sizeMapping.adaptiveGemmNTAB != 0
                                 && ntab.nta == 4)
                                    ? 1
                                    : 0;
        const uint32_t ntbBit = (useNtabBits && sizeMapping.adaptiveGemmNTAB != 0
                                 && ntab.ntb == 4)
                                    ? 1
                                    : 0;

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
                    internalArg1 = internalArg1
                                   | ((wgmxccsplitk & 0x3FF) << 22)
                                   | ((wgmxccchunk & 0xFF) << 14)
                                   | ((wgmxcc & 0xF) << 10)
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
                  + ", InternalArgsSupport.version="
                  + std::to_string(internalArgsSupport.version) + ", AdaptiveGemmNTAB="
                  + std::to_string(sizeMapping.adaptiveGemmNTAB)
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
            size_t           sum          = (staggerUMapping & 0x7) << 13;
            size_t           sus          = staggerMask1 & (staggerUStrideShift << 8);
            size_t           su           = mask8 & staggerU;
            if(Debug::Instance().disableStaggerU())
                su = 0;
            su           = su | sus;
            su           = su | sum;
            internalArg0 = internalArg0 | (su << 16);
        }
        else if(T_Debug && Debug::Instance().disableStaggerU())
            std::cout << "solution doesn't support configurable staggerU" << std::endl;

        args.template append<uint32_t>("internalArgs", internalArg0);

        if(internalArgsSupport.version >= 1)
        {
            args.template append<int32_t>("internalArgs1", internalArg1);
            args.template append<uint32_t>("numWorkGroups", numWorkGroups);
        }
    }

    void ContractionSolution::calculateGrid(dim3&                               workGroupSize,
                                            dim3&                               numWorkGroups,
                                            ContractionSolution::Problem const& problem) const
    {
        workGroupSize.x = sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y
                          * sizeMapping.workGroupSize.z;
        workGroupSize.y = 1;
        workGroupSize.z = 1;

        numWorkGroups.x = 1;
        numWorkGroups.y = 1;

        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
        {
            numWorkGroups.x *= problem.freeSizeA(i);
        }
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
        {
            numWorkGroups.y *= problem.freeSizeB(i);
        }

        numWorkGroups.z = 1;
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
        {
            if(sizeMapping.packBatchDims & 0x1)
                numWorkGroups.x *= problem.batchSize(i);
            if(sizeMapping.packBatchDims & 0x2)
                numWorkGroups.y *= problem.batchSize(i);
            if(!sizeMapping.packBatchDims)
                numWorkGroups.z *= problem.batchSize(i);
        }

        if(problem.transposeC01())
            std::swap(numWorkGroups.x, numWorkGroups.y);

        numWorkGroups.x = CeilDivide(numWorkGroups.x, sizeMapping.macroTile.x);
        numWorkGroups.y = CeilDivide(numWorkGroups.y, sizeMapping.macroTile.y);
    }

    template <bool T_Debug>
    KernelInvocation
        ContractionSolution::generateSingleCall(ContractionSolution::Problem const& problem,
                                                ContractionInputs const&            inputs,
                                                Hardware const&                     hardware,
                                                StreamKSettings const&              sk,
                                                GSUSettings const&                  gsuSettings) const
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

        if(sizeMapping.streamK != 0)
        {
            if(sizeMapping.streamKForceDPOnly != 0
               && (sizeMapping.clusterDim.x > 1 || sizeMapping.clusterDim.y > 1))
            {
                // ForceDPOnly cluster multicast [Cs, Ck]: launch a grid spanning
                // the full M x N tile space -- gridX = nWG0 (M-tiles), gridY = nWG1
                // (N-tiles), gridZ = batch -- so the kernel's StreamKIdx fold
                // (StreamK.preLoop) gives each work-group exactly one tile and the
                // Cs X-peers of a cluster always land M-adjacent (sharing B). A 1-D
                // [Cs, 1] cluster is the Ck == 1 case of the same launch. The
                // round-up below pads non-multiple extents; sk.grid == tiles here.
                rv.numWorkGroups.x = problemNumGroupTiles.x; // nWG0 (M-tiles)
                // rv.numWorkGroups.y already = nWG1 * gsu (N-tiles); z stays batch.
            }
            else
            {
                // Linear Stream-K launch (no cluster, or ForceDPOnly=0).
                rv.numWorkGroups.x = sk.grid;
                rv.numWorkGroups.y = 1;
                rv.numWorkGroups.z = 1;
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
        // clusterDim, so round up. The grid set above holds the REAL extents and
        // need not be a cluster multiple. The extra padded work-groups early-exit
        // in the kernel prologue (StreamK.streamKClusterPadEarlyExit on the
        // ForceDPOnly cluster path) BEFORE the -3 cluster barrier, so their
        // WAVEDONE decrements the barrier's live member count, and the surviving
        // peers' broadcast masks are trimmed to the present lanes
        // (computeMulticastMaskReduction).
        //
        // Only the ForceDPOnly cluster multicast needs this: it is the path whose
        // grid spans the real M x N tile space and whose padded peers have a
        // pad-exit. A ForceDPOnly==0 Stream-K cluster keeps develop's launch --
        // its 1-D sk.grid is not a tile space and it has no pad-exit, so rounding
        // up would only add work-groups that run the whole Stream-K prologue
        // before falling out on an empty iteration range.
        bool skClusterMulticast = sizeMapping.streamK != 0
                                  && sizeMapping.streamKForceDPOnly != 0 && enableCluster;
        if(enableCluster && (sizeMapping.streamK == 0 || skClusterMulticast))
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
                = calculateAutoWGM(problem, &hardware, sk.grid);
            auto [autoStaggerUMapping, autoStaggerU, autoStaggerUStrideShift]
                = calculateAutoStaggerU(problem, &hardware, sk.grid, autoWGM);
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
                kernelArgs<T_Debug, false>( 1,
                                            3,
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
            }
            else
            {
                kernelArgs<T_Debug, false>( 1,
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
            }
        }
        singleCallArgs<T_Debug, true>(
            problem, inputs, 0, &hardware, problemNumGroupTiles, rv.numWorkGroups, rv.args, sk);

        if(gsuSettings.globalAccumulation == 3 || sizeMapping.adaptiveGemmGSUA == 1) // MBSK or MB with AdaptiveGemmGSUA
        {
            rv.args.append<void const*>("dstD", inputs.d);
            // MBSK: synchronizer address, MB: null address
            rv.args.append<void const*>("Synchronizer",
                                        gsuSettings.globalAccumulation == 3
                                        ? inputs.Synchronizer
                                        : NULL);
            rv.args.append<uint32_t>("GSUSync", 0);
        }

        // Batch offset support for General Batched GEMM (SupportUserArgs kernels).
        // Appended at the tail, after the dstD/Synchronizer block, to match the
        // kernel signature order (see Signature.py).
        if(!problemType.groupedGemm && sizeMapping.customKernelName.empty())
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

    template <typename KA>
    void ContractionSolution::calculateSingleCallWorkGroupItems(
        std::vector<Problem> const& problems,
        const TensileLite::dim3&    workGroupSize,
        TensileLite::dim3&          numWorkGroups,
        TensileLite::dim3&          numWorkItems,
        KA&                         h_args,
        uint32_t                    autoGsuVal) const
    {

        uint32_t wgLeft  = 0;
        uint32_t wgRight = 0;

        for(int idx = 0; idx < problems.size(); idx++)
        {
            if constexpr(!std::is_same<KA, KernelArgumentsCounter>::value)
            {
                auto problem = problems[idx];

                numWorkGroups.x = 1;
                numWorkGroups.y = 1;
                numWorkGroups.z = 1;

                for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
                {
                    numWorkGroups.x *= problem.freeSizeA(i);
                }

                for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
                {
                    numWorkGroups.y *= problem.freeSizeB(i);
                }

                for(size_t i = 0; i < problem.batchIndices().size(); i++)
                {
                    if(sizeMapping.packBatchDims & 0x1)
                        numWorkGroups.x *= problem.batchSize(i);
                    if(sizeMapping.packBatchDims & 0x2)
                        numWorkGroups.y *= problem.batchSize(i);
                    if(!sizeMapping.packBatchDims)
                        numWorkGroups.z *= problem.batchSize(i);
                }

                if(problem.transposeC01())
                    std::swap(numWorkGroups.x, numWorkGroups.y);

                numWorkGroups.x = CeilDivide(numWorkGroups.x, sizeMapping.macroTile.x);
                numWorkGroups.y = CeilDivide(numWorkGroups.y, sizeMapping.macroTile.y);

                uint32_t gsu
                    = problem.getParams().gsu() > 0 ? problem.getParams().gsu() : autoGsuVal;
                if(gsu > 0)
                    numWorkGroups.y *= gsu;

                numWorkItems.x += (workGroupSize.x * numWorkGroups.x * workGroupSize.y
                                   * numWorkGroups.y * workGroupSize.z * numWorkGroups.z);

                if constexpr(std::is_same<KA, KernelArguments>::value)
                {
                    wgRight = numWorkItems.x / workGroupSize.x / workGroupSize.y / workGroupSize.z;
                    h_args.template append<uint32_t>("wgTable", wgLeft);
                    wgLeft = wgRight;
                }
            }
            else
            {
                if constexpr(!std::is_same<KA, int>::value)
                    h_args.template append<uint32_t>("wgTable", 0);
            }
        }
    }

    template <bool T_Debug, typename KA>
    KernelInvocation ContractionSolution::generateSingleCallGroupedGemm(
        std::vector<ContractionSolution::Problem> const& problems,
        ContractionSolution::GroupedInputs const&        inputs,
        Hardware const&                                  hardware,
        KA&                                              h_args,
        void const*                                      userArgs) const
    {
        KernelInvocation rv;
        rv.isSingleCall = true;

        if constexpr(!std::is_same<KA, KernelArgumentsCounter>::value)
        {
            rv.kernelName = kernelName;

            rv.args = KernelArguments(T_Debug);

            rv.workGroupSize.x = sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y
                                 * sizeMapping.workGroupSize.z;
            rv.workGroupSize.y = 1;
            rv.workGroupSize.z = 1;

            rv.numWorkItems.x = 0;
            rv.numWorkItems.y = 1;
            rv.numWorkItems.z = 1;

            rv.sharedMemBytes = 0;
        }
        auto autoGsuVal = calculateAutoGSU(problems[0], &hardware);
        calculateSingleCallWorkGroupItems(
            problems, rv.workGroupSize, rv.numWorkGroups, rv.numWorkItems, h_args, autoGsuVal);

        uint32_t workspaceOffsetInByte
            = this->requiredHostWorkspaceSizePerProblem * problems.size();
        if constexpr(!std::is_same<KA, int>::value)
        {
            for(int idx = 0; idx < problems.size(); idx++)
            {
                auto            problem = problems[idx];
                StreamKSettings sk;
                // Grouped gemm currently not supported in SK
                // But this code path is run to calculate to determine if solution is supported
                // Set SK grid to 1 for now to avoid 0 division
                sk.grid = 1;
                singleCallArgs<T_Debug, false>(problem,
                                               inputs.grouped[idx],
                                               workspaceOffsetInByte,
                                               &hardware,
                                               rv.numWorkGroups,
                                               rv.numWorkGroups,
                                               h_args,
                                               sk);

                if(sizeMapping.globalAccumulation == 3 || sizeMapping.adaptiveGemmGSUA == 1) // MBSK or MB with AdaptiveGemmGSUA
                {
                    h_args.template append<void const*>("dstD", inputs.grouped[idx].d);
                    // MBSK: synchronizer address, MB: null address
                    h_args.template append<void const*>("Synchronizer",
                                                        sizeMapping.globalAccumulation == 3
                                                        ? inputs.grouped[idx].Synchronizer
                                                        : NULL);
                    h_args.template append<uint32_t>("GSUSync", 0);
                }

                if constexpr(std::is_same<KA, KernelArguments>::value)
                    workspaceOffsetInByte += requiredWorkspaceSize(problem, hardware);
            }
        }

        if constexpr(!std::is_same<KA, KernelArgumentsCounter>::value)
        {
            auto [autoWGM, autoWGMXCC, autoWGMXCCCHUNK, autoWGMXCCSPLITK]
                = calculateAutoWGM(problems[0], &hardware, 0);
            auto [autoStaggerUMapping, autoStaggerU, autoStaggerUStrideShift]
                = calculateAutoStaggerU(problems[0], &hardware, 0, autoWGM);
            AdaptiveGemmNTAB ntab = calculateAdaptiveGemmNTAB(problems[0], &hardware);

            if(internalArgsSupport.useUniversalArgs)
            {
                KERNELARGTYPE argType = KERNELARGTYPE::HBM;
                if(userArgs != nullptr)
                {
                    argType = KERNELARGTYPE::USERARGS;
                }
                kernelArgs<T_Debug, false>(problems.size(),
                                           (uint32_t)argType,
                                           rv.args,
                                           getNumWorkGroups(rv),
                                           &hardware,
                                           problems[0].getParams(),
                                           autoWGM,
                                           autoWGMXCC,
                                           autoWGMXCCCHUNK,
                                           autoWGMXCCSPLITK,
                                           autoStaggerUMapping,
                                           autoStaggerU,
                                           autoStaggerUStrideShift,
                                           autoGsuVal,
                                           ntab);
                // For user input
                if(argType == KERNELARGTYPE::USERARGS)
                {
                    rv.args.append<void const*>("DeviceUserArguments", userArgs);
                }
                else
                {
                    rv.args.append<void const*>("argsPtr", (void*)inputs.ws);
                }
            }
            else
            {
                rv.args.append<uint32_t>("gemm_count", problems.size());
                // For user input
                rv.args.append<void const*>("DeviceUserArguments", userArgs);
                rv.args.append<void const*>("argsPtr", (void*)inputs.ws);
                rv.args.append<uint32_t>("numWorkGroups",
                                         rv.numWorkItems.x / rv.workGroupSize.x / rv.workGroupSize.y
                                             / rv.workGroupSize.z);
                kernelArgs<T_Debug, true>(0,
                                          (uint32_t)KERNELARGTYPE::NORMAL,
                                          rv.args,
                                          0,
                                          &hardware,
                                          problems[0].getParams(),
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

            rv.args.append<void const*>("Synchronizer", (void*)inputs.grouped[0].Synchronizer);
            rv.args.append<void const*>(
                "Workspace",
                (uint8_t*)inputs.ws + this->requiredHostWorkspaceSizePerProblem * problems.size());
            rv.codeObjectFile = codeObjectFilename.load();
        }

        return rv;
    }

    template <bool T_Debug>
    KernelInvocation
        ContractionSolution::generateBetaOnlyCall(Problem const&           problem,
                                                  ContractionInputs const& inputs) const
    {
        TensorDescriptor const& c               = problem.c();
        TensorDescriptor const& d               = problem.d();
        bool                    enableFactorDim = false;

        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        rv.kernelName = betaOnlyKernelName(problem);

        rv.workGroupSize.x = 256;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        size_t wiX = 1;
        size_t wiY = 1;
        size_t wiZ = 1;
        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            wiX *= problem.freeSizeA(i);
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            wiY *= problem.freeSizeB(i);
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
            wiZ *= problem.batchSize(i);

        rv.numWorkGroups.x = CeilDivide(wiX * wiY * wiZ, rv.workGroupSize.x);
        rv.numWorkGroups.y = 1;
        rv.numWorkGroups.z = 1;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(sizeMapping.globalAccumulation)
            rv.args.append<void*>("WS", inputs.ws);
        else if(problemType.stridedBatched)
            rv.args.append<void*>("D", inputs.d);
        else
            rv.args.append<void const* const*>("batchD", inputs.batchD);

        if(problemType.stridedBatched)
            rv.args.append<void const*>("C", inputs.c);
        else
            rv.args.append<void const* const*>("batchC", inputs.batchC);

        if(problemType.useBias && sizeMapping.globalAccumulation == 0 && (!problemType.useGradient))
        {
            if(problemType.stridedBatched)
                rv.args.append<void const*>("bias", inputs.bias);
            else
                rv.args.append<void const* const*>("batchBias", inputs.batchBias);
            if(problemType.useBias == 3)
                enableFactorDim = true;
        }
        if((!problemType.useScaleAB.empty()) && sizeMapping.globalAccumulation == 0)
        {
            rv.args.append<void const*>("scaleA", inputs.scaleA);
            rv.args.append<void const*>("scaleB", inputs.scaleB);
        }
        if(problemType.useScaleCD && sizeMapping.globalAccumulation == 0)
        {
            rv.args.append<void const*>("scaleC", inputs.scaleC);
            rv.args.append<void const*>("scaleD", inputs.scaleD);
        }
        if(problemType.useScaleAlphaVec && sizeMapping.globalAccumulation == 0)
        {
            rv.args.append<void const*>("scaleAlphaVec", inputs.scaleAlphaVec);
            if(problemType.useScaleAlphaVec == 3)
                enableFactorDim = true;
        }

        if(sizeMapping.globalAccumulation)
        {
            size_t stride = d.sizes()[0];
            for(size_t i = 1; i < d.dimensions(); i++)
            {
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideW", i),
                                         d.sizes()[i] == 1 ? 0 : stride);
                stride *= d.sizes()[i];
            }
        }
        else
        {
            for(size_t i = 1; i < d.dimensions(); i++)
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideD", i),
                                         d.sizes()[i] == 1 ? 0 : d.strides()[i]);
        }

        for(size_t i = 1; i < c.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideC", i),
                                     c.sizes()[i] == 1 ? 0 : c.strides()[i]);

        if(problemType.useBias && sizeMapping.globalAccumulation == 0 && (!problemType.useGradient))
        {
            TensorDescriptor const& bias = problem.tensor(ContractionProblemGemm::TENSOR::BIAS);
            rv.args.append<uint32_t>(
                "strideBias",
                problem.useBias() && bias.dimensions() ? bias.strides()[bias.dimensions() - 1] : 0);
        }

        if(enableFactorDim)
            rv.args.template append<uint32_t>("factorDim",
                                              (uint32_t)problem.getParams().factorDim());

        int idx = 0;
        for(auto size : problem.d().sizes())
        {
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("size_", idx), size);
            idx++;
        }

        rv.args.append("beta", inputs.beta, problem.betaType());

        if(problemType.useGateResidual)
        {
            if(problemType.stridedBatched)
                rv.args.template append<void const*>("gateResidual", inputs.gateResidual);
            else
                rv.args.template append<void const* const*>("batchGateResidual",
                                                         inputs.batchGateResidual);
            bool hasGate = problem.useGateResidual();
            rv.args.template append<uint32_t>(
                "gate_type",
                static_cast<uint32_t>(
                    hasGate ? problem.tensor(ContractionProblemGemm::TENSOR::GATE_RESIDUAL).dataType()
                            : problemType.gateResidualDataTypeWhiteList.at(0)));

            TensorDescriptor const& gate
                = problem.tensor(ContractionProblemGemm::TENSOR::GATE_RESIDUAL);
            for(size_t i = 1; i < d.dimensions(); i++)
                rv.args.template append<uint32_t>(concatenate_if<T_Debug>("strideGate", i),
                                               hasGate ? gate.strides()[i] : 0);
        }
        //Pass along code object dependency
        rv.codeObjectFile = codeObjectFilename.load();

        return rv;
    }

    template <bool T_Debug>
    KernelInvocation ContractionSolution::generateBetaOnlyCallGroupedGemm(
        std::vector<ContractionSolution::Problem> const& problems,
        ContractionSolution::GroupedInputs const&        inputs) const
    {
        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        rv.kernelName = betaOnlyKernelName(problems[0]);

        rv.workGroupSize.x = 256;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        rv.codeObjectFile = codeObjectFilename.load();

        return rv;
    }

    std::string ContractionSolution::betaOnlyKernelName(Problem const& problem) const
    {
        std::string name = concatenate(
            "C", problem.cNames(), "_", DataTypeInfo::Get(problem.d().dataType()).abbrev);

        if(problemType.groupedGemm)
        {
            name += "_GG";
        }
        else if(!problemType.stridedBatched)
        {
            name += "_GB";
        }

        int factorDim = 0;
        if(sizeMapping.globalAccumulation == 0)
        {
            if(!problemType.useGradient)
                factorDim = problemType.useScaleAlphaVec | problemType.useBias;
            else
                factorDim = problemType.useScaleAlphaVec;
        }
        if(problemType.useBias && sizeMapping.globalAccumulation == 0 && (!problemType.useGradient))
        {
            auto s = rocisa::TypeAbbrev(problem.bias().dataType());
            name += ("_Bias" + s);
        }
        if(factorDim == 2)
            name += "_FDN";
        else if(factorDim == 3)
            name += "_FDMN";

        if(sizeMapping.globalAccumulation)
        {
            name += "_GA";
        }
        if(problemType.useGateResidual)
        {
            name += ("_GateR");
        }
        return name;
    }

    template <bool T_Debug, typename KA>
    void ContractionSolution::outputConversionCallArgs(ContractionSolution::Problem const& problem,
                                                       ContractionInputs const&            inputs,
                                                       uint32_t const&        workspaceOffsetInByte,
                                                       KA&                    args,
                                                       StreamKSettings const& sk,
                                                       uint32_t               autoGsuVal,
                                                       uint32_t               additionalPaddingPerBatchGeneralBatch) const
    {
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();
        TensorDescriptor const& e = problem.tensor(ContractionProblemGemm::TENSOR::E);

        if(problemType.useE)
        {
            if(problemType.stridedBatched)
                args.template append<void*>("E", inputs.e);
            else
                args.template append<void const* const*>("batchE", 0);
        }

        if(problemType.stridedBatched)
            args.template append<void*>("D", inputs.d);
        else
            args.template append<void const* const*>("batchD", inputs.batchD);

        args.template append<void*>("WS", (uint8_t*)inputs.ws + workspaceOffsetInByte);

        if(problemType.stridedBatched)
            args.template append<void const*>("C", inputs.c);
        else
            args.template append<void const* const*>("batchC", inputs.batchC);

        bool useBias = false;
        if(problemType.useBias)
        {
            if(!problemType.useGradient)
            {
                if(problemType.stridedBatched)
                    args.template append<void const*>("bias", inputs.bias);
                else
                    args.template append<void const* const*>("batchBias", inputs.batchBias);
                useBias = true;
            }
            else
            {
                for(auto it : problemType.biasSrcWhiteList)
                {
                    if(it == ContractionProblemGemm::TENSOR::A
                       || it == ContractionProblemGemm::TENSOR::B)
                    {
                        if(problemType.stridedBatched)
                            args.template append<void*>("bias", const_cast<void*>(inputs.bias));
                        else
                            args.template append<void**>("batchBias",
                                                         const_cast<void**>(inputs.batchBias));
                        useBias = true;
                        break;
                    }
                }
            }
        }

        if(!problemType.useScaleAB.empty()) // GSU dep
        {
            args.template append<void const*>("scaleA", inputs.scaleA);
            args.template append<void const*>("scaleB", inputs.scaleB);
        }
        if(problemType.useScaleCD) // GSU dep
        {
            args.template append<void const*>("scaleC", inputs.scaleC);
            args.template append<void const*>("scaleD", inputs.scaleD);
        }
        if(problemType.useScaleAlphaVec) // GSU dep
        {
            args.template append<void const*>("scaleAlphaVec", inputs.scaleAlphaVec);
        }

        if(problemType.useGateResidual)
            args.template append<void const*>("gateResidual", inputs.gateResidual);

        if(sizeMapping.globalAccumulation == 2 || sizeMapping.streamK > 0)
            args.append("alpha", inputs.alpha, problem.alphaType());
        else
            args.append("alpha", 1.0f, problem.betaType());

        if((sizeMapping.globalAccumulation == 2 || sizeMapping.streamK > 0) and problemType.useBeta)
            args.append("beta", inputs.beta, problem.betaType());
        else
            args.append("beta", 0.0f, problem.betaType());

        if((problemType.activationType != ActivationType::None) && sizeMapping.activationFused)
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

        if(problemType.useE)
            for(size_t i = 1; i < e.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideE", i),
                                               e.strides()[i]);

        for(size_t i = 1; i < d.dimensions(); i++)
            args.template append<uint32_t>(concatenate_if<T_Debug>("strideD", i), d.strides()[i]);

        uint32_t wsStride = d.sizes()[0];
        for(size_t i = 1; i < d.dimensions(); i++)
        {
            args.template append<uint32_t>(concatenate_if<T_Debug>("strideW", i), wsStride);
            wsStride *= d.sizes()[i];
        }

        for(size_t i = 1; i < c.dimensions(); i++)
            args.template append<uint32_t>(concatenate_if<T_Debug>("strideC", i), c.strides()[i]);

        if(problemType.useGateResidual)
        {
            TensorDescriptor const& gate
                = problem.tensor(ContractionProblemGemm::TENSOR::GATE_RESIDUAL);
            bool hasGate = problem.useGateResidual();
            for(size_t i = 1; i < c.dimensions(); i++)
                args.template append<uint32_t>(concatenate_if<T_Debug>("strideGate", i),
                                               hasGate ? gate.strides()[i] : 0);
        }

        if(useBias)
        {
            TensorDescriptor const& bias = problem.tensor(ContractionProblemGemm::TENSOR::BIAS);
            args.template append<uint32_t>(
                "strideBias",
                problem.useBias() && bias.dimensions() ? bias.strides()[bias.dimensions() - 1] : 0);
        }

        int i = 0;
        for(auto size : problem.d().sizes())
        {
            args.template append<uint32_t>(concatenate_if<T_Debug>("size_", i), size);
            i++;
        }
        uint32_t gsu
            = sizeMapping.globalAccumulation == 1
                  ? 1
                  : (problem.getParams().gsu() > 0 ? problem.getParams().gsu() : autoGsuVal);

        if(sizeMapping.streamK > 0)
        {
            auto tiles = problem.getNumTiles(sizeMapping, 1);
            // Avoid 0 division when tiles is 0 (e.g. zero-sized dimension in grouped gemm)
            if(tiles > 0)
                gsu = sk.grid / tiles;
        }

        args.template append<uint32_t>(concatenate_if<T_Debug>("gsu"), gsu);
        // Added the extra check for useScaleAlphaVec with value 3 to match the condition in KernelWriterConversion.py for expecting
        // argument in the kernel side.
        if((useBias && problemType.useBias == 3) || problemType.useScaleAlphaVec == 3)
        {
            args.template append<uint32_t>("factorDim", (uint32_t)problem.getParams().factorDim());
        }
        // Adding the batchmode kernel argument for post GSU kernel to determine 
        // how to index the batch dimension in Strided Batch versus General Batched.
        if(problemType.groupedGemm == false && sizeMapping.customKernelName.empty())
        {
            ContractionProblemGemm::BATCHMODE batchMode = problem.batchMode();
            args.template append<uint32_t>("batchMode", static_cast<uint32_t>(batchMode));
            args.template append<uint32_t>("additionalPaddingPerBatch", additionalPaddingPerBatchGeneralBatch);

            // The HIP-compiled conversion kernel lays out these int64_t params on
            // 8-byte-aligned kernarg slots. Match that alignment on the host so the
            // bytes line up; a bare append() leaves them 4-byte-shifted when the
            // preceding args don't end on an 8-byte boundary (e.g. the non-HAS
            // variant), causing the kernel to read the neighboring offset into the
            // high dword of the address and fault.
            args.template appendAligned<int64_t>("batchOffsetD", inputs.batchOffsetD);
            args.template appendAligned<int64_t>("batchOffsetC", inputs.batchOffsetC);
        }

    }

    template <bool T_Debug>
    KernelInvocation
        ContractionSolution::generateOutputConversionCall(Problem const&           problem,
                                                          ContractionInputs const& inputs,
                                                          StreamKSettings const&   sk,
                                                          uint32_t                 autoGsuVal) const
    {
        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        rv.workGroupSize.x = 256;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        size_t wiX = 1;
        size_t wiY = 1;
        size_t wiZ = 1;
        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            wiX *= problem.freeSizeA(i);
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            wiY *= problem.freeSizeB(i);
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
            wiZ *= problem.batchSize(i);

        size_t vw = 1;
        if(wiX * wiY * wiZ > 2048)
        {
            //reach threashhold to trigger wider load
            if(problem.freeSizeA(0) % 4 == 0
               && DataTypeInfo::Get(problemType.aType).elementSize
                      < DataTypeInfo::Get(rocisa::DataType::Double).elementSize)
                vw = 4;
            else if(problem.freeSizeA(0) % 2 == 0
                    && DataTypeInfo::Get(problemType.aType).elementSize
                           < DataTypeInfo::Get(rocisa::DataType::ComplexDouble).elementSize)
                vw = 2;
        }

        uint32_t gsu
            = sizeMapping.globalAccumulation == 1
                  ? 1
                  : (problem.getParams().gsu() > 0 ? problem.getParams().gsu() : autoGsuVal);

        if(sizeMapping.streamK > 0)
        {
            // If using post kernel with stream-k then it is doing parallel reduciton
            // Calculate the splitting factor
            auto tiles = problem.getNumTiles(sizeMapping, 1);
            gsu        = sk.grid / tiles;
        }
        rv.kernelName = outputConversionKernelName(problem, inputs, vw, gsu);
        int additionalPaddingPerBatchGeneralBatch = 0;
        if(problem.batchMode() == ContractionProblemGemm::BATCHMODE::STRIDED)
            rv.numWorkGroups.x = CeilDivide(wiX * wiY * wiZ, rv.workGroupSize.x * vw);
        else
        {
            rv.numWorkGroups.x = CeilDivide(wiX * wiY, rv.workGroupSize.x * vw) * wiZ;
            int extra_work_items = (wiX * wiY) % (rv.workGroupSize.x * vw);
            additionalPaddingPerBatchGeneralBatch = extra_work_items > 0 ? (rv.workGroupSize.x * vw) - extra_work_items : 0;
        }
        rv.numWorkGroups.y = 1;
        rv.numWorkGroups.z = 1;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        outputConversionCallArgs<T_Debug>(problem, inputs, 0, rv.args, sk, autoGsuVal, additionalPaddingPerBatchGeneralBatch);

        //@TODO determine if this is needed, may not end up in the same code object file
        rv.codeObjectFile = codeObjectFilename.load();

        if(problemType.stochasticRounding)
        {
            // generate seed from random generator
            std::random_device                      rd;
            std::mt19937                            gen(rd());
            std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
            uint32_t                                seed = distribution(gen);
            rv.args.append<uint32_t>("RNDSeed", seed);
        }
        return rv;
    }

    template <typename KA>
    void ContractionSolution::calculateConversionCallWorkGroupItems(
        std::vector<ContractionSolution::Problem> const& problems,
        size_t&                                          vw,
        const TensileLite::dim3&                         workGroupSize,
        TensileLite::dim3&                               numWorkGroups,
        TensileLite::dim3&                               numWorkItems,
        KA&                                              h_args) const
    {
        if constexpr(std::is_same<KA, KernelArguments>::value)
        {
            size_t wi_count = 0;
            for(int idx = 0; idx < problems.size(); idx++)
            {
                auto problem = problems[idx];

                size_t wiX = 1;
                size_t wiY = 1;
                size_t wiZ = 1;
                for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
                    wiX *= problem.freeSizeA(i);
                for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
                    wiY *= problem.freeSizeB(i);
                for(size_t i = 0; i < problem.batchIndices().size(); i++)
                    wiZ *= problem.batchSize(i);

                wi_count += (wiX * wiY * wiZ);
            }

            //reach threashhold to trigger wider load
            if(wi_count > 2048)
            {
                bool not4 = false;
                bool not2 = false;
                for(int idx = 0; idx < problems.size(); idx++)
                {
                    auto problem = problems[idx];
                    if(problem.freeSizeA(0) % 4 != 0
                       && DataTypeInfo::Get(problemType.aType).elementSize
                              < DataTypeInfo::Get(rocisa::DataType::Double).elementSize)
                        not4 = true;
                    if(problem.freeSizeA(0) % 2 != 0)
                        not2 = true;
                }

                if(!not4)
                    vw = 4;
                else if(!not2)
                    vw = 2;
            }
        }

        int32_t  wiLeft  = 0;
        uint32_t wiRight = 0;
        for(int idx = 0; idx < problems.size(); idx++)
        {
            if constexpr(!std::is_same<KA, KernelArgumentsCounter>::value)
            {
                auto problem = problems[idx];

                size_t wiX = 1;
                size_t wiY = 1;
                size_t wiZ = 1;
                for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
                    wiX *= problem.freeSizeA(i);
                for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
                    wiY *= problem.freeSizeB(i);
                for(size_t i = 0; i < problem.batchIndices().size(); i++)
                    wiZ *= problem.batchSize(i);

                numWorkGroups.x = CeilDivide(wiX * wiY * wiZ, workGroupSize.x * vw);

                numWorkItems.x += workGroupSize.x * numWorkGroups.x;

                if constexpr(std::is_same<KA, KernelArguments>::value)
                {
                    wiRight = numWorkItems.x;
                    h_args.template append<uint32_t>("wiTable", wiLeft);
                    wiLeft = wiRight;
                }
            }
            else
            {
                h_args.template append<uint32_t>("wiTable", wiLeft);
            }
        }

        if constexpr(std::is_same<KA, KernelArguments>::value)
        {
            numWorkGroups.y = 1;
            numWorkGroups.z = 1;
            numWorkItems.y  = workGroupSize.y * numWorkGroups.y;
            numWorkItems.z  = workGroupSize.z * numWorkGroups.z;
        }
    }

    template <bool T_Debug, typename KA>
    KernelInvocation ContractionSolution::generateOutputConversionCallGroupedGemm(
        std::vector<ContractionSolution::Problem> const& problems,
        ContractionSolution::GroupedInputs const&        inputs,
        Hardware const&                                  hardware,
        KA&                                              h_args) const
    {
        KernelInvocation rv;
        uint32_t         previousArgsSpaceOffsetInByte = 0;

        size_t vw = 1;
        if constexpr(std::is_same<KA, KernelArguments>::value)
        {
            previousArgsSpaceOffsetInByte = h_args.size();

            rv.args = KernelArguments(T_Debug);

            rv.args.reserve(512, 64);

            rv.workGroupSize.x = 256;
            rv.workGroupSize.y = 1;
            rv.workGroupSize.z = 1;

            rv.numWorkItems.x = 0;
        }

        calculateConversionCallWorkGroupItems(
            problems, vw, rv.workGroupSize, rv.numWorkGroups, rv.numWorkItems, h_args);

        uint32_t autoGsuVal = calculateAutoGSU(problems[0], &hardware);
        uint32_t gsu        = sizeMapping.globalAccumulation == 1
                                  ? 1
                                  : (problems[0].getParams().gsu() > 0 ? problems[0].getParams().gsu()
                                                                       : autoGsuVal);

        if constexpr(std::is_same<KA, KernelArguments>::value)
        {
            rv.kernelName = outputConversionKernelName(problems[0], inputs.grouped[0], vw, gsu);
        }

        uint32_t workspaceOffsetInByte
            = this->requiredHostWorkspaceSizePerProblem * problems.size();
        for(int idx = 0; idx < problems.size(); idx++)
        {
            auto            problem = problems[idx];
            StreamKSettings sk;
            outputConversionCallArgs<T_Debug>(
                problem, inputs.grouped[idx], workspaceOffsetInByte, h_args, sk, autoGsuVal);
            if constexpr(std::is_same<KA, KernelArguments>::value)
                workspaceOffsetInByte += requiredWorkspaceSize(problem, hardware);
        }

        if constexpr(std::is_same<KA, KernelArguments>::value)
        {
            uint8_t* d_args = (uint8_t*)(inputs.ws) + previousArgsSpaceOffsetInByte;
            rv.args.append<uint8_t*>("wiTablePtr", d_args);
            // For user input
            rv.args.append<void const*>("DeviceUserArguments", nullptr);
            rv.args.append<uint8_t*>("argsPtr", d_args + problems.size() * sizeof(uint32_t));
            rv.args.append<uint32_t>("gemm_count", problems.size());
            rv.codeObjectFile = codeObjectFilename.load();
        }

        return rv;
    }

    template <bool T_Debug>
    KernelInvocation ContractionSolution::updateUserArgsOutputConversionCallGroupedGemm(
        std::vector<ContractionSolution::Problem> const& problems,
        const void*                                      userArgs,
        const void*                                      workspace) const
    {
        KernelInvocation rv;
        uint32_t         previousArgsSpaceOffsetInByte = 0;
        // FIXME: Need to find a way to offset the arg spaces

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        size_t vw = 1;

        rv.workGroupSize.x = 256;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        rv.numWorkItems.x = 0;

        int h_args = 0; // Dummy value
        calculateConversionCallWorkGroupItems(
            problems, vw, rv.workGroupSize, rv.numWorkGroups, rv.numWorkItems, h_args);

        // FIXME: No problem and input for kernel name
        // rv.kernelName = outputConversionKernelName(
        //     problems[0], inputs.grouped[0], vw, sizeMapping.globalSplitU);

        uint8_t* d_args = (uint8_t*)workspace + previousArgsSpaceOffsetInByte;
        rv.args.append<uint8_t*>("wiTablePtr", d_args);
        // For user input
        rv.args.append<void const*>("DeviceUserArguments", nullptr);
        rv.args.append<uint8_t*>("argsPtr", d_args + problems.size() * sizeof(uint32_t));
        rv.args.append<uint32_t>("gemm_count", problems.size());
        rv.codeObjectFile = codeObjectFilename.load();

        return rv;
    }

    std::string ContractionSolution::outputConversionKernelName(Problem const&           problem,
                                                                ContractionInputs const& inputs,
                                                                size_t                   vw,
                                                                size_t                   gsu) const
    {
        auto inputTypeStr = (problem.a().dataType() == rocisa::DataType::Int8
                             || problem.a().dataType() == rocisa::DataType::Int32)
                                ? DataTypeInfo::Get(rocisa::DataType::Int32).abbrev
                            : problem.a().dataType() == rocisa::DataType::Double
                                ? DataTypeInfo::Get(rocisa::DataType::Double).abbrev
                                : DataTypeInfo::Get(rocisa::DataType::Float).abbrev;

        std::string name = concatenate("C",
                                       problem.cNames(),
                                       "_",
                                       inputTypeStr,
                                       DataTypeInfo::Get(problem.d().dataType()).abbrev);

        if(problemType.groupedGemm)
        {
            name += "_GG";
        }
        else if(!problemType.stridedBatched)
        {
            name += "_GB";
        }

        if(problemType.useBias)
        {
            auto s = rocisa::TypeAbbrev(problem.bias().dataType());
            if(problemType.useGradient)
            {
                if(problem.biasSrc() == ContractionProblemGemm::TENSOR::D)
                    s = rocisa::TypeAbbrev(problem.computeType());
                if(inputs.bias != nullptr)
                {
                    const char* alpha[5] = {"A", "B", "C", "D", "E"};
                    std::string ss;
                    for(auto it : problemType.biasSrcWhiteList)
                    {
                        if(it < 5)
                        {
                            ss += alpha[it];
                        }
                    }
                    name += ("_DBias" + s + "_BiasSrc" + ss);
                }
            }
            else
            {
                name += ("_Bias" + s);
            }
        }

        int factorDim
            = std::max(problemType.useGradient ? 0 : problemType.useBias, problemType.useScaleAlphaVec);
        if(factorDim)
        {
            if(factorDim == 2)
                name += ("_FDN");
            else if(factorDim == 3)
                name += ("_FDMN");
        }

        if(problemType.useE)
        {
            auto s = rocisa::TypeAbbrev(
                problem.tensors()[ContractionProblemGemm::TENSOR::E].dataType());
            if(problemType.useGradient)
            {
                name += ("_Grad" + s);
            }
            else
            {
                name += ("_Aux" + s);
            }
        }

        if(problemType.useGateResidual)
        {
            auto gateDtype = problem.useGateResidual()
                                 ? problem.tensor(ContractionProblemGemm::TENSOR::GATE_RESIDUAL).dataType()
                                 : problemType.gateResidualDataTypeWhiteList.at(0);
            name += ("_Gate" + rocisa::TypeAbbrev(gateDtype));
        }

        if(problemType.activationType != ActivationType::None)
        {
            if(problemType.activationType == ActivationType::All)
            {
                name += "_A";
            }
            else if(problemType.activationType == ActivationType::Hipblaslt_all)
            {
                name += "_HA";
            }
            else
            {
                std::string actName = ToString(problemType.activationType);
                std::transform(actName.begin(), actName.end(), actName.begin(), ::toupper);
                name += actName;
            }

            name += rocisa::TypeAbbrev(problemType.activationComputeDataType);

            if(problemType.activationNoGuard)
            {
                name += "ng";
            }
        }

        if(problemType.useScaleAB == "Scalar")
        {
            name += ("_ScaleAB");
        }
        else if(problemType.useScaleAB == "Vector")
        {
            name += ("_ScaleABVec");
        }
        if(problemType.useScaleCD)
        {
            name += ("_ScaleCD");
        }

        if(problemType.useScaleAlphaVec)
        {
            name += ("_ScaleAlphaVec");
        }

        uint32_t gsuTemp = gsu - 1;
        gsuTemp |= gsuTemp >> 1;
        gsuTemp |= gsuTemp >> 2;
        gsuTemp |= gsuTemp >> 4;
        gsuTemp |= gsuTemp >> 8;
        gsuTemp |= gsuTemp >> 16;
        gsuTemp++;

        name += "_PostGSU"
                + std::to_string(
                    std::min(static_cast<decltype(sizeMapping.globalSplitUPGR)>(gsuTemp),
                             sizeMapping.globalSplitUPGR));

        name += "_VW" + std::to_string(vw);
        if(problemType.useGateResidual)
        {
            name += "_GateR";
        }
        return name;
    }

    template <bool T_Debug>
    KernelInvocation
        ContractionSolution::generateReductionCall(Problem const&           problem,
                                                   ContractionInputs const& inputs) const
    {
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();
        TensorDescriptor const& e = problem.tensor(ContractionProblemGemm::TENSOR::E);

        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        size_t threads = 256;
        size_t mt0     = 256;
        size_t mt1     = 1;
        size_t vw      = 1;
        // TODO: Currently only support bias reduction
        if(problem.d().sizes()[1] >= 8192)
        {
            threads = 1024;
            mt1     = 32;
            vw      = 4;
        }
        else if(problem.d().sizes()[1] >= 32)
        {
            mt1 = 32;
        }
        else
        {
            // MT1 should be the power of 2 to match setting in tensileLite
            mt1 = static_cast<int>(pow(2, ceil(log2(mt1))));
            if(mt1 == 0)
                mt1 = 1;
        }
        mt0 = threads / mt1;

        rv.kernelName = outputReductionKernelName(problem, inputs, mt0, mt1, vw);

        rv.workGroupSize.x = threads;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        // TODO: Currently only support bias reduction
        rv.numWorkGroups.x = CeilDivide(problem.d().sizes()[0], (mt0 * vw));
        rv.numWorkGroups.y = 1;
        rv.numWorkGroups.z = 1;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        // FIXME: Need to check the formula for batch > 1
        rv.args.append<void*>("WS", inputs.ws);
        rv.args.append<void const*>("bias", inputs.bias);
        for(size_t i = 0; i < 2; i++)
        {
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("size_", i), problem.d().sizes()[i]);
        }
        rv.args.append<uint32_t>("strideDJ", d.sizes()[0]);

        //@TODO determine if this is needed, may not end up in the same code object file
        rv.codeObjectFile = codeObjectFilename.load();

        return rv;
    }

    std::string ContractionSolution::outputReductionKernelName(Problem const&           problem,
                                                               ContractionInputs const& inputs,
                                                               size_t                   mt0,
                                                               size_t                   mt1,
                                                               size_t                   vw) const
    {
        auto&       biasTensor = problem.tensor(ContractionProblemGemm::TENSOR::BIAS);
        std::string name       = concatenate("D",
                                       problem.dNames(),
                                       "_",
                                       DataTypeInfo::Get(biasTensor.dataType()).abbrev,
                                       DataTypeInfo::Get(problem.betaType()).abbrev);
        name += concatenate("_MT", mt0, "x", mt1);
        name += concatenate("_VW", vw);
        name += "_Reduction";

        return name;
    }

    std::vector<KernelInvocation> ContractionSolution::solve(ContractionProblem const& problem,
                                                             ProblemInputs const&      inputs,
                                                             Hardware const&           hardware,
                                                             void*       hipHostMemory,
                                                             size_t      hipHostMemorySize,
                                                             hipStream_t stream) const
    {
        if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(&problem))
        {
            auto gemmInputs = dynamic_cast<ContractionInputs const*>(&inputs);
            return solve((*gemmProblem), (*gemmInputs), hardware);
        }
        else if(auto groupedProblem = dynamic_cast<ContractionProblemGroupedGemm const*>(&problem))
        {
            auto& gemms         = groupedProblem->gemms;
            auto  groupedInputs = dynamic_cast<ContractionGroupedInputs const*>(&inputs);
            return solveGroupedGemm(
                gemms, (*groupedInputs), hardware, hipHostMemory, hipHostMemorySize, stream);
        }
        else
        {
            throw std::runtime_error("Failed to cast problem type.");
        }
    }

    // For Tensile debugging, will allocate and initialize DeviceUserArguments with the problems and inputs.
    std::vector<KernelInvocation>
        ContractionSolution::solveTensileGPU(ContractionProblem const& problem,
                                             ProblemInputs const&      inputs,
                                             Hardware const&           hardware,
                                             void**                    dUA,
                                             void**                    dUAHost,
                                             void*                     hipHostMemory,
                                             size_t                    hipHostMemorySize,
                                             hipStream_t               stream) const
    {
        // Since we now use universal args, we block globalSplitU here if using UserArgs
        if((sizeMapping.globalSplitU > 1 || sizeMapping.globalSplitU == -1)
           && sizeMapping.globalAccumulation != 3)
        {
            KernelInvocation dummyrv;
            dummyrv.kernelName = "";

            dummyrv.args = KernelArguments(false);

            dummyrv.workGroupSize.x = 1;
            dummyrv.workGroupSize.y = 1;
            dummyrv.workGroupSize.z = 1;

            dummyrv.numWorkItems.x = 1;
            dummyrv.numWorkItems.y = 1;
            dummyrv.numWorkItems.z = 1;

            dummyrv.sharedMemBytes = 0;
            return {dummyrv};
        }
        if(auto groupedProblem = dynamic_cast<ContractionProblemGroupedGemm const*>(&problem))
        {
            auto& gemms         = groupedProblem->gemms;
            auto  groupedInputs = dynamic_cast<ContractionGroupedInputs const*>(&inputs);
            return solveTensileGroupedGemmGPU(gemms,
                                              (*groupedInputs),
                                              hardware,
                                              dUA,
                                              dUAHost,
                                              hipHostMemory,
                                              hipHostMemorySize,
                                              stream);
        }
        else
        {
            throw std::runtime_error("Failed to cast problem type.");
        }
    }

    namespace
    {
        size_t getSKGridImpl(ContractionSolution const& self,
                             ContractionProblemGemm const& problem,
                             Hardware const&               hardware,
                             size_t                        tiles,
                             origami::reduction_t          reductionStrat,
                             bool const*                   sk5EffectiveDynamic,
                             bool*                         outFixedGridUsed      = nullptr,
                             bool*                         outTreeBoundsFallback = nullptr,
                             bool*                         outClusterDPGridClamp = nullptr,
                             size_t*                       outSelectedGrid       = nullptr);

        // Reconcile the reduction strategy with the grid that was finally
        // chosen. The strategy is picked BEFORE the grid -- getSKReduction()
        // feeds getSKGridImpl() as an input -- so the grid can still land on a
        // splitting factor F = grid / tiles of 1, which parallel reduction
        // cannot express: it splits each output tile across F workgroups and
        // sums the partials in a second kernel, so F < 2 leaves it nothing to
        // reduce (solve() rejects it outright). Ways the grid gets there:
        //
        //   * the uniform-summation-order F-star snap in getSKGridImpl() finds
        //     no admissible F >= 2 (workspace too small for 2 * tiles partial
        //     tiles, or ItersPerTile / F below MinItersPerCU) and falls back to
        //     the all-full grid == tiles;
        //   * the same snap sees g0 < tiles and snaps up to tiles;
        //   * skFixedGrid / skMaxCUs / skGridMultiplier / the analytical grid
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
        // immediately after getSKGridImpl(), so query and launch cannot
        // disagree about which reduction is in play, and therefore cannot
        // disagree about the workspace it needs.
        inline origami::reduction_t streamKReconcileReduction(
            origami::reduction_t reductionStrat, size_t skGrid, size_t tiles)
        {
            if(reductionStrat == origami::reduction_t::parallel
               && (tiles == 0 || (skGrid / tiles) < 2))
                return origami::reduction_t::tree;
            return reductionStrat;
        }
    }

    std::vector<KernelInvocation>
        ContractionSolution::solve(ContractionSolution::Problem const& problem,
                                   ContractionSolution::Inputs const&  inputs,
                                   Hardware const&                     hardware) const
    {
        calculateAutoGSU(problem, &hardware);
        if(Debug::Instance().printWinningKernelName())
            std::cout << "Running kernel: " << this->KernelName()
                      << " [MatchingTag: " << this->matchingTag() << "]" << std::endl;

        // retreive alpha/beta type set via setAlpha/BetaType()
        auto alphaType = problem.alphaType();
        auto betaType  = problem.betaType();

        // TODO: Some gtests are passing the "problem" without actually defining the
        // alpha/beta type (alphaType and betaType remain None).
        // Until we fix those gtests, we need to keep this condition to adjust the missing
        // alpha/beta data types.
        if(alphaType == rocisa::DataType::None)
        {
            alphaType = problemType.aType == rocisa::DataType::BFloat16 ? rocisa::DataType::Float
                                                                        : problemType.dType;
        }
        if(betaType == rocisa::DataType::None)
        {
            betaType = alphaType;
        }

        bool debug = Debug::Instance().printKernelArguments() || this->kernelArgsLog;

        int boundSize = 1;
        for(size_t i = 0; i < problem.boundIndices().size(); i++)
            boundSize *= problem.boundSize(i);

        // Check for nullptrs if alpha is non-zero.
        if((!CompareValue(inputs.alpha, (double)0) && (boundSize != 0))
           && ((problem.stridedBatched() && (inputs.a == nullptr || inputs.b == nullptr))
               || (!problem.stridedBatched()
                   && (inputs.batchA == nullptr || inputs.batchB == nullptr))))
        {
            std::string matrixID = inputs.a == nullptr ? "A" : "B";
            std::string msg      = std::string("Unsupported nullptr for ") + matrixID
                              + std::string(" when (Alpha !=0) && (K != 0)\n");
            throw std::runtime_error(msg.c_str());
        }

        // Check if alpha matches problem definition
        if(problem.alphaRestriction() != ScalarValue::Any
           && problem.alphaRestriction() != toScalarValueEnum(inputs.alpha))
        {
            std::stringstream inputValue;
            inputValue << ToString(inputs.alpha);
            std::string msg = std::string("Alpha value ") + inputValue.str()
                              + std::string(" doesn't match that set in problem: ")
                              + ToString(problem.alphaRestriction());
            throw std::runtime_error(msg.c_str());
        }

        // Check if beta matches problem definition
        if(problem.betaRestriction() != ScalarValue::Any
           && problem.betaRestriction() != toScalarValueEnum(inputs.beta))
        {
            std::stringstream inputValue;
            inputValue << ToString(inputs.beta);
            std::string msg = std::string("Beta value ") + inputValue.str()
                              + std::string(" doesn't match that set in problem: ")
                              + ToString(problem.betaRestriction());
            throw std::runtime_error(msg.c_str());
        }

        if(problem.cEqualsD() && inputs.c != inputs.d)
            throw std::runtime_error(
                "ContractionProblemGemm has cEqualsD set, but pointers for c and d are not equal");

        std::vector<KernelInvocation> rv;

        auto autoGsuVal = calculateAutoGSU(problem, &hardware);
        auto gsu        = problem.getParams().gsu() > 0 ? problem.getParams().gsu() : autoGsuVal;
        if(gsu > 1 && sizeMapping.globalAccumulation != 2 && sizeMapping.globalAccumulation != 3)
        {
            if(debug)
                rv.push_back(generateBetaOnlyCall<true>(problem, inputs));
            else
                rv.push_back(generateBetaOnlyCall<false>(problem, inputs));
        }

        StreamKSettings  sk;
        StreamKDecisions skDecisions;
        if(sizeMapping.streamK > 0)
        {
            auto tiles = problem.getNumTiles(sizeMapping, 1);
            // Single source of truth: compute the StreamK launch decisions once
            // (also consumed by the diagnostic launch summary and unit tests) and
            // consume them here. computeStreamKDecisions() owns the reduction /
            // grid / workspace-DP-fallback logic and reuses the same helpers, so
            // the decisions ARE what is launched.
            //
            // The dynamic-queue guard below consumes skDecisions.isDynamic, so
            // this call has to precede it -- which moves getSKReduction(),
            // getSKGridImpl() and the workspace sizing ahead of that rejection.
            // That is safe rather than free: the call mutates no solution or
            // problem state and launches nothing, so the guard still throws before
            // any kernel-arg packing or kernel launch. It is not, however,
            // output-free -- the helpers it calls can write TENSILE_DB diagnostics
            // to stderr (streamK5EffectiveDynamic's SK5 mode-selection note under
            // 0x100000, getSKGridImpl's CU-occupancy note under
            // printPropertyEvaluation). Neither is reachable on a guard-rejected
            // launch as the code stands (the guard only fires on the dynamic-queue
            // path, where the reduction is forced to tree without calling
            // getSKReduction and getSKGridImpl takes its diagnostic-free
            // branches), but a future edit that prints from those helpers would
            // start emitting for solutions this guard goes on to reject.
            skDecisions = computeStreamKDecisions(problem, hardware);
            // Defensive: dynamic-queue / work-stealing StreamK solutions are
            // excluded from selection on devices whose runtime XCD count is not
            // a power of two or does not equal the baked per-XCD queue count
            // (see streamKDynamicQueueSupported() wired into softwarePredicate).
            // The normal path therefore never reaches solve() for such a
            // solution; a different (SK3-static / non-StreamK) solution serves
            // the GEMM instead. If we DO get here it means the software
            // predicate was bypassed (e.g. an explicit select-by-index), so
            // reject EXPLICITLY rather than silently running the fixed-mask
            // kernel with a mismatched queue count (which would corrupt
            // results).
            const bool dynamicQueuePath = skDecisions.isDynamic;
            if(dynamicQueuePath && streamKDynamicQueueUnsupported(hardware))
            {
                warnStreamKDynamicQueueUnsupportedOnce(hardware);
                // Fail EARLY -- before workspace sizing / kernel-arg packing --
                // when NUM_XCD is unknown (baked queue count == 0, e.g. missing
                // analyticalHardware). Sizing the per-XCD counter region with a
                // 0 queue count would under-allocate the workspace the kernel
                // writes; reject with an actionable message instead.
                if(streamKBakedQueueCount(hardware) == 0)
                    throw std::runtime_error(
                        "hipBLASLt Error: StreamK dynamic-queue (work-stealing) requires a known "
                        "NUM_XCD (analyticalHardware unavailable); refusing to size the per-XCD "
                        "counter workspace with an unknown queue count. "
                        "Select a non-work-stealing solution instead.");
                throw std::runtime_error(
                    "hipBLASLt Error: StreamK dynamic-queue (work-stealing) solution selected on a "
                    "device whose XCD count is not a power of two or does not equal the compiled "
                    "per-XCD queue count; this kernel is unsupported here. "
                    "Select a non-work-stealing solution instead.");
            }

            // Consume the reduction / grid decided above. resolveStreamKSettings()
            // is the single source of truth for what is launched: it runs the same
            // reduction selection, the same getSKGridImpl() call and the same
            // workspace-insufficient DP fallback that computeStreamKDecisions()
            // reports on, and additionally applies streamKReconcileReduction() so
            // requiredWorkspaceSize() and the launch cannot disagree.
            // computeStreamKDecisions() mirrors that reconciliation at the same
            // point, so the snapshot above and the settings below carry the same
            // (reduction, grid) pair; the re-wire after the guard re-asserts it.
            sk = resolveStreamKSettings(problem, hardware);

            // Defensive only: resolveStreamKSettings() runs
            // streamKReconcileReduction() on the final grid, which demotes
            // parallel to tree exactly when this condition would hold, so this
            // is unreachable. Kept so a future path that builds a
            // StreamKSettings without that reconciliation still fails loudly
            // rather than launching a fixup kernel with nothing to reduce.
            if(tiles != 0 && sk.reduction == origami::reduction_t::parallel && sk.grid / tiles < 2)
            {
                throw std::runtime_error("hipblasLT Error: Cannot use Parallel reduction with "
                                         "StreamK kernel with splitting factor < 2\n");
            }

            // Wire the FINAL launch grid and reduction from the values solve()
            // actually launches with (post all fallbacks, post reconciliation), so
            // the diagnostic summary can never drift from the real launch.
            skDecisions.finalGrid = sk.grid;
            skDecisions.skGrid    = sk.grid;
            skDecisions.reduction = sk.reduction;
            if(Debug::Instance().printStreamKLaunchSummary())
                printStreamKLaunchSummary(std::cerr, problem, skDecisions);
        }

        GSUSettings gsuSettings;
        gsuSettings.globalAccumulation = problem.getAccumulation(hardware, sizeMapping, gsu);

        // Evaluated immediately before dispatch: this is the first point at which
        // every value the kernel will see is final. The StreamK block above can
        // still rewrite sk.grid and sk.reduction, and globalAccumulation is only
        // resolved on the line above.
        checkUniformSummationOrder(
            problem, hardware, sk, gsuSettings.globalAccumulation, gsu, inputs.Synchronizer);

        if(debug)
            rv.push_back(generateSingleCall<true>(problem, inputs, hardware, sk, gsuSettings));
        else
            rv.push_back(generateSingleCall<false>(problem, inputs, hardware, sk, gsuSettings));

        if((gsu > 1 && gsuSettings.globalAccumulation && gsuSettings.globalAccumulation != 3)
           || sk.reduction == origami::reduction_t::parallel)
        {
            if(debug)
                rv.push_back(generateOutputConversionCall<true>(problem, inputs, sk, autoGsuVal));
            else
                rv.push_back(generateOutputConversionCall<false>(problem, inputs, sk, autoGsuVal));
        }

        // The reduction of A is done in ConversionKernel when GSU > 1 in MultipleBuffer mode
        if(problemType.useBias && problemType.useGradient
           && (problem.biasSrc() == ContractionProblemGemm::TENSOR::D))
        {
            if(problem.d().dimensions() != 3)
            {
                throw std::runtime_error("Currently only supports bias reduction (m x n x batch)");
            }
            // Skip if output is null
            if(inputs.bias != nullptr)
            {
                if(debug)
                    rv.push_back(generateReductionCall<true>(problem, inputs));
                else
                    rv.push_back(generateReductionCall<false>(problem, inputs));
            }
        }

        return rv;
    }

    std::vector<KernelInvocation> ContractionSolution::solveGroupedGemm(
        std::vector<ContractionSolution::Problem> const& problems,
        ContractionSolution::GroupedInputs const&        inputs,
        Hardware const&                                  hardware,
        void*                                            hipHostMemory,
        size_t                                           hipHostMemorySize,
        hipStream_t                                      stream) const
    {
        if(Debug::Instance().printWinningKernelName())
            std::cout << "Running kernel: " << this->KernelName()
                      << " [MatchingTag: " << this->matchingTag() << "]" << std::endl;

        // retreive alpha/beta type set via setAlpha/BetaType()
        auto alphaType = problems[0].alphaType();
        auto betaType  = problems[0].betaType();

        // TODO: Some gtests are passing the "problem" without actually defining the
        // alpha/beta type (alphaType and betaType remain None).
        // Until we fix those gtests, we need to keep this condition to adjust the missing
        // alpha/beta data types.
        if(alphaType == rocisa::DataType::None)
        {
            alphaType = problemType.aType == rocisa::DataType::BFloat16 ? rocisa::DataType::Float
                                                                        : problemType.dType;
        }
        if(betaType == rocisa::DataType::None)
        {
            betaType = alphaType;
        }

        bool debug = Debug::Instance().printKernelArguments() || this->kernelArgsLog;

        auto autoGsuVal = calculateAutoGSU(problems[0], &hardware);
        auto gsu = problems[0].getParams().gsu() > 0 ? problems[0].getParams().gsu() : autoGsuVal;

        // Check for nullptrs if alpha is non-zero.
        for(int idx = 0; idx < problems.size(); idx++)
        {
            int boundSize = 1;
            for(size_t i = 0; i < problems[idx].boundIndices().size(); i++)
                boundSize *= problems[idx].boundSize(i);

            const auto n = problems[idx].freeSizeB(0);

            if(n && ((!CompareValue(inputs.grouped[idx].alpha, (double)0)) && (boundSize != 0))
               && ((problems[idx].stridedBatched()
                    && (inputs.grouped[idx].a == nullptr || inputs.grouped[idx].b == nullptr))))
            {
                std::string matrixID = inputs.grouped[idx].a == nullptr ? "A" : "B";
                std::string msg      = std::string("Unsupported nullptr for ") + matrixID
                                  + std::string(" when (Alpha !=0) && (K != 0)\n");
                throw std::runtime_error(msg.c_str());
            }

            // Check if alpha matches problem definition
            if(problems[idx].alphaRestriction() != ScalarValue::Any
               && problems[idx].alphaRestriction() != toScalarValueEnum(inputs.grouped[idx].alpha))
            {
                std::stringstream inputValue;
                inputValue << ToString(inputs.grouped[idx].alpha);
                std::string msg = std::string("Alpha value ") + inputValue.str()
                                  + std::string(" doesn't match that set in problem: ")
                                  + ToString(problems[idx].alphaRestriction());
                throw std::runtime_error(msg.c_str());
            }

            // Check if beta matches problem definition
            if(problems[idx].betaRestriction() != ScalarValue::Any
               && problems[idx].betaRestriction() != toScalarValueEnum(inputs.grouped[idx].beta))
            {
                std::stringstream inputValue;
                inputValue << ToString(inputs.grouped[idx].beta);
                std::string msg = std::string("Beta value ") + inputValue.str()
                                  + std::string(" doesn't match that set in problem: ")
                                  + ToString(problems[idx].betaRestriction());
                throw std::runtime_error(msg.c_str());
            }

            if(problems[idx].cEqualsD() && inputs.grouped[idx].c != inputs.grouped[idx].d)
                throw std::runtime_error(
                    "ContractionProblem has cEqualsD set, but pointers for c and d are not equal");

            // The grouped path never resolves StreamK: generateSingleCallGroupedGemm()
            // packs skGrid == 0 and reads sizeMapping.globalAccumulation directly,
            // so those are the values the gate must see. Any StreamK solution
            // reaching here therefore fails the grid divisibility check.
            checkUniformSummationOrder(problems[idx],
                                       hardware,
                                       StreamKSettings{},
                                       sizeMapping.globalAccumulation,
                                       gsu,
                                       inputs.grouped[idx].Synchronizer);
        }

        std::vector<KernelInvocation> rv;
        auto                          h_args = KernelArguments(debug);
        if(hipHostMemory)
        {
            h_args.useExternalPointer(hipHostMemory, hipHostMemorySize);
        }
        h_args.reserve(32768, 8192);

        // if((sizeMapping.globalSplitU > 1 || sizeMapping.globalSplitU == -1) && sizeMapping.globalAccumulation != 2)
        // {
        //     if(debug)
        //         rv.push_back(generateBetaOnlyCallGroupedGemm<true>(problems, inputs));
        //     else
        //         rv.push_back(generateBetaOnlyCallGroupedGemm<false>(problems, inputs));
        // }

        if(debug)
            rv.push_back(generateSingleCallGroupedGemm<true>(problems, inputs, hardware, h_args));
        else
            rv.push_back(generateSingleCallGroupedGemm<false>(problems, inputs, hardware, h_args));

        if(sizeMapping.globalAccumulation == 2 && gsu > 1)
        {
            if(debug)
                rv.push_back(generateOutputConversionCallGroupedGemm<true>(
                    problems, inputs, hardware, h_args));
            else
                rv.push_back(generateOutputConversionCallGroupedGemm<false>(
                    problems, inputs, hardware, h_args));
        }

        if(debug)
        {
            std::cout << "Grouped gemm argsPtr kernels: " << std::endl;
            for(auto& kernel : rv)
            {
                std::cout << kernel.kernelName << std::endl;
            }
            std::cout << h_args;
        }

        if(hipHostMemory && hipHostMemorySize < h_args.size())
            throw std::runtime_error("Insufficient host memory size.");

        uint8_t*    d_args = (uint8_t*)inputs.ws;
        const void* tmpMem = hipHostMemory ? hipHostMemory : h_args.data();

        HIP_CHECK_EXC(hipMemcpyAsync(
            d_args, tmpMem, h_args.size() * sizeof(uint8_t), hipMemcpyHostToDevice, stream));

        return rv;
    }

    std::vector<KernelInvocation>
        ContractionSolution::solveGroupedGemmGPU(std::vector<Problem> const& problems,
                                                 GroupedInputs const&        inputs,
                                                 Hardware const&             hardware,
                                                 const void*                 dUA,
                                                 const void*                 workspace,
                                                 hipStream_t                 stream) const
    {
        if(!problemType.supportDeviceUserArguments)
        {
            throw std::runtime_error("Currently this solution does not support user args.");
        }

        auto gsu = problems[0].getParams().gsu() > 0 ? problems[0].getParams().gsu()
                                                     : calculateAutoGSU(problems[0], &hardware);

        // Shares generateSingleCallGroupedGemm(), so the same arguments as in
        // solveGroupedGemm() apply.
        for(size_t idx = 0; idx < problems.size(); idx++)
            checkUniformSummationOrder(problems[idx],
                                       hardware,
                                       StreamKSettings{},
                                       sizeMapping.globalAccumulation,
                                       gsu,
                                       inputs.grouped[idx].Synchronizer);

        std::vector<KernelInvocation> rv;

        bool debug = Debug::Instance().printKernelArguments() || this->kernelArgsLog;

        // Here we only update the pointer
        int h_args = 1; // Dummy
        if(debug)
            rv.push_back(
                generateSingleCallGroupedGemm<true>(problems, inputs, hardware, h_args, dUA));
        else
            rv.push_back(
                generateSingleCallGroupedGemm<false>(problems, inputs, hardware, h_args, dUA));

        if((sizeMapping.globalAccumulation && gsu > 1) && (sizeMapping.globalAccumulation != 3))
        {
            if(debug)
                rv.push_back(
                    updateUserArgsOutputConversionCallGroupedGemm<true>(problems, dUA, workspace));
            else
                rv.push_back(
                    updateUserArgsOutputConversionCallGroupedGemm<false>(problems, dUA, workspace));
        }

        return rv;
    }

    // For Tensile debugging, will allocate and initialize DeviceUserArguments with the problems and inputs.
    std::vector<KernelInvocation>
        ContractionSolution::solveTensileGroupedGemmGPU(std::vector<Problem> const& problems,
                                                        GroupedInputs const&        inputs,
                                                        Hardware const&             hardware,
                                                        void**                      dUA,
                                                        void**                      dUAHost,
                                                        void*                       hipHostMemory,
                                                        size_t      hipHostMemorySize,
                                                        hipStream_t stream) const
    {
        calculateAutoGSU(problems[0], &hardware);
        // Allocate and copy data to dUA
        if(problems[0].activationType() == ActivationType::None
           || (problems[0].activationType() != ActivationType::None
               && problems[0].activationComputeType() == rocisa::DataType::Float))
        {
            auto requiredSize = sizeof(DeviceUserArguments<float>) * problems.size();
            static_cast<void>(hipHostMalloc(dUAHost, requiredSize, 0));
            setDeviceUserArgs(problems, inputs, (DeviceUserArguments<float>*)(*dUAHost));
            static_cast<void>(hipMalloc(dUA, requiredSize));
            static_cast<void>(hipMemcpy(*dUA, *dUAHost, requiredSize, hipMemcpyHostToDevice));
            static_cast<void>(hipDeviceSynchronize());
        }
        else
        {
            throw std::runtime_error("Unsupported Device memory type.");
        }

        return solveGroupedGemmGPU(problems, inputs, hardware, *dUA, inputs.ws, stream);
    }

    void ContractionSolution::relaseDeviceUserArgs(void* dUA, void* dUAHost)
    {
        static_cast<void>(hipFree(dUA));
        static_cast<void>(hipFree(dUAHost));
    }

    ContractionSolution::StaticPerformanceModel
        ContractionSolution::staticPerformanceModel(double M,
                                                    double N,
                                                    double K,
                                                    double NumBatches,
                                                    double MT0,
                                                    double MT1,
                                                    double NumCUs,
                                                    double TotalGranularity,
                                                    int    GlobalSplitU) const
    {
        StaticPerformanceModel spm;

        int beta      = (int)problemType.useBeta;
        int betaReads = 0, betaWrites = 0;
        if(GlobalSplitU == 1)
        {
            if(beta != 0.0)
                betaReads = 1.0;
        }
        else
        {
            if(beta == 0)
                betaWrites = 1; // zero output
            else if(beta != 1.0) // if 1.0, just atomic update output
            {
                // if not 1.0, read, scale, write, then atomic update in kernel
                betaReads  = 1; // initial read for scale
                betaWrites = 1; // writeback after scale
            }
        }

        auto aInfo = DataTypeInfo::Get(problemType.aType);
        auto bInfo = DataTypeInfo::Get(problemType.bType);
        auto cInfo = DataTypeInfo::Get(problemType.cType);
        auto dInfo = DataTypeInfo::Get(problemType.dType);

        spm.memReadBytesA = multiplyElementSize((NumBatches * M * N * K) / MT1, aInfo.elementSize);
        spm.memReadBytesB = multiplyElementSize((NumBatches * M * N * K) / MT0, bInfo.elementSize);
        spm.memReadBytesC = multiplyElementSize((NumBatches * M * N) * betaReads, cInfo.elementSize);

        if(GlobalSplitU == 1)
            spm.memWriteBytesD = multiplyElementSize((NumBatches * M * N) * (1 + betaWrites), dInfo.elementSize);
        else
        {
            bool   hardwareAtomic   = false; // TODO-model
            double atomicOperations = hardwareAtomic ? 2 : 3; // read-mod-write or cas  //TODO-model
            double atomicCollisions = 1.0; // TODO-could be based on K, GSU
            spm.memWriteBytesD      = multiplyElementSize((NumBatches * M * N)
                                 * (betaWrites + atomicOperations * atomicCollisions)
                                 , dInfo.elementSize);
        }
        spm.memReadBytes   = spm.memReadBytesA + spm.memReadBytesB + spm.memReadBytesC;
        spm.memGlobalReads = divideElementSize(spm.memReadBytesA, aInfo.elementSize)
                             + divideElementSize(spm.memReadBytesB, bInfo.elementSize)
                             + divideElementSize(spm.memReadBytesC, cInfo.elementSize);
        spm.memGlobalWrites = divideElementSize(spm.memWriteBytesD, dInfo.elementSize);

        return spm;
    }

    bool ContractionSolution::checkInternalArgumentsSupport(ContractionProblem const& problem,
                                                            std::ostream&             stream,
                                                            bool                      debug) const
    {
        bool pass = true;

        if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(&problem))
        {
            if(!internalArgsSupport.gsu && gemmProblem->getParams().gsu() != 0)
            {
                if(debug)
                {
                    stream << "This solution does not support custom gsu." << std::endl;
                }
                pass = false;
            }
            if(!internalArgsSupport.wgm && gemmProblem->getParams().wgm() != 0)
            {
                if(debug)
                {
                    stream << "This solution does not support custom wgm." << std::endl;
                }
                pass = false;
            }
        }
        else if(auto groupedProblem = dynamic_cast<ContractionProblemGroupedGemm const*>(&problem))
        {
            if(gemmProblem->getParams().gsu() != 0)
            {
                if(debug)
                {
                    stream << "Currently grouped gemm does not support custom arguments tuning."
                           << std::endl;
                }
                pass = false;
            }
            if(!internalArgsSupport.wgm && gemmProblem->getParams().wgm() != 0)
            {
                if(debug)
                {
                    stream << "This solution does not support custom wgm." << std::endl;
                }
                pass = false;
            }
        }
        else
        {
            pass = false;
            throw std::runtime_error("Failed to cast problem type.");
        }
        return pass;
    }

    size_t ContractionSolution::requiredWorkspaceSize(Problem const&  problem,
                                                      Hardware const& hardware) const
    {
        size_t size = 0;
        // TODO: Pass GSU from problem and change value[2] to gsu if gsu != default value
        size_t gsu
            = problem.getParams().gsu() > 0 ? problem.getParams().gsu() : sizeMapping.globalSplitU;

        if(sizeMapping.streamK > 0 && sizeMapping.streamKAtomic == 0)
        {
            // SK doesn't care gsu
            if(gsu > 1)
            {
                std::cerr << "Warning: Stream-K Data Parallel does not support GSU > 1, "
                          << "setting GSU to 1." << std::endl;
                gsu = 1;
            }
            const bool streamKDP = Debug::Instance().useStreamKDataParrallel();
            const bool forceDPOnly = sizeMapping.streamKForceDPOnly != 0;
            auto       tiles     = problem.getNumTiles(sizeMapping, 1);
            if(tiles > 0) // Grouped GEMM reports 0 tiles
            {
                const bool effectiveDynamic = (sizeMapping.streamK == 5)
                                                  ? streamK5EffectiveDynamic(problem, hardware)
                                                  : false;
                // Mirror solve()'s SK4 / SK5-dynamic tree forcing so the grid
                // and workspace reported here match the launch reduction.
                origami::reduction_t reductionStrat;
                if(sizeMapping.streamK == 4)
                    reductionStrat = origami::reduction_t::tree;
                else if(sizeMapping.streamK == 5)
                    reductionStrat = effectiveDynamic ? origami::reduction_t::tree
                                                      : getSKReduction(problem, hardware);
                else
                    reductionStrat = getSKReduction(problem, hardware);
                size_t skGrid = getSKGridImpl(*this,
                                              problem,
                                              hardware,
                                              tiles,
                                              reductionStrat,
                                              sizeMapping.streamK == 5 ? &effectiveDynamic
                                                                       : nullptr);
                // A grid with fewer than two workgroups per tile cannot carry
                // parallel reduction. Reconcile with the SAME helper
                // resolveStreamKSettings() uses, on the same triple, so the
                // size reported here is the size the launch actually needs.
                reductionStrat = streamKReconcileReduction(reductionStrat, skGrid, tiles);
                // Get space required for partial tiles=
                if(reductionStrat == origami::reduction_t::parallel)
                {
                    size_t splitk         = skGrid / tiles;
                    size_t idealWorkspace = requiredWorkspaceSizeGsu(problem, hardware, splitk);
                    if(idealWorkspace <= problem.workspaceSize())
                        size += idealWorkspace;
                }
                else if(skGrid > 0 && (tiles % skGrid != 0 && !streamKDP && !forceDPOnly))
                {
                    size_t idealWorkspace = partialTileSize(skGrid);
                    // Reserve the per-XCD work-queue region for the dynamic-queue
                    // path. Sized as (per-queue stride) * (baked per-XCD queue
                    // count), both from origami: the stride is the L2 cache-line
                    // size (get_default_cache_line_bytes, 128 B for gfx942/gfx950)
                    // so each counter owns its line (no false sharing), and the
                    // queue count is the per-arch XCD count (e.g. 8 for
                    // gfx942/gfx950). This may slightly over-report on a device
                    // that falls back to tree reduction (e.g. MI300A), which is
                    // safe (never under-sized).
                    if(sizeMapping.streamK == 4
                       || (sizeMapping.streamK == 5 && effectiveDynamic))
                        idealWorkspace
                            += streamKPerQueueStrideBytes(hardware) * streamKBakedQueueCount(hardware);
                    // If given workspace is less than ideal, we can fall back to DP mode
                    // Performance will likely be lower, but the kernel can run if workspace is unavailable
                    if(idealWorkspace <= problem.workspaceSize())
                        size += idealWorkspace;
                }
            }
        }
        else
        {
            // TODO: Pass GSU from problem and change value[2] to gsu if gsu != default value
            size_t gsu = problem.getParams().gsu() > 0 ? problem.getParams().gsu()
                                                       : calculateAutoGSU(problem, &hardware);
            size += requiredWorkspaceSizeGsu(problem, hardware, gsu);
        }
        return size;
    }

    size_t ContractionSolution::requiredWorkspaceSizeGsu(Problem const&  problem,
                                                         Hardware const& hardware,
                                                         size_t          gsu) const
    {
        size_t size = 0;

        size_t gsuMultiplier = gsu > 1 ? gsu : 0;
        size_t batch         = problem.d().sizes()[2];
        size_t tiles         = problem.getNumTiles(sizeMapping, gsu) * batch;
        size_t tileSize
            = sizeMapping.macroTile.x * sizeMapping.macroTile.y * sizeMapping.workspaceSizePerElemC;
        size_t bufSize = gsu > 1 ? tiles * tileSize : 0;
        size += bufSize;

        if(problemType.useGradient && problemType.useBias
           && problem.getParams().biasEnum() != rocisa::DataType::None)
        {
            if(problem.biasSrc() == ContractionProblemGemm::TENSOR::A)
            {
                size += problem.freeSizeA(0) * sizeMapping.workspaceSizePerElemBias * gsuMultiplier;
            }
            else if(problem.biasSrc() == ContractionProblemGemm::TENSOR::B)
            {
                size += problem.freeSizeB(0) * sizeMapping.workspaceSizePerElemBias * gsuMultiplier;
            }
            else if(problem.biasSrc() == ContractionProblemGemm::TENSOR::D && (gsuMultiplier == 0))
            {
                size += problem.d().totalLogicalElements() * problem.computeTypeElementSize() * gsu;
            }
        }

        // workspace for amaxD
        if(problemType.outputAmaxD)
        {
            auto numWGS = getNumWorkGroups(problem, sizeMapping);
            size += multiplyElementSize(numWGS, problem.amaxd().elementBytes());
        }

        return size;
    }

    size_t
        ContractionSolution::requiredWorkspaceSizeGroupedGemm(std::vector<Problem> const& problems,
                                                              Hardware const& hardware) const
    {
        size_t sizeInByte = 0;

        for(int i = 0; i < problems.size(); i++)
        {
            auto problem = problems[i];
            sizeInByte += requiredWorkspaceSize(problem, hardware);
        }
        ContractionGroupedInputs inputs;
        for(int i = 0; i < problems.size(); i++)
        {
            ContractionInputs unit;
            inputs.grouped.push_back(unit);
        }
        auto h_args = KernelArgumentsCounter();
        generateSingleCallGroupedGemm<false>(problems, inputs, hardware, h_args);
        if(sizeMapping.globalAccumulation)
            generateOutputConversionCallGroupedGemm<false>(problems, inputs, hardware, h_args);
        sizeInByte += h_args.size();
        return sizeInByte;
    }

    size_t ContractionSolution::requiredHostSizeGroupedGemmSingle(Problem const&  problem,
                                                                  Hardware const& hardware) const
    {
        if(!problemType.groupedGemm)
            return 0;

        std::vector<Problem> tmpProblem;
        tmpProblem.emplace_back(problem);
        ContractionGroupedInputs inputs;
        for(int i = 0; i < tmpProblem.size(); i++)
        {
            ContractionInputs unit;
            inputs.grouped.push_back(unit);
        }
        auto h_args = KernelArgumentsCounter();
        generateSingleCallGroupedGemm<false>(tmpProblem, inputs, hardware, h_args);
        if(sizeMapping.globalAccumulation)
            generateOutputConversionCallGroupedGemm<false>(tmpProblem, inputs, hardware, h_args);
        return h_args.size();
    }

    size_t ContractionSolution::requiredSynchronizerSize(Problem const&  problem,
                                                         Hardware const& hardware) const
    {
        if(sizeMapping.globalAccumulation == 3)
        {
            size_t batch = problem.d().sizes()[2];
            size_t tiles = problem.getNumTiles(sizeMapping, 1) * batch;
            return tiles * sizeMapping.synchronizerSizePerWG * sizeof(int);
        }
        return 0;
    }

    namespace
    {
        // Forward declaration: defined with the other uniform-summation-order Stream-K
        // helpers later in this file. getSKReduction consults it when deciding
        // whether parallel remains eligible under uniform summation order, and
        // wants only the yes/no, so obstacleToken defaults to unrequested.
        std::string streamKUniformSummationOrderObstacle(
            SizeMapping const&                      sizeMapping,
            ContractionSolution::ProblemType const& problemType,
            ContractionSolution::Problem const&     problem,
            char const**                            obstacleToken = nullptr);
    } // namespace

    origami::reduction_t ContractionSolution::getSKReduction(Problem const&  problem,
                                                             Hardware const& hardware) const
    {
        auto reductionStrat = origami::reduction_t::tree;

        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
        assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);

        if(!sizeMapping.customKernelName.empty() || handwrittenCustomKernel())
        {
            // Custom kernels currently only support single-kernel (tree) reduction.
            // YAML records set sizeMapping.customKernelName; handwritten kernels
            // are identified by customKernel.name && !generated.
            reductionStrat = origami::reduction_t::tree;
        }
        else if(sizeMapping.streamKForceDPOnly != 0)
        {
            reductionStrat = origami::reduction_t::tree;
        }
        else
        {
            if(static_cast<origami::grid_selection_t>(pAMDGPU->skDynamicGrid)
               != origami::grid_selection_t::k_split_aware)
            {
                return reductionStrat;
            }

            size_t x     = 1;
            size_t y     = 1;
            size_t z     = 1;
            size_t batch = 1;
            for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            {
                x *= problem.freeSizeA(i);
            }
            for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            {
                y *= problem.freeSizeB(i);
            }
            for(size_t i = 0; i < problem.boundIndices().size(); ++i)
            {
                z *= problem.boundSize(i);
            }
            for(size_t i = 0; i < problem.batchIndices().size(); ++i)
            {
                batch *= problem.batchSize(i);
            }
            hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
            TENSILE_ASSERT_EXC(hipAMDGPU != nullptr);
            TENSILE_ASSERT_EXC(hipAMDGPU->analyticalHardware != nullptr);

            origami::problem_t origami_problem = {
                .size  = {x, y, z},
                .batch = batch,
                // CU budget hint; 0 = use all CUs.
                .num_cus = static_cast<size_t>(problem.getParams().smCountTarget()),
            };
            origami::config_t origami_config = {
                .mt = {static_cast<size_t>(sizeMapping.macroTile.x),
                       static_cast<size_t>(sizeMapping.macroTile.y),
                       static_cast<size_t>(sizeMapping.depthU)},
            };

            reductionStrat = origami::streamk::select_reduction(
                origami_problem,
                *(hipAMDGPU->analyticalHardware),
                origami_config,
                static_cast<origami::grid_selection_t>(pAMDGPU->skDynamicGrid));
        }

        // Under USO + static two-tile packing, admit origami's parallel
        // reduction when the launch is eligible (non-atomic and no static
        // obstacles). Final grid / F checks run at the launch gate once
        // getSKGridImpl has sized the grid. Otherwise refuse parallel and
        // force tree so query (requiredWorkspaceSize) and launch (solve)
        // stay on the steered tree path. Mirror the gate's
        // staticTwoTilePacking predicate.
        if(problem.getParams().uniformSummationOrder())
        {
            const bool effectiveDynamic
                = (sizeMapping.streamK == 5) ? streamK5EffectiveDynamic(problem, hardware)
                                             : false;
            const bool staticTwoTilePacking
                = (sizeMapping.streamK == 3)
                  || (sizeMapping.streamK == 5 && !effectiveDynamic);
            if(staticTwoTilePacking)
            {
                const bool keepParallel
                    = reductionStrat == origami::reduction_t::parallel
                      && sizeMapping.streamKAtomic == 0
                      && streamKUniformSummationOrderObstacle(
                             sizeMapping, problemType, problem)
                             .empty();
                if(!keepParallel)
                    reductionStrat = origami::reduction_t::tree;
            }
        }

        return reductionStrat;
    }

    bool ContractionSolution::streamK5EffectiveDynamic(Problem const&  problem,
                                                       Hardware const& hardware) const
    {
        bool        effectiveDynamic = false;
        const char* reasonStr        = "default";

        const int sk5DebugMode = Debug::Instance().streamK5ForceMode();
        if(sk5DebugMode == 0)
        {
            effectiveDynamic = false;
            reasonStr        = "force-env";
        }
        else if(sk5DebugMode == 1)
        {
            effectiveDynamic = true;
            reasonStr        = "force-env";
        }
        else
        {
            // -1 (or any non 0/1) -> respect the API attribute and run the
            // original requested-mode logic.
            bool      runHeuristic  = false;
            const int requestedMode = problem.getParams().streamKTileSchedulingMode();
            switch(requestedMode)
            {
            case 1: // ON -> dynamic (SK4) path
                effectiveDynamic = true;
                reasonStr        = "api-on";
                break;
            case 0: // OFF -> static (SK3) unless sm_count_target engages heuristic
                if(problem.getParams().smCountTarget() <= 0)
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
                size_t x = 1, y = 1, z = 1, batchSz = 1;
                for(size_t i = 0; i < problem.freeIndicesA().size(); ++i)
                    x *= problem.freeSizeA(i);
                for(size_t i = 0; i < problem.freeIndicesB().size(); ++i)
                    y *= problem.freeSizeB(i);
                for(size_t i = 0; i < problem.boundIndices().size(); ++i)
                    z *= problem.boundSize(i);
                for(size_t i = 0; i < problem.batchIndices().size(); ++i)
                    batchSz *= problem.batchSize(i);

                origami::problem_t origami_problem = {
                    .size  = {x, y, z},
                    .batch = batchSz,
                    // CU budget hint; 0 = use all CUs.
                    .num_cus = static_cast<size_t>(problem.getParams().smCountTarget()),
                };
                origami::config_t origami_config = {
                    .mt = {static_cast<size_t>(sizeMapping.macroTile.x),
                           static_cast<size_t>(sizeMapping.macroTile.y),
                           static_cast<size_t>(sizeMapping.depthU)},
                    .occupancy = std::max(sizeMapping.CUOccupancy, static_cast<int>(1)),
                };

                hip::HipAMDGPU const* hipAMDGPU = dynamic_cast<hip::HipAMDGPU const*>(&hardware);
                TENSILE_ASSERT_EXC(hipAMDGPU != nullptr);
                TENSILE_ASSERT_EXC(hipAMDGPU->analyticalHardware != nullptr);
                const auto autoMode = origami::streamk::select_hybrid_mode(
                    origami_problem,
                    *(hipAMDGPU->analyticalHardware),
                    origami_config,
                    static_cast<size_t>(problem.getParams().smCountTarget()));
                effectiveDynamic = autoMode == origami::hybrid_mode_t::dynamic;
            }
        }

        if(Debug::Instance().printStreamKModeSelection())
        {
            const int   forceMode = Debug::Instance().streamK5ForceMode();
            const int   requested = problem.getParams().streamKTileSchedulingMode();
            const int   smCount   = problem.getParams().smCountTarget();
            const char* reqStr
                = (requested == 0 ? "OFF" : requested == 1 ? "ON" : requested == 2 ? "AUTO" : "?");
            std::cerr << "TensileLite::DEBUG: SK5 hybrid mode for kernel '" << this->kernelName
                      << "': requested=" << reqStr << " forceEnv=" << forceMode
                      << " smCountTarget=" << smCount << " reason=" << reasonStr << " -> effective="
                      << (effectiveDynamic ? "dynamic(SK4/work-queue)" : "static(SK3/static-tile)")
                      << "\n";
        }

        return effectiveDynamic;
    }

    bool ContractionSolution::streamKDynamicQueueSupported(Problem const&  problem,
                                                           Hardware const& hardware) const
    {
        // Only StreamK solutions can ever take the dynamic-queue / work-stealing
        // path; everything else (SK3-static, non-StreamK) is always selectable.
        if(sizeMapping.streamK != 4 && sizeMapping.streamK != 5)
            return true;

        // Fast/common path: on hardware whose runtime XCD count is a power of
        // two AND equals the baked per-XCD queue count, the fixed queue masking
        // is valid, so nothing is excluded. Unknown hardware (missing analytical
        // info / no baked count) is treated as UNSUPPORTED by the predicate, so
        // it falls through to the reject-and-continue path below rather than
        // being kept. Checked before streamK5EffectiveDynamic() so the mainline
        // gfx942(MI300X)/gfx950 path stays cheap.
        if(!streamKDynamicQueueUnsupported(hardware))
            return true;

        // Runtime XCD count is not a power of two or does not equal the baked
        // per-XCD queue count. Only the dynamic-queue sub-path is affected: an
        // SK5 solution that resolves to the static (SK3) sub-path for this
        // problem stays valid and selectable.
        const bool dynamicQueue
            = (sizeMapping.streamK == 4)
              || (sizeMapping.streamK == 5 && streamK5EffectiveDynamic(problem, hardware));
        if(!dynamicQueue)
            return true;

        // Reject-and-continue: exclude this dynamic-queue / work-stealing
        // solution from selection (return false) and warn the user ONCE so they
        // are informed rather than silently degraded to tree reduction. Because
        // this is a selection-time predicate, other solutions (SK3-static,
        // non-StreamK) remain available to serve the GEMM.
        warnStreamKDynamicQueueUnsupportedOnce(hardware);
        return false;
    }

    bool ContractionSolution::handwrittenCustomKernel() const
    {
        return !customKernel.name.empty() && !customKernel.generated;
    }

    StreamKSettings ContractionSolution::resolveStreamKSettings(Problem const&  problem,
                                                                Hardware const& hardware) const
    {
        StreamKSettings sk;
        if(sizeMapping.streamK == 0)
            return sk;

        auto tiles = problem.getNumTiles(sizeMapping, 1);
        const bool effectiveDynamic = (sizeMapping.streamK == 5)
                                          ? streamK5EffectiveDynamic(problem, hardware)
                                          : false;
        const bool dynamicQueuePath
            = (sizeMapping.streamK == 4)
              || (sizeMapping.streamK == 5 && effectiveDynamic);

        if(sizeMapping.streamK == 4)
            sk.reduction = origami::reduction_t::tree;
        else if(sizeMapping.streamK == 5)
            sk.reduction = effectiveDynamic ? origami::reduction_t::tree
                                            : getSKReduction(problem, hardware);
        else
            sk.reduction = getSKReduction(problem, hardware);
        sk.streamKTileSchedulingMode = problem.getParams().streamKTileSchedulingMode();
        sk.smCountTarget             = problem.getParams().smCountTarget();
        sk.grid = getSKGridImpl(*this,
                                problem,
                                hardware,
                                tiles,
                                sk.reduction,
                                sizeMapping.streamK == 5 ? &effectiveDynamic : nullptr);
        // Same reconciliation, same helper, same triple as
        // requiredWorkspaceSize(). Must run before the workspace-fit fallback
        // below so that fallback sees the reduction the launch will use.
        sk.reduction = streamKReconcileReduction(sk.reduction, sk.grid, tiles);

        const bool streamKDP   = Debug::Instance().useStreamKDataParrallel();
        const bool forceDPOnly = sizeMapping.streamKForceDPOnly != 0;
        if(sk.grid > 0
           && (sk.reduction == origami::reduction_t::parallel
               || (tiles % sk.grid != 0 && !streamKDP && !forceDPOnly)))
        {
            size_t idealWorkspace = partialTileSize(sk.grid);
            // SK4 and SK5-dynamic need the per-XCD work-queue region; SK5-static
            // sizes like standalone SK3. The region is sized as (per-queue
            // stride) * (baked per-XCD queue count), both sourced from origami:
            // the stride is the L2 cache-line size (get_default_cache_line_bytes,
            // 128 B for gfx942/gfx950) so each counter owns its line (no false
            // sharing), and the queue count is the per-arch XCD count. The
            // acceptance guard requires runtime NUM_XCD == baked, so this equals
            // cacheLineBytes * NUM_XCD for every device that reaches here.
            if(dynamicQueuePath)
                idealWorkspace
                    += streamKPerQueueStrideBytes(hardware) * streamKBakedQueueCount(hardware);
            // If given workspace is less than ideal, we can fall back to DP mode
            // Performance will likely be lower, but the kernel can run if workspace is unavailable.
            // (The non-power-of-two XCD case is handled earlier by explicit
            // rejection, not by a silent fall back to tree reduction.)
            if(idealWorkspace > problem.workspaceSize())
            {
                sk.reduction = origami::reduction_t::tree;
                sk.grid      = tiles;
            }
        }

        return sk;
    }

    namespace
    {
        // Thread-local tally of the uniform-summation-order selection filter.
        //
        // Capacity is fixed and the token strings are borrowed rather than
        // copied, so recording a rejection never allocates. The token set is
        // closed and small (one literal per clause of the launch-obstacle
        // helper), so a linear scan is cheaper than any keyed container and
        // keeps the report in a deterministic insertion order.
        constexpr size_t uniformSummationOrderTallyCapacity = 32;

        struct UniformSummationOrderTally
        {
            char const* tokens[uniformSummationOrderTallyCapacity] = {};
            size_t      counts[uniformSummationOrderTallyCapacity] = {};
            size_t      distinct                                   = 0;
            // Candidates that reached the uniform-summation-order filter, i.e.
            // that had already satisfied every other selection predicate.
            size_t examined = 0;
            size_t refused  = 0;
        };

        UniformSummationOrderTally& uniformSummationOrderTally()
        {
            static thread_local UniformSummationOrderTally tally;
            return tally;
        }

        void uniformSummationOrderTallyRecord(char const* token)
        {
            UniformSummationOrderTally& tally = uniformSummationOrderTally();
            ++tally.refused;

            // Every clause tags itself, so a null token means a clause was
            // added without one. Naming that case is better than dropping the
            // rejection out of the total.
            if(token == nullptr)
                token = "Untagged";

            for(size_t i = 0; i < tally.distinct; ++i)
            {
                if(std::strcmp(tally.tokens[i], token) == 0)
                {
                    ++tally.counts[i];
                    return;
                }
            }

            if(tally.distinct == uniformSummationOrderTallyCapacity)
            {
                // Unreachable while the token set is smaller than the capacity;
                // folding rather than dropping keeps the counts summing to
                // refused if a future clause pushes it over.
                ++tally.counts[uniformSummationOrderTallyCapacity - 1];
                return;
            }

            tally.tokens[tally.distinct] = token;
            tally.counts[tally.distinct] = 1;
            ++tally.distinct;
        }

        // Statically-knowable reasons a StreamK launch cannot be shown
        // row-uniform: no resolved grid, reduction strategy or Synchronizer
        // pointer is consulted, so the launch-obstacle helper (used by both
        // checkUniformSummationOrder() and uniformSummationOrderSupported())
        // can share one implementation and cannot disagree. Returns an empty
        // string when nothing here objects.
        //
        // Each rejection also names itself through obstacleToken (see the
        // declaration of uniformSummationOrderLaunchObstacle) so a diagnostic
        // never has to recover the reason by matching on the prose.
        //
        // None of these fires for any solution in the shipped tuned logic. They
        // fence configurations that are expressible but were never audited for
        // this guarantee.
        std::string streamKUniformSummationOrderObstacle(
            SizeMapping const&                        sizeMapping,
            ContractionSolution::ProblemType const&   problemType,
            ContractionSolution::Problem const&       problem,
            char const**                              obstacleToken)
        {
            auto refuse = [&](char const* token, std::string detail) -> std::string {
                if(obstacleToken != nullptr)
                    *obstacleToken = token;
                return detail;
            };

            // Atomic fixup accumulates partial tiles in arrival order.
            if(sizeMapping.streamKAtomic != 0)
                return refuse("StreamKAtomic", "StreamKAtomic=1");

            // The partials write and the fixup read both build their store
            // state coordinate-agnostically (lane-linear addressing, no
            // coord0/coord1 term), which is what makes an out-of-range lane
            // unable to influence an in-range one. UseInitialStridesCD turns
            // optSingleColVgpr/optSharedColVgpr off outright
            // (AsmStoreState.py), giving every element its own address calc, so
            // that premise no longer holds. Note the parameter is mapOptional
            // in the solution serialization: a record omitting it reads false,
            // so this rejection is only as good as the logic files.
            if(problemType.useInitialStridesCD)
                return refuse("UseInitialStridesCD",
                              "UseInitialStridesCD=1 disables the coordinate-agnostic store "
                              "addressing the StreamK partials path relies on");

            // Same premise, different switch: more than one packed index in
            // dimension 0 selects per-column address VGPRs and more than one in
            // dimension 1 selects per-element ones (AsmStoreState.py). This is
            // the expression the file already uses for the same question in
            // projectedPerformance()/calculateDimensionM().
            if(problem.freeIndicesA().size() > 1 || (sizeMapping.packBatchDims & 0x1))
                return refuse("PackedC0Index",
                              "a packed C0 index set disables the coordinate-agnostic store "
                              "addressing the StreamK partials path relies on");
            if(problem.freeIndicesB().size() > 1 || (sizeMapping.packBatchDims & 0x2))
                return refuse("PackedC1Index",
                              "a packed C1 index set disables the coordinate-agnostic store "
                              "addressing the StreamK partials path relies on");

            // WaveSplitK and LocalSplitU are mutually exclusive projections of
            // WorkGroup[2] (Solution.py), so WorkGroup[2] > 1 with LocalSplitU
            // == 1 is exactly WaveSplitK. Its redundant-lane store mask covers
            // the D/TD stores, not the WS partials store StreamK uses, so all
            // NumWaveSplitK lanes would write the same workspace address. No
            // shipped solution combines the two; keeping the check inside the
            // StreamK scope is deliberate, since every shipped WaveSplitK
            // solution has StreamK=0 and refusing those would buy nothing.
            if(sizeMapping.workGroupSize.z > 1 && sizeMapping.LocalSplitU <= 1)
                return refuse("WaveSplitK",
                              "WaveSplitK (WorkGroup[2]="
                                  + std::to_string(sizeMapping.workGroupSize.z)
                                  + " with LocalSplitU=" + std::to_string(sizeMapping.LocalSplitU)
                                  + ") does not mask redundant lanes on the StreamK partials "
                                    "store");

            // MX block scaling. Detected from the problem type, never from a
            // kernel-name substring: thousands of shipped f32 records carry an
            // _MX_ token that comes from F32XdlMathOp=XFloat32 and have no
            // scale tensor at all, and MXBlock* is only ever emitted at
            // problem-type level in the logic files.
            if(problemType.mxBlockA != 0 || problemType.mxBlockB != 0)
            {
                // The granule size below is derived for one specific geometry.
                // Change the swizzle format, the block size or MatrixInstK and
                // 256 silently becomes the wrong number rather than a violated
                // one, so pin the envelope the derivation covers. All shipped
                // MX solutions satisfy all three.
                if(problemType.mxScaleFormat != 1)
                    return refuse("MXScaleFormat",
                                  "MX scale format " + std::to_string(problemType.mxScaleFormat)
                                      + " under StreamK is not audited for uniform summation "
                                        "order");
                if((problemType.mxBlockA != 0 && problemType.mxBlockA != 32)
                   || (problemType.mxBlockB != 0 && problemType.mxBlockB != 32))
                    return refuse("MXBlockSize",
                                  "an MX block size other than 32 under StreamK is not audited "
                                  "for uniform summation order");
                if(sizeMapping.matrixInstruction[2] != 128)
                    return refuse("MXMatrixInstK",
                                  "MX with MatrixInstK="
                                      + std::to_string(sizeMapping.matrixInstruction[2])
                                      + " under StreamK is not audited for uniform summation "
                                        "order");

                // The gfx950 HostPreSwizzle scale layout packs 32 M/N rows x 8
                // MX blocks into one 256-byte granule addressed lane-linearly,
                // so byte position inside a granule encodes both a row and a
                // K-block. StreamK shifts the scale SRD by a scalar
                // StreamKLocalStart*DepthU bytes; unless that is a whole number
                // of granules it permutes rows against K-blocks and different
                // rows of one tile consume scales from different K positions.
                // Solution validation already forces this for MX fp4 (duUnit =
                // numSubIterK * MatrixInstK * LocalSplitU = 256) but admits
                // DepthU=128 for MX fp8, so assert it here.
                constexpr size_t mxScaleSwizzleGranuleK = 256; // lrSubtileShape[1](2) * instK(128)
                if(sizeMapping.depthU % mxScaleSwizzleGranuleK != 0)
                    return refuse("MXScaleSwizzleGranule",
                                  "DepthU=" + std::to_string(sizeMapping.depthU)
                                      + " is not a multiple of the MX scale swizzle granule (256 "
                                        "K elements), so a StreamK K-cut can land inside a "
                                        "granule");
            }

            return {};
        }
    }

    bool ContractionSolution::uniformSummationOrderSupported(Problem const&  problem,
                                                             Hardware const& hardware) const
    {
        if(!problem.getParams().uniformSummationOrder())
            return true;

        StreamKSettings sk;
        uint32_t        gsu        = 0;
        size_t          resolvedGA = 0;

        if(problem.groupedGemm())
        {
            // generateSingleCallGroupedGemm() packs skGrid == 0 and reads
            // sizeMapping.globalAccumulation directly. Matching that here is
            // what keeps a grouped StreamK winner from surviving selection and
            // then failing the launch gate on grid == 0.
            gsu        = problem.getParams().gsu() > 0 ? problem.getParams().gsu()
                                                       : calculateAutoGSU(problem, &hardware);
            resolvedGA = sizeMapping.globalAccumulation;
        }
        else
        {
            const uint32_t autoGsuVal = calculateAutoGSU(problem, &hardware);
            gsu = problem.getParams().gsu() > 0 ? problem.getParams().gsu() : autoGsuVal;
            sk  = resolveStreamKSettings(problem, hardware);
            resolvedGA = problem.getAccumulation(hardware, sizeMapping, gsu);
        }

        // The Synchronizer pointer is not allocated at heuristic time; skip
        // that one launch-only clause. Everything else the gate would refuse
        // is dropped here so findTopSolutions / isAlgoSupported return only
        // kernels this problem can launch under USO.
        //
        // The token is collected only to be tallied: a lookup that ends with no
        // admissible solution reports which clauses emptied it, which is the
        // only way to tell "uniform summation order refused everything" from
        // "nothing matched this problem in the first place". Tallying is off
        // unless TENSILE_DB bit 0x200000 is set and never changes the verdict.
        char const*       obstacleToken = nullptr;
        const std::string obstacle      = uniformSummationOrderLaunchObstacle(
            problem, hardware, sk, resolvedGA, gsu, nullptr, false, &obstacleToken);

        if(Debug::Instance().printNoSolutionUniformSummationOrder())
        {
            ++uniformSummationOrderTally().examined;
            if(!obstacle.empty())
                uniformSummationOrderTallyRecord(obstacleToken);
        }

        return obstacle.empty();
    }

    void uniformSummationOrderSelectionTallyReset()
    {
        uniformSummationOrderTally() = UniformSummationOrderTally{};
    }

    std::string uniformSummationOrderSelectionTallyReport(size_t& examined, size_t& refused)
    {
        UniformSummationOrderTally const& tally = uniformSummationOrderTally();
        examined                                = tally.examined;
        refused                                 = tally.refused;

        if(tally.distinct == 0)
            return "none";

        // Most frequent first, insertion order breaking ties, so the same run
        // always renders the same string.
        std::vector<size_t> order(tally.distinct);
        for(size_t i = 0; i < tally.distinct; ++i)
            order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return tally.counts[a] > tally.counts[b];
        });

        std::string report;
        for(size_t i = 0; i < order.size(); ++i)
        {
            if(i != 0)
                report += ',';
            report += tally.tokens[order[i]];
            report += ':';
            report += std::to_string(tally.counts[order[i]]);
        }
        return report;
    }

    std::string ContractionSolution::uniformSummationOrderLaunchObstacle(
        Problem const&         problem,
        Hardware const&        hardware,
        StreamKSettings const& sk,
        size_t                 resolvedGlobalAccumulation,
        uint32_t               gsu,
        void const*            synchronizer,
        bool                   requireSynchronizer,
        char const**           obstacleToken) const
    {
        auto refuse = [&](char const* token, std::string detail) -> std::string {
            if(obstacleToken != nullptr)
                *obstacleToken = token;
            return detail;
        };

        if(handwrittenCustomKernel())
            return refuse("CustomKernel",
                          "custom kernel " + customKernel.name
                              + " is not supported under uniform summation order");

        if(sizeMapping.streamK != 0)
        {
            // The statically-knowable obstacles (atomic fixup,
            // UseInitialStridesCD, packed C0/C1, WaveSplitK, MX scale granule).
            // That helper tags itself, so the token is already set when it
            // objects.
            const std::string obstacle = streamKUniformSummationOrderObstacle(
                sizeMapping, problemType, problem, obstacleToken);
            if(!obstacle.empty())
                return obstacle;

            // Load-bearing rather than defensive: the grouped-GEMM callers pass
            // a default-constructed StreamKSettings, so this is what stands
            // between them and a division by zero below.
            if(sk.grid == 0)
                return refuse("StreamKGridZero", "the resolved StreamK grid is 0");

            // StreamK=5 hybrid runs the static (SK3) packing unless its
            // sub-mode resolves to dynamic. generateSingleCall() decides this
            // from the same problem and hardware, so the gate and the packer
            // cannot disagree about which ABI is in play.
            const bool effectiveDynamic = (sizeMapping.streamK == 5)
                                              ? streamK5EffectiveDynamic(problem, hardware)
                                              : false;
            const bool staticTwoTilePacking
                = (sizeMapping.streamK == 3) || (sizeMapping.streamK == 5 && !effectiveDynamic);

            // Batch-inclusive tile count, matching the value the StreamK
            // resolution and the kernel-arg packing both use.
            const size_t tiles = problem.getNumTiles(sizeMapping, 1);

            if(sk.reduction == origami::reduction_t::parallel)
            {
                // Parallel packs AddressFlags == 0 and does not use the tree
                // static-split model. Admit only when the shared helper proves
                // the launch is row-uniform; otherwise fail closed.
                if(!streamKParallelReductionRowUniform(
                       sk, sizeMapping.streamKAtomic, staticTwoTilePacking, tiles))
                    return refuse(
                        "ParallelReductionNotRowUniform",
                        "the resolved StreamK parallel reduction is not row-uniform: tiles="
                            + std::to_string(tiles) + " grid=" + std::to_string(sk.grid)
                            + " streamKAtomic=" + std::to_string(sizeMapping.streamKAtomic)
                            + " staticTwoTilePacking=" + (staticTwoTilePacking ? "1" : "0"));
            }
            else if(sk.reduction == origami::reduction_t::tree)
            {
                if(staticTwoTilePacking)
                {
                    // Clamped exactly as generateSingleCall() clamps it, so K==0
                    // resolves the same way on both sides.
                    const size_t itersPerTile
                        = std::max(size_t{1}, problem.getItersPerTile(sizeMapping));

                    AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
                    if(pAMDGPU == nullptr)
                        return refuse("HardwareNotAMDGPU",
                                      "the StreamK split cannot be recomputed for this hardware");

                    // The same helper generateSingleCall() packs from, so the gate
                    // reasons about the split the kernel actually performs.
                    const StreamKStaticSplit split
                        = streamKStaticSplit(tiles,
                                             itersPerTile,
                                             sk.grid,
                                             pAMDGPU != nullptr ? pAMDGPU->skFullTiles : 1,
                                             sizeMapping.streamKForceDPOnly != 0);

                    if(!streamKStaticSplitRowUniform(split,
                                                     tiles,
                                                     itersPerTile,
                                                     sk.grid,
                                                     internalArgsSupport.perTileExtraIters))
                        return refuse(
                            "StaticSplitNotRowUniform",
                            "the StreamK split is not row-uniform: tiles=" + std::to_string(tiles)
                                + " itersPerTile=" + std::to_string(itersPerTile)
                                + " grid=" + std::to_string(sk.grid)
                                + " skTiles=" + std::to_string(split.skTiles)
                                + " skItersPerWG=" + std::to_string(split.skItersPerWG)
                                + " extraIters=" + std::to_string(split.extraIters)
                                + " perTileExtraIters="
                                + (internalArgsSupport.perTileExtraIters ? "1" : "0"));
                }
                else if(tiles % sk.grid != 0)
                {
                    // SK4 and SK5-dynamic pack SKTiles/SKSplit/SKItersPerWI, which
                    // the derivation above does not describe. They are additionally
                    // held to SKTiles == 0 below, so the divisibility test is
                    // redundant for them, but redundant and fail-closed is the
                    // right side to err on.
                    return refuse("GridDoesNotDivideTiles",
                                  "StreamK grid " + std::to_string(sk.grid)
                                      + " does not divide the tile count "
                                      + std::to_string(tiles));
                }

                // Mirrors the arg-packing condition: the ws/Flags pair is only
                // appended for these kernels, and the device reads AddressFlags == 0
                // as a request for the parallel reduction path.
                if(requireSynchronizer && sizeMapping.streamKAtomic == 0
                   && sizeMapping.streamKForceDPOnly == 0 && synchronizer == nullptr)
                    return refuse("SynchronizerNull",
                                  "the StreamK Synchronizer/Flags pointer is null");

                // The dynamic-queue variants are row-uniform only while every output
                // tile stays data-parallel, i.e. the packed SKTiles is 0.
                if(sizeMapping.streamK == 4 || effectiveDynamic)
                {
                    AMDGPU const*  pAMDGPU       = dynamic_cast<AMDGPU const*>(&hardware);
                    const int      overrideTiles = pAMDGPU != nullptr ? pAMDGPU->skTiles : -1;
                    const uint32_t skTiles
                        = overrideTiles > -1 ? static_cast<uint32_t>(overrideTiles) : 0u;
                    if(skTiles != 0)
                        return refuse("DynamicQueueSKTiles",
                                      "the dynamic-queue StreamK path is packing SKTiles="
                                          + std::to_string(skTiles) + " rather than 0");
                }
            }
            else
            {
                return refuse("UnknownReduction",
                              "the resolved StreamK reduction is neither tree nor parallel");
            }
        }

        // The resolved value, which can differ from sizeMapping.globalAccumulation
        // when AdaptiveGemmGSUA is enabled, is the one the kernel sees. 2
        // (MultipleBuffer), 3 (MultipleBufferSingleKernel) and 4 (PartialsBuffer)
        // are always row-uniform; 0 (none) and 1 (SingleBuffer) only accumulate
        // atomically once the effective GSU splits K, and are a plain write to D
        // at GSU 1.
        const bool accumulationRowUniform
            = resolvedGlobalAccumulation == 2 || resolvedGlobalAccumulation == 3
              || resolvedGlobalAccumulation == 4
              || ((resolvedGlobalAccumulation == 0 || resolvedGlobalAccumulation == 1) && gsu <= 1);
        if(!accumulationRowUniform)
            return refuse("GlobalAccumulation",
                          "resolved GlobalAccumulation="
                              + std::to_string(resolvedGlobalAccumulation)
                              + " with GSU=" + std::to_string(gsu) + " is not row-uniform");

        // Recomputes exactly what generateSingleCall() packs. The clamp in
        // calculateAutoStaggerU() should already have forced this to 0; checking
        // it anyway is what catches a future path that bypasses the clamp.
        const int32_t autoWGM = std::get<0>(calculateAutoWGM(problem, &hardware, sk.grid));
        const size_t  resolvedStaggerU
            = std::get<1>(calculateAutoStaggerU(problem, &hardware, sk.grid, autoWGM));
        if(resolvedStaggerU != 0)
            return refuse("ResolvedStaggerU",
                          "the resolved StaggerU is " + std::to_string(resolvedStaggerU)
                              + " rather than 0");

        // Only a handwritten custom kernel can carry a stagger the host cannot
        // reach. A generated kernel takes StaggerU exclusively from the packed
        // internal argument: the solution's StaggerU survives code generation
        // only as an SGPR-pool sizing input, so two solutions differing only in
        // StaggerU -- or only in SupportCustomStaggerU -- emit byte-identical
        // assembly, and every generated wrap site unpacks the runtime value.
        // The clamp above therefore does reach it, and !internalArgsSupport
        // .staggerU says nothing about whether the kernel staggers; it only
        // says the host declines to write the field, which leaves it 0.
        //
        // A handwritten custom kernel is one with a non-empty customKernel.name
        // that the generator did not produce (customKernel.generated == false);
        // see handwrittenCustomKernel(). Those are frozen assembly and may bake
        // a literal stagger into the loop, which no host-side clamp can undo.
        //
        // Defense in depth rather than a live path: the CustomKernel clause at
        // the top of this function already refuses every handwritten custom
        // kernel outright, so this cannot fire today. It is kept, like the
        // ResolvedStaggerU check above, so that relaxing that clause to admit
        // individually vetted custom kernels cannot silently admit one whose
        // stagger is compiled in.
        if(handwrittenCustomKernel() && sizeMapping.staggerU != 0)
            return refuse("CompiledInStaggerU",
                          "this kernel has a compiled-in StaggerU="
                              + std::to_string(sizeMapping.staggerU)
                              + " that the host cannot clamp");

        return {};
    }

    void ContractionSolution::checkUniformSummationOrder(Problem const&         problem,
                                                         Hardware const&        hardware,
                                                         StreamKSettings const& sk,
                                                         size_t      resolvedGlobalAccumulation,
                                                         uint32_t    gsu,
                                                         void const* synchronizer) const
    {
        if(!problem.getParams().uniformSummationOrder())
            return;

        const std::string reason = uniformSummationOrderLaunchObstacle(
            problem, hardware, sk, resolvedGlobalAccumulation, gsu, synchronizer, true);
        if(!reason.empty())
        {
            throw UniformSummationOrderError(
                "hipBLASLt Error: solution '" + this->kernelName
                + "' cannot guarantee uniform summation order for this launch: " + reason);
        }
    }

    namespace
    {
        size_t getSKGridImpl(ContractionSolution const& self,
                             ContractionProblemGemm const& problem,
                             Hardware const&               hardware,
                             size_t                        tiles,
                             origami::reduction_t          reductionStrat,
                             bool const*                   sk5EffectiveDynamic,
                             bool*                         outFixedGridUsed,
                             bool*                         outTreeBoundsFallback,
                             bool*                         outClusterDPGridClamp,
                             size_t*                       outSelectedGrid)
        {
            if(outFixedGridUsed)
                *outFixedGridUsed = false;
            if(outTreeBoundsFallback)
                *outTreeBoundsFallback = false;
            if(outClusterDPGridClamp)
                *outClusterDPGridClamp = false;
            size_t     skGrid    = tiles; // Fallback
            const bool streamKDP = Debug::Instance().useStreamKDataParrallel();
            if(streamKDP)
                skGrid = tiles;

            // If K==0, run kernel as DP with Alpha=0 to skip main loop and apply beta*c
            size_t z = 1;
            for(size_t i = 0; i < problem.boundIndices().size(); ++i)
            {
                z *= problem.boundSize(i);
            }
            if(z == 0)
                skGrid = tiles;

            AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);

            assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);
            size_t cuCount = pAMDGPU->computeUnitCount;

            // User-specified grid size for Stream-K kernel.
            if(pAMDGPU->skFixedGrid > 0)
            {
                skGrid = pAMDGPU->skFixedGrid;
                if(outFixedGridUsed)
                    *outFixedGridUsed = true;
            }
            else if(pAMDGPU->skDynamicGrid > 0)
            {
                const bool sk5UsesSK4Grid
                    = (self.sizeMapping.streamK == 5)
                      && (sk5EffectiveDynamic != nullptr
                              ? *sk5EffectiveDynamic
                              : self.streamK5EffectiveDynamic(problem, hardware));
                if(self.sizeMapping.streamK == 4 || sk5UsesSK4Grid)
                {
                    // Limit workgroups per CU to 3
                    // TODO Verify this limit is best
                    auto kernelOccupancy = std::min(self.sizeMapping.CUOccupancy, 3);
                    auto maxGrid         = cuCount * kernelOccupancy;
                    if(pAMDGPU->skMaxCUs > 0)
                    {
                        maxGrid = std::min(maxGrid, static_cast<size_t>(pAMDGPU->skMaxCUs));
                    }
                    // TODO Calculate total work items when dynamic queue works with stream-k
                    // For now, all work items are full tiles
                    auto workItems = tiles;
                    // Select grid to use all CUs, unless number of work items is less
                    skGrid = std::min(workItems, maxGrid);
                }
                else
                {
                    size_t x     = 1;
                    size_t y     = 1;
                    size_t batch = 1;
                    for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
                    {
                        x *= problem.freeSizeA(i);
                    }
                    for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
                    {
                        y *= problem.freeSizeB(i);
                    }
                    for(size_t i = 0; i < problem.batchIndices().size(); ++i)
                    {
                        batch *= problem.batchSize(i);
                    }
                    hip::HipAMDGPU const* hipAMDGPU
                        = dynamic_cast<hip::HipAMDGPU const*>(&hardware);

                    // Fold both CU budgets into origami_problem.num_cus (the single
                    // source of truth select_grid_size derives its budget from).
                    // smCountTarget and skMaxCUs each use 0 to mean "no cap"; take the
                    // tighter (minimum) positive cap so the analytical path honors both.
                    auto   smt       = problem.getParams().smCountTarget(); // int, 0 = no cap
                    auto   skm       = pAMDGPU->skMaxCUs;                   // int, 0 = no cap
                    size_t budget    = 0;                                  // 0 = use all CUs
                    if(smt > 0)
                        budget = static_cast<size_t>(smt);
                    if(skm > 0)
                        budget = (budget == 0) ? static_cast<size_t>(skm)
                                               : std::min(budget, static_cast<size_t>(skm));

                    origami::problem_t origami_problem = {
                        .size        = {x, y, z},
                        .batch       = batch,
                        // CU budget hint; 0 = use all CUs.
                        .num_cus     = budget,
                        .a_transpose = problem.transA() ? origami::transpose_t::T
                                                        : origami::transpose_t::N,
                        .b_transpose = problem.transB() ? origami::transpose_t::T
                                                        : origami::transpose_t::N,
                        .a_dtype     = datatypeToAnalyticalDatatype(problem.a().dataType()),
                        .b_dtype     = datatypeToAnalyticalDatatype(problem.b().dataType()),
                        .mi_dtype    = datatypeToAnalyticalDatatype(problem.computeInputTypeA()),
                    };
                    if(Debug::Instance().printPropertyEvaluation() && self.sizeMapping.CUOccupancy <= 0)
                    {
                        std::cerr << "TensileLite::DEBUG: sizeMapping.CUOccupancy="
                                  << self.sizeMapping.CUOccupancy
                                  << " (<=0) for kernel '" << self.kernelName
                                  << "'; clamping to 1 for origami grid selection.\n";
                    }
                    origami::config_t origami_config = {
                        .mt = {static_cast<size_t>(self.sizeMapping.macroTile.x),
                               static_cast<size_t>(self.sizeMapping.macroTile.y),
                               static_cast<size_t>(self.sizeMapping.depthU)},
                        .mi = {static_cast<size_t>(self.sizeMapping.matrixInstruction[0]),
                               static_cast<size_t>(self.sizeMapping.matrixInstruction[1]),
                               static_cast<size_t>(self.sizeMapping.matrixInstruction[2])},
                        .occupancy = std::max(self.sizeMapping.CUOccupancy, static_cast<int>(1)),
                        .workgroup_mapping         = self.sizeMapping.workGroupMapping,
                        .workspace_size            = problem.workspaceSize(),
                        .workspace_size_per_elem_c = self.sizeMapping.workspaceSizePerElemC,
                        .reduction_strategy        = reductionStrat,
                    };

                    TENSILE_ASSERT_EXC(hipAMDGPU->analyticalHardware != nullptr);

                    skGrid = origami::streamk::select_grid_size(
                        origami_problem,
                        *(hipAMDGPU->analyticalHardware),
                        origami_config,
                        static_cast<origami::grid_selection_t>(pAMDGPU->skDynamicGrid));
                }
            }
            // Limit the CUs Stream-K is launched on either max or the specified,
            // whichever is minimum.
            else if(pAMDGPU->skMaxCUs > 0)
            {
                skGrid = std::min(cuCount, static_cast<size_t>(pAMDGPU->skMaxCUs));
            }

            // Multiply the cuCount with a constant factor (c), and launch
            // c * cuCount number of workgroups for Stream-K.
            else if(pAMDGPU->skGridMultiplier > 1)
            {
                skGrid = cuCount * pAMDGPU->skGridMultiplier;
            }

            // If no option is specified, launch exactly cuCount worth of workgroups.
            else
            {
                skGrid = cuCount;
            }

            // Under uniform-summation-order + static two-tile packing, snap the chain output
            // g0 onto an admissible uniform grid. F-star is never-upward
            // (g0 > tiles). When g0 < tiles, snap up to tiles (all-full): mixed
            // GridDividesTiles (tiles % g0 == 0) is two-tile DP+SK that gfx950
            // does not store (workspace skipped, SK D rows stay poison). Must
            // run before the magic-division guard below so that guard still
            // validates the final grid (a snap after it can emit out-of-range
            // itersPerWG). Same ABI predicate as checkUniformSummationOrder.
            if(problem.getParams().uniformSummationOrder() && tiles > 0 && skGrid > 0)
            {
                const bool effectiveDynamic
                    = (self.sizeMapping.streamK == 5)
                      && (sk5EffectiveDynamic != nullptr
                              ? *sk5EffectiveDynamic
                              : self.streamK5EffectiveDynamic(problem, hardware));
                const bool staticTwoTilePacking
                    = (self.sizeMapping.streamK == 3)
                      || (self.sizeMapping.streamK == 5 && !effectiveDynamic);
                if(staticTwoTilePacking)
                {
                    const size_t g0 = skGrid;
                    // Same clamp as the packer / gate: K==0 yields I==0 otherwise.
                    const size_t I
                        = std::max(size_t{1}, problem.getItersPerTile(self.sizeMapping));
                    // Matches origami::streamk MinItersPerCU (streamk.cpp).
                    constexpr size_t MinItersPerCU = 8;
                    const bool perTileExtraIters = self.internalArgsSupport.perTileExtraIters;

                    // The flag-region bound is a constraint ON this selection, not
                    // a correction applied after it. Every grid this snap can emit
                    // for F >= 2 is tiles * F > tiles, so tiles % skGrid == tiles
                    // != 0 and the flag clamp below would fire on exactly these
                    // grids -- rewriting tiles * F to StreamKFlagElements, which is
                    // not in general a multiple of tiles and so is refused by
                    // checkUniformSummationOrder as a non-row-uniform static split.
                    // Folding the bound in here instead means the search picks a
                    // uniform grid that already satisfies it and the clamp becomes
                    // a no-op, so the flag-region invariant is enforced exactly as
                    // before and never has to rewrite a uniform grid.
                    //
                    // Conditioned on the same three predicates the clamp uses, so a
                    // launch that never touches the flag region is not constrained
                    // by its size. The clamp's fourth predicate (tiles % skGrid !=
                    // 0) is implied for every F >= 2 candidate and so is omitted.
                    // The all-full grid (F == 1, skGrid == tiles) satisfies
                    // tiles % skGrid == 0, uses no flag region at all, and is
                    // therefore admissible at any tile count -- which is what keeps
                    // FStar = 1 a valid floor even when tiles itself exceeds the
                    // bound.
                    const bool flagRegionBinds
                        = self.sizeMapping.streamKAtomic == 0
                          && self.sizeMapping.streamKForceDPOnly == 0
                          && reductionStrat != origami::reduction_t::parallel;

                    if(g0 > tiles)
                    {
                        const size_t F0    = g0 / tiles;
                        size_t       FStar = 1; // always admissible (all-full)
                        for(size_t F = F0; F >= 2; --F)
                        {
                            // Tree all-partial without per-tile extras needs
                            // F | I. Parallel extras are per PartialIdx and
                            // tile-symmetric without that capability bit, so
                            // skip the divisibility requirement for parallel.
                            if(reductionStrat != origami::reduction_t::parallel
                               && !perTileExtraIters && (I % F) != 0)
                                continue;
                            if((I / F) < MinItersPerCU)
                                continue;
                            // F==1 needs no partials; for F>=2 require workspace fit.
                            if(self.partialTileSize(tiles * F) > problem.workspaceSize())
                                continue;
                            // Stay inside the flag region these grids will use.
                            if(flagRegionBinds
                               && (tiles * F) > static_cast<size_t>(StreamKFlagElements))
                                continue;
                            FStar = F;
                            break;
                        }
                        skGrid = tiles * FStar;
                    }
                    else if(g0 < tiles)
                    {
                        skGrid = tiles;
                    }

                    if(skGrid != g0
                       && (pAMDGPU->skFixedGrid > 0 || pAMDGPU->skMaxCUs > 0
                           || pAMDGPU->skGridMultiplier > 1))
                    {
                        warnStreamKUniformityGridSnapOnce(g0, skGrid);
                    }
                }
            }

            // Grid selected by the config/CU/override logic, captured before the
            // "reset to tiles" tree-fixup-bounds fallback below.
            //
            // Captured AFTER the uniform-summation-order snap above, because that
            // snap is grid SELECTION under uniform summation order -- it is how an
            // admissible uniform grid is chosen -- and not one of the post-selection
            // fallbacks the launch summary attributes with `changedBy`. Capturing
            // ahead of it would report selected != final with changedBy = none,
            // which is exactly the unattributed-rewrite misreport the out-params
            // exist to prevent. The snap emits its own
            // warnStreamKUniformityGridSnapOnce() note when it overrides an
            // explicitly requested grid, so the override is still observable.
            if(outSelectedGrid)
                *outSelectedGrid = skGrid;

            // Tree-fixup uses scalarUInt24DivideAndRemainder (dividend < 2^24, divisor < 2^16).
            // If we exceed those bounds, fall back to DP.
            if(reductionStrat == origami::reduction_t::tree)
            {
                size_t itersPerTile = problem.getItersPerTile(self.sizeMapping);
                size_t itersPerWG   = tiles * itersPerTile / skGrid;

                if(itersPerTile >= 65536 || itersPerWG >= 65536
                   || (tiles * itersPerTile) >= 16777216)
                {
                    skGrid = tiles;
                    if(outTreeBoundsFallback)
                        *outTreeBoundsFallback = true;
                }
            }

            // StreamK ForceDPOnly cluster multicast (gfx1250, ClusterDim-driven):
            // one work-group per output tile (not a K-split), so skGrid == tiles.
            // The launch pads up to the cluster dims and the boundary peers
            // pad-exit (StreamK.preLoop) when the size is not a cluster multiple.
            if(self.sizeMapping.streamK == 3 && self.sizeMapping.streamKForceDPOnly
               && (static_cast<size_t>(self.sizeMapping.clusterDim.x)
                   * static_cast<size_t>(self.sizeMapping.clusterDim.y))
                      > 1)
            {
                skGrid = tiles;
                if(outClusterDPGridClamp)
                    *outClusterDPGridClamp = true;
            }

            // The flag region holds one int per Stream-K workgroup, indexed by
            // CTA id ("flag offset based on CTA index" in StreamK.py), so a grid
            // wider than the region would have workgroups writing past the end
            // of their own block. Every caller reaches skGrid through here and
            // this is the last write, so it is the one place the bound has to
            // hold.
            //
            // Only the launches that actually reach the flags are bounded:
            //
            //   - atomic and ForceDPOnly kernels never take the pointer at all
            //     (see the Flags kernarg in singleCallArgs)
            //   - parallel reduction is passed Flags == nullptr, and the kernel
            //     branches on that to skip the flag protocol
            //   - tiles % skGrid == 0 spreads the tiles evenly over the
            //     workgroups, so no partial tiles exist to fix up. This is the
            //     same test the workspace sizing uses to decide whether partials
            //     exist, and it is what excludes every data-parallel path: they
            //     all arrive at skGrid == tiles, whether from K == 0, from the
            //     tree-fixup bound above, or from the data-parallel debug knob.
            //
            // Clamping outside those cases would shrink a grid that has no flag
            // region to overrun, and against the data-parallel fallbacks it
            // would do real harm: the tree-fixup bound sets skGrid = tiles
            // precisely to leave Stream-K behind, and cutting that back would
            // return the launch to the K-split it was escaping. No current part
            // reaches the bound on the paths that are checked; gfx950 has 256
            // CUs and picks 224.
            const bool usesFlagRegion = self.sizeMapping.streamKAtomic == 0
                                        && self.sizeMapping.streamKForceDPOnly == 0
                                        && reductionStrat != origami::reduction_t::parallel
                                        && skGrid > 0 && (tiles % skGrid) != 0;

            if(usesFlagRegion && skGrid > static_cast<size_t>(StreamKFlagElements))
            {
                if(Debug::Instance().printPropertyEvaluation())
                {
                    std::cerr << "TensileLite::DEBUG: kernel '" << self.kernelName
                              << "' StreamK grid " << skGrid << " exceeds the "
                              << StreamKFlagElements << "-entry flag region (tiles=" << tiles
                              << "); clamping the grid to " << StreamKFlagElements << ".\n";
                }
                skGrid = StreamKFlagElements;
            }

            return skGrid;
        }
    } // namespace

    size_t ContractionSolution::getSKGrid(Problem const&       problem,
                                          Hardware const&      hardware,
                                          size_t               tiles,
                                          origami::reduction_t reductionStrat) const
    {
        return getSKGridImpl(*this, problem, hardware, tiles, reductionStrat, nullptr);
    }

    size_t ContractionSolution::partialTileSize(size_t skGrid) const
    {
        size_t size = 0;

        size_t tileSize
            = sizeMapping.macroTile.x * sizeMapping.macroTile.y * sizeMapping.workspaceSizePerElemC;
        size += tileSize * skGrid; // Partials tile per WG
        // TODO batches
        // TODO round up for alignment?

        return size;
    }

    // Single source of truth for the StreamK launch decisions. The reduction, grid,
    // and workspace/DP fallback computed here are the values solve() launches with:
    // solve() reads them straight out of the returned snapshot into StreamKSettings
    // and applies no further StreamK sizing of its own. Existing helpers
    // (streamK5EffectiveDynamic, getSKReduction, getSKGridImpl, partialTileSize) are
    // reused rather than re-derived.
    //
    // Two mirrors here must be kept in sync with code elsewhere:
    //   * the reserve-or-not workspace rule below duplicates the one in
    //     ContractionSolution::requiredWorkspaceSize(), which is what the caller
    //     allocates from;
    //   * skTiles/skSplit/totalItems duplicate the kernel-arg packing in makeArgs().
    //
    // The partials-workspace guard reserves iff (reduction==parallel || tiles%grid!=0),
    // sized as partialTileSize(grid) (+ the per-XCD work-queue region on the dynamic
    // path); the reservation does not depend on dynamicPartialsSlots. The
    // dynamicPartialsSlots field is still populated (skTiles*skSplit, computed
    // locally) purely for reporting.
    StreamKDecisions
        ContractionSolution::computeStreamKDecisions(Problem const&  problem,
                                                     Hardware const& hardware) const
    {
        StreamKDecisions d;
        d.streamKMode = sizeMapping.streamK;
        if(sizeMapping.streamK <= 0)
            return d;

        const size_t tiles = problem.getNumTiles(sizeMapping, 1);
        d.tiles            = tiles;

        const bool effectiveDynamic
            = (sizeMapping.streamK == 5) ? streamK5EffectiveDynamic(problem, hardware) : false;
        d.effectiveDynamic = effectiveDynamic;

        // Reduction strategy. SK4 and SK5-resolved-dynamic are unconditionally tree;
        // everything else asks getSKReduction(). Note requiredWorkspaceSize() always
        // asks getSKReduction() and has no such special case -- see the note above.
        origami::reduction_t reduction;
        if(sizeMapping.streamK == 4)
            reduction = origami::reduction_t::tree;
        else if(sizeMapping.streamK == 5)
            reduction = effectiveDynamic ? origami::reduction_t::tree
                                         : getSKReduction(problem, hardware);
        else
            reduction = getSKReduction(problem, hardware);

        // Grid -- reuses getSKGridImpl (same call solve() makes), and captures the
        // fixed-grid / tree-bounds fallbacks plus the pre-tree-bounds "selected"
        // grid via the optional out-params.
        bool         fixedGridUsed  = false;
        bool         treeBounds     = false;
        bool         clusterDPClamp = false;
        size_t       selectedGrid   = 0;
        const size_t gridInitial    = getSKGridImpl(*this,
                                                 problem,
                                                 hardware,
                                                 tiles,
                                                 reduction,
                                                 sizeMapping.streamK == 5 ? &effectiveDynamic
                                                                          : nullptr,
                                                 &fixedGridUsed,
                                                 &treeBounds,
                                                 &clusterDPClamp,
                                                 &selectedGrid);
        d.selectedGrid            = selectedGrid;
        d.skGridPreFallback       = gridInitial;
        d.fixedGridUsed           = fixedGridUsed;
        d.treeBoundsFallbackFired = treeBounds;
        d.clusterDPGridClamped    = clusterDPClamp;

        size_t grid = gridInitial;

        // Same reconciliation, same helper, same triple as
        // resolveStreamKSettings() -- which is what solve() actually launches --
        // and as requiredWorkspaceSize(). It has to run here, before the
        // workspace-fit fallback below, for the same reason it does there: the
        // fallback's `reduction == parallel` disjunct must see the reduction the
        // launch will use. Omitting it would let this snapshot report a
        // workspaceDP fallback (and a grid) that the launch never took, whenever
        // the grid lands on a splitting factor below 2 with parallel selected.
        reduction = streamKReconcileReduction(reduction, grid, tiles);

        const bool streamKDP   = Debug::Instance().useStreamKDataParrallel();
        const bool forceDPOnly = sizeMapping.streamKForceDPOnly != 0;
        d.streamKDP            = streamKDP;
        d.forceDPOnly          = forceDPOnly;

        const bool isDynamic
            = (sizeMapping.streamK == 4) || (sizeMapping.streamK == 5 && effectiveDynamic);
        d.isDynamic = isDynamic;

        d.numQueues           = streamKBakedQueueCount(hardware);
        d.givenWorkspaceBytes = problem.workspaceSize();

        // Workspace / DP fallback. Reserve iff (reduction==parallel ||
        // tiles%grid!=0), sized by grid (not by dynamicSlots). This is the same
        // reserve-or-not rule requiredWorkspaceSize() implements independently, so
        // the two must be changed together.
        size_t idealWorkspace = 0;
        bool   needPartials   = false;
        if(grid > 0
           && (reduction == origami::reduction_t::parallel
               || (tiles % grid != 0 && !streamKDP && !forceDPOnly)))
        {
            needPartials   = true;
            idealWorkspace = partialTileSize(grid);
            if(isDynamic)
                idealWorkspace
                    += streamKPerQueueStrideBytes(hardware) * streamKBakedQueueCount(hardware);
            if(idealWorkspace > problem.workspaceSize())
            {
                reduction                  = origami::reduction_t::tree;
                grid                       = tiles;
                d.workspaceDPFallbackFired = true;
            }
        }

        d.reduction              = reduction;
        d.skGrid                 = grid;
        d.finalGrid              = grid;
        d.idealWorkspaceBytes    = idealWorkspace;
        d.requiredWorkspaceBytes = (needPartials && !d.workspaceDPFallbackFired) ? idealWorkspace : 0;
        d.workspaceAllocated     = d.requiredWorkspaceBytes > 0;
        d.dpOnly                 = streamKDP || forceDPOnly || d.workspaceDPFallbackFired;

        // skTiles / skSplit / totalItems -- mirror the kernel-arg packing in
        // makeArgs, using the FINAL (post-fallback) grid and reduction.
        const size_t itersPerTile = std::max(size_t{1}, problem.getItersPerTile(sizeMapping));
        if(isDynamic)
        {
            AMDGPU const* pAMDGPU   = dynamic_cast<AMDGPU const*>(&hardware);
            int           overrideT = pAMDGPU ? pAMDGPU->skTiles : -1;
            int           overrideS = pAMDGPU ? pAMDGPU->skSplit : -1;
            uint32_t      skTiles   = 0;
            uint32_t      skSplit   = 2;
            if(overrideT > -1)
                skTiles = static_cast<uint32_t>(overrideT);
            if(overrideS > -1)
                skSplit = static_cast<uint32_t>(overrideS);
            uint32_t skItersPerWI = CeilDivide(static_cast<uint32_t>(itersPerTile), skSplit);
            skSplit               = CeilDivide(static_cast<uint32_t>(itersPerTile), skItersPerWI);
            d.skTiles             = skTiles;
            d.skSplit             = skSplit;
            d.totalItems          = (tiles - skTiles) + static_cast<size_t>(skTiles) * skSplit;
        }
        else if(reduction == origami::reduction_t::parallel && tiles > 0)
        {
            uint32_t skSplit = static_cast<uint32_t>(grid / tiles);
            d.skSplit        = skSplit;
            d.skTiles        = skSplit; // parallel path packs skTiles = skSplit
            d.totalItems     = tiles;
        }
        else
        {
            // Reuse the shared static-split helper rather than repeating its
            // arithmetic: makeArgs() packs skTiles from exactly this call, so the
            // report cannot drift from the launch.
            AMDGPU const*            pAMDGPU   = dynamic_cast<AMDGPU const*>(&hardware);
            const int                fullTiles = pAMDGPU ? pAMDGPU->skFullTiles : 1;
            const StreamKStaticSplit split
                = streamKStaticSplit(tiles, itersPerTile, grid, fullTiles, forceDPOnly);
            d.skTiles    = split.skTiles;
            d.skSplit    = 1;
            d.totalItems = tiles;
        }

        // Informational only (see field doc): skTiles*skSplit slot count for the
        // dynamic path, computed LOCALLY here. It does NOT feed the allocation
        // guard above.
        d.dynamicPartialsSlots = isDynamic ? static_cast<size_t>(d.skTiles) * d.skSplit : 0;

        d.partialsPresent = d.skTiles > 0;
        return d;
    }

    void ContractionSolution::printStreamKLaunchSummary(std::ostream&           os,
                                                        Problem const&          problem,
                                                        StreamKDecisions const& d) const
    {
        // Everything printed below already lives in the snapshot, so the problem is
        // currently unused. It stays in the signature because a launch summary is
        // naturally reported per (solution, problem) pair and the obvious next
        // additions -- the GEMM sizes, transposes, and problem-level StreamK params
        // -- are only reachable from here; keeping it avoids churning every call
        // site and every test when one of those is added.
        (void)problem;
        auto reductionStr = [](origami::reduction_t r) {
            return r == origami::reduction_t::parallel ? "parallel(DP)" : "tree";
        };
        const char* modeStr = "?";
        switch(d.streamKMode)
        {
        case 0: modeStr = "none"; break;
        case 3: modeStr = "SK3(static)"; break;
        case 4: modeStr = "SK4(dynamic)"; break;
        case 5:
            modeStr = d.effectiveDynamic ? "SK5->dynamic(SK4)" : "SK5->static(SK3)";
            break;
        default: modeStr = "SK?"; break;
        }

        // Which fallback (if any) turned the initially-selected grid into the
        // final launch grid. Reported alongside selectedGrid vs finalGrid.
        // Ordered latest-clamp-wins, i.e. the reverse of the order they are applied:
        // the workspace-DP fallback runs last (in computeStreamKDecisions, after
        // getSKGridImpl returns); inside getSKGridImpl the ForceDPOnly cluster
        // multicast clamp runs after the tree-bounds fallback, which in turn runs
        // after the skFixedGrid override. So the first matching branch below names
        // the clamp that actually produced finalGrid.
        const char* gridChangedBy = "none";
        if(d.workspaceDPFallbackFired)
            gridChangedBy = "workspaceDP";
        else if(d.clusterDPGridClamped)
            gridChangedBy = "clusterDPMulticast";
        else if(d.treeBoundsFallbackFired)
            gridChangedBy = "treeBounds";
        else if(d.fixedGridUsed)
            gridChangedBy = "fixedGrid";

        // Which mechanism (if any) makes this launch data-parallel-only. More than
        // one can be set at once (e.g. the debug override on a force-DP-only
        // kernel), so the ladder reports the most specific explanation first:
        // the compile-time kernel param, then the process-wide debug override,
        // then the runtime workspace fallback -- from "this kernel is always DP"
        // to "this particular launch had to give up on StreamK".
        const char* dpOnlySource = "none";
        if(d.forceDPOnly)
            dpOnlySource = "forceDPOnly(param)";
        else if(d.streamKDP)
            dpOnlySource = "streamKDP(debug)";
        else if(d.workspaceDPFallbackFired)
            dpOnlySource = "workspaceDP(runtime)";

        // Human-readable byte annotation, e.g. "1245184 (1.19 MiB)". SIZE_MAX is
        // reported as "unbounded": that is ContractionProblem's default workspace
        // size (see ContractionProblem.hpp m_workspaceSize), meaning
        // setWorkspaceSize() was never called and the workspace is uncapped, so
        // printing it as a byte count would be nonsense.
        auto humanUnit = [](size_t bytes) -> std::string {
            static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
            double              v      = static_cast<double>(bytes);
            int                 u      = 0;
            while(v >= 1024.0 && u < 4)
            {
                v /= 1024.0;
                ++u;
            }
            std::ostringstream o;
            if(u == 0)
                o << bytes << " B";
            else
            {
                o.setf(std::ios::fixed);
                o.precision(2);
                o << v << " " << units[u];
            }
            return o.str();
        };
        auto fmtBytes = [&](size_t bytes) -> std::string {
            std::ostringstream o;
            o << bytes << " (" << humanUnit(bytes) << ")";
            return o.str();
        };
        auto fmtGiven = [&](size_t bytes) -> std::string {
            if(bytes == std::numeric_limits<size_t>::max())
                return "unbounded";
            return fmtBytes(bytes);
        };

        // Output shape: one header line carrying the fixed token
        // "TensileLite::StreamK LAUNCH SUMMARY" and the kernel name, followed by
        // labeled sections. Each section header sits on its own line and its
        // fields are indented beneath it as "key = value" pairs, one per line,
        // with the '=' column-aligned within the section. Keep the header token
        // and the field key spellings stable: log scrapers and the substring
        // assertions in tests/StreamKLaunchSummary_test.cpp match on them.
        //
        // 'field' prints one indented, '='-aligned pair. 'width' is the per-section
        // key column width chosen so every '=' in that section lines up.
        auto field = [&os](int width, const char* key, std::string const& value) {
            os << "    " << std::left << std::setw(width) << key << " = " << value << "\n";
        };
        auto yn = [](bool b) -> std::string { return b ? "yes" : "no"; };

        os << "TensileLite::StreamK LAUNCH SUMMARY  kernel='" << this->kernelName << "'\n";

        os << "  mode:\n";
        field(9, "mode", modeStr);
        field(9, "streamK", std::to_string(d.streamKMode));
        field(9, "reduction", reductionStr(d.reduction));
        field(9, "isDynamic", yn(d.isDynamic));

        os << "  grid:\n";
        field(11, "selected", std::to_string(d.selectedGrid));
        field(11, "final", std::to_string(d.finalGrid));
        field(11, "changedBy", gridChangedBy);
        // preFallback is only interesting when a clamp inside getSKGridImpl
        // (tree-bounds fixup, or the ForceDPOnly cluster multicast) already moved
        // the grid away from what was selected, before the workspace-DP fallback.
        if(d.skGridPreFallback != d.selectedGrid)
            field(11, "preFallback", std::to_string(d.skGridPreFallback));

        os << "  tiles:\n";
        field(10, "tiles", std::to_string(d.tiles));
        field(10, "skTiles", std::to_string(d.skTiles));
        field(10, "skSplit", std::to_string(d.skSplit));
        field(10, "totalItems", std::to_string(d.totalItems));
        field(10, "partials", yn(d.partialsPresent));

        os << "  workspace:\n";
        field(9, "allocated", yn(d.workspaceAllocated));
        field(9, "required", fmtBytes(d.requiredWorkspaceBytes));
        field(9, "ideal", fmtBytes(d.idealWorkspaceBytes));
        field(9, "given", fmtGiven(d.givenWorkspaceBytes));

        os << "  dp-only:\n";
        field(11, "dpOnly", yn(d.dpOnly));
        field(11, "source", dpOnlySource);
        field(11, "forceDPOnly", yn(d.forceDPOnly));
        field(11, "streamKDP", yn(d.streamKDP));

        // Work-queue fields describe the per-XCD dynamic work-queue synchronizer,
        // which only exists on the dynamic path (SK4, or SK5 resolved dynamic).
        // On non-dynamic paths (SK3 / SK5-static / parallel-DP / dp-only) these
        // values are meaningless, so they are printed as NA rather than a
        // misleading number. Gated on the SAME isDynamic predicate solve() uses;
        // the struct still holds the raw counts.
        os << "  work-queue:\n";
        if(d.isDynamic)
        {
            field(20, "numQueues(NUM_XCD)", std::to_string(d.numQueues));
            field(20, "dynamicPartialsSlots", std::to_string(d.dynamicPartialsSlots));
        }
        else
        {
            os << "    NA (work-queues not used)\n";
        }

        os << "  fallbacks:\n";
        field(19, "fixedGrid", yn(d.fixedGridUsed));
        field(19, "workspaceDPFallback", yn(d.workspaceDPFallbackFired));
        field(19, "treeBoundsFallback", yn(d.treeBoundsFallbackFired));
        field(19, "clusterDPMulticast", yn(d.clusterDPGridClamped));
    }

    float ContractionSolution::computeGranularity(float x)
    {
        return x / ceil(x);
    }

    ContractionSolution::Granularities
        ContractionSolution::computeGranularities(Hardware const& hardware,
                                                  double          M,
                                                  double          N,
                                                  double          K,
                                                  double          NumBatches,
                                                  uint32_t        autoGsuVal) const
    {
        ContractionSolution::Granularities granularities;

        double MT0 = sizeMapping.macroTile.x;
        double MT1 = sizeMapping.macroTile.y;

        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
        assert(pAMDGPU);

        double NumCUs        = pAMDGPU->computeUnitCount;
        double wavefrontSize = pAMDGPU->wavefrontSize;
        double simdPerCu     = pAMDGPU->simdPerCu;

        double GlobalSplitU = autoGsuVal;
        double LocalSplitU  = sizeMapping.workGroupSize.z;

        granularities.MT0 = MT0;
        granularities.MT1 = MT1;
        granularities.GSU = GlobalSplitU;
        granularities.LSU = LocalSplitU;
        granularities.CUs = NumCUs;

        granularities.numTiles0 = M / MT0;
        granularities.numTiles1 = N / MT1;

        granularities.tile0Granularity = computeGranularity(granularities.numTiles0);
        granularities.tile1Granularity = computeGranularity(granularities.numTiles1);

        granularities.tilesPerCu
            = (NumBatches * ceil(granularities.numTiles0) * ceil(granularities.numTiles1))
              / (NumCUs / GlobalSplitU / LocalSplitU);

        granularities.totalTiles    = ceil(granularities.numTiles0) * ceil(granularities.numTiles1);
        granularities.natTilesPerCu = NumBatches * granularities.totalTiles / NumCUs;
        granularities.suTilesPerCu  = (granularities.totalTiles * GlobalSplitU) / NumCUs;
        granularities.suCuGranularity = computeGranularity(granularities.suTilesPerCu);

        granularities.waveGranularity = std::min(
            1.00,
            static_cast<double>(floor(granularities.tilesPerCu + 1.0) * sizeMapping.workGroupSize.x
                                * sizeMapping.workGroupSize.y * sizeMapping.workGroupSize.z)
                / pAMDGPU->wavefrontSize / pAMDGPU->simdPerCu);

        granularities.waves
            = ceil((sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y) / wavefrontSize);

        granularities.suWavesPerSimdx2
            = (granularities.suTilesPerCu * granularities.waves) / (2 * simdPerCu);
        granularities.suWaveGranularity
            = granularities.suWavesPerSimdx2 * ceil(granularities.suWavesPerSimdx2);

        double nat_tiles_per_cu
            = NumBatches * ceil(granularities.numTiles0) * ceil(granularities.numTiles1) / NumCUs;
        granularities.natCuGranularity = ceil(nat_tiles_per_cu) * ceil(nat_tiles_per_cu) / NumCUs;

        granularities.cuGranularity = computeGranularity(granularities.tilesPerCu);

        granularities.totalGranularity
            = granularities.tile0Granularity * granularities.tile1Granularity
              * granularities.cuGranularity * granularities.waveGranularity;

        granularities.totalTileAwareGranularity
            = granularities.tile0Granularity * granularities.tile1Granularity
              * granularities.suCuGranularity * granularities.suWaveGranularity;

        return granularities;
    }

    ContractionSolution::ProjectedPerformance
        ContractionSolution::projectedPerformance(Problem const&  problem,
                                                  Hardware const& hardware) const
    {
        ProjectedPerformance pp;

        double M = 1.0, N = 1.0;
        if(problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesA(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                M *= problem.a().sizes()[*pi];
        }
        else
            M = problem.freeSizeA(0);

        if(problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesB(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                N *= problem.b().sizes()[*pi];
        }
        else
            N = problem.freeSizeB(0);

        double NumBatches = 1;
        if(sizeMapping.packBatchDims == 0)
        {
            for(size_t i = 0; i < problem.batchIndices().size(); i++)
                NumBatches *= problem.batchSize(i);
        }
        double K = problem.boundSize(0); // TODO - fix for multiple summations

        pp.granularities = ContractionSolution::computeGranularities(
            hardware, M, N, K, NumBatches, calculateAutoGSU(problem, &hardware));

        auto it = ideals.begin();

        int    closestKMeasure     = std::numeric_limits<int>::max();
        double closestKPerformance = 0.0;

        while(it != ideals.end())
        {
            int myK       = it->first;
            int myMeasure = std::abs(myK - K);
            if(myMeasure < closestKMeasure)
            {
                closestKMeasure     = myMeasure;
                closestKPerformance = it->second;
            }
            it++;
        }

        double MT0    = pp.granularities.MT0;
        double MT1    = pp.granularities.MT1;
        double NumCUs = pp.granularities.CUs;

        double GlobalSplitU         = pp.granularities.GSU;
        double IdealGranularityPerf = closestKPerformance;

        pp.staticModel = staticPerformanceModel(
            M, N, K, NumBatches, MT0, MT1, NumCUs, pp.granularities.totalGranularity, GlobalSplitU);

        pp.speedGFlops = IdealGranularityPerf * pp.granularities.totalGranularity;
        pp.CUs         = NumCUs;

        return pp;
    }

    double ContractionSolution::calculateDimensionM(Problem const& problem) const
    {
        double M = 1.0;
        if(problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesA(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                M *= problem.a().sizes()[*pi];
        }
        else
        {
            M = problem.freeSizeA(0);
        }
        return M;
    }

    double ContractionSolution::calculateDimensionN(Problem const& problem) const
    {
        double N = 1.0;
        if(problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesB(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                N *= problem.b().sizes()[*pi];
        }
        else
            N = problem.freeSizeB(0);
        return N;
    }

    double ContractionSolution::calculateNumBatches(Problem const& problem) const
    {
        double NumBatches = 1;
        if(sizeMapping.packBatchDims == 0)
        {
            for(size_t i = 0; i < problem.batchIndices().size(); i++)
                NumBatches *= problem.batchSize(i);
        }
        return NumBatches;
    }

    origami::data_type_t ContractionSolution::getOrigamiDatatype(Problem const& problem) const
    {
        return datatypeToAnalyticalDatatype(problem.computeInputTypeA());
    }

    std::ostream& operator<<(std::ostream&                                      stream,
                             ContractionSolution::StaticPerformanceModel const& spm)
    {
        return stream << " memReadBytesA=" << spm.memReadBytesA
                      << " memReadBytesB=" << spm.memReadBytesB
                      << " memReadBytesC=" << spm.memReadBytesC
                      << " memWriteBytesD=" << spm.memWriteBytesD;
    }

    std::ostream& operator<<(std::ostream&                                    stream,
                             ContractionSolution::ProjectedPerformance const& pp)
    {
        return stream << " numTiles0=" << pp.granularities.numTiles0
                      << " numTiles1=" << pp.granularities.numTiles1
                      << " tilesPerCu=" << pp.granularities.tilesPerCu

                      << " totalGranularity=" << pp.granularities.totalGranularity
                      << " tile0Granularity=" << pp.granularities.tile0Granularity
                      << " tile1Granularity=" << pp.granularities.tile1Granularity
                      << " cuGranularity=" << pp.granularities.cuGranularity
                      << " waveGranularity=" << pp.granularities.waveGranularity

                      << " speedGFlops=" << pp.speedGFlops

                      << " staticModel=[ " << pp.staticModel << " ]";
    }

    std::ostream& operator<<(std::ostream& stream, BufferLoadCheckPacket const& st)
    {
        return stream << " shiftPtrElemA=" << st.shiftPtrElemA
                      << " shiftPtrElemB=" << st.shiftPtrElemB << " depthUorMT0=" << st.depthUorMT0
                      << " depthUorMT1=" << st.depthUorMT1;
    }
} // namespace TensileLite
