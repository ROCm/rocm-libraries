/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_GEMM_KERNEL_BASE_DISPATCH_IMPL_HPP
#define ROCWMMA_GEMM_KERNEL_BASE_DISPATCH_IMPL_HPP

#include "gemm_kernel_base.hpp"
#include "helper_macros.hpp"

namespace rocwmma
{
    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    template <template <uint32_t, uint32_t, uint32_t, uint32_t> class TestGuard>
    bool GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::dispatchGuard() const
    {

        // The test guard will be dispatched against 4 runtime params:
        // - TBlockX [32, 64, 128, 256]
        // - TBlockY [1, 2, 4]
        // - Wave Size [32, 64]
        // - Arch [gfx908, gfx90a, gfx942, gfx950, gfx1100, gfx1101, gfx1102, gfx1200, gfx1201]
        auto dispatchGuardFunc = [this]() {
            bool dispatchResult = false;

            auto waveSize   = DeviceInfo::instance()->warpSize();
            auto deviceArch = DeviceInfo::instance()->getGcnArch();

#define CASE_IMPL_ASSIGN4(TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID) \
    dispatchResult = TestGuard<TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID>::enableRun();

#define SWITCH_BODY_TBLOCK_X(TBLOCK_Y, WAVE_SIZE, ARCH_ID) \
    ROCWMMA_SWITCH_BODY4_ARG4(                             \
        mTBlockX, CASE_IMPL_ASSIGN4, 32u, 64u, 128u, 256u, TBLOCK_Y, WAVE_SIZE, ARCH_ID)

#define SWITCH_BODY_TBLOCK_Y(WAVE_SIZE, ARCH_ID) \
    ROCWMMA_SWITCH_BODY3_ARG3(mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u, WAVE_SIZE, ARCH_ID)

#define SWITCH_BODY_WAVE_SIZE(ARCH_ID) \
    ROCWMMA_SWITCH_BODY2_ARG2(         \
        waveSize, SWITCH_BODY_TBLOCK_Y, HipDevice::Wave32, HipDevice::Wave64, ARCH_ID)

#define DISPATCH_GUARD_BODY                          \
    ROCWMMA_SWITCH_BODY9_ARG1(deviceArch,            \
                              SWITCH_BODY_WAVE_SIZE, \
                              HipDevice::GFX908,     \
                              HipDevice::GFX90A,     \
                              HipDevice::GFX942,     \
                              HipDevice::GFX950,     \
                              HipDevice::GFX1100,    \
                              HipDevice::GFX1101,    \
                              HipDevice::GFX1102,    \
                              HipDevice::GFX1200,    \
                              HipDevice::GFX1201)

            DISPATCH_GUARD_BODY

#undef CASE_IMPL_ASSIGN4
#undef SWITCH_BODY_TBLOCK_X
#undef SWITCH_BODY_TBLOCK_Y
#undef SWITCH_BODY_WAVE_SIZE
#undef DISPATCH_GUARD_BODY

            return dispatchResult;
        };

        // Finally, execute and return the dispatch guard result
        return dispatchGuardFunc();
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    template <template <uint32_t, uint32_t, uint32_t, uint32_t> class KernelClass>
    auto GemmKernelBase<BlockM,
                        BlockN,
                        BlockK,
                        InputT,
                        OutputT,
                        ComputeT,
                        LayoutA,
                        LayoutB,
                        LayoutC,
                        LayoutD>::dispatchKernelFunc() const -> KernelFunc
    {
        // The kernel function will be dispatched against 4 runtime params:
        // - TBlockX [32, 64, 128, 256]
        // - TBlockY [1, 2, 4]
        // - Wave Size [32, 64]
        // - Arch [gfx908, gfx90a, gfx940, gfx941, gfx942, gfx950 gfx1100, gfx1101, gfx1102, gfx1200, gfx1201]
        auto dispatchKernel = [this]() {
            auto waveSize   = DeviceInfo::instance()->warpSize();
            auto deviceArch = DeviceInfo::instance()->getGcnArch();

            // Runtime dispatcher to assign compile time TBlock params.
            auto result = typename std::decay_t<decltype(*this)>::KernelFunc(nullptr);

#define CASE_IMPL_ASSIGN4(TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID) \
    result = KernelClass<TBLOCK_X, TBLOCK_Y, WAVE_SIZE, ARCH_ID>::generate();

#define SWITCH_BODY_TBLOCK_X(TBLOCK_Y, WAVE_SIZE, ARCH_ID) \
    ROCWMMA_SWITCH_BODY4_ARG4(                             \
        mTBlockX, CASE_IMPL_ASSIGN4, 32u, 64u, 128u, 256u, TBLOCK_Y, WAVE_SIZE, ARCH_ID)

#define SWITCH_BODY_TBLOCK_Y(WAVE_SIZE, ARCH_ID) \
    ROCWMMA_SWITCH_BODY3_ARG3(mTBlockY, SWITCH_BODY_TBLOCK_X, 1u, 2u, 4u, WAVE_SIZE, ARCH_ID)

#define SWITCH_BODY_WAVE_SIZE(ARCH_ID) \
    ROCWMMA_SWITCH_BODY2_ARG2(         \
        waveSize, SWITCH_BODY_TBLOCK_Y, HipDevice::Wave32, HipDevice::Wave64, ARCH_ID)

#define DISPATCH_KERNEL_FUNC_BODY                    \
    ROCWMMA_SWITCH_BODY9_ARG1(deviceArch,            \
                              SWITCH_BODY_WAVE_SIZE, \
                              HipDevice::GFX908,     \
                              HipDevice::GFX90A,     \
                              HipDevice::GFX942,     \
                              HipDevice::GFX950,     \
                              HipDevice::GFX1100,    \
                              HipDevice::GFX1101,    \
                              HipDevice::GFX1102,    \
                              HipDevice::GFX1200,    \
                              HipDevice::GFX1201)

            DISPATCH_KERNEL_FUNC_BODY

#undef CASE_IMPL_ASSIGN4
#undef SWITCH_BODY_TBLOCK_X
#undef SWITCH_BODY_TBLOCK_Y
#undef SWITCH_BODY_WAVE_SIZE
#undef DISPATCH_KERNEL_FUNC_BODY

            return result;
        };

        return dispatchKernel();
    }

} // namespace rocwmma

#endif // ROCWMMA_GEMM_KERNEL_BASE_DISPATCH_IMPL_HPP
