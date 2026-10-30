/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <gtest/gtest.h>

#include <rocwmma/rocwmma.hpp>

#include "wmma_target_backend.hpp"

// The characterization expectations hold only in the host pass: they were
// recorded there, and the unsupported register types depend on the wave size
// of the pass being compiled. Device passes check that dependency below.
#if !defined(__HIP_DEVICE_COMPILE__)
#include "wmma_characterization.hpp"
#else
namespace
{
    using namespace rocwmma;

    // Device-pass counterpart of the host-only unsupported register types: the
    // unsupported primary template sizes its registers from the wave size of the
    // pass being compiled, not from the explicit target, and sizes the
    // accumulator as BlockM * BlockM.
    template <typename Op, typename ARegs, typename BRegs, typename CRegs, typename DRegs>
    constexpr bool unsupportedWithRegs()
    {
        return !MmaTraits<Op>::is_supported
               && is_same_v<typename Op::ARegsT, ARegs> && is_same_v<typename Op::BRegsT, BRegs>
               && is_same_v<typename Op::CRegsT, CRegs> && is_same_v<typename Op::DRegsT, DRegs>;
    }

    // No target has a 16x32 WMMA, and gfx908 has no WMMA at all.
    using F16Rect     = detail::amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 32u, 16u>;
    using F16OnGfx908 = detail::amdgcn_wmma<float16_t,
                                            float16_t,
                                            float32_t,
                                            16u,
                                            16u,
                                            16u,
                                            Constants::AMDGCN_ARCH_ID_GFX908>;

#if ROCWMMA_WAVE32_MODE
    static_assert(Constants::AMDGCN_WAVE_SIZE == 32u);
    static_assert(unsupportedWithRegs<F16Rect,
                                      VecT<float32_t, 4>,
                                      VecT<float32_t, 8>,
                                      VecT<float32_t, 8>,
                                      VecT<float32_t, 8>>());
    static_assert(unsupportedWithRegs<F16OnGfx908,
                                      VecT<float32_t, 4>,
                                      VecT<float32_t, 4>,
                                      VecT<float32_t, 8>,
                                      VecT<float32_t, 8>>());
#else
    static_assert(Constants::AMDGCN_WAVE_SIZE == 64u);
    static_assert(unsupportedWithRegs<F16Rect,
                                      VecT<float32_t, 2>,
                                      VecT<float32_t, 4>,
                                      VecT<float32_t, 4>,
                                      VecT<float32_t, 4>>());
    static_assert(unsupportedWithRegs<F16OnGfx908,
                                      VecT<float32_t, 2>,
                                      VecT<float32_t, 2>,
                                      VecT<float32_t, 4>,
                                      VecT<float32_t, 4>>());
#endif
}
#endif

TEST(WmmaCharacterization, HostCompileTime)
{
    // All checks are static_asserts in wmma_characterization.hpp and are
    // evaluated when this TU is compiled (host pass). Reaching this point at
    // runtime means they all held.
    SUCCEED();
}
