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
#ifndef ROCWMMA_TEST_WMMA_TARGET_BACKEND_HPP
#define ROCWMMA_TEST_WMMA_TARGET_BACKEND_HPP

#include <rocwmma/internal/mma_selector.hpp>
#include <rocwmma/internal/wmma_impl.hpp>

namespace rocwmma
{
    // Test-only: binds an explicit target into the six-parameter backend that
    // MmaSelector accepts, so the production selection algorithm can be
    // exercised for any target from a single (e.g. host) compilation pass.
    template <uint32_t TargetId>
    struct wmma_backend_for
    {
        template <typename InputTA,
                  typename InputTB,
                  typename ComputeT,
                  uint32_t BlockM,
                  uint32_t BlockN,
                  uint32_t BlockK>
        using type
            = detail::amdgcn_wmma<InputTA, InputTB, ComputeT, BlockM, BlockN, BlockK, TargetId>;
    };

    template <uint32_t TargetId,
              typename InputTA,
              typename InputTB,
              typename ComputeT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockKTest>
    using SelectedWmmaFor = typename MmaSelector<wmma_backend_for<TargetId>::template type,
                                                 InputTA,
                                                 InputTB,
                                                 ComputeT,
                                                 BlockM,
                                                 BlockN,
                                                 BlockKTest>::SelectedOp;

} // namespace rocwmma

#endif // ROCWMMA_TEST_WMMA_TARGET_BACKEND_HPP
