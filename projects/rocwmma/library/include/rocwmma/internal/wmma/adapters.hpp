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
#ifndef ROCWMMA_WMMA_ADAPTERS_HPP
#define ROCWMMA_WMMA_ADAPTERS_HPP

#include "../vector_util.hpp"
#include "descriptor.hpp"

namespace rocwmma
{
    namespace detail
    {
        // Serves a requested K smaller than the native K of operation Op by
        // zero-extending A and B. C and D are unchanged, and the A and B registers
        // keep Op's element types with proportionally fewer elements:
        //     RequestedCount = NativeCount * BlockK / NativeK
        // Padding doubles K one step at a time until it reaches the native K.
        template <typename Op, uint32_t BlockK>
        struct wmma_pad_k
        {
        private:
            constexpr static uint32_t NativeK = Op::Key::BlockK;

            static_assert(BlockK > 0u && BlockK < NativeK && NativeK % BlockK == 0u
                              && ((NativeK / BlockK) & (NativeK / BlockK - 1u)) == 0u,
                          "rocWMMA: a padded K must be the native K divided by a power of two");

            template <typename RegsT>
            using rebind_t = VecT<typename VecTraits<RegsT>::DataT,
                                  VecTraits<RegsT>::size() * BlockK / NativeK>;

            static_assert((VecTraits<typename Op::ARegsT>::size() * BlockK) % NativeK == 0u
                              && (VecTraits<typename Op::BRegsT>::size() * BlockK) % NativeK == 0u,
                          "rocWMMA: padded A or B registers would not be whole");

            // The operation that serves twice this K
            using NextOp = conditional_t<BlockK * 2u == NativeK, Op, wmma_pad_k<Op, BlockK * 2u>>;

        public:
            using ARegsT = rebind_t<typename Op::ARegsT>;
            using BRegsT = rebind_t<typename Op::BRegsT>;
            using CRegsT = typename Op::CRegsT;
            using DRegsT = typename Op::DRegsT;

            ROCWMMA_DEVICE static inline auto
                exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                return NextOp::exec(concat(regsA, ARegsT{0}), concat(regsB, BRegsT{0}), regsC);
            }
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_ADAPTERS_HPP
