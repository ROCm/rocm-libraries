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
#ifndef ROCWMMA_WMMA_DESCRIPTOR_HPP
#define ROCWMMA_WMMA_DESCRIPTOR_HPP

#include "../types.hpp"
#include "../utility/type_traits.hpp"

namespace rocwmma
{
    namespace detail
    {
        // Identifies a WMMA operation as seen by callers: input types, accumulator
        // type and block sizes.
        template <typename InputTA_In,
                  typename InputTB_In,
                  typename ComputeT_In,
                  uint32_t BlockM_In,
                  uint32_t BlockN_In,
                  uint32_t BlockK_In>
        struct wmma_key
        {
            using InputTA  = InputTA_In;
            using InputTB  = InputTB_In;
            using ComputeT = ComputeT_In;

            constexpr static uint32_t BlockM = BlockM_In;
            constexpr static uint32_t BlockN = BlockN_In;
            constexpr static uint32_t BlockK = BlockK_In;

            // The same operation with a different K
            template <uint32_t BlockK>
            using with_block_k
                = wmma_key<InputTA_In, InputTB_In, ComputeT_In, BlockM_In, BlockN_In, BlockK>;
        };

        // A set of K values, e.g. the smaller K values a native operation serves by
        // zero-padding A and B.
        template <uint32_t... Ks>
        struct wmma_k_list
        {
            constexpr static bool contains(uint32_t k)
            {
                return ((k == Ks) || ... || false);
            }
        };

        // A list of native operation descriptors, e.g. the capabilities of a target.
        template <typename... Ops>
        struct wmma_op_list
        {
        };

        // Lets a static_assert in a template depend on a template parameter.
        template <typename T>
        constexpr bool wmma_dependent_false_v = false;

        // Control operands of the WMMA builtins. Descriptors name the value they
        // pass at each position.
        enum struct WmmaInputSign : bool
        {
            Unsigned = false,
            Signed   = true
        };

        // Half of each 32-bit register that receives a 16-bit result in the
        // unpacked (gfx11) accumulator layout.
        enum struct WmmaAccumHalf : bool
        {
            Low  = false,
            High = true
        };

        enum struct WmmaReuse : bool
        {
            None  = false,
            Reuse = true
        };

        // A and B input modifier (gfx1250)
        enum struct WmmaInputNeg : bool
        {
            None   = false,
            Negate = true
        };

        // C input modifier (gfx1250)
        enum struct WmmaCMod : int16_t
        {
            None   = 0,
            Negate = 1,
            Abs    = 2,
            NegAbs = 3
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_DESCRIPTOR_HPP
