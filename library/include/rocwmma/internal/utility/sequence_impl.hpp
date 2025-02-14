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

#ifndef ROCWMMA_UTILITY_SEQUENCE_IMPL_HPP
#define ROCWMMA_UTILITY_SEQUENCE_IMPL_HPP

#include "type_traits.hpp"

namespace rocwmma
{
    namespace detail
    {
        template <typename Int, Int... Ints>
        struct integer_sequence
        {
            using value_type = Int;
            constexpr integer_sequence() {}
            static constexpr size_t size() noexcept
            {
                return sizeof...(Ints);
            }
        };

        template <size_t... Indices>
        using index_sequence = integer_sequence<size_t, Indices...>;

        namespace
        {
            // Merge two integer sequences, adding an offset to the right-hand side.
            template <typename Offset, typename Lhs, typename Rhs>
            struct merge;

            template <typename Int, Int Offset, Int... Lhs, Int... Rhs>
            struct merge<integral_constant<Int, Offset>,
                         integer_sequence<Int, Lhs...>,
                         integer_sequence<Int, Rhs...>>
            {
                using type = integer_sequence<Int, Lhs..., (Offset + Rhs)...>;
            };

            template <typename Int, typename N>
            struct log_make_sequence
            {
                using L    = integral_constant<Int, N::value / 2>;
                using R    = integral_constant<Int, N::value - L::value>;
                using type = typename merge<L,
                                            typename log_make_sequence<Int, L>::type,
                                            typename log_make_sequence<Int, R>::type>::type;
            };

            // An empty sequence.
            template <typename Int>
            struct log_make_sequence<Int, integral_constant<Int, 0>>
            {
                using type = integer_sequence<Int>;
            };

            // A single-element sequence.
            template <typename Int>
            struct log_make_sequence<Int, integral_constant<Int, 1>>
            {
                using type = integer_sequence<Int, 0>;
            };
        }

        template <typename Int, Int N>
        using make_integer_sequence =
            typename log_make_sequence<Int, integral_constant<Int, N>>::type;

        template <size_t N>
        using make_index_sequence = make_integer_sequence<size_t, N>;

        template <size_t Offset, size_t Stride, size_t... Is>
        ROCWMMA_HOST_DEVICE constexpr inline auto
            make_offset_index_sequence_helper(index_sequence<Is...>)
        {
            return index_sequence<(Offset + Is * Stride)...>{};
        }

        template <size_t N, size_t Offset = 0u, size_t Stride = 1u>
        ROCWMMA_HOST_DEVICE constexpr inline auto make_offset_index_sequence()
        {
            return make_offset_index_sequence_helper<Offset, Stride>(make_index_sequence<N>{});
        }

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_SEQUENCE_IMPL_HPP
