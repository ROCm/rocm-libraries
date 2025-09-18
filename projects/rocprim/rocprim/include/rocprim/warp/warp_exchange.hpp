// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_WARP_WARP_EXCHANGE_HPP_
#define ROCPRIM_WARP_WARP_EXCHANGE_HPP_

#include <cassert>
#include <type_traits>
#include <utility>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../functional.hpp"
#include "../intrinsics.hpp"
#include "../intrinsics/warp_shuffle.hpp"
#include "../types.hpp"
#include <rocprim/functional.hpp>
#include <rocprim/intrinsics/thread.hpp>

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief The \p warp_exchange class is a warp level parallel primitive which provides
/// methods for rearranging items partitioned across threads in a warp.
///
/// \tparam T the input type.
/// \tparam ItemsPerThread the number of items contributed by each thread.
/// \tparam VirtualWaveSize the number of threads in a warp.
///
/// \par Overview
/// * The \p warp_exchange class supports the following rearrangement methods:
///   * Transposing a blocked arrangement to a striped arrangement.
///   * Transposing a striped arrangement to a blocked arrangement.
///
/// \par Examples
/// \parblock
/// In the example an exchange operation is performed on a warp of 8 threads, using type
/// \p int with 4 items per thread.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     constexpr unsigned int threads_per_block = 128;
///     constexpr unsigned int threads_per_warp  =   8;
///     constexpr unsigned int items_per_thread  =   4;
///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
///     // specialize warp_exchange for int, warp of 8 threads and 4 items per thread
///     using warp_exchange_int = rocprim::warp_exchange<int, items_per_thread, threads_per_warp>;
///     // allocate storage in shared memory
///     __shared__ warp_exchange_int::storage_type storage[warps_per_block];
///
///     int items[items_per_thread];
///     ...
///     warp_exchange_int w_exchange;
///     w_exchange.blocked_to_striped(items, items, storage[warp_id]);
///     ...
/// }
/// \endcode
/// \endparblock
template<class T,
         unsigned int ItemsPerThread,
         unsigned int VirtualWaveSize = ::rocprim::arch::wavefront::min_size(),
         ::rocprim::arch::wavefront::target TargetWaveSize
         = ::rocprim::arch::wavefront::get_target()>
class warp_exchange
{
    static_assert(::rocprim::detail::is_power_of_two(VirtualWaveSize),
                  "Logical warp size must be a power of two.");

public:
    ROCPRIM_INLINE ROCPRIM_HOST_DEVICE warp_exchange()
    {
        detail::check_virtual_wave_size<VirtualWaveSize>();
    }

private:
    struct storage_type_
    {
        uninitialized_array<T, VirtualWaveSize * ItemsPerThread> buffer;
    };

    template<int NumEntries, int IdX, class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void Foreach(const T (&input)[ItemsPerThread],
                 U (&output)[ItemsPerThread],
                 const int xor_bit_set)
    {
        // To prevent double work for IdX and IdX + NumEntries
        if(NumEntries != 0 && (IdX / NumEntries) % 2 == 0)
        {
            const T send_val = (xor_bit_set ? input[IdX] : input[IdX + NumEntries]);
            const T recv_val
                = ::rocprim::detail::warp_swizzle_shuffle(send_val, NumEntries, VirtualWaveSize);
            (xor_bit_set ? output[IdX] : output[IdX + NumEntries]) = recv_val;
        }
    }

    template<int NumEntries, class U, int... Ids>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void Foreach(const T (&input)[ItemsPerThread],
                 U (&output)[ItemsPerThread],
                 const std::integer_sequence<int, Ids...>,
                 const bool xor_bit_set)
    {
        // To create a static inner loop that executes the code with
        // the values [0, 1, ..., ItemsPerThread-1, ItemsPerThread] as IdX
        int ignored[] = {((Foreach<NumEntries, Ids>(input, output, xor_bit_set)), 0)...};
        (void)ignored;
    }

    template<unsigned int MaxIter, class U, int... Iter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void TransposeImpl(const T (&input)[ItemsPerThread],
                       U (&output)[ItemsPerThread],
                       const unsigned int lane_id,
                       const std::integer_sequence<int, Iter...>)
    {
        // To create a static outer loop that executes the code with
        // the values [ItemsPerThread/2, ItemsPerThread/4, ..., 1, 0] as NumEntries
        int ignored[]
            = {(Foreach<(1 << (MaxIter - Iter))>(input,
                                                 output,
                                                 std::make_integer_sequence<int, ItemsPerThread>{},
                                                 lane_id & (1 << (MaxIter - Iter))),
                0)...};
        (void)ignored;
    }

    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void Transpose(const T (&input)[ItemsPerThread],
                   U (&output)[ItemsPerThread],
                   const unsigned int lane_id)
    {
        constexpr unsigned int n_iter = rocprim::Log2<ItemsPerThread>::VALUE;
        TransposeImpl<n_iter - 1>(input,
                                  output,
                                  lane_id,
                                  std::make_integer_sequence<int, n_iter>{});
    }

    template<unsigned int Width, typename U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    U warp_rotate_right(const U& input, const unsigned int n)
    {
        const int lane_id = ::rocprim::lane_id();

        // Calculate which lane to get data from
        // For right rotation by n in width w:
        // if we're at position p, we want data from (p + n) % w
        const int warp_base    = (lane_id / Width) * Width;
        const int logical_lane = lane_id % Width;
        const int src_lane     = warp_base + ((logical_lane - n + Width) % Width);
        return ::rocprim::warp_shuffle(input, src_lane, Width);
    }

    template<unsigned int Width, typename U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    U warp_rotate_left(const U& input, const unsigned int n)
    {
        const int lane_id = ::rocprim::lane_id();

        // Calculate which lane to get data from
        // For left rotation by n in width w:
        // if we're at position p, we want data from (p - n + w) % w
        const int warp_base    = (lane_id / Width) * Width;
        const int logical_lane = lane_id % Width;
        const int src_lane     = warp_base + ((logical_lane + n) % Width);
        return ::rocprim::warp_shuffle(input, src_lane, Width);
    }

    // Conditions for blocked to striped and striped to blocked
    struct conditions
    {
        static constexpr bool is_equal_size = VirtualWaveSize == ItemsPerThread;

        static constexpr bool is_quad_compatible_bs
            = ItemsPerThread % ROCPRIM_QUAD_SIZE == 0
              && ItemsPerThread % (VirtualWaveSize / ROCPRIM_QUAD_SIZE) == 0;

        static constexpr bool is_quad_compatible_sb
            = ItemsPerThread % ROCPRIM_QUAD_SIZE == 0
              && ItemsPerThread % (VirtualWaveSize / ROCPRIM_QUAD_SIZE) == 0
              // this config is not performant for the DPP quad_perm implementation
              && !(VirtualWaveSize == 64 && ItemsPerThread == 32);

        static constexpr bool warp_divide_items = ItemsPerThread % VirtualWaveSize == 0;

        static constexpr bool items_divide_warp = VirtualWaveSize % ItemsPerThread == 0;
    };

    enum class ImplementationType
    {
        Unknown         = 0,
        EqualSize       = 1,
        QuadCompatible  = 2,
        WarpDivideItems = 3,
        ItemsDivideWarp = 4
    };

    template<class U>
    struct implementation_selector_bs
    {
        static constexpr ImplementationType value
            = conditions::is_equal_size           ? ImplementationType::EqualSize
              : conditions::is_quad_compatible_bs ? ImplementationType::QuadCompatible
              : conditions::warp_divide_items     ? ImplementationType::WarpDivideItems
              : conditions::items_divide_warp     ? ImplementationType::ItemsDivideWarp
                                                  : ImplementationType::Unknown;
    };

    template<class U>
    struct implementation_selector_sb
    {
        static constexpr ImplementationType value
            = conditions::is_equal_size           ? ImplementationType::EqualSize
              : conditions::is_quad_compatible_sb ? ImplementationType::QuadCompatible
              : conditions::warp_divide_items     ? ImplementationType::WarpDivideItems
              : conditions::items_divide_warp     ? ImplementationType::ItemsDivideWarp
                                                  : ImplementationType::Unknown;
    };

    // Rearrangement of items for blocked to striped and striped to blocked
    // algorithms
    template<unsigned int Width, bool BlockedToStriped, class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void rearrange_items(unsigned int flat_id,
                         U (&input)[ItemsPerThread],
                         U (&output)[ItemsPerThread])
    {
        constexpr unsigned int ipt_div_width = ItemsPerThread / Width;

        unsigned int m = flat_id % Width;
        unsigned int n = 0;
        ROCPRIM_UNROLL
        while(n < ItemsPerThread)
        {
            ROCPRIM_UNROLL
            for(unsigned int j = 0; j < ipt_div_width; j++)
            {
                unsigned int dst_idx = BlockedToStriped ? n : m + j * Width;
                unsigned int src_idx = BlockedToStriped ? m + j * Width : n;

                output[dst_idx] = input[src_idx];

                n++;
            }
            m = (m - 1) % Width;
        }
    }

    // Case 1: Equal size
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<implementation_selector_bs<U>::value
                            == ImplementationType::EqualSize>::type
        blocked_to_striped_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        const unsigned int    lane_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        T                     temp[ItemsPerThread];
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            temp[i] = input[i];
        }
        Transpose(temp, temp, lane_id);
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = temp[i];
        }
    }

    // Case 2: Quad compatible
    // Works only when ItemsPerThread % ROCPRIM_QUAD_SIZE == 0 &&
    //                 ItemsPerThread % (VirtualWaveSize / ROCPRIM_QUAD_SIZE) == 0
    //
    // FIRST PART: Going from blocked to striped at the quad level using DPP quad_perm and
    //             item rearrangements
    //
    // The following is an example with IPT = 8. The patter can be extended to any multiple of
    // 4 items per thread. In the first rotation, we will have IPT/4 consecutive times the rotation
    // pattern x0-x1-x2-x3. In the item rearrangement, items are rearranged to be in ascending order.
    // In the second rotation, each rotation will happen IPT/4 times before decreasing by one until 0.
    //
    // First quad-level rotation         Item rearrangement
    // ----------------------------      ------------------
    // | 0   8   16  24 |                | 0   8   16  24 |
    // | 1   9   17  25 |  x1  0x93      | 25  1   9   17 |
    // | 2   10  18  26 |  x2  0x4E      | 18  26  2   10 |
    // | 3   11  19  27 |  x3  0x39      | 11  19  27  3  |
    // | 4   12  20  28 |                | 4   12  20  28 |
    // | 5   13  21  29 |  x1  0x93      | 29  5   13  21 |
    // | 6   14  22  30 |  x2  0x4E      | 22  30  6   14 |
    // | 7   15  23  31 |  x3  0x39      | 15  23  31  7  |

    // Second quad-level rotation        Goal Order
    // ---------------------------       ------------------
    // | 0   1   2   3  |                | 0   1   2   3  |
    // | 4   5   6   7  |                | 4   5   6   7  |
    // | 11  8   9   10 | x3  0x93       | 8   9   10  11 |
    // | 15  12  13  14 | x3  0x93       | 12  13  14  15 |
    // | 18  19  16  17 | x2  0x4E       | 16  17  18  19 |
    // | 22  23  20  21 | x2  0x4E       | 20  21  22  23 |
    // | 25  26  27  24 | x1  0x39       | 24  25  26  27 |
    // | 29  30  31  28 | x1  0x39       | 28  29  30  31 |

    // SECOND PART: Going from blocked to striped at the warp level using shuffle and item rearrangements.
    // We follow the same permutations as in part one but each rotation is mutiplied by 4. We basically
    // abstract one item to a stripe of 4 items and follow the same steps.
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<implementation_selector_bs<U>::value
                            == ImplementationType::QuadCompatible>::type
        blocked_to_striped_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        constexpr unsigned int NUM_QUADS         = VirtualWaveSize / ROCPRIM_QUAD_SIZE;
        constexpr unsigned int IPT_DIV_QUAD_SIZE = ItemsPerThread / ROCPRIM_QUAD_SIZE;

        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();

        U values1[ItemsPerThread];
        U values2[ItemsPerThread];
        U values3[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            values1[i] = input[i];
        }

        // First quad-level permutations
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            unsigned int i_mod4 = i % 4;
            switch(i_mod4)
            {
                case 1: // quad_perm:[3,0,1,2]
                    values1[i] = ::rocprim::detail::warp_move_dpp<U, 0x93>(values1[i]);
                    break;
                case 2: // quad_perm:[2,3,0,1]
                    values1[i] = ::rocprim::detail::warp_move_dpp<U, 0x4E>(values1[i]);
                    break;
                case 3: // quad_perm:[1,2,3,0]
                    values1[i] = ::rocprim::detail::warp_move_dpp<U, 0x39>(values1[i]);
                    break;
            }
        }

        rearrange_items<ROCPRIM_QUAD_SIZE, true>(flat_id, values1, values2);

        // Second quad-level permutations
        unsigned int i               = IPT_DIV_QUAD_SIZE;
        unsigned int quad_perm_index = 2;
        while(i < ItemsPerThread)
        {
            ROCPRIM_UNROLL
            for(unsigned int j = 0; j < IPT_DIV_QUAD_SIZE; j++)
            {
                switch(quad_perm_index)
                {
                    case 2: // quad_perm:[1,2,3,0]
                        values2[i] = ::rocprim::detail::warp_move_dpp<U, 0x39>(values2[i]);
                        break;
                    case 1: // quad_perm:[2,3,0,1]
                        values2[i] = ::rocprim::detail::warp_move_dpp<U, 0x4E>(values2[i]);
                        break;
                    case 0: // quad_perm:[3,0,1,2]
                        values2[i] = ::rocprim::detail::warp_move_dpp<U, 0x93>(values2[i]);
                        break;
                }
                i++;
            }
            quad_perm_index--;
        }

        if constexpr(VirtualWaveSize > ROCPRIM_QUAD_SIZE)
        {
            // First warp rotation
            ROCPRIM_UNROLL
            for(unsigned int i = 1; i < ItemsPerThread; i++)
            {
                unsigned int i_mod_quad_warp = i % NUM_QUADS;
                if(i_mod_quad_warp != 0)
                {
                    const unsigned int total_rotation = i_mod_quad_warp * ROCPRIM_QUAD_SIZE;
                    values2[i] = warp_rotate_right<VirtualWaveSize>(values2[i], total_rotation);
                }
            }

            rearrange_items<NUM_QUADS, true>(flat_id / ROCPRIM_QUAD_SIZE, values2, values3);

            // Second warp rotation
            constexpr unsigned int items_per_quad_warp = ItemsPerThread / NUM_QUADS;
            unsigned int           remaining_rotations = NUM_QUADS - 1;
            ROCPRIM_UNROLL
            for(unsigned int i = items_per_quad_warp; i < ItemsPerThread; i += items_per_quad_warp)
            {
                ROCPRIM_UNROLL
                for(unsigned int j = 0; j < items_per_quad_warp && (i + j) < ItemsPerThread; j++)
                {
                    const unsigned int rotation = remaining_rotations * ROCPRIM_QUAD_SIZE;
                    values3[i + j] = warp_rotate_right<VirtualWaveSize>(values3[i + j], rotation);
                }
                remaining_rotations--;
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = values3[i];
        }
    }

    // Case 3: Warp divides items
    // Similar logic of case 2 but only at a warp level using only shuffle
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<implementation_selector_bs<U>::value
                            == ImplementationType::WarpDivideItems>::type
        blocked_to_striped_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        const unsigned int     flat_id      = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        constexpr unsigned int ipt_div_warp = ItemsPerThread / VirtualWaveSize;
        U                      values1[ItemsPerThread];
        U                      values2[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            values1[i] = input[i];
        }

        // First warp rotation
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            const unsigned int rotations = i % VirtualWaveSize;
            if(rotations != 0)
            {
                values1[i] = warp_rotate_right<VirtualWaveSize>(values1[i], rotations);
            }
        }

        rearrange_items<VirtualWaveSize, true>(flat_id, values1, values2);

        // Second warp rotation
        unsigned int rotations = VirtualWaveSize - 1;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i += ipt_div_warp)
        {
            ROCPRIM_UNROLL
            for(unsigned int j = 0; j < ipt_div_warp; j++)
            {
                values2[i] = warp_rotate_right<VirtualWaveSize>(values2[i], rotations);
                rotations--;
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = values2[i];
        }
    }

    // Case 4: Items divide warp
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<implementation_selector_bs<U>::value
                            == ImplementationType::ItemsDivideWarp>::type
        blocked_to_striped_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        U                  work_array[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int dst_idx = 0; dst_idx < ItemsPerThread; dst_idx++)
        {
            ROCPRIM_UNROLL
            for(unsigned int src_idx = 0; src_idx < ItemsPerThread; src_idx++)
            {
                const auto value = ::rocprim::warp_shuffle(
                    input[src_idx],
                    flat_id / ItemsPerThread + dst_idx * (VirtualWaveSize / ItemsPerThread),
                    VirtualWaveSize);
                if(src_idx == flat_id % ItemsPerThread)
                {
                    work_array[dst_idx] = value;
                }
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = work_array[i];
        }
    }

    // Case 1: Equal size
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<implementation_selector_sb<U>::value
                            == ImplementationType::EqualSize>::type
        striped_to_blocked_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        blocked_to_striped_shuffle_impl(input, output);
    }

    // Case 2: Quad compatible
    // Works only when ItemsPerThread % ROCPRIM_QUAD_SIZE == 0 &&
    //                 ItemsPerThread % (VirtualWaveSize / ROCPRIM_QUAD_SIZE) == 0
    //
    // The logic of this implementation is the inverse of blocked to striped.
    // Check comments of blocked to striped case 2 for more details.
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<implementation_selector_sb<U>::value
                            == ImplementationType::QuadCompatible>::type
        striped_to_blocked_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        constexpr unsigned int NUM_QUADS         = VirtualWaveSize / ROCPRIM_QUAD_SIZE;
        constexpr unsigned int IPT_DIV_QUAD_SIZE = ItemsPerThread / ROCPRIM_QUAD_SIZE;

        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();

        U values1[ItemsPerThread];
        U values2[ItemsPerThread];
        U values3[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            values1[i] = input[i];
            values2[i] = input[i];
        }

        if constexpr(VirtualWaveSize > ROCPRIM_QUAD_SIZE)
        {
            // First warp rotation
            constexpr unsigned int items_per_quad_warp = ItemsPerThread / NUM_QUADS;
            unsigned int           remaining_rotations = NUM_QUADS - 1;
            ROCPRIM_UNROLL
            for(unsigned int i = items_per_quad_warp; i < ItemsPerThread; i += items_per_quad_warp)
            {
                ROCPRIM_UNROLL
                for(unsigned int j = 0; j < items_per_quad_warp && (i + j) < ItemsPerThread; j++)
                {
                    const unsigned int rotation = remaining_rotations * ROCPRIM_QUAD_SIZE;
                    values1[i + j] = warp_rotate_left<VirtualWaveSize>(values1[i + j], rotation);
                }
                remaining_rotations--;
            }

            rearrange_items<NUM_QUADS, false>(flat_id / ROCPRIM_QUAD_SIZE, values1, values2);

            // Second warp rotation
            ROCPRIM_UNROLL
            for(unsigned int i = 1; i < ItemsPerThread; i++)
            {
                unsigned int i_mod_quad_warp = i % NUM_QUADS;
                if(i_mod_quad_warp != 0)
                {
                    const unsigned int total_rotation = i_mod_quad_warp * ROCPRIM_QUAD_SIZE;
                    values2[i] = warp_rotate_left<VirtualWaveSize>(values2[i], total_rotation);
                }
            }
        }

        // First quad-level permutations
        unsigned int i               = IPT_DIV_QUAD_SIZE;
        unsigned int quad_perm_index = 0;
        while(i < ItemsPerThread)
        {
            ROCPRIM_UNROLL
            for(unsigned int j = 0; j < IPT_DIV_QUAD_SIZE; j++)
            {
                switch(quad_perm_index)
                {
                    case 2: // quad_perm:[1,2,3,0]
                        values2[i] = ::rocprim::detail::warp_move_dpp<U, 0x39>(values2[i]);
                        break;
                    case 1: // quad_perm:[2,3,0,1]
                        values2[i] = ::rocprim::detail::warp_move_dpp<U, 0x4E>(values2[i]);
                        break;
                    case 0: // quad_perm:[3,0,1,2]
                        values2[i] = ::rocprim::detail::warp_move_dpp<U, 0x93>(values2[i]);
                        break;
                }
                i++;
            }
            quad_perm_index++;
        }

        rearrange_items<ROCPRIM_QUAD_SIZE, false>(flat_id, values2, values3);

        // Second quad-level permutations
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            unsigned int i_mod4 = i % 4;
            switch(i_mod4)
            {
                case 3: // quad_perm:[3,0,1,2]
                    values3[i] = ::rocprim::detail::warp_move_dpp<U, 0x93>(values3[i]);
                    break;
                case 2: // quad_perm:[2,3,0,1]
                    values3[i] = ::rocprim::detail::warp_move_dpp<U, 0x4E>(values3[i]);
                    break;
                case 1: // quad_perm:[1,2,3,0]
                    values3[i] = ::rocprim::detail::warp_move_dpp<U, 0x39>(values3[i]);
                    break;
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = values3[i];
        }
    }

    // Case 3: Warp divides items
    // Similar logic of case 2 but only at a warp level using only shuffle
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<implementation_selector_sb<U>::value
                            == ImplementationType::WarpDivideItems>::type
        striped_to_blocked_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        const unsigned int     flat_id      = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        constexpr unsigned int ipt_div_warp = ItemsPerThread / VirtualWaveSize;
        U                      values1[ItemsPerThread];
        U                      values2[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            values1[i] = input[i];
        }

        // First warp rotation
        unsigned int rotations = VirtualWaveSize - 1;
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i += ipt_div_warp)
        {
            ROCPRIM_UNROLL
            for(unsigned int j = 0; j < ipt_div_warp; j++)
            {
                values1[i] = warp_rotate_left<VirtualWaveSize>(values1[i], rotations);
                rotations--;
            }
        }

        rearrange_items<VirtualWaveSize, false>(flat_id, values1, values2);

        // Second warp rotation
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            const unsigned int rotations = i % VirtualWaveSize;
            if(rotations != 0)
            {
                values2[i] = warp_rotate_left<VirtualWaveSize>(values2[i], rotations);
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = values2[i];
        }
    }

    // Case 4: Items divide warp
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<implementation_selector_sb<U>::value
                            == ImplementationType::ItemsDivideWarp>::type
        striped_to_blocked_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        U                  work_array[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int dst_idx = 0; dst_idx < ItemsPerThread; dst_idx++)
        {
            ROCPRIM_UNROLL
            for(unsigned int src_idx = 0; src_idx < ItemsPerThread; src_idx++)
            {
                const auto value = ::rocprim::warp_shuffle(input[src_idx],
                                                           (ItemsPerThread * flat_id + dst_idx)
                                                               % VirtualWaveSize,
                                                           VirtualWaveSize);
                if(flat_id / (VirtualWaveSize / ItemsPerThread) == src_idx)
                {
                    work_array[dst_idx] = value;
                }
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = work_array[i];
        }
    }

public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by the related parallel primitive.
    ///
    /// Depending on the implementation the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    using storage_type = storage_type_; // only for Doxygen

    /// \brief Transposes a blocked arrangement of items to a striped arrangement
    /// across the warp, using temporary storage.
    ///
    /// \tparam U [inferred] the output type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     constexpr unsigned int threads_per_block = 128;
    ///     constexpr unsigned int threads_per_warp  =   8;
    ///     constexpr unsigned int items_per_thread  =   4;
    ///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
    ///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
    ///     // specialize warp_exchange for int, warp of 8 threads and 4 items per thread
    ///     using warp_exchange_int = rocprim::warp_exchange<int, items_per_thread, threads_per_warp>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_exchange_int::storage_type storage[warps_per_block];
    ///
    ///     int items[items_per_thread];
    ///     ...
    ///     warp_exchange_int w_exchange;
    ///     w_exchange.blocked_to_striped(items, items, storage[warp_id]);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer.emplace(flat_id * ItemsPerThread + i, input[i]);
        }
        ::rocprim::wave_barrier();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[i * VirtualWaveSize + flat_id];
        }
    }

    /// \brief Transposes a blocked arrangement of items to a striped arrangement
    /// across the warp, using warp shuffle operations.
    /// Uses an optimized implementation for when VirtualWaveSize is equal to ItemsPerThread.
    /// Caution: this API is experimental. Performance might not be consistent.
    /// One of these following conditions must be satisfied:
    ///     1. VirtualWaveSize is equal to ItemsPerThread
    ///     2. ItemsPerThread % ROCPRIM_QUAD_SIZE == 0 && ItemsPerThread % (VirtualWaveSize / ROCPRIM_QUAD_SIZE) == 0
    ///     3. ItemsPerThread is divisible by VirtualWaveSize
    ///     4. VirtualWaveSize is divisible by ItemsPerThread
    ///
    /// \tparam U [inferred] the output type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     constexpr unsigned int threads_per_block = 128;
    ///     constexpr unsigned int threads_per_warp  =   8;
    ///     constexpr unsigned int items_per_thread  =   4;
    ///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
    ///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
    ///     // specialize warp_exchange for int, warp of 8 threads and 4 items per thread
    ///     using warp_exchange_int = rocprim::warp_exchange<int, items_per_thread, threads_per_warp>;
    ///
    ///     int items[items_per_thread];
    ///     ...
    ///     warp_exchange_int w_exchange;
    ///     w_exchange.blocked_to_striped_shuffle(items, items);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_striped_shuffle(const T (&input)[ItemsPerThread], U (&output)[ItemsPerThread])
    {
        static_assert(
            conditions::is_equal_size || conditions::is_quad_compatible_bs
                || conditions::warp_divide_items || conditions::items_divide_warp,
            "Input constraints violated. Please see documentation for allowed configurations.");

        blocked_to_striped_shuffle_impl(input, output);
    }

    /// \brief Transposes a striped arrangement of items to a blocked arrangement
    /// across the warp, using temporary storage.
    ///
    /// \tparam U [inferred] the output type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     constexpr unsigned int threads_per_block = 128;
    ///     constexpr unsigned int threads_per_warp  =   8;
    ///     constexpr unsigned int items_per_thread  =   4;
    ///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
    ///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
    ///     // specialize warp_exchange for int, warp of 8 threads and 4 items per thread
    ///     using warp_exchange_int = rocprim::warp_exchange<int, threads_per_warp, items_per_thread>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_exchange_int::storage_type storage[warps_per_block];
    ///
    ///     int items[items_per_thread];
    ///     ...
    ///     warp_exchange_int w_exchange;
    ///     w_exchange.striped_to_blocked(items, items, storage[warp_id]);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer.emplace(i * VirtualWaveSize + flat_id, input[i]);
        }
        ::rocprim::wave_barrier();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[flat_id * ItemsPerThread + i];
        }
    }

    /// \brief Transposes a striped arrangement of items to a blocked arrangement
    /// across the warp, using warp shuffle operations.
    /// Uses an optimized implementation for when VirtualWaveSize is equal to ItemsPerThread.
    /// Caution: this API is experimental. Performance might not be consistent.
    /// One of these following conditions must be satisfied:
    ///     1. VirtualWaveSize is equal to ItemsPerThread
    ///     2. ItemsPerThread % ROCPRIM_QUAD_SIZE == 0 && ItemsPerThread % (VirtualWaveSize / ROCPRIM_QUAD_SIZE) == 0
    ///     3. ItemsPerThread is divisible by VirtualWaveSize
    ///     4. VirtualWaveSize is divisible by ItemsPerThread
    ///
    /// \tparam U [inferred] the output type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     constexpr unsigned int threads_per_block = 128;
    ///     constexpr unsigned int threads_per_warp  =   8;
    ///     constexpr unsigned int items_per_thread  =   4;
    ///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
    ///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
    ///     // specialize warp_exchange for int, warp of 8 threads and 4 items per thread
    ///     using warp_exchange_int = rocprim::warp_exchange<int, items_per_thread, threads_per_warp>;
    ///
    ///     int items[items_per_thread];
    ///     ...
    ///     warp_exchange_int w_exchange;
    ///     w_exchange.striped_to_blocked_shuffle(items, items);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void striped_to_blocked_shuffle(const T (&input)[ItemsPerThread], U (&output)[ItemsPerThread])
    {
        static_assert(
            conditions::is_equal_size || conditions::is_quad_compatible_sb
                || conditions::warp_divide_items || conditions::items_divide_warp,
            "Input constraints violated. Please see documentation for allowed configurations.");

        striped_to_blocked_shuffle_impl(input, output);
    }

    /// \brief Orders \p input values according to ranks using temporary storage,
    /// then writes the values to \p output in a striped manner.
    /// No values in \p ranks should exists that exceed \p VirtualWaveSize*ItemsPerThread-1 .
    /// \tparam U [inferred] the output type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [in] ranks array containing the positions.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     constexpr unsigned int threads_per_block = 128;
    ///     constexpr unsigned int threads_per_warp  =   8;
    ///     constexpr unsigned int items_per_thread  =   4;
    ///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
    ///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
    ///     // specialize warp_exchange for int, warp of 8 threads and 4 items per thread
    ///     using warp_exchange_int = rocprim::warp_exchange<int, items_per_thread, threads_per_warp>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_exchange_int::storage_type storage[warps_per_block];
    ///
    ///     int items[items_per_thread];
    ///
    ///     // data-type of `ranks` should be able to contain warp_size*items_per_thread unique elements
    ///     // unsigned short is sufficient for up to 1024*64 elements
    ///     unsigned short ranks[items_per_thread];
    ///     ...
    ///     warp_exchange_int w_exchange;
    ///     w_exchange.scatter_to_striped(items, items, ranks, storage[warp_id]);
    ///     ...
    /// }
    /// \endcode
    template<class U, class OffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const OffsetT (&ranks)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer.emplace(ranks[i], input[i]);
        }
        ::rocprim::wave_barrier();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            unsigned int item_offset = (i * VirtualWaveSize) + flat_id;
            output[i]                = storage_buffer[item_offset];
        }
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<typename T, unsigned int ItemsPerThread, unsigned int VirtualWaveSize>
class warp_exchange<T, ItemsPerThread, VirtualWaveSize, ::rocprim::arch::wavefront::target::dynamic>
{
private:
    using warp_exchange_wave32 = warp_exchange<T,
                                               ItemsPerThread,
                                               VirtualWaveSize,
                                               ::rocprim::arch::wavefront::target::size32>;
    using warp_exchange_wave64 = warp_exchange<T,
                                               ItemsPerThread,
                                               VirtualWaveSize,
                                               ::rocprim::arch::wavefront::target::size64>;
    using dispatch
        = ::rocprim::detail::dispatch_wave_size<warp_exchange_wave32, warp_exchange_wave64>;

public:
    using storage_type = typename dispatch::storage_type;

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto blocked_to_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.blocked_to_striped(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto blocked_to_striped_shuffle(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.blocked_to_striped_shuffle(args...); },
                   args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto striped_to_blocked(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.striped_to_blocked(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto striped_to_blocked_shuffle(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.striped_to_blocked_shuffle(args...); },
                   args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scatter_to_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.scatter_to_striped(args...); }, args...);
    }
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_EXCHANGE_HPP_
