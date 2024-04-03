// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
/// \tparam T - the input type.
/// \tparam ItemsPerThread - the number of items contributed by each thread.
/// \tparam WarpSize - the number of threads in a warp.
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
template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize = ::rocprim::device_warp_size()
>
class warp_exchange
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    struct storage_type_
    {
        T buffer[WarpSize * ItemsPerThread];
    };

    template<int NumEntries, int IdX, class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE void Foreach(const T (&input)[ItemsPerThread],
                                               U (&output)[ItemsPerThread],
                                               const int xor_bit_set)
    {
        // To prevent double work for IdX and IdX + NumEntries
        if(NumEntries != 0 && (IdX / NumEntries) % 2 == 0)
        {
            const T send_val = (xor_bit_set ? input[IdX] : input[IdX + NumEntries]);
            const T recv_val
                = ::rocprim::detail::warp_swizzle_shuffle(send_val, NumEntries, WarpSize);
            (xor_bit_set ? output[IdX] : output[IdX + NumEntries]) = recv_val;
        }
    }

    template<int NumEntries, class U, int... Ids>
    ROCPRIM_DEVICE ROCPRIM_INLINE void Foreach(const T (&input)[ItemsPerThread],
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
    ROCPRIM_DEVICE ROCPRIM_INLINE void TransposeImpl(const T (&input)[ItemsPerThread],
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
    ROCPRIM_DEVICE ROCPRIM_INLINE void Transpose(const T (&input)[ItemsPerThread],
                                                 U (&output)[ItemsPerThread],
                                                 const unsigned int lane_id)
    {
        constexpr unsigned int n_iter = rocprim::Log2<ItemsPerThread>::VALUE;
        TransposeImpl<n_iter - 1>(input,
                                  output,
                                  lane_id,
                                  std::make_integer_sequence<int, n_iter>{});
    }

    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        blocked_striped_shuffle_efficient_impl(const T (&input)[ItemsPerThread],
                                               U (&output)[ItemsPerThread])
    {
        static constexpr bool IS_ARCH_WARP = WarpSize == ::rocprim::device_warp_size();
        const unsigned int    flat_lane_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        const unsigned int    lane_id = IS_ARCH_WARP ? flat_lane_id : (flat_lane_id % WarpSize);
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

    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        blocked_to_striped_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        U                  work_array[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int dst_idx = 0; dst_idx < ItemsPerThread; dst_idx++)
        {
            ROCPRIM_UNROLL
            for(unsigned int src_idx = 0; src_idx < ItemsPerThread; src_idx++)
            {
                const auto value = ::rocprim::warp_shuffle(
                    input[src_idx],
                    flat_id / ItemsPerThread + dst_idx * (WarpSize / ItemsPerThread),
                    WarpSize);
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

    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE void
        striped_to_blocked_shuffle_impl(const T (&input)[ItemsPerThread],
                                        U (&output)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        U                  work_array[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int dst_idx = 0; dst_idx < ItemsPerThread; dst_idx++)
        {
            ROCPRIM_UNROLL
            for(unsigned int src_idx = 0; src_idx < ItemsPerThread; src_idx++)
            {
                const auto value
                    = ::rocprim::warp_shuffle(input[src_idx],
                                              (ItemsPerThread * flat_id + dst_idx) % WarpSize,
                                              WarpSize);
                if(flat_id / (WarpSize / ItemsPerThread) == src_idx)
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
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
    using storage_type = detail::raw_storage<storage_type_>;
    #else
    using storage_type = storage_type_; // only for Doxygen
    #endif

    /// \brief Transposes a blocked arrangement of items to a striped arrangement
    /// across the warp, using temporary storage.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
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
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[flat_id * ItemsPerThread + i] = input[i];
        }
        ::rocprim::wave_barrier();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[i * WarpSize + flat_id];
        }
    }

    /// \brief Transposes a blocked arrangement of items to a striped arrangement
    /// across the warp, using warp shuffle operations.
    /// Uses an optimized implementation for when WarpSize is equal to ItemsPerThread.
    /// Caution: this API is experimental. Performance might not be consistent.
    /// ItemsPerThread must be a divisor of WarpSize.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE void blocked_to_striped_shuffle(const T (&input)[ItemsPerThread],
                                                                  U (&output)[ItemsPerThread])
    {
        static_assert(
            WarpSize % ItemsPerThread == 0,
            "ItemsPerThread must be a divisor of WarpSize to use blocked_to_striped_shuffle");
        if(WarpSize == ItemsPerThread)
        {
            blocked_striped_shuffle_efficient_impl(input, output);
        }
        else
        {
            blocked_to_striped_shuffle_impl(input, output);
        }
    }

    /// \brief Transposes a striped arrangement of items to a blocked arrangement
    /// across the warp, using temporary storage.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
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
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[i * WarpSize + flat_id] = input[i];
        }
        ::rocprim::wave_barrier();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[flat_id * ItemsPerThread + i];
        }
    }

    /// \brief Transposes a striped arrangement of items to a blocked arrangement
    /// across the warp, using warp shuffle operations.
    /// Uses an optimized implementation for when WarpSize is equal to ItemsPerThread.
    /// Caution: this API is experimental. Performance might not be consistent.
    /// ItemsPerThread must be a divisor of WarpSize.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE void striped_to_blocked_shuffle(const T (&input)[ItemsPerThread],
                                                                  U (&output)[ItemsPerThread])
    {
        static_assert(
            WarpSize % ItemsPerThread == 0,
            "ItemsPerThread must be a divisor of WarpSize to use striped_to_blocked_shuffle");

        if(WarpSize == ItemsPerThread)
        {
            blocked_striped_shuffle_efficient_impl(input, output);
        }
        else
        {
            striped_to_blocked_shuffle_impl(input, output);
        }
    }

    /// \brief Orders \p input values according to ranks using temporary storage,
    /// then writes the values to \p output in a striped manner.
    /// No values in \p ranks should exists that exceed \p WarpSize*ItemsPerThread-1 .
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] ranks - array containing the positions.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
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
    void scatter_to_striped(
            const T (&input)[ItemsPerThread],
            U (&output)[ItemsPerThread],
            const OffsetT (&ranks)[ItemsPerThread],
            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        ROCPRIM_UNROLL
        for (unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[ranks[i]] = input[i];
        }
        ::rocprim::wave_barrier();

        ROCPRIM_UNROLL
        for (unsigned int i = 0; i < ItemsPerThread; i++)
        {
            unsigned int item_offset = (i * WarpSize) + flat_id;
            output[i] = storage_.buffer[item_offset];
        }
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_EXCHANGE_HPP_
