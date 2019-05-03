// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
#define ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief The \p block_exchange class is a block level parallel primitive which provides
/// methods for rearranging items partitioned across threads in a block.
///
/// \tparam T - the input type.
/// \tparam BlockSize - the number of threads in a block.
/// \tparam ItemsPerThread - the number of items contributed by each thread.
///
/// \par Overview
/// * The \p block_exchange class supports the following rearrangement methods:
///   * Transposing a blocked arrangement to a striped arrangement.
///   * Transposing a striped arrangement to a blocked arrangement.
///   * Transposing a blocked arrangement to a warp-striped arrangement.
///   * Transposing a warp-striped arrangement to a blocked arrangement.
///   * Scattering items to a blocked arrangement.
///   * Scattering items to a striped arrangement.
/// * Data is automatically be padded to ensure zero bank conflicts.
///
/// \par Examples
/// \parblock
/// In the examples exchange operation is performed on block of 128 threads, using type
/// \p int with 8 items per thread.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
///     // allocate storage in shared memory
///     __shared__ block_exchange_int::storage_type storage;
///
///     int items[8];
///     ...
///     block_exchange_int b_exchange;
///     b_exchange.blocked_to_striped(items, items, storage);
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
class block_exchange
{
    // Select warp size
    static constexpr unsigned int warp_size =
        detail::get_min_warp_size(BlockSize, ::rocprim::warp_size());
    // Number of warps in block
    static constexpr unsigned int warps_no = (BlockSize + warp_size - 1) / warp_size;

    // Minimize LDS bank conflicts for power-of-two strides, i.e. when items accessed
    // using `thread_id * ItemsPerThread` pattern where ItemsPerThread is power of two
    // (all exchanges from/to blocked).
    static constexpr bool has_bank_conflicts =
        ItemsPerThread >= 2 && ::rocprim::detail::is_power_of_two(ItemsPerThread);
    static constexpr unsigned int banks_no = ::rocprim::detail::get_lds_banks_no();
    static constexpr unsigned int bank_conflicts_padding =
        has_bank_conflicts ? (BlockSize * ItemsPerThread / banks_no) : 0;

    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    struct storage_type_
    {
        T buffer[BlockSize * ItemsPerThread + bank_conflicts_padding];
    };

public:

    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
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
    /// across the thread block.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    template<class U>
    ROCPRIM_DEVICE inline
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        blocked_to_striped(input, output, storage);
    }

    /// \brief Transposes a blocked arrangement of items to a striped arrangement
    /// across the thread block, using temporary storage.
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
    ///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
    ///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_exchange_int::storage_type storage;
    ///
    ///     int items[8];
    ///     ...
    ///     block_exchange_int b_exchange;
    ///     b_exchange.blocked_to_striped(items, items, storage);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE inline
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(flat_id * ItemsPerThread + i)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(i * BlockSize + flat_id)];
        }
    }

    /// \brief Transposes a striped arrangement of items to a blocked arrangement
    /// across the thread block.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    template<class U>
    ROCPRIM_DEVICE inline
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        striped_to_blocked(input, output, storage);
    }

    /// \brief Transposes a striped arrangement of items to a blocked arrangement
    /// across the thread block, using temporary storage.
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
    ///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
    ///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_exchange_int::storage_type storage;
    ///
    ///     int items[8];
    ///     ...
    ///     block_exchange_int b_exchange;
    ///     b_exchange.striped_to_blocked(items, items, storage);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE inline
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(i * BlockSize + flat_id)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(flat_id * ItemsPerThread + i)];
        }
    }

    /// \brief Transposes a blocked arrangement of items to a warp-striped arrangement
    /// across the thread block.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    template<class U>
    ROCPRIM_DEVICE inline
    void blocked_to_warp_striped(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        blocked_to_warp_striped(input, output, storage);
    }

    /// \brief Transposes a blocked arrangement of items to a warp-striped arrangement
    /// across the thread block, using temporary storage.
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
    ///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
    ///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_exchange_int::storage_type storage;
    ///
    ///     int items[8];
    ///     ...
    ///     block_exchange_int b_exchange;
    ///     b_exchange.blocked_to_warp_striped(items, items, storage);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE inline
    void blocked_to_warp_striped(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread],
                                 storage_type& storage)
    {
        constexpr unsigned int items_per_warp = warp_size * ItemsPerThread;
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int current_warp_size = get_current_warp_size();
        const unsigned int offset = warp_id * items_per_warp;
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(offset + lane_id * ItemsPerThread + i)] = input[i];
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(offset + i * current_warp_size + lane_id)];
        }
    }

    /// \brief Transposes a warp-striped arrangement of items to a blocked arrangement
    /// across the thread block.
    ///
    /// \tparam U - [inferred] the output type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    template<class U>
    ROCPRIM_DEVICE inline
    void warp_striped_to_blocked(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        warp_striped_to_blocked(input, output, storage);
    }

    /// \brief Transposes a warp-striped arrangement of items to a blocked arrangement
    /// across the thread block, using temporary storage.
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
    ///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
    ///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_exchange_int::storage_type storage;
    ///
    ///     int items[8];
    ///     ...
    ///     block_exchange_int b_exchange;
    ///     b_exchange.warp_striped_to_blocked(items, items, storage);
    ///     ...
    /// }
    /// \endcode
    template<class U>
    ROCPRIM_DEVICE inline
    void warp_striped_to_blocked(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread],
                                 storage_type& storage)
    {
        constexpr unsigned int items_per_warp = warp_size * ItemsPerThread;
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id();
        const unsigned int current_warp_size = get_current_warp_size();
        const unsigned int offset = warp_id * items_per_warp;
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(offset + i * current_warp_size + lane_id)] = input[i];
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(offset + lane_id * ItemsPerThread + i)];
        }
    }

    /// \brief Scatters items to a blocked arrangement based on their ranks
    /// across the thread block.
    ///
    /// \tparam U - [inferred] the output type.
    /// \tparam Offset - [inferred] the rank type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [out] ranks - array that has rank of data.
    template<class U, class Offset>
    ROCPRIM_DEVICE inline
    void scatter_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        scatter_to_blocked(input, output, ranks, storage);
    }

    /// \brief Scatters items to a blocked arrangement based on their ranks
    /// across the thread block, using temporary storage.
    ///
    /// \tparam U - [inferred] the output type.
    /// \tparam Offset - [inferred] the rank type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [out] ranks - array that has rank of data.
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
    ///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
    ///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_exchange_int::storage_type storage;
    ///
    ///     int items[8];
    ///     int ranks[8];
    ///     ...
    ///     block_exchange_int b_exchange;
    ///     b_exchange.scatter_to_blocked(items, items, ranks, storage);
    ///     ...
    /// }
    /// \endcode
    template<class U, class Offset>
    ROCPRIM_DEVICE inline
    void scatter_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            storage_.buffer[index(rank)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(flat_id * ItemsPerThread + i)];
        }
    }

    /// \brief Scatters items to a striped arrangement based on their ranks
    /// across the thread block.
    ///
    /// \tparam U - [inferred] the output type.
    /// \tparam Offset - [inferred] the rank type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [out] ranks - array that has rank of data.
    template<class U, class Offset>
    ROCPRIM_DEVICE inline
    void scatter_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        scatter_to_striped(input, output, ranks, storage);
    }

    /// \brief Scatters items to a striped arrangement based on their ranks
    /// across the thread block, using temporary storage.
    ///
    /// \tparam U - [inferred] the output type.
    /// \tparam Offset - [inferred] the rank type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [out] ranks - array that has rank of data.
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
    ///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
    ///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_exchange_int::storage_type storage;
    ///
    ///     int items[8];
    ///     int ranks[8];
    ///     ...
    ///     block_exchange_int b_exchange;
    ///     b_exchange.scatter_to_striped(items, items, ranks, storage);
    ///     ...
    /// }
    /// \endcode
    template<class U, class Offset>
    ROCPRIM_DEVICE inline
    void scatter_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            storage_.buffer[rank] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[i * BlockSize + flat_id];
        }
    }

    /// \brief Scatters items to a striped arrangement based on their ranks
    /// across the thread block, guarded by rank.
    ///
    /// \par Overview
    /// * Items with rank -1 are not scattered.
    ///
    /// \tparam U - [inferred] the output type.
    /// \tparam Offset - [inferred] the rank type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] ranks - array that has rank of data.
    template<class U, class Offset>
    ROCPRIM_DEVICE inline
    void scatter_to_striped_guarded(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        scatter_to_striped_guarded(input, output, ranks, storage);
    }

    /// \brief Scatters items to a striped arrangement based on their ranks
    /// across the thread block, guarded by rank, using temporary storage.
    ///
    /// \par Overview
    /// * Items with rank -1 are not scattered.
    ///
    /// \tparam U - [inferred] the output type.
    /// \tparam Offset - [inferred] the rank type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] ranks - array that has rank of data.
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
    ///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
    ///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_exchange_int::storage_type storage;
    ///
    ///     int items[8];
    ///     int ranks[8];
    ///     ...
    ///     block_exchange_int b_exchange;
    ///     b_exchange.scatter_to_striped_guarded(items, items, ranks, storage);
    ///     ...
    /// }
    /// \endcode
    template<class U, class Offset>
    ROCPRIM_DEVICE inline
    void scatter_to_striped_guarded(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            if(rank >= 0)
            {
                storage_.buffer[rank] = input[i];
            }
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[i * BlockSize + flat_id];
        }
    }

    /// \brief Scatters items to a striped arrangement based on their ranks
    /// across the thread block, with a flag to denote validity.
    ///
    /// \tparam U - [inferred] the output type.
    /// \tparam Offset - [inferred] the rank type.
    /// \tparam ValidFlag - [inferred] the validity flag type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] ranks - array that has rank of data.
    /// \param [in] is_valid - array that has flags to denote validity.
    template<class U, class Offset, class ValidFlag>
    ROCPRIM_DEVICE inline
    void scatter_to_striped_flagged(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    const ValidFlag (&is_valid)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        scatter_to_striped_flagged(input, output, ranks, is_valid, storage);
    }

    /// \brief Scatters items to a striped arrangement based on their ranks
    /// across the thread block, with a flag to denote validity, using temporary
    /// storage.
    ///
    /// \tparam U - [inferred] the output type.
    /// \tparam Offset - [inferred] the rank type.
    /// \tparam ValidFlag - [inferred] the validity flag type.
    ///
    /// \param [in] input - array that data is loaded from.
    /// \param [out] output - array that data is loaded to.
    /// \param [in] ranks - array that has rank of data.
    /// \param [in] is_valid - array that has flags to denote validity.
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
    ///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
    ///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_exchange_int::storage_type storage;
    ///
    ///     int items[8];
    ///     int ranks[8];
    ///     int flags[8];
    ///     ...
    ///     block_exchange_int b_exchange;
    ///     b_exchange.scatter_to_striped_flagged(items, items, ranks, flags, storage);
    ///     ...
    /// }
    /// \endcode
    template<class U, class Offset, class ValidFlag>
    ROCPRIM_DEVICE inline
    void scatter_to_striped_flagged(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    const ValidFlag (&is_valid)[ItemsPerThread],
                                    storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            if(is_valid[i])
            {
                storage_.buffer[rank] = input[i];
            }
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[i * BlockSize + flat_id];
        }
    }

private:

    ROCPRIM_DEVICE inline
    unsigned int get_current_warp_size() const
    {
        const unsigned int warp_id = ::rocprim::warp_id();
        return (warp_id == warps_no - 1)
            ? (BlockSize % warp_size > 0 ? BlockSize % warp_size : warp_size)
            : warp_size;
    }

    // Change index to minimize LDS bank conflicts if necessary
    ROCPRIM_DEVICE inline
    unsigned int index(unsigned int n)
    {
        // Move every 32-bank wide "row" (32 banks * 4 bytes) by one item
        return has_bank_conflicts ? (n + n / banks_no) : n;
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
