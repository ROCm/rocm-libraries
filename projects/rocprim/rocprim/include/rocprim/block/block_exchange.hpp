// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../functional.hpp"
#include "../intrinsics.hpp"
#include "../intrinsics/arch.hpp"
#include "../types.hpp"

#include "config.hpp"

#include <cstddef>

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief The \p block_exchange class is a block level parallel primitive which provides
/// methods for rearranging items partitioned across threads in a block.
///
/// \tparam T the input type.
/// \tparam BlockSize the number of threads in a block.
/// \tparam ItemsPerThread the number of items contributed by each thread.
/// \tparam PaddingHint a hint that decides when to use padding. May not always be applicable.
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
template<class T,
         unsigned int                       BlockSizeX,
         unsigned int                       ItemsPerThread,
         unsigned int                       BlockSizeY  = 1,
         unsigned int                       BlockSizeZ  = 1,
         block_padding_hint                 PaddingHint = block_padding_hint::avoid_conflicts,
         ::rocprim::arch::wavefront::target TargetWaveSize
         = ::rocprim::arch::wavefront::get_target()>
class block_exchange
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    // Select warp size
    static constexpr unsigned int warp_size = ::rocprim::detail::get_min_warp_size(
        BlockSize, ::rocprim::arch::wavefront::size_from_target<TargetWaveSize>());
    // Number of warps in block
    static constexpr unsigned int warps_no = ::rocprim::detail::ceiling_div(BlockSize, warp_size);
    static constexpr unsigned int banks_no = ::rocprim::detail::get_lds_banks_no();
    static constexpr unsigned int buffer_size
        = static_cast<unsigned int>(rocprim::max(size_t{1}, size_t{4} / sizeof(T)));

    struct unpadded_config
    {
        static constexpr bool         has_bank_conflicts = false;
        static constexpr unsigned int padding            = 0;
    };

    struct padded_config
    {
        // Minimize LDS bank conflicts for power-of-two strides, i.e. when items accessed
        // using `thread_id * ItemsPerThread` pattern where ItemsPerThread is power of two
        // (all exchanges from/to blocked).
        static constexpr bool has_bank_conflicts
            = ItemsPerThread >= 2 && ::rocprim::detail::is_power_of_two(ItemsPerThread);
        static constexpr unsigned int padding
            = has_bank_conflicts ? (BlockSize * ItemsPerThread / banks_no) : 0;
    };

    template<typename Config>
    struct build_config : Config
    {
        static constexpr unsigned int storage_count = BlockSize * ItemsPerThread + Config::padding;
        static constexpr unsigned int storage_size  = sizeof(T) * storage_count;
        static constexpr unsigned int occupancy     = detail::get_min_lds_size() / storage_size;
    };

    using config = detail::select_block_padding_config<PaddingHint,
                                                       build_config<padded_config>,
                                                       build_config<unpadded_config>>;

    static constexpr bool         has_bank_conflicts     = config::has_bank_conflicts;
    static constexpr unsigned int bank_conflicts_padding = config::padding;
    static constexpr unsigned int storage_count          = config::storage_count;

    struct storage_type_
    {
        uninitialized_array<T, storage_count, 16> buffer;
    };

public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implementation the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    using storage_type = storage_type_;

    /// \brief Transposes a blocked arrangement of items to a striped arrangement
    /// across the thread block.
    ///
    /// \tparam U [inferred] the output type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        blocked_to_striped(input, output, storage);
    }

    /// \brief Transposes a blocked arrangement of items to a striped arrangement
    /// across the thread block, using temporary storage.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer.emplace(index(flat_id * ItemsPerThread + i), input[i]);
        }
        ::rocprim::syncthreads();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[index(i * BlockSize + flat_id)];
        }
    }

    /// \brief Transposes a striped arrangement of items to a blocked arrangement
    /// across the thread block.
    ///
    /// \tparam U [inferred] the output type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        striped_to_blocked(input, output, storage);
    }

    /// \brief Transposes a striped arrangement of items to a blocked arrangement
    /// across the thread block, using temporary storage.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer.emplace(index(i * BlockSize + flat_id), input[i]);
        }
        ::rocprim::syncthreads();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[index(flat_id * ItemsPerThread + i)];
        }
    }

    /// \brief Transposes a blocked arrangement of items to a warp-striped arrangement
    /// across the thread block.
    ///
    /// \tparam U [inferred] the output type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void blocked_to_warp_striped(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        blocked_to_warp_striped(input, output, storage);
    }

    /// \brief Transposes a blocked arrangement of items to a warp-striped arrangement
    /// across the thread block, using temporary storage.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_warp_striped(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread],
                                 storage_type& storage)
    {
        constexpr unsigned int items_per_warp = warp_size * ItemsPerThread;
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const unsigned int current_warp_size = get_current_warp_size();
        const unsigned int     offset            = warp_id * items_per_warp;

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer.emplace(index(offset + lane_id * ItemsPerThread + i), input[i]);
        }

        ::rocprim::wave_barrier();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[index(offset + i * current_warp_size + lane_id)];
        }
    }

    /// \brief Transposes a warp-striped arrangement of items to a blocked arrangement
    /// across the thread block.
    ///
    /// \tparam U [inferred] the output type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    template<class U>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void warp_striped_to_blocked(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        warp_striped_to_blocked(input, output, storage);
    }

    /// \brief Transposes a warp-striped arrangement of items to a blocked arrangement
    /// across the thread block, using temporary storage.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void warp_striped_to_blocked(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread],
                                 storage_type& storage)
    {
        constexpr unsigned int items_per_warp = warp_size * ItemsPerThread;
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const unsigned int current_warp_size = get_current_warp_size();
        const unsigned int     offset            = warp_id * items_per_warp;

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer.emplace(index(offset + i * current_warp_size + lane_id), input[i]);
        }

        ::rocprim::wave_barrier();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[index(offset + lane_id * ItemsPerThread + i)];
        }
    }

    /// \brief Scatters items to a blocked arrangement based on their ranks
    /// across the thread block.
    ///
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [out] ranks array that has rank of data.
    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void scatter_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        scatter_to_blocked(input, output, ranks, storage);
    }

    /// \brief Gathers items from a striped arrangement based on their ranks
    /// across the thread block.
    ///
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [out] ranks array that has rank of data.
    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void gather_from_striped(const T (&input)[ItemsPerThread],
                                   U (&output)[ItemsPerThread],
                                   const Offset (&ranks)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        gather_from_striped(input, output, ranks, storage);
    }

    /// \brief Scatters items to a blocked arrangement based on their ranks
    /// across the thread block, using temporary storage.
    ///
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [out] ranks array that has rank of data.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            storage.buffer.emplace(index(rank), input[i]);
        }
        ::rocprim::syncthreads();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[index(flat_id * ItemsPerThread + i)];
        }
    }

    /// \brief Gathers items from a striped arrangement based on their ranks
    /// across the thread block, using temporary storage.
    ///
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [out] ranks array that has rank of data.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    template <class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void gather_from_striped(const T (&input)[ItemsPerThread],
                             U (&output)[ItemsPerThread],
                             const Offset (&ranks)[ItemsPerThread],
                             storage_type& storage)
    {
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage.buffer.emplace(index(i * BlockSize + flat_id), input[i]);
        }
        ::rocprim::syncthreads();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            output[i]         = storage_buffer[index(rank)];
        }
    }

    /// \brief Scatters items to a striped arrangement based on their ranks
    /// across the thread block.
    ///
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [out] ranks array that has rank of data.
    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
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
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [out] ranks array that has rank of data.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            storage.buffer.emplace(rank, input[i]);
        }
        ::rocprim::syncthreads();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[i * BlockSize + flat_id];
        }
    }

    /// \brief Scatters items to a *warp* striped arrangement based on their ranks
    /// across the thread block, using temporary storage.
    ///
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [out] ranks array that has rank of data.
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
    ///     // specialize block_exchange for int, block of 128 threads and 8 items per thread
    ///     using block_exchange_int = rocprim::block_exchange<int, 128, 8>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_exchange_int::storage_type storage;
    ///
    ///     int items[8];
    ///     int ranks[8];
    ///     ...
    ///     block_exchange_int b_exchange;
    ///     b_exchange.scatter_to_warp_striped(items, items, ranks, storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int VirtualWaveSize = arch::wavefront::size_from_target<TargetWaveSize>(),
             class U,
             class Offset>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_warp_striped(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread],
                                 const Offset (&ranks)[ItemsPerThread],
                                 storage_type& storage)
    {
        static_assert(detail::is_power_of_two(VirtualWaveSize)
                          && VirtualWaveSize <= arch::wavefront::max_size(),
                      "VirtualWaveSize must be a power of two and equal or less"
                      "than the size of hardware warp.");
        assert(VirtualWaveSize <= arch::wavefront::size());

        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const unsigned int thread_id     = detail::logical_lane_id<VirtualWaveSize>();
        const unsigned int warp_id       = flat_id / VirtualWaveSize;
        const unsigned int warp_offset   = warp_id * VirtualWaveSize * ItemsPerThread;
        const unsigned int thread_offset = thread_id + warp_offset;

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            storage.buffer.emplace(index(rank), input[i]);
        }

        ::rocprim::syncthreads();

        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[index(thread_offset + i * VirtualWaveSize)];
        }
    }

    /// \brief Scatters items to a striped arrangement based on their ranks
    /// across the thread block, guarded by rank.
    ///
    /// \par Overview
    /// * Items with rank -1 are not scattered.
    ///
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [in] ranks array that has rank of data.
    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
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
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [in] ranks array that has rank of data.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_striped_guarded(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    storage_type& storage)
    {
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            if(rank >= 0)
            {
                storage.buffer.emplace(rank, input[i]);
            }
        }
        ::rocprim::syncthreads();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[i * BlockSize + flat_id];
        }
    }

    /// \brief Scatters items to a striped arrangement based on their ranks
    /// across the thread block, with a flag to denote validity.
    ///
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    /// \tparam ValidFlag [inferred] the validity flag type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [in] ranks array that has rank of data.
    /// \param [in] is_valid array that has flags to denote validity.
    template<class U, class Offset, class ValidFlag>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
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
    /// \tparam U [inferred] the output type.
    /// \tparam Offset [inferred] the rank type.
    /// \tparam ValidFlag [inferred] the validity flag type.
    ///
    /// \param [in] input array that data is loaded from.
    /// \param [out] output array that data is loaded to.
    /// \param [in] ranks array that has rank of data.
    /// \param [in] is_valid array that has flags to denote validity.
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
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_striped_flagged(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    const ValidFlag (&is_valid)[ItemsPerThread],
                                    storage_type& storage)
    {
        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            if(is_valid[i])
            {
                storage.buffer.emplace(rank, input[i]);
            }
        }
        ::rocprim::syncthreads();
        const auto& storage_buffer = storage.buffer.get_unsafe_array();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_buffer[i * BlockSize + flat_id];
        }
    }

private:

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int get_current_warp_size() const
    {
        const unsigned int warp_id = ::rocprim::warp_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        return (warp_id == warps_no - 1)
            ? (BlockSize % warp_size > 0 ? BlockSize % warp_size : warp_size)
            : warp_size;
    }

    // Change index to minimize LDS bank conflicts if necessary
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int index(unsigned int n)
    {
        // Move every 32-bank wide "row" (32 banks * 4 bytes) by one item
        return has_bank_conflicts ? (n + (n / (banks_no * buffer_size)) * buffer_size) : n;
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<typename T,
         unsigned int       BlockSizeX,
         unsigned int       ItemsPerThread,
         unsigned int       BlockSizeY,
         unsigned int       BlockSizeZ,
         block_padding_hint PaddingHint>
class block_exchange<T,
                     BlockSizeX,
                     ItemsPerThread,
                     BlockSizeY,
                     BlockSizeZ,
                     PaddingHint,
                     ::rocprim::arch::wavefront::target::dynamic>
{
private:
    using block_exchange_wave32 = block_exchange<T,
                                                 BlockSizeX,
                                                 ItemsPerThread,
                                                 BlockSizeY,
                                                 BlockSizeZ,
                                                 PaddingHint,
                                                 ::rocprim::arch::wavefront::target::size32>;
    using block_exchange_wave64 = block_exchange<T,
                                                 BlockSizeX,
                                                 ItemsPerThread,
                                                 BlockSizeY,
                                                 BlockSizeZ,
                                                 PaddingHint,
                                                 ::rocprim::arch::wavefront::target::size64>;
    using dispatch
        = ::rocprim::detail::dispatch_wave_size<block_exchange_wave32, block_exchange_wave64>;

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
    auto striped_to_blocked(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.striped_to_blocked(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto blocked_to_warp_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.blocked_to_warp_striped(args...); },
                   args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto warp_striped_to_blocked(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.warp_striped_to_blocked(args...); },
                   args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scatter_to_blocked(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.scatter_to_blocked(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto gather_from_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.gather_from_striped(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scatter_to_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.scatter_to_striped(args...); }, args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scatter_to_warp_striped(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.scatter_to_warp_striped(args...); },
                   args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scatter_to_striped_guarded(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.scatter_to_striped_guarded(args...); },
                   args...);
    }

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scatter_to_striped_flagged(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.scatter_to_striped_flagged(args...); },
                   args...);
    }
};

#endif

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
