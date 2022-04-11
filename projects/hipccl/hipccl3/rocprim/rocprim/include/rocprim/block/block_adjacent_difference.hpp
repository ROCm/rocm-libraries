/******************************************************************************
* Copyright (c) 2011, Duane Merrill.  All rights reserved.
* Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
* Modifications Copyright (c) 2022, Advanced Micro Devices, Inc.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the NVIDIA CORPORATION nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

#ifndef ROCPRIM_BLOCK_BLOCK_ADJACENT_DIFFERENCE_HPP_
#define ROCPRIM_BLOCK_BLOCK_ADJACENT_DIFFERENCE_HPP_


#include "detail/block_adjacent_difference_impl.hpp"

#include "../config.hpp"
#include "../detail/various.hpp"



/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief The \p block_adjacent_difference class is a block level parallel primitive which provides
/// methods for applying binary functions for pairs of consecutive items partition across a thread
/// block.
///
/// \tparam T - the input type.
/// \tparam BlockSize - the number of threads in a block.
///
/// \par Overview
/// * There are two types of flags:
///   * Head flags.
///   * Tail flags.
/// * The above flags are used to differentiate items from their predecessors or successors.
/// * E.g. Head flags are convenient for differentiating disjoint data segments as part of a
/// segmented reduction/scan.
///
/// \par Examples
/// \parblock
/// In the examples discontinuity operation is performed on block of 128 threads, using type
/// \p int.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize discontinuity for int and a block of 128 threads
///     using block_adjacent_difference_int = rocprim::block_adjacent_difference<int, 128>;
///     // allocate storage in shared memory
///     __shared__ block_adjacent_difference_int::storage_type storage;
///
///     // segment of consecutive items to be used
///     int input[8];
///     ...
///     int head_flags[8];
///     block_adjacent_difference_int b_discontinuity;
///     using flag_op_type = typename rocprim::greater<int>;
///     b_discontinuity.flag_heads(head_flags, input, flag_op_type(), storage);
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_adjacent_difference
#ifndef DOXYGEN_SHOULD_SKIP_THIS // hide implementation detail from documentation
    : private detail::block_adjacent_difference_impl<T, BlockSizeX, BlockSizeY, BlockSizeZ>
#endif // DOXYGEN_SHOULD_SKIP_THIS
{
    using base_type = detail::block_adjacent_difference_impl<T, BlockSizeX, BlockSizeY, BlockSizeZ>;

    static constexpr unsigned BlockSize = base_type::BlockSize;
    // Struct used for creating a raw_storage object for this primitive's temporary storage.
    struct storage_type_
    {
        typename base_type::storage_type left;
        typename base_type::storage_type right;
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
    using storage_type = storage_type_;
    #endif

    /// \brief Tags \p head_flags that indicate discontinuities between items partitioned
    /// across the thread block, where the first item has no reference and is always
    /// flagged.
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use subtract_left() or block_discontinuity::flag_heads() instead.
    /// \tparam ItemsPerThread - [inferred] the number of items to be processed by
    /// each thread.
    /// \tparam Flag - [inferred] the flag type.
    /// \tparam FlagOp - [inferred] type of binary function used for flagging.
    ///
    /// \param [out] head_flags - array that contains the head flags.
    /// \param [in] input - array that data is loaded from.
    /// \param [in] flag_op - binary operation function object that will be used for flagging.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt> or <tt>bool (const T& a, const T& b, unsigned int b_index);</tt>.
    /// The signature does not need to have <tt>const &</tt>, but function object
    /// must not modify the objects passed to it.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reuse
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_adjacent_difference_int = rocprim::block_adjacent_difference<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_adjacent_difference_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     ...
    ///     int head_flags[8];
    ///     block_adjacent_difference_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads(head_flags, input, flag_op_type(), storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_left or block_discontinuity.flag_heads instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;
        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, head_flags, flag_op, input[0] /* predecessor */, storage.get().left);
    }

    /// \overload
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use subtract_left() or block_discontinuity::flag_heads() instead.
    /// This overload does not take a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_left or block_discontinuity.flag_heads instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads(head_flags, input, flag_op, storage);
    }

    /// \brief Tags \p head_flags that indicate discontinuities between items partitioned
    /// across the thread block, where the first item of the first thread is compared against
    /// a \p tile_predecessor_item.
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use subtract_left() or block_discontinuity::flag_heads() instead.
    ///
    /// \tparam ItemsPerThread - [inferred] the number of items to be processed by
    /// each thread.
    /// \tparam Flag - [inferred] the flag type.
    /// \tparam FlagOp - [inferred] type of binary function used for flagging.
    ///
    /// \param [out] head_flags - array that contains the head flags.
    /// \param [in] tile_predecessor_item - first tile item from thread to be compared
    /// against.
    /// \param [in] input - array that data is loaded from.
    /// \param [in] flag_op - binary operation function object that will be used for flagging.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt> or <tt>bool (const T& a, const T& b, unsigned int b_index);</tt>.
    /// The signature does not need to have <tt>const &</tt>, but function object
    /// must not modify the objects passed to it.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reuse
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_adjacent_difference_int = rocprim::block_adjacent_difference<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_adjacent_difference_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     int tile_item = 0;
    ///     if (threadIdx.x == 0)
    ///     {
    ///         tile_item = ...
    ///     }
    ///     ...
    ///     int head_flags[8];
    ///     block_adjacent_difference_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads(head_flags, tile_item, input, flag_op_type(),
    ///                                storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_left or block_discontinuity.flag_heads instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    T tile_predecessor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;
        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, head_flags, flag_op, tile_predecessor_item, storage.get().left);
    }

    /// \overload
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use subtract_left() or block_discontinuity::flag_heads() instead.
    ///
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_left or block_discontinuity.flag_heads instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    T tile_predecessor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads(head_flags, tile_predecessor_item, input, flag_op, storage);
    }

    /// \brief Tags \p tail_flags that indicate discontinuities between items partitioned
    /// across the thread block, where the last item has no reference and is always
    /// flagged.
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use subtract_right() or block_discontinuity::flag_tails() instead.
    ///
    /// \tparam ItemsPerThread - [inferred] the number of items to be processed by
    /// each thread.
    /// \tparam Flag - [inferred] the flag type.
    /// \tparam FlagOp - [inferred] type of binary function used for flagging.
    ///
    /// \param [out] tail_flags - array that contains the tail flags.
    /// \param [in] input - array that data is loaded from.
    /// \param [in] flag_op - binary operation function object that will be used for flagging.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt> or <tt>bool (const T& a, const T& b, unsigned int b_index);</tt>.
    /// The signature does not need to have <tt>const &</tt>, but function object
    /// must not modify the objects passed to it.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reuse
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_adjacent_difference_int = rocprim::block_adjacent_difference<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_adjacent_difference_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     ...
    ///     int tail_flags[8];
    ///     block_adjacent_difference_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_tails(tail_flags, input, flag_op_type(), storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_right or block_discontinuity.flag_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags       = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_successor = false;
        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, tail_flags, flag_op, input[0] /* successor */, storage.get().right);
    }

    /// \overload
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use subtract_right() or block_discontinuity::flag_tails() instead.
    ///
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_right or block_discontinuity.flag_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_tails(tail_flags, input, flag_op, storage);
    }

    /// \brief Tags \p tail_flags that indicate discontinuities between items partitioned
    /// across the thread block, where the last item of the last thread is compared against
    /// a \p tile_successor_item.
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use subtract_right() or block_discontinuity::flag_tails() instead.
    ///
    /// \tparam ItemsPerThread - [inferred] the number of items to be processed by
    /// each thread.
    /// \tparam Flag - [inferred] the flag type.
    /// \tparam FlagOp - [inferred] type of binary function used for flagging.
    ///
    /// \param [out] tail_flags - array that contains the tail flags.
    /// \param [in] tile_successor_item - last tile item from thread to be compared
    /// against.
    /// \param [in] input - array that data is loaded from.
    /// \param [in] flag_op - binary operation function object that will be used for flagging.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt> or <tt>bool (const T& a, const T& b, unsigned int b_index);</tt>.
    /// The signature does not need to have <tt>const &</tt>, but function object
    /// must not modify the objects passed to it.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reuse
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_adjacent_difference_int = rocprim::block_adjacent_difference<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_adjacent_difference_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     int tile_item = 0;
    ///     if (threadIdx.x == 0)
    ///     {
    ///         tile_item = ...
    ///     }
    ///     ...
    ///     int tail_flags[8];
    ///     block_adjacent_difference_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_tails(tail_flags, tile_item, input, flag_op_type(),
    ///                                storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_right or block_discontinuity.flag_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    T tile_successor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags       = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_successor = true;
        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, tail_flags, flag_op, tile_successor_item, storage.get().right);
    }

    /// \overload
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use subtract_right() or block_discontinuity::flag_tails() instead.
    ///
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_right or block_discontinuity.flag_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    T tile_successor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_tails(tail_flags, tile_successor_item, input, flag_op, storage);
    }

    /// \brief Tags both \p head_flags and\p tail_flags that indicate discontinuities
    /// between items partitioned across the thread block.
    ///
    /// \tparam ItemsPerThread - [inferred] the number of items to be processed by
    /// each thread.
    /// \tparam Flag - [inferred] the flag type.
    /// \tparam FlagOp - [inferred] type of binary function used for flagging.
    ///
    /// \param [out] head_flags - array that contains the head flags.
    /// \param [out] tail_flags - array that contains the tail flags.
    /// \param [in] input - array that data is loaded from.
    /// \param [in] flag_op - binary operation function object that will be used for flagging.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt> or <tt>bool (const T& a, const T& b, unsigned int b_index);</tt>.
    /// The signature does not need to have <tt>const &</tt>, but function object
    /// must not modify the objects passed to it.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reuse
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_adjacent_difference_int = rocprim::block_adjacent_difference<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_adjacent_difference_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     ...
    ///     int head_flags[8];
    ///     int tail_flags[8];
    ///     block_adjacent_difference_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads_and_tails(head_flags, tail_flags, input,
    ///                                          flag_op_type(), storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;
        static constexpr auto with_successor   = false;

        // Copy items in case head_flags is aliased with input
        T items[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i) {
            items[i] = input[i];
        }

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            items, head_flags, flag_op, items[0] /*predecessor*/, storage.get().left);

        base_type::template apply_right<as_flags, reversed, with_successor>(
            items, tail_flags, flag_op, items[0] /*successor*/, storage.get().right);
    }

    /// \overload
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use block_discontinuity::flag_heads_and_tails() instead.
    ///
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads_and_tails(head_flags, tail_flags, input, flag_op, storage);
    }

    /// \brief Tags both \p head_flags and\p tail_flags that indicate discontinuities
    /// between items partitioned across the thread block, where the last item of the
    /// last thread is compared against a \p tile_successor_item.
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use block_discontinuity::flag_heads_and_tails() instead.
    ///
    /// \tparam ItemsPerThread - [inferred] the number of items to be processed by
    /// each thread.
    /// \tparam Flag - [inferred] the flag type.
    /// \tparam FlagOp - [inferred] type of binary function used for flagging.
    ///
    /// \param [out] head_flags - array that contains the head flags.
    /// \param [out] tail_flags - array that contains the tail flags.
    /// \param [in] tile_successor_item - last tile item from thread to be compared
    /// against.
    /// \param [in] input - array that data is loaded from.
    /// \param [in] flag_op - binary operation function object that will be used for flagging.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt> or <tt>bool (const T& a, const T& b, unsigned int b_index);</tt>.
    /// The signature does not need to have <tt>const &</tt>, but function object
    /// must not modify the objects passed to it.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reuse
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_adjacent_difference_int = rocprim::block_adjacent_difference<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_adjacent_difference_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     int tile_item = 0;
    ///     if (threadIdx.x == 0)
    ///     {
    ///         tile_item = ...
    ///     }
    ///     ...
    ///     int head_flags[8];
    ///     int tail_flags[8];
    ///     block_adjacent_difference_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads_and_tails(head_flags, tail_flags, tile_item,
    ///                                          input, flag_op_type(),
    ///                                          storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;
        static constexpr auto with_successor   = true;

        // Copy items in case head_flags is aliased with input
        T items[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i) {
            items[i] = input[i];
        }

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            items, head_flags, flag_op, items[0] /*predecessor*/, storage.get().left);

        base_type::template apply_right<as_flags, reversed, with_successor>(
            items, tail_flags, flag_op, tile_successor_item, storage.get().right);
    }

    /// \overload
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use block_discontinuity::flag_heads_and_tails() instead.
    ///
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads_and_tails(head_flags, tail_flags, tile_successor_item, input, flag_op, storage);
    }

    /// \brief Tags both \p head_flags and\p tail_flags that indicate discontinuities
    /// between items partitioned across the thread block, where the first item of the
    /// first thread is compared against a \p tile_predecessor_item.
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use block_discontinuity::flag_heads_and_tails() instead.
    ///
    /// \tparam ItemsPerThread - [inferred] the number of items to be processed by
    /// each thread.
    /// \tparam Flag - [inferred] the flag type.
    /// \tparam FlagOp - [inferred] type of binary function used for flagging.
    ///
    /// \param [out] head_flags - array that contains the head flags.
    /// \param [in] tile_predecessor_item - first tile item from thread to be compared
    /// against.
    /// \param [out] tail_flags - array that contains the tail flags.
    /// \param [in] input - array that data is loaded from.
    /// \param [in] flag_op - binary operation function object that will be used for flagging.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt> or <tt>bool (const T& a, const T& b, unsigned int b_index);</tt>.
    /// The signature does not need to have <tt>const &</tt>, but function object
    /// must not modify the objects passed to it.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reuse
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_adjacent_difference_int = rocprim::block_adjacent_difference<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_adjacent_difference_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     int tile_item = 0;
    ///     if (threadIdx.x == 0)
    ///     {
    ///         tile_item = ...
    ///     }
    ///     ...
    ///     int head_flags[8];
    ///     int tail_flags[8];
    ///     block_adjacent_difference_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads_and_tails(head_flags, tile_item, tail_flags,
    ///                                          input, flag_op_type(),
    ///                                          storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;
        static constexpr auto with_successor   = false;

        // Copy items in case head_flags is aliased with input
        T items[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i) {
            items[i] = input[i];
        }

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            items, head_flags, flag_op, tile_predecessor_item, storage.get().left);

        base_type::template apply_right<as_flags, reversed, with_successor>(
            items, tail_flags, flag_op, items[0] /*successor*/, storage.get().right);
    }

    /// \overload
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use block_discontinuity::flag_heads_and_tails() instead.
    ///
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, input, flag_op, storage);
    }

    /// \brief Tags both \p head_flags and\p tail_flags that indicate discontinuities
    /// between items partitioned across the thread block, where the first and last items of
    /// the first and last thread is compared against a \p tile_predecessor_item and
    /// a \p tile_successor_item.
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use block_discontinuity::flag_heads_and_tails() instead.
    ///
    /// \tparam ItemsPerThread - [inferred] the number of items to be processed by
    /// each thread.
    /// \tparam Flag - [inferred] the flag type.
    /// \tparam FlagOp - [inferred] type of binary function used for flagging.
    ///
    /// \param [out] head_flags - array that contains the head flags.
    /// \param [in] tile_predecessor_item - first tile item from thread to be compared
    /// against.
    /// \param [out] tail_flags - array that contains the tail flags.
    /// \param [in] tile_successor_item - last tile item from thread to be compared
    /// against.
    /// \param [in] input - array that data is loaded from.
    /// \param [in] flag_op - binary operation function object that will be used for flagging.
    /// The signature of the function should be equivalent to the following:
    /// <tt>bool f(const T &a, const T &b);</tt> or <tt>bool (const T& a, const T& b, unsigned int b_index);</tt>.
    /// The signature does not need to have <tt>const &</tt>, but function object
    /// must not modify the objects passed to it.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reuse
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_adjacent_difference_int = rocprim::block_adjacent_difference<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_adjacent_difference_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     int tile_predecessor_item = 0;
    ///     int tile_successor_item = 0;
    ///     if (threadIdx.x == 0)
    ///     {
    ///         tile_predecessor_item = ...
    ///         tile_successor_item = ...
    ///     }
    ///     ...
    ///     int head_flags[8];
    ///     int tail_flags[8];
    ///     block_adjacent_difference_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item,
    ///                                          tail_flags, tile_successor_item,
    ///                                          input, flag_op_type(),
    ///                                          storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;
        static constexpr auto with_successor   = true;

        // Copy items in case head_flags is aliased with input
        T items[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i) {
            items[i] = input[i];
        }

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            items, head_flags, flag_op, tile_predecessor_item, storage.get().left);

        base_type::template apply_right<as_flags, reversed, with_successor>(
            items, tail_flags, flag_op, tile_successor_item, storage.get().right);
    }

    /// \overload
    /// \deprecated The flags API of block_adjacent_difference is deprecated,
    /// use block_discontinuity::flag_heads_and_tails() instead.
    ///
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads_and_tails(
            head_flags, tile_predecessor_item, tail_flags, tile_successor_item,
            input, flag_op, storage
        );
    }

    /// \brief Apply a function to each consecutive pair of elements partitioned across threads in
    /// the block and write the output to the position of the left item.
    ///
    /// The first item in the first thread is copied from the input then for the rest the following
    /// code applies.
    /// \code
    /// // For each i in [1, block_size * ItemsPerThread) across threads in a block
    /// output[i] = op(input[i], input[i-1]);
    /// \endcode
    ///
    /// \tparam Output - [inferred] the type of output, must be assignable from the result of `op`
    /// \tparam ItemsPerThread - [inferred] the number of items processed by each thread
    /// \tparam BinaryFunction - [inferred] the type of the function to apply
    /// \param [in] input - array that data is loaded from partitioned across the threads in the block
    /// \param [out] output - array where the result of function application will be written to
    /// \param [in] op - binary function applied to the items.
    /// The signature of the function should be equivalent to the following:
    /// `bool f(const T &a, const T &b)` The signature does not need to have
    /// `const &` but the function object must not modify the objects passed to it.
    /// \param storage reference to a temporary storage object of type #storage_type
    /// \par Storage reuse
    /// Synchronization barrier should be placed before `storage` is reused
    /// or repurposed: `__syncthreads()` or \link syncthreads() rocprim::syncthreads() \endlink.
    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_left(const T (&input)[ItemsPerThread],
                                                     Output (&output)[ItemsPerThread],
                                                     const BinaryFunction op,
                                                     storage_type&        storage)
    {
        static constexpr auto as_flags         = false;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, output, op, input[0] /* predecessor */, storage.get().left);
    }

    /// \brief Apply a function to each consecutive pair of elements partitioned across threads in
    /// the block and write the output to the position of the left item, with an explicit item before 
    /// the tile.
    ///
    /// \code
    /// // For the first item on the first thread use the tile predecessor
    /// output[0] = op(input[0], tile_predecessor)
    /// // For other items, i in [1, block_size * ItemsPerThread) across threads in a block
    /// output[i] = op(input[i], input[i-1]);
    /// \endcode
    ///
    /// \tparam Output - [inferred] the type of output, must be assignable from the result of `op`
    /// \tparam ItemsPerThread - [inferred] the number of items processed by each thread
    /// \tparam BinaryFunction - [inferred] the type of the function to apply
    /// \param [in] input - array that data is loaded from partitioned across the threads in the block
    /// \param [out] output - array where the result of function application will be written to
    /// \param [in] op - binary function applied to the items.
    /// The signature of the function should be equivalent to the following:
    /// `bool f(const T &a, const T &b)` The signature does not need to have
    /// `const &` but the function object must not modify the objects passed to it.
    /// \param [in] tile_predecessor - the item before the tile, will be used as the input 
    /// of the first application of `op`
    /// \param storage - reference to a temporary storage object of type #storage_type
    /// \par Storage reuse
    /// Synchronization barrier should be placed before `storage` is reused
    /// or repurposed: `__syncthreads()` or \link syncthreads() rocprim::syncthreads() \endlink.
    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_left(const T (&input)[ItemsPerThread],
                                                     Output (&output)[ItemsPerThread],
                                                     const BinaryFunction op,
                                                     const T              tile_predecessor,
                                                     storage_type&        storage)
    {
        static constexpr auto as_flags         = false;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, output, op, tile_predecessor, storage.get().left);
    }

    /// \brief Apply a function to each consecutive pair of elements partitioned across threads in
    /// the block and write the output to the position of the left item, in a partial tile.
    ///
    /// \code
    /// output[0] = input[0]
    /// // For each item i in [1, valid_items) across threads in a block
    /// output[i] = op(input[i], input[i-1]);
    /// // Just copy "invalid" items in [valid_items, block_size * ItemsPerThread)
    /// output[i] = input[i]
    /// \endcode
    ///
    /// \tparam Output - [inferred] the type of output, must be assignable from the result of `op`
    /// \tparam ItemsPerThread - [inferred] the number of items processed by each thread
    /// \tparam BinaryFunction - [inferred] the type of the function to apply
    /// \param [in] input - array that data is loaded from partitioned across the threads in the block
    /// \param [out] output - array where the result of function application will be written to
    /// \param [in] op - binary function applied to the items.
    /// The signature of the function should be equivalent to the following:
    /// `bool f(const T &a, const T &b)` The signature does not need to have
    /// `const &` but the function object must not modify the objects passed to it.
    /// \param [in] valid_items - number of items in the block which are considered "valid" and will
    /// be used. Must be less or equal to `BlockSize` * `ItemsPerThread`
    /// \param storage - reference to a temporary storage object of type #storage_type
    /// \par Storage reuse
    /// Synchronization barrier should be placed before `storage` is reused
    /// or repurposed: `__syncthreads()` or \link syncthreads() rocprim::syncthreads() \endlink.
    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_left_partial(const T (&input)[ItemsPerThread],
                                                             Output (&output)[ItemsPerThread],
                                                             const BinaryFunction op,
                                                             const unsigned int   valid_items,
                                                             storage_type&        storage)
    {
        static constexpr auto as_flags         = false;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;

        base_type::template apply_left_partial<as_flags, reversed, with_predecessor>(
            input, output, op, input[0] /* predecessor */, valid_items, storage.get().left);
    }

    /// \brief Apply a function to each consecutive pair of elements partitioned across threads in
    /// the block and write the output to the position of the left item, in a partial tile with a
    /// predecessor.
    ///
    /// This combines subtract_left_partial() with a tile predecessor.
    /// \tparam Output - [inferred] the type of output, must be assignable from the result of `op`
    /// \tparam ItemsPerThread - [inferred] the number of items processed by each thread
    /// \tparam BinaryFunction - [inferred] the type of the function to apply
    /// \param [in] input - array that data is loaded from partitioned across the threads in the block
    /// \param [out] output - array where the result of function application will be written to
    /// \param [in] op - binary function applied to the items.
    /// The signature of the function should be equivalent to the following:
    /// `bool f(const T &a, const T &b)` The signature does not need to have
    /// `const &` but the function object must not modify the objects passed to it.
    /// \param [in] tile_predecessor - the item before the tile, will be used as the input 
    /// of the first application of `op`
    /// \param [in] valid_items - number of items in the block which are considered "valid" and will
    /// be used. Must be less or equal to `BlockSize` * `ItemsPerThread`
    /// \param storage - reference to a temporary storage object of type #storage_type
    /// \par Storage reuse
    /// Synchronization barrier should be placed before `storage` is reused
    /// or repurposed: `__syncthreads()` or \link syncthreads() rocprim::syncthreads() \endlink.
    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_left_partial(const T (&input)[ItemsPerThread],
                                                             Output (&output)[ItemsPerThread],
                                                             const BinaryFunction op,
                                                             const T              tile_predecessor,
                                                             const unsigned int   valid_items,
                                                             storage_type&        storage)
    {
        static constexpr auto as_flags         = false;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;

        base_type::template apply_left_partial<as_flags, reversed, with_predecessor>(
            input, output, op, tile_predecessor, valid_items, storage.get().left);
    }

    /// \brief Apply a function to each consecutive pair of elements partitioned across threads in
    /// the block and write the output to the position of the right item.
    ///
    /// The last item in the last thread is copied from the input then for the rest the following
    /// code applies.
    /// \code
    /// // For each i in [0, block_size * ItemsPerThread - 1) across threads in a block
    /// output[i] = op(input[i], input[i+1]);
    /// \endcode
    ///
    /// \tparam Output - [inferred] the type of output, must be assignable from the result of `op`
    /// \tparam ItemsPerThread - [inferred] the number of items processed by each thread
    /// \tparam BinaryFunction - [inferred] the type of the function to apply
    /// \param [in] input - array that data is loaded from partitioned across the threads in the block
    /// \param [out] output - array where the result of function application will be written to
    /// \param [in] op - binary function applied to the items.
    /// The signature of the function should be equivalent to the following:
    /// `bool f(const T &a, const T &b)` The signature does not need to have
    /// `const &` but the function object must not modify the objects passed to it.
    /// \param storage - reference to a temporary storage object of type #storage_type
    /// \par Storage reuse
    /// Synchronization barrier should be placed before `storage` is reused
    /// or repurposed: `__syncthreads()` or \link syncthreads() rocprim::syncthreads() \endlink.
    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_right(const T (&input)[ItemsPerThread],
                                                      Output (&output)[ItemsPerThread],
                                                      const BinaryFunction op,
                                                      storage_type&        storage)
    {
        static constexpr auto as_flags       = false;
        static constexpr auto reversed       = false;
        static constexpr auto with_successor = false;

        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, output, op, input[0] /* successor */, storage.get().right);
    }

    /// \brief Apply a function to each consecutive pair of elements partitioned across threads in
    /// the block and write the output to the position of the right item, with an explicit item after 
    /// the tile.
    ///
    /// \code
    /// // For each items i in [0, block_size * ItemsPerThread - 1) across threads in a block
    /// output[i] = op(input[i], input[i+1]);
    /// // For the last item on the last thread use the tile successor
    /// output[block_size * ItemsPerThread - 1] =
    ///      op(input[block_size * ItemsPerThread - 1], tile_successor)
    /// \endcode
    ///
    /// \tparam Output - [inferred] the type of output, must be assignable from the result of `op`
    /// \tparam ItemsPerThread - [inferred] the number of items processed by each thread
    /// \tparam BinaryFunction - [inferred] the type of the function to apply
    /// \param [in] input - array that data is loaded from partitioned across the threads in the block
    /// \param [out] output - array where the result of function application will be written to
    /// \param [in] op - binary function applied to the items.
    /// The signature of the function should be equivalent to the following:
    /// `bool f(const T &a, const T &b)` The signature does not need to have
    /// `const &` but the function object must not modify the objects passed to it.
    /// \param [in] tile_successor - the item after the tile, will be used as the input 
    /// of the last application of `op`
    /// \param storage - reference to a temporary storage object of type #storage_type
    /// \par Storage reuse
    /// Synchronization barrier should be placed before `storage` is reused
    /// or repurposed: `__syncthreads()` or \link syncthreads() rocprim::syncthreads() \endlink.
    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_right(const T (&input)[ItemsPerThread],
                                                      Output (&output)[ItemsPerThread],
                                                      const BinaryFunction op,
                                                      const T              tile_successor,
                                                      storage_type&        storage)
    {
        static constexpr auto as_flags       = false;
        static constexpr auto reversed       = false;
        static constexpr auto with_successor = true;

        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, output, op, tile_successor, storage.get().right);
    }

    /// \brief Apply a function to each consecutive pair of elements partitioned across threads in
    /// the block and write the output to the position of the right item, in a partial tile.
    ///
    /// \code
    /// // For each item i in [0, valid_items) across threads in a block
    /// output[i] = op(input[i], input[i + 1]);
    /// // Just copy "invalid" items in [valid_items, block_size * ItemsPerThread)
    /// output[i] = input[i]
    /// \endcode
    ///
    /// \tparam Output - [inferred] the type of output, must be assignable from the result of `op`
    /// \tparam ItemsPerThread - [inferred] the number of items processed by each thread
    /// \tparam BinaryFunction - [inferred] the type of the function to apply
    /// \param [in] input - array that data is loaded from partitioned across the threads in the block
    /// \param [out] output - array where the result of function application will be written to
    /// \param [in] op - binary function applied to the items.
    /// The signature of the function should be equivalent to the following:
    /// `bool f(const T &a, const T &b)` The signature does not need to have
    /// `const &` but the function object must not modify the objects passed to it.
    /// \param [in] valid_items - number of items in the block which are considered "valid" and will
    /// be used. Must be less or equal to `BlockSize` * `ItemsPerThread`
    /// \param storage - reference to a temporary storage object of type #storage_type
    /// \par Storage reuse
    /// Synchronization barrier should be placed before `storage` is reused
    /// or repurposed: `__syncthreads()` or \link syncthreads() rocprim::syncthreads() \endlink.
    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_right_partial(const T (&input)[ItemsPerThread],
                                                              Output (&output)[ItemsPerThread],
                                                              const BinaryFunction op,
                                                              const unsigned int   valid_items,
                                                              storage_type&        storage)
    {
        static constexpr auto as_flags = false;
        static constexpr auto reversed = false;

        base_type::template apply_right_partial<as_flags, reversed>(
            input, output, op, valid_items, storage.get().right);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_ADJACENT_DIFFERENCE_HPP_
