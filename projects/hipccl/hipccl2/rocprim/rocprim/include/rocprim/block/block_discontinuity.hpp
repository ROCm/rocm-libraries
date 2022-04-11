// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_
#define ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_


#include "detail/block_adjacent_difference_impl.hpp"

#include "../config.hpp"
#include "../detail/various.hpp"



/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief The \p block_discontinuity class is a block level parallel primitive which provides
/// methods for flagging items that are discontinued within an ordered set of items across
/// threads in a block.
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
///     using block_discontinuity_int = rocprim::block_discontinuity<int, 128>;
///     // allocate storage in shared memory
///     __shared__ block_discontinuity_int::storage_type storage;
///
///     // segment of consecutive items to be used
///     int input[8];
///     ...
///     int head_flags[8];
///     block_discontinuity_int b_discontinuity;
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
class block_discontinuity
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
    ///
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
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_discontinuity_int = rocprim::block_discontinuity<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_discontinuity_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     ...
    ///     int head_flags[8];
    ///     block_discontinuity_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads(head_flags, input, flag_op_type(), storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = false;
        static constexpr auto with_predecessor = false;
        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, head_flags, flag_op, input[0] /* predecessor */, storage.get().left);
    }

    /// \overload
    /// This overload does not take a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
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
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_discontinuity_int = rocprim::block_discontinuity<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_discontinuity_int::storage_type storage;
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
    ///     block_discontinuity_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads(head_flags, tile_item, input, flag_op_type(),
    ///                                storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    T tile_predecessor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = false;
        static constexpr auto with_predecessor = true;
        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, head_flags, flag_op, tile_predecessor_item, storage.get().left);
    }

    /// \overload
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
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
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_discontinuity_int = rocprim::block_discontinuity<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_discontinuity_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     ...
    ///     int tail_flags[8];
    ///     block_discontinuity_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_tails(tail_flags, input, flag_op_type(), storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags       = true;
        static constexpr auto reversed       = false;
        static constexpr auto with_successor = false;
        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, tail_flags, flag_op, input[0] /* successor */, storage.get().right);
    }

    /// \overload
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
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
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_discontinuity_int = rocprim::block_discontinuity<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_discontinuity_int::storage_type storage;
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
    ///     block_discontinuity_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_tails(tail_flags, tile_item, input, flag_op_type(),
    ///                                storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    T tile_successor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags       = true;
        static constexpr auto reversed       = false;
        static constexpr auto with_successor = true;
        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, tail_flags, flag_op, tile_successor_item, storage.get().right);
    }

    /// \overload
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
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
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_discontinuity_int = rocprim::block_discontinuity<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_discontinuity_int::storage_type storage;
    ///
    ///     // segment of consecutive items to be used
    ///     int input[8];
    ///     ...
    ///     int head_flags[8];
    ///     int tail_flags[8];
    ///     block_discontinuity_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads_and_tails(head_flags, tail_flags, input,
    ///                                          flag_op_type(), storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = false;
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
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
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
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_discontinuity_int = rocprim::block_discontinuity<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_discontinuity_int::storage_type storage;
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
    ///     block_discontinuity_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads_and_tails(head_flags, tail_flags, tile_item,
    ///                                          input, flag_op_type(),
    ///                                          storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = false;
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
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
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
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_discontinuity_int = rocprim::block_discontinuity<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_discontinuity_int::storage_type storage;
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
    ///     block_discontinuity_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads_and_tails(head_flags, tile_item, tail_flags,
    ///                                          input, flag_op_type(),
    ///                                          storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = false;
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
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
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
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Example.
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize discontinuity for int and a block of 128 threads
    ///     using block_discontinuity_int = rocprim::block_discontinuity<int, 128>;
    ///     // allocate storage in shared memory
    ///     __shared__ block_discontinuity_int::storage_type storage;
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
    ///     block_discontinuity_int b_discontinuity;
    ///     using flag_op_type = typename rocprim::greater<int>;
    ///     b_discontinuity.flag_heads_and_tails(head_flags, tile_predecessor_item,
    ///                                          tail_flags, tile_successor_item,
    ///                                          input, flag_op_type(),
    ///                                          storage);
    ///     ...
    /// }
    /// \endcode
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
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
        static constexpr auto reversed         = false;
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
    /// This overload does not accept a reference to temporary storage, instead it is declared as
    /// part of the function itself. Note that this does NOT decrease the shared memory requirements
    /// of a kernel using this function.
    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
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
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_
