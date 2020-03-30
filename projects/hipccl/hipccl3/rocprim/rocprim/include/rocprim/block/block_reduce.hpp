// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_
#define ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_reduce_warp_reduce.hpp"
#include "detail/block_reduce_raking_reduce.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Available algorithms for block_reduce primitive.
enum class block_reduce_algorithm
{
    /// \brief A warp_reduce based algorithm.
    using_warp_reduce,
    /// \brief An algorithm which limits calculations to a single hardware warp.
    raking_reduce,
    /// \brief Default block_reduce algorithm.
    default_algorithm = using_warp_reduce,
};

namespace detail
{

// Selector for block_reduce algorithm which gives block reduce implementation
// type based on passed block_reduce_algorithm enum
template<block_reduce_algorithm Algorithm>
struct select_block_reduce_impl;

template<>
struct select_block_reduce_impl<block_reduce_algorithm::using_warp_reduce>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
    using type = block_reduce_warp_reduce<T, BlockSizeX, BlockSizeY, BlockSizeZ>;
};

template<>
struct select_block_reduce_impl<block_reduce_algorithm::raking_reduce>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
    using type = block_reduce_raking_reduce<T, BlockSizeX, BlockSizeY, BlockSizeZ>;
};

} // end namespace detail

/// \brief The block_reduce class is a block level parallel primitive which provides methods
/// for performing reductions operations on items partitioned across threads in a block.
///
/// \tparam T - the input/output type.
/// \tparam BlockSize - the number of threads in a block.
/// \tparam Algorithm - selected reduce algorithm, block_reduce_algorithm::default_algorithm by default.
///
/// \par Overview
/// * Supports non-commutative reduce operators. However, a reduce operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Computation can more efficient when:
///   * \p ItemsPerThread is greater than one,
///   * \p T is an arithmetic type,
///   * reduce operation is simple addition operator, and
///   * the number of threads in the block is a multiple of the hardware warp size (see rocprim::warp_size()).
/// * block_reduce has two alternative implementations: \p block_reduce_algorithm::using_warp_reduce
///   and block_reduce_algorithm::raking_reduce.
///
/// \par Examples
/// \parblock
/// In the examples reduce operation is performed on block of 192 threads, each provides
/// one \p int value, result is returned using the same variable as for input.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize warp_reduce for int and logical warp of 192 threads
///     using block_reduce_int = rocprim::block_reduce<int, 192>;
///     // allocate storage in shared memory
///     __shared__ block_reduce_int::storage_type storage;
///
///     int value = ...;
///     // execute reduce
///     block_reduce_int().reduce(
///         value, // input
///         value, // output
///         storage
///     );
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int BlockSizeX,
    block_reduce_algorithm Algorithm = block_reduce_algorithm::default_algorithm,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_reduce
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_reduce_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ>
#endif
{
    using base_type = typename detail::select_block_reduce_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ>;
public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords <tt>__shared__</tt>. It can be aliased to
    /// an externally allocated memory, or be a part of a union type with other storage types
    /// to increase shared memory reusability.
    using storage_type = typename base_type::storage_type;

    /// \brief Performs reduction across threads in a block.
    ///
    /// \tparam BinaryFunction - type of binary function used for reduce. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] reduce_op - binary operation function object that will be used for reduce.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// The examples present min reduce operations performed on a block of 256 threads,
    /// each provides one \p float value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 256
    /// {
    ///     // specialize block_reduce for float and block of 256 threads
    ///     using block_reduce_f = rocprim::block_reduce<float, 256>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_reduce_float::storage_type storage;
    ///
    ///     float input = ...;
    ///     float output;
    ///     // execute min reduce
    ///     block_reduce_float().reduce(
    ///         input,
    ///         output,
    ///         storage,
    ///         rocprim::minimum<float>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{1, -2, 3, -4, ..., 255, -256}</tt>, then
    /// \p output value will be <tt>{-256}</tt>.
    /// \endparblock
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void reduce(T input,
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, storage, reduce_op);
    }

    /// \overload
    /// \brief Performs reduction across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam BinaryFunction - type of binary function used for reduce. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] reduce_op - binary operation function object that will be used for reduce.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void reduce(T input,
                T& output,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, reduce_op);
    }

    /// \brief Performs reduction across threads in a block.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for reduce. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] reduce_op - binary operation function object that will be used for reduce.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// The examples present maximum reduce operations performed on a block of 128 threads,
    /// each provides two \p long value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 128
    /// {
    ///     // specialize block_reduce for long and block of 128 threads
    ///     using block_reduce_f = rocprim::block_reduce<long, 128>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_reduce_long::storage_type storage;
    ///
    ///     long input[2] = ...;
    ///     long output[2];
    ///     // execute max reduce
    ///     block_reduce_long().reduce(
    ///         input,
    ///         output,
    ///         storage,
    ///         rocprim::maximum<long>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{-1, 2, -3, 4, ..., -255, 256}</tt>, then
    /// \p output value will be <tt>{256}</tt>.
    /// \endparblock
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, storage, reduce_op);
    }

    /// \overload
    /// \brief Performs reduction across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for reduce. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] reduce_op - binary operation function object that will be used for reduce.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, reduce_op);
    }

    /// \brief Performs reduction across threads in a block.
    ///
    /// \tparam BinaryFunction - type of binary function used for reduce. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] valid_items - number of items that will be reduced in the block.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] reduce_op - binary operation function object that will be used for reduce.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    ///
    /// \par Examples
    /// \parblock
    /// The examples present min reduce operations performed on a block of 256 threads,
    /// each provides one \p float value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 256
    /// {
    ///     // specialize block_reduce for float and block of 256 threads
    ///     using block_reduce_f = rocprim::block_reduce<float, 256>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_reduce_float::storage_type storage;
    ///
    ///     float input = ...;
    ///     unsigned int valid_items = 250;
    ///     float output;
    ///     // execute min reduce
    ///     block_reduce_float().reduce(
    ///         input,
    ///         output,
    ///         valid_items,
    ///         storage,
    ///         rocprim::minimum<float>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    /// \endparblock
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void reduce(T input,
                T& output,
                unsigned int valid_items,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, valid_items, storage, reduce_op);
    }

    /// \overload
    /// \brief Performs reduction across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for reduce. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] valid_items - number of items that will be reduced in the block.
    /// \param [in] reduce_op - binary operation function object that will be used for reduce.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void reduce(T input,
                T& output,
                unsigned int valid_items,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, valid_items, reduce_op);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_
