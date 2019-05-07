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

#ifndef ROCPRIM_BLOCK_BLOCK_SCAN_HPP_
#define ROCPRIM_BLOCK_BLOCK_SCAN_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_scan_warp_scan.hpp"
#include "detail/block_scan_reduce_then_scan.hpp"

/// \addtogroup blockmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Available algorithms for block_scan primitive.
enum class block_scan_algorithm
{
    /// \brief A warp_scan based algorithm.
    using_warp_scan,
    /// \brief An algorithm which limits calculations to a single hardware warp.
    reduce_then_scan,
    /// \brief Default block_scan algorithm.
    default_algorithm = using_warp_scan,
};

namespace detail
{

// Selector for block_scan algorithm which gives block scan implementation
// type based on passed block_scan_algorithm enum
template<block_scan_algorithm Algorithm>
struct select_block_scan_impl;

template<>
struct select_block_scan_impl<block_scan_algorithm::using_warp_scan>
{
    template<class T, unsigned int BlockSize>
    using type = block_scan_warp_scan<T, BlockSize>;
};

template<>
struct select_block_scan_impl<block_scan_algorithm::reduce_then_scan>
{
    template<class T, unsigned int BlockSize>
    // When BlockSize is less than hardware warp size block_scan_warp_scan performs better than
    // block_scan_reduce_then_scan by specializing for warps
    using type = typename std::conditional<
                    (BlockSize <= ::rocprim::warp_size()),
                    block_scan_warp_scan<T, BlockSize>,
                    block_scan_reduce_then_scan<T, BlockSize>
                 >::type;
};

} // end namespace detail

/// \brief The block_scan class is a block level parallel primitive which provides methods
/// for performing inclusive and exclusive scan operations of items partitioned across
/// threads in a block.
///
/// \tparam T - the input/output type.
/// \tparam BlockSize - the number of threads in a block.
/// \tparam Algorithm - selected scan algorithm, block_scan_algorithm::default_algorithm by default.
///
/// \par Overview
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Computation can more efficient when:
///   * \p ItemsPerThread is greater than one,
///   * \p T is an arithmetic type,
///   * scan operation is simple addition operator, and
///   * the number of threads in the block is a multiple of the hardware warp size (see rocprim::warp_size()).
/// * block_scan has two alternative implementations: \p block_scan_algorithm::using_warp_scan
///   and block_scan_algorithm::reduce_then_scan.
///
/// \par Examples
/// \parblock
/// In the examples scan operation is performed on block of 192 threads, each provides
/// one \p int value, result is returned using the same variable as for input.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize warp_scan for int and logical warp of 192 threads
///     using block_scan_int = rocprim::block_scan<int, 192>;
///     // allocate storage in shared memory
///     __shared__ block_scan_int::storage_type storage;
///
///     int value = ...;
///     // execute inclusive scan
///     block_scan_int().inclusive_scan(
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
    unsigned int BlockSize,
    block_scan_algorithm Algorithm = block_scan_algorithm::default_algorithm
>
class block_scan
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_scan_impl<Algorithm>::template type<T, BlockSize>
#endif
{
    using base_type = typename detail::select_block_scan_impl<Algorithm>::template type<T, BlockSize>;
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

    /// \brief Performs inclusive scan across threads in a block.
    ///
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present inclusive min scan operations performed on a block of 256 threads,
    /// each provides one \p float value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 256
    /// {
    ///     // specialize block_scan for float and block of 256 threads
    ///     using block_scan_f = rocprim::block_scan<float, 256>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_float::storage_type storage;
    ///
    ///     float input = ...;
    ///     float output;
    ///     // execute inclusive min scan
    ///     block_scan_float().inclusive_scan(
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
    /// \p output values in will be <tt>{1, -2, -2, -4, ..., -254, -256}</tt>.
    /// \endparblock
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::inclusive_scan(input, output, storage, scan_op);
    }

    /// \overload
    /// \brief Performs inclusive scan across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::inclusive_scan(input, output, scan_op);
    }

    /// \brief Performs inclusive scan and reduction across threads in a block.
    ///
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [out] reduction - result of reducing of all \p input values in a block.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present inclusive min scan operations performed on a block of 256 threads,
    /// each provides one \p float value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 256
    /// {
    ///     // specialize block_scan for float and block of 256 threads
    ///     using block_scan_f = rocprim::block_scan<float, 256>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_float::storage_type storage;
    ///
    ///     float input = ...;
    ///     float output;
    ///     float reduction;
    ///     // execute inclusive min scan
    ///     block_scan_float().inclusive_scan(
    ///         input,
    ///         output,
    ///         reduction,
    ///         storage,
    ///         rocprim::minimum<float>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{1, -2, 3, -4, ..., 255, -256}</tt>, then
    /// \p output values in will be <tt>{1, -2, -2, -4, ..., -254, -256}</tt>, and the \p reduction will
    /// be <tt>-256</tt>.
    /// \endparblock
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::inclusive_scan(input, output, reduction, storage, scan_op);
    }

    /// \overload
    /// \brief Performs inclusive scan and reduction across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [out] reduction - result of reducing of all \p input values in a block.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::inclusive_scan(input, output, reduction, scan_op);
    }

    /// \brief Performs inclusive scan across threads in a block, and uses
    /// \p prefix_callback_op to generate prefix value for the whole block.
    ///
    /// \tparam PrefixCallback - type of the unary function object used for generating
    /// block-wide prefix value for the scan operation.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in,out] prefix_callback_op - function object for generating block prefix value.
    /// The signature of the \p prefix_callback_op should be equivalent to the following:
    /// <tt>T f(const T &block_reduction);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    /// The object will be called by the first warp of the block with block reduction of
    /// \p input values as input argument. The result of the first thread will be used as the
    /// block-wide prefix.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present inclusive prefix sum operations performed on a block of 256 threads,
    /// each thread provides one \p int value.
    ///
    /// \code{.cpp}
    ///
    /// struct my_block_prefix
    /// {
    ///     int prefix;
    ///
    ///     __device__ my_block_prefix(int prefix) : prefix(prefix) {}
    ///
    ///     __device__ int operator()(int block_reduction)
    ///     {
    ///         int old_prefix = prefix;
    ///         prefix = prefix + block_reduction;
    ///         return old_prefix;
    ///     }
    /// };
    ///
    /// __global__ void example_kernel(...) // hipBlockDim_x = 256
    /// {
    ///     // specialize block_scan for int and block of 256 threads
    ///     using block_scan_f = rocprim::block_scan<int, 256>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_int::storage_type storage;
    ///
    ///     // init prefix functor
    ///     my_block_prefix prefix_callback(10);
    ///
    ///     int input;
    ///     int output;
    ///     // execute inclusive prefix sum
    ///     block_scan_int().inclusive_scan(
    ///         input,
    ///         output,
    ///         storage,
    ///         prefix_callback,
    ///         rocprim::plus<int>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{1, 1, 1, ..., 1}</tt>, then
    /// \p output values in will be <tt>{11, 12, 13, ..., 266}</tt>, and the \p prefix will
    /// be <tt>266</tt>.
    /// \endparblock
    template<
        class PrefixCallback,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        base_type::inclusive_scan(input, output, storage, prefix_callback_op, scan_op);
    }

    /// \brief Performs inclusive scan across threads in a block.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present inclusive maximum scan operations performed on a block of 128 threads,
    /// each provides two \p long value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 128
    /// {
    ///     // specialize block_scan for long and block of 128 threads
    ///     using block_scan_f = rocprim::block_scan<long, 128>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_long::storage_type storage;
    ///
    ///     long input[2] = ...;
    ///     long output[2];
    ///     // execute inclusive min scan
    ///     block_scan_long().inclusive_scan(
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
    /// \p output values in will be <tt>{-1, 2, 2, 4, ..., 254, 256}</tt>.
    /// \endparblock
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], storage, scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, storage, scan_op);
        }
    }

    /// \overload
    /// \brief Performs inclusive scan across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, scan_op);
        }
    }

    /// \brief Performs inclusive scan and reduction across threads in a block.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [out] reduction - result of reducing of all \p input values in a block.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present inclusive maximum scan operations performed on a block of 128 threads,
    /// each provides two \p long value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 128
    /// {
    ///     // specialize block_scan for long and block of 128 threads
    ///     using block_scan_f = rocprim::block_scan<long, 128>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_long::storage_type storage;
    ///
    ///     long input[2] = ...;
    ///     long output[2];
    ///     long reduction;
    ///     // execute inclusive min scan
    ///     block_scan_long().inclusive_scan(
    ///         input,
    ///         output,
    ///         reduction,
    ///         storage,
    ///         rocprim::maximum<long>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{-1, 2, -3, 4, ..., -255, 256}</tt>, then
    /// \p output values in will be <tt>{-1, 2, 2, 4, ..., 254, 256}</tt> and the \p reduction will be \p 256.
    /// \endparblock
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], reduction, storage, scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, reduction, storage, scan_op);
        }
    }

    /// \overload
    /// \brief Performs inclusive scan and reduction across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [out] reduction - result of reducing of all \p input values in a block.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], reduction, scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, reduction, scan_op);
        }
    }

    /// \brief Performs inclusive scan across threads in a block, and uses
    /// \p prefix_callback_op to generate prefix value for the whole block.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam PrefixCallback - type of the unary function object used for generating
    /// block-wide prefix value for the scan operation.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in,out] prefix_callback_op - function object for generating block prefix value.
    /// The signature of the \p prefix_callback_op should be equivalent to the following:
    /// <tt>T f(const T &block_reduction);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    /// The object will be called by the first warp of the block with block reduction of
    /// \p input values as input argument. The result of the first thread will be used as the
    /// block-wide prefix.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present inclusive prefix sum operations performed on a block of 128 threads,
    /// each thread provides two \p int value.
    ///
    /// \code{.cpp}
    ///
    /// struct my_block_prefix
    /// {
    ///     int prefix;
    ///
    ///     __device__ my_block_prefix(int prefix) : prefix(prefix) {}
    ///
    ///     __device__ int operator()(int block_reduction)
    ///     {
    ///         int old_prefix = prefix;
    ///         prefix = prefix + block_reduction;
    ///         return old_prefix;
    ///     }
    /// };
    ///
    /// __global__ void example_kernel(...) // hipBlockDim_x = 128
    /// {
    ///     // specialize block_scan for int and block of 128 threads
    ///     using block_scan_f = rocprim::block_scan<int, 128>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_int::storage_type storage;
    ///
    ///     // init prefix functor
    ///     my_block_prefix prefix_callback(10);
    ///
    ///     int input[2] = ...;
    ///     int output[2];
    ///     // execute inclusive prefix sum
    ///     block_scan_int().inclusive_scan(
    ///         input,
    ///         output,
    ///         storage,
    ///         prefix_callback,
    ///         rocprim::plus<int>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{1, 1, 1, ..., 1}</tt>, then
    /// \p output values in will be <tt>{11, 12, 13, ..., 266}</tt>, and the \p prefix will
    /// be <tt>266</tt>.
    /// \endparblock
    template<
        unsigned int ItemsPerThread,
        class PrefixCallback,
        class BinaryFunction
    >
    ROCPRIM_DEVICE inline
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], storage, prefix_callback_op, scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, storage, prefix_callback_op, scan_op);
        }
    }

    /// \brief Performs exclusive scan across threads in a block.
    ///
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] init - initial value used to start the exclusive scan. Should be the same
    /// for all threads in a block.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present exclusive min scan operations performed on a block of 256 threads,
    /// each provides one \p float value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 256
    /// {
    ///     // specialize block_scan for float and block of 256 threads
    ///     using block_scan_f = rocprim::block_scan<float, 256>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_float::storage_type storage;
    ///
    ///     float init = ...;
    ///     float input = ...;
    ///     float output;
    ///     // execute exclusive min scan
    ///     block_scan_float().exclusive_scan(
    ///         input,
    ///         output,
    ///         init,
    ///         storage,
    ///         rocprim::minimum<float>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{1, -2, 3, -4, ..., 255, -256}</tt>
    /// and \p init is \p 0, then \p output values in will be <tt>{0, 0, -2, -2, -4, ..., -254, -254}</tt>.
    /// \endparblock
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::exclusive_scan(input, output, init, storage, scan_op);
    }

    /// \overload
    /// \brief Performs exclusive scan across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] init - initial value used to start the exclusive scan. Should be the same
    /// for all threads in a block.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::exclusive_scan(input, output, init, scan_op);
    }

    /// \brief Performs exclusive scan and reduction across threads in a block.
    ///
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] init - initial value used to start the exclusive scan. Should be the same
    /// for all threads in a block.
    /// \param [out] reduction - result of reducing of all \p input values in a block.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present exclusive min scan operations performed on a block of 256 threads,
    /// each provides one \p float value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 256
    /// {
    ///     // specialize block_scan for float and block of 256 threads
    ///     using block_scan_f = rocprim::block_scan<float, 256>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_float::storage_type storage;
    ///
    ///     float init = 0;
    ///     float input = ...;
    ///     float output;
    ///     float reduction;
    ///     // execute exclusive min scan
    ///     block_scan_float().exclusive_scan(
    ///         input,
    ///         output,
    ///         init,
    ///         reduction,
    ///         storage,
    ///         rocprim::minimum<float>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{1, -2, 3, -4, ..., 255, -256}</tt>
    /// and \p init is \p 0, then \p output values in will be <tt>{0, 0, -2, -2, -4, ..., -254, -254}</tt>
    /// and the \p reduction will be \p -256.
    /// \endparblock
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::exclusive_scan(input, output, init, reduction, storage, scan_op);
    }

    /// \overload
    /// \brief Performs exclusive scan and reduction across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] init - initial value used to start the exclusive scan. Should be the same
    /// for all threads in a block.
    /// \param [out] reduction - result of reducing of all \p input values in a block.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::exclusive_scan(input, output, init, reduction, scan_op);
    }

    /// \brief Performs exclusive scan across threads in a block, and uses
    /// \p prefix_callback_op to generate prefix value for the whole block.
    ///
    /// \tparam PrefixCallback - type of the unary function object used for generating
    /// block-wide prefix value for the scan operation.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in,out] prefix_callback_op - function object for generating block prefix value.
    /// The signature of the \p prefix_callback_op should be equivalent to the following:
    /// <tt>T f(const T &block_reduction);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    /// The object will be called by the first warp of the block with block reduction of
    /// \p input values as input argument. The result of the first thread will be used as the
    /// block-wide prefix.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present exclusive prefix sum operations performed on a block of 256 threads,
    /// each thread provides one \p int value.
    ///
    /// \code{.cpp}
    ///
    /// struct my_block_prefix
    /// {
    ///     int prefix;
    ///
    ///     __device__ my_block_prefix(int prefix) : prefix(prefix) {}
    ///
    ///     __device__ int operator()(int block_reduction)
    ///     {
    ///         int old_prefix = prefix;
    ///         prefix = prefix + block_reduction;
    ///         return old_prefix;
    ///     }
    /// };
    ///
    /// __global__ void example_kernel(...) // hipBlockDim_x = 256
    /// {
    ///     // specialize block_scan for int and block of 256 threads
    ///     using block_scan_f = rocprim::block_scan<int, 256>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_int::storage_type storage;
    ///
    ///     // init prefix functor
    ///     my_block_prefix prefix_callback(10);
    ///
    ///     int input;
    ///     int output;
    ///     // execute exclusive prefix sum
    ///     block_scan_int().exclusive_scan(
    ///         input,
    ///         output,
    ///         storage,
    ///         prefix_callback,
    ///         rocprim::plus<int>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{1, 1, 1, ..., 1}</tt>, then
    /// \p output values in will be <tt>{10, 11, 12, 13, ..., 265}</tt>, and the \p prefix will
    /// be <tt>266</tt>.
    /// \endparblock
    template<
        class PrefixCallback,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void exclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        base_type::exclusive_scan(input, output, storage, prefix_callback_op, scan_op);
    }

    /// \brief Performs exclusive scan across threads in a block.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] init - initial value used to start the exclusive scan. Should be the same
    /// for all threads in a block.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present exclusive maximum scan operations performed on a block of 128 threads,
    /// each provides two \p long value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 128
    /// {
    ///     // specialize block_scan for long and block of 128 threads
    ///     using block_scan_f = rocprim::block_scan<long, 128>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_long::storage_type storage;
    ///
    ///     long init = ...;
    ///     long input[2] = ...;
    ///     long output[2];
    ///     // execute exclusive min scan
    ///     block_scan_long().exclusive_scan(
    ///         input,
    ///         output,
    ///         init,
    ///         storage,
    ///         rocprim::maximum<long>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{-1, 2, -3, 4, ..., -255, 256}</tt>
    /// and \p init is 0, then \p output values in will be <tt>{0, 0, 2, 2, 4, ..., 254, 254}</tt>.
    /// \endparblock
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], init, storage, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, init, storage, scan_op);
        }
    }

    /// \overload
    /// \brief Performs exclusive scan across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] init - initial value used to start the exclusive scan. Should be the same
    /// for all threads in a block.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], init, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, init, scan_op);
        }
    }

    /// \brief Performs exclusive scan and reduction across threads in a block.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] init - initial value used to start the exclusive scan. Should be the same
    /// for all threads in a block.
    /// \param [out] reduction - result of reducing of all \p input values in a block.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present exclusive maximum scan operations performed on a block of 128 threads,
    /// each provides two \p long value.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // hipBlockDim_x = 128
    /// {
    ///     // specialize block_scan for long and block of 128 threads
    ///     using block_scan_f = rocprim::block_scan<long, 128>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_long::storage_type storage;
    ///
    ///     long init = ...;
    ///     long input[2] = ...;
    ///     long output[2];
    ///     long reduction;
    ///     // execute exclusive min scan
    ///     block_scan_long().exclusive_scan(
    ///         input,
    ///         output,
    ///         init,
    ///         reduction,
    ///         storage,
    ///         rocprim::maximum<long>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{-1, 2, -3, 4, ..., -255, 256}</tt>
    /// and \p init is 0, then \p output values in will be <tt>{0, 0, 2, 2, 4, ..., 254, 254}</tt>
    /// and the \p reduction will be \p 256.
    /// \endparblock
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], init, reduction, storage, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, init, reduction, storage, scan_op);
        }
    }

    /// \overload
    /// \brief Performs exclusive scan and reduction across threads in a block.
    ///
    /// * This overload does not accept storage argument. Required shared memory is
    /// allocated by the method itself.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] init - initial value used to start the exclusive scan. Should be the same
    /// for all threads in a block.
    /// \param [out] reduction - result of reducing of all \p input values in a block.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], init, reduction, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, init, reduction, scan_op);
        }
    }

    /// \brief Performs exclusive scan across threads in a block, and uses
    /// \p prefix_callback_op to generate prefix value for the whole block.
    ///
    /// \tparam ItemsPerThread - number of items in the \p input array.
    /// \tparam PrefixCallback - type of the unary function object used for generating
    /// block-wide prefix value for the scan operation.
    /// \tparam BinaryFunction - type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - reference to an array containing thread input values.
    /// \param [out] output - reference to a thread output array. May be aliased with \p input.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in,out] prefix_callback_op - function object for generating block prefix value.
    /// The signature of the \p prefix_callback_op should be equivalent to the following:
    /// <tt>T f(const T &block_reduction);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    /// The object will be called by the first warp of the block with block reduction of
    /// \p input values as input argument. The result of the first thread will be used as the
    /// block-wide prefix.
    /// \param [in] scan_op - binary operation function object that will be used for scan.
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
    /// The examples present exclusive prefix sum operations performed on a block of 128 threads,
    /// each thread provides two \p int value.
    ///
    /// \code{.cpp}
    ///
    /// struct my_block_prefix
    /// {
    ///     int prefix;
    ///
    ///     __device__ my_block_prefix(int prefix) : prefix(prefix) {}
    ///
    ///     __device__ int operator()(int block_reduction)
    ///     {
    ///         int old_prefix = prefix;
    ///         prefix = prefix + block_reduction;
    ///         return old_prefix;
    ///     }
    /// };
    ///
    /// __global__ void example_kernel(...) // hipBlockDim_x = 128
    /// {
    ///     // specialize block_scan for int and block of 128 threads
    ///     using block_scan_f = rocprim::block_scan<int, 128>;
    ///     // allocate storage in shared memory for the block
    ///     __shared__ block_scan_int::storage_type storage;
    ///
    ///     // init prefix functor
    ///     my_block_prefix prefix_callback(10);
    ///
    ///     int input[2] = ...;
    ///     int output[2];
    ///     // execute exclusive prefix sum
    ///     block_scan_int().exclusive_scan(
    ///         input,
    ///         output,
    ///         storage,
    ///         prefix_callback,
    ///         rocprim::plus<int>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block are <tt>{1, 1, 1, ..., 1}</tt>, then
    /// \p output values in will be <tt>{10, 11, 12, 13, ..., 265}</tt>, and the \p prefix will
    /// be <tt>266</tt>.
    /// \endparblock
    template<
        unsigned int ItemsPerThread,
        class PrefixCallback,
        class BinaryFunction
    >
    ROCPRIM_DEVICE inline
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], storage, prefix_callback_op, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, storage, prefix_callback_op, scan_op);
        }
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group blockmodule

#endif // ROCPRIM_BLOCK_BLOCK_SCAN_HPP_
