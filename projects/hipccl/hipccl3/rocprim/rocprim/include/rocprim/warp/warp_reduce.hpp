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

#ifndef ROCPRIM_WARP_WARP_REDUCE_HPP_
#define ROCPRIM_WARP_WARP_REDUCE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "detail/warp_reduce_crosslane.hpp"
#include "detail/warp_reduce_shared_mem.hpp"

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Select warp_reduce implementation based WarpSize
template<class T, unsigned int WarpSize, bool UseAllReduce>
struct select_warp_reduce_impl
{
    typedef typename std::conditional<
        // can we use crosslane (DPP or shuffle-based) implementation?
        detail::is_warpsize_shuffleable<WarpSize>::value,
        detail::warp_reduce_crosslane<T, WarpSize, UseAllReduce>, // yes
        detail::warp_reduce_shared_mem<T, WarpSize, UseAllReduce> // no
    >::type type;
};

} // end namespace detail

/// \brief The warp_reduce class is a warp level parallel primitive which provides methods
/// for performing reduction operations on items partitioned across threads in a hardware
/// warp.
///
/// \tparam T - the input/output type.
/// \tparam WarpSize - the size of logical warp size, which can be equal to or less than
/// the size of hardware warp (see rocprim::warp_size()). Reduce operations are performed
/// separately within groups determined by WarpSize.
/// \tparam UseAllReduce - input parameter to determine whether to broadcast final reduction
/// value to all threads (default is false).
///
/// \par Overview
/// * \p WarpSize must be equal to or less than the size of hardware warp (see
/// rocprim::warp_size()). If it is less, reduce is performed separately within groups
/// determined by WarpSize. \n
/// For example, if \p WarpSize is 4, hardware warp is 64, reduction will be performed in logical
/// warps grouped like this: `{ {0, 1, 2, 3}, {4, 5, 6, 7 }, ..., {60, 61, 62, 63} }`
/// (thread is represented here by its id within hardware warp).
/// * Logical warp is a group of \p WarpSize consecutive threads from the same hardware warp.
/// * Supports non-commutative reduce operators. However, a reduce operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Number of threads executing warp_reduce's function must be a multiple of \p WarpSize;
/// * All threads from a logical warp must be in the same hardware warp.
///
/// \par Examples
/// \parblock
/// In the examples reduce operation is performed on groups of 16 threads, each provides
/// one \p int value, result is returned using the same variable as for input. Hardware
/// warp size is 64. Block (tile) size is 64.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize warp_reduce for int and logical warp of 16 threads
///     using warp_reduce_int = rocprim::warp_reduce<int, 16>;
///     // allocate storage in shared memory
///     __shared__ warp_reduce_int::storage_type temp[4];
///
///     int logical_warp_id = hipThreadIdx_x/16;
///     int value = ...;
///     // execute reduce
///     warp_reduce_int().reduce(
///         value, // input
///         value, // output
///         temp[logical_warp_id]
///     );
///     ...
/// }
/// \endcode
/// \endparblock
template<
    class T,
    unsigned int WarpSize = warp_size(),
    bool UseAllReduce = false
>
class warp_reduce
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_warp_reduce_impl<T, WarpSize, UseAllReduce>::type
#endif
{
    using base_type = typename detail::select_warp_reduce_impl<T, WarpSize, UseAllReduce>::type;

    // Check if WarpSize is correct
    static_assert(WarpSize <= warp_size(), "WarpSize can't be greater than hardware warp size.");

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

    /// \brief Performs reduction across threads in a logical warp.
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
    /// In the examples reduce operation is performed on groups of 16 threads, each provides
    /// one \p int value, result is returned using the same variable as for input. Hardware
    /// warp size is 64. Block (tile) size is 64.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize warp_reduce for int and logical warp of 16 threads
    ///     using warp_reduce_int = rocprim::warp_reduce<int, 16>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_reduce_int::storage_type temp[4];
    ///
    ///     int logical_warp_id = hipThreadIdx_x/16;
    ///     int value = ...;
    ///     // execute reduction
    ///     warp_reduce_int().reduce(
    ///         value, // input
    ///         value, // output
    ///         temp[logical_warp_id],
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
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, storage, reduce_op);
    }

    /// \brief Performs reduction across threads in a logical warp.
    ///
    /// \tparam BinaryFunction - type of binary function used for reduce. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] valid_items - number of items that will be reduced in the warp.
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
    /// In the examples reduce operation is performed on groups of 16 threads, each provides
    /// one \p int value, result is returned using the same variable as for input. Hardware
    /// warp size is 64. Block (tile) size is 64.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...)
    /// {
    ///     // specialize warp_reduce for int and logical warp of 16 threads
    ///     using warp_reduce_int = rocprim::warp_reduce<int, 16>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_reduce_int::storage_type temp[4];
    ///
    ///     int logical_warp_id = hipThreadIdx_x/16;
    ///     int value = ...;
    ///     int valid_items = 4;
    ///     // execute reduction
    ///     warp_reduce_int().reduce(
    ///         value, // input
    ///         value, // output
    ///         valid_items,
    ///         temp[logical_warp_id]
    ///     );
    ///     ...
    /// }
    /// \endcode
    /// \endparblock
    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void reduce(T input,
                T& output,
                int valid_items,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, valid_items, storage, reduce_op);
    }

    /// \brief Performs head-segmented reduction across threads in a logical warp.
    ///
    /// \tparam Flag - type of head flags. Must be contextually convertible to \p bool.
    /// \tparam BinaryFunction - type of binary function used for reduce. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] flag - thread head flag, \p true flags mark beginnings of segments.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] reduce_op - binary operation function object that will be used for reduce.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    template<class Flag, class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void head_segmented_reduce(T input,
                               T& output,
                               Flag flag,
                               storage_type& storage,
                               BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::head_segmented_reduce(input, output, flag, storage, reduce_op);
    }

    /// \brief Performs tail-segmented reduction across threads in a logical warp.
    ///
    /// \tparam Flag - type of tail flags. Must be contextually convertible to \p bool.
    /// \tparam BinaryFunction - type of binary function used for reduce. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input - thread input value.
    /// \param [out] output - reference to a thread output value. May be aliased with \p input.
    /// \param [in] flag - thread tail flag, \p true flags mark ends of segments.
    /// \param [in] storage - reference to a temporary storage object of type storage_type.
    /// \param [in] reduce_op - binary operation function object that will be used for reduce.
    /// The signature of the function should be equivalent to the following:
    /// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
    /// <tt>const &</tt>, but function object must not modify the objects passed to it.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    template<class Flag, class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE inline
    void tail_segmented_reduce(T input,
                               T& output,
                               Flag flag,
                               storage_type& storage,
                               BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::tail_segmented_reduce(input, output, flag, storage, reduce_op);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_REDUCE_HPP_
