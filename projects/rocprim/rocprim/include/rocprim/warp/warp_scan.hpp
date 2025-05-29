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

#ifndef ROCPRIM_WARP_WARP_SCAN_HPP_
#define ROCPRIM_WARP_WARP_SCAN_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../functional.hpp"
#include "../intrinsics.hpp"
#include "../types.hpp"

#include "detail/warp_scan_crosslane.hpp"
#include "detail/warp_scan_shared_mem.hpp"

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Select warp_scan implementation based VirtualWaveSize
template<class T, unsigned int VirtualWaveSize>
struct select_warp_scan_impl
{
    using type = typename std::conditional<
        // can we use crosslane (DPP or shuffle-based) implementation?
        detail::is_warpsize_shuffleable<VirtualWaveSize>::value,
        detail::warp_scan_crosslane<T, VirtualWaveSize>, // yes
        detail::warp_scan_shared_mem<T, VirtualWaveSize> // no
        >::type;
};

} // end namespace detail

/// \brief The warp_scan class is a warp level parallel primitive which provides methods
/// for performing inclusive and exclusive scan operations of items partitioned across
/// threads in a hardware warp.
///
/// \tparam T the input/output type.
/// \tparam VirtualWaveSize the size of logical warp size, which can be equal to or less than
/// the size of hardware warp (see rocprim::arch::wavefront::min_size()). Scan operations are performed
/// separately within groups determined by VirtualWaveSize.
///
/// \par Overview
/// * \p VirtualWaveSize must be equal to or less than the size of hardware warp (see
/// rocprim::arch::wavefront::min_size()). If it is less, scan is performed separately within groups
/// determined by VirtualWaveSize. \n
/// For example, if \p VirtualWaveSize is 4, hardware warp is 64, scan will be performed in logical
/// warps grouped like this: `{ {0, 1, 2, 3}, {4, 5, 6, 7 }, ..., {60, 61, 62, 63} }`
/// (thread is represented here by its id within hardware warp).
/// * Logical warp is a group of \p VirtualWaveSize consecutive threads from the same hardware warp.
/// * Supports non-commutative scan operators. However, a scan operator should be
/// associative. When used with non-associative functions the results may be non-deterministic
/// and/or vary in precision.
/// * Number of threads executing warp_scan's function must be a multiple of \p VirtualWaveSize;
/// * All threads from a logical warp must be in the same hardware warp.
///
/// \par Examples
/// \parblock
/// In the examples scan operation is performed on groups of 16 threads, each provides
/// one \p int value, result is returned using the same variable as for input. Hardware
/// warp size is 64. Block (tile) size is 64.
///
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // specialize warp_scan for int and logical warp of 16 threads
///     using warp_scan_int = rocprim::warp_scan<int, 16>;
///     // allocate storage in shared memory
///     __shared__ warp_scan_int::storage_type temp[4];
///
///     int logical_warp_id = threadIdx.x/16;
///     int value = ...;
///     // execute inclusive scan
///     warp_scan_int().inclusive_scan(
///         value, // input
///         value, // output
///         temp[logical_warp_id]
///     );
///     ...
/// }
/// \endcode
/// \endparblock
template<class T,
         unsigned int            VirtualWaveSize = arch::wavefront::min_size(),
         arch::wavefront::target TargetWaveSize  = arch::wavefront::get_target()>
class warp_scan
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_warp_scan_impl<T, VirtualWaveSize>::type
#endif
{
    using base_type = typename detail::select_warp_scan_impl<T, VirtualWaveSize>::type;

    // Check if VirtualWaveSize is valid for the targets
    static_assert(VirtualWaveSize <= ROCPRIM_MAX_WARP_SIZE,
                  "VirtualWaveSize can't be greater than hardware warp size.");

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

    /// \brief Performs inclusive scan across threads in a logical warp.
    ///
    /// \tparam BinaryFunction type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input thread input value.
    /// \param [out] output reference to a thread output value. May be aliased with \p input.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op binary operation function object that will be used for scan.
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
    /// The examples present inclusive min scan operations performed on groups of 32 threads,
    /// each provides one \p float value, result is returned using the same variable as for input.
    /// Hardware warp size is 64. Block (tile) size is 256.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // blockDim.x = 256
    /// {
    ///     // specialize warp_scan for float and logical warp of 32 threads
    ///     using warp_scan_f = rocprim::warp_scan<float, 32>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_scan_float::storage_type temp[8]; // 256/32 = 8
    ///
    ///     int logical_warp_id = threadIdx.x/32;
    ///     float value = ...;
    ///     // execute inclusive min scan
    ///     warp_scan_float().inclusive_scan(
    ///         value, // input
    ///         value, // output
    ///         temp[logical_warp_id],
    ///         rocprim::minimum<float>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the input values across threads in a block/tile are <tt>{1, -2, 3, -4, ..., 255, -256}</tt>, then
    /// output values in the first logical warp will be <tt>{1, -2, -2, -4, ..., -32},</tt> in the second:
    /// <tt>{33, -34, -34, -36, ..., -64}</tt> etc.
    /// \endparblock
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T              input,
                        T&             output,
                        storage_type&  storage,
                        BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
                return;
            }
        }
        base_type::inclusive_scan(input, output, storage, scan_op);
    }

    /// \brief Performs inclusive scan across threads in a logical warp.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T, T&, storage_type&, BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size "
                                 ". Aborting warp sort.");
        return;
    }

    /// \brief Performs seeded inclusive scan across threads in a logical warp.
    ///
    /// \tparam BinaryFunction type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input thread input value.
    /// \param [out] output reference to a thread output value. May be aliased with \p input.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] init initial value to seed the inclusive scan.
    /// \param [in] scan_op binary operation function object that will be used for scan.
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
    /// The examples present seeded inclusive min scan operations performed on groups of 32 threads,
    /// each provides one \p float value, result is returned using the same variable as for input.
    /// Hardware warp size is 64. Block (tile) size is 256.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // blockDim.x = 256
    /// {
    ///     // specialize warp_scan for float and logical warp of 32 threads
    ///     using warp_scan_f = rocprim::warp_scan<float, 32>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_scan_float::storage_type temp[8]; // 256/32 = 8
    ///
    ///     int logical_warp_id = threadIdx.x/32;
    ///     float value = ...;
    ///     float init  = ...;
    ///     // execute inclusive min scan
    ///     warp_scan_float().inclusive_scan(
    ///         value, // input
    ///         value, // output
    ///         temp[logical_warp_id],
    ///         init,
    ///         rocprim::minimum<float>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the input values across threads in a block/tile are <tt>{1, -2, 3, -4, ..., 255, -256}</tt>
    /// and the value for seeding the scan is -1, then output values in the first logical warp will be
    /// <tt>{-1, -2, -2, -4, ..., -32},</tt>, in the second: <tt>{-1, -34, -34, -36, ..., -64}</tt>, etc.
    /// \endparblock
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T              input,
                        T&             output,
                        storage_type&  storage,
                        T              init,
                        BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
            }
        }
        base_type::inclusive_scan(input, output, storage, scan_op, init);
    }

    /// \brief Performs seeded inclusive scan across threads in a logical warp.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T, T&, storage_type&, T, BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size "
                                 ". Aborting warp sort.");
        return;
    }

    /// \brief Performs inclusive scan and reduction across threads in a logical warp.
    ///
    /// \tparam BinaryFunction type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input thread input value.
    /// \param [out] output reference to a thread output value. May be aliased with \p input.
    /// \param [out] reduction result of reducing of all \p input values in logical warp.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op binary operation function object that will be used for scan.
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
    /// The examples present inclusive prefix sum operations performed on groups of 64 threads,
    /// each thread provides one \p int value. Hardware warp size is 64. Block (tile) size is 256.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // blockDim.x = 256
    /// {
    ///     // specialize warp_scan for int and logical warp of 64 threads
    ///     using warp_scan_int = rocprim::warp_scan<int, 64>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_scan_int::storage_type temp[4]; // 256/64 = 4
    ///
    ///     int logical_warp_id = threadIdx.x/64;
    ///     int input = ...;
    ///     int output, reduction;
    ///     // inclusive prefix sum
    ///     warp_scan_int().inclusive_scan(
    ///         input,
    ///         output,
    ///         reduction,
    ///         temp[logical_warp_id]
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block/tile are <tt>{1, 1, 1, 1, ..., 1, 1}</tt>, then
    /// \p output values in the every logical warp will be <tt>{1, 2, 3, 4, ..., 64}</tt>.
    /// The \p reduction will be equal \p 64.
    /// \endparblock
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T              input,
                        T&             output,
                        T&             reduction,
                        storage_type&  storage,
                        BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
            }
        }
        base_type::inclusive_scan(input, output, reduction, storage, scan_op);
    }

    /// \brief Performs inclusive scan and reduction across threads in a logical warp.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T, T&, T&, storage_type&, BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size "
                                 ". Aborting warp sort.");
        return;
    }

    /// \brief Performs seeded inclusive scan and reduction across threads in a logical warp.
    ///
    /// \tparam BinaryFunction type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input thread input value.
    /// \param [out] output reference to a thread output value. May be aliased with \p input.
    /// \param [out] reduction result of reducing of all \p input values in logical warp.
    /// This does not include \p init.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] init initial value to seed the inclusive scan.
    /// \param [in] scan_op binary operation function object that will be used for scan.
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
    /// The examples present seeded inclusive prefix sum operations performed on groups of 64 threads,
    /// each thread provides one \p int value. Hardware warp size is 64. Block (tile) size is 256.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // blockDim.x = 256
    /// {
    ///     // specialize warp_scan for int and logical warp of 64 threads
    ///     using warp_scan_int = rocprim::warp_scan<int, 64>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_scan_int::storage_type temp[4]; // 256/64 = 4
    ///
    ///     int logical_warp_id = threadIdx.x/64;
    ///     int input = ...;
    ///     int init  = ...;
    ///     int output, reduction;
    ///     // inclusive prefix sum
    ///     warp_scan_int().inclusive_scan(
    ///         input,
    ///         output,
    ///         reduction,
    ///         temp[logical_warp_id],
    ///         init
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the \p input values across threads in a block/tile are <tt>{1, 1, 1, 1, ..., 1, 1}</tt>
    /// and the value for seeding the scan is 1, then
    /// \p output values in the every logical warp will be <tt>{2, 3, 4, 5, ..., 65}</tt>.
    /// The \p reduction will be equal \p 65.
    /// \endparblock
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T              input,
                        T&             output,
                        T&             reduction,
                        storage_type&  storage,
                        T              init,
                        BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
            }
        }
        base_type::inclusive_scan(input, output, reduction, storage, scan_op, init);
    }

    /// \brief Performs seeded inclusive scan and reduction across threads in a logical warp.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T, T&, T&, storage_type&, T, BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size "
                                 ". Aborting warp sort.");
        return;
    }

    /// \brief Performs exclusive scan across threads in a logical warp.
    ///
    /// \tparam BinaryFunction type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input thread input value.
    /// \param [out] output reference to a thread output value. May be aliased with \p input.
    /// \param [in] init initial value used to start the exclusive scan. Should be the same
    /// for all threads in a logical warp.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op binary operation function object that will be used for scan.
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
    /// The examples present exclusive min scan operations performed on groups of 32 threads,
    /// each provides one \p float value, result is returned using the same variable as for input.
    /// Hardware warp size is 64. Block (tile) size is 256.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // blockDim.x = 256
    /// {
    ///     // specialize warp_scan for float and logical warp of 32 threads
    ///     using warp_scan_f = rocprim::warp_scan<float, 32>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_scan_float::storage_type temp[8]; // 256/32 = 8
    ///
    ///     int logical_warp_id = threadIdx.x/32;
    ///     float value = ...;
    ///     // execute exclusive min scan
    ///     warp_scan_float().exclusive_scan(
    ///         value, // input
    ///         value, // output
    ///         100.0f, // init
    ///         temp[logical_warp_id],
    ///         rocprim::minimum<float>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the initial value is \p 100 and input values across threads in a block/tile are
    /// <tt>{1, -2, 3, -4, ..., 255, -256}</tt>, then output values in the first logical
    /// warp will be <tt>{100, 1, -2, -2, -4, ..., -30},</tt> in the second:
    /// <tt>{100, 33, -34, -34, -36, ..., -62}</tt> etc.
    /// \endparblock
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T              input,
                        T&             output,
                        T              init,
                        storage_type&  storage,
                        BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
            }
        }
        base_type::exclusive_scan(input, output, init, storage, scan_op);
    }

    /// \brief Performs exclusive scan across threads in a logical warp.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T, T&, T, storage_type&, BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size "
                                 ". Aborting warp sort.");
        return;
    }

    /// \brief Performs exclusive scan and reduction across threads in a logical warp.
    ///
    /// \tparam BinaryFunction type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input thread input value.
    /// \param [out] output reference to a thread output value. May be aliased with \p input.
    /// \param [in] init initial value used to start the exclusive scan. Should be the same
    /// for all threads in a logical warp.
    /// \param [out] reduction result of reducing of all \p input values in logical warp.
    /// \p init value is not included in the reduction.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op binary operation function object that will be used for scan.
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
    /// The examples present exclusive prefix sum operations performed on groups of 64 threads,
    /// each thread provides one \p int value. Hardware warp size is 64. Block (tile) size is 256.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // blockDim.x = 256
    /// {
    ///     // specialize warp_scan for int and logical warp of 64 threads
    ///     using warp_scan_int = rocprim::warp_scan<int, 64>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_scan_int::storage_type temp[4]; // 256/64 = 4
    ///
    ///     int logical_warp_id = threadIdx.x/64;
    ///     int input = ...;
    ///     int output, reduction;
    ///     // exclusive prefix sum
    ///     warp_scan_int().exclusive_scan(
    ///         input,
    ///         output,
    ///         10, // init
    ///         reduction,
    ///         temp[logical_warp_id]
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the initial value is \p 10 and \p input values across threads in a block/tile are
    /// <tt>{1, 1, ..., 1, 1}</tt>, then \p output values in every logical warp will be
    /// <tt>{10, 11, 12, 13, ..., 73}</tt>. The \p reduction will be 64.
    /// \endparblock
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T              input,
                        T&             output,
                        T              init,
                        T&             reduction,
                        storage_type&  storage,
                        BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
            }
        }
        base_type::exclusive_scan(input, output, init, reduction, storage, scan_op);
    }

    /// \brief Performs exclusive scan and reduction across threads in a logical warp.
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T, T&, T, T&, storage_type&, BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size "
                                 ". Aborting warp sort.");
        return;
    }

    /// \brief Performs exclusive scan without an initial value across threads in a logical warp
    /// \tparam BinaryFunction binary function used for scan
    /// \param input Thread input value
    /// \param[out] output Reference to thread output value. Each threads value for the scan will
    /// be written to it. May be aliased with `input`. The value written is unspecified for the first
    /// thread of each logical warp.
    /// \param [in] storage Reference to a temporary storage object of type storage_type.
    /// \param scan_op The function object used to combine elements used for the scan
    template<class BinaryFunction                 = ::rocprim::plus<>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T              input,
                        T&             output,
                        storage_type&  storage,
                        BinaryFunction scan_op = BinaryFunction())
#ifndef DOXYGEN_DOCUMENTATION_BUILD
        -> std::enable_if_t<FunctionVirtualWaveSize <= arch::wavefront::max_size()>
#else
        -> void
#endif
    {
        base_type::exclusive_scan(input, output, storage, scan_op);
    }

    /// \cond
    template<class BinaryFunction                 = ::rocprim::plus<>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T /*input*/,
                        T& /*output*/,
                        storage_type& /*storage*/,
                        BinaryFunction /*scan_op*/ = BinaryFunction())
        -> std::enable_if_t<(FunctionVirtualWaveSize > arch::wavefront::max_size())>
    {
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size."
                                 " Aborting warp scan.");
    }
    /// \endcond

    /// \brief Performs exclusive scan and reduction without an initial value across threads in
    /// a logical warp
    /// \tparam BinaryFunction binary function used for scan
    /// \param input Thread input value
    /// \param[out] output Reference to thread output value. Each threads value for the scan will
    /// be written to it. May be aliased with `input`. The value written is unspecified for the first
    /// thread of each logical warp.
    /// \param[out] reduction Result of reducing of all `input` values in the logical warp.
    /// \param [in] storage Reference to a temporary storage object of type storage_type.
    /// \param scan_op The function object used to combine elements used for the scan
    template<class BinaryFunction                 = ::rocprim::plus<>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T              input,
                        T&             output,
                        storage_type&  storage,
                        T&             reduction,
                        BinaryFunction scan_op = BinaryFunction())
#ifndef DOXYGEN_DOCUMENTATION_BUILD
        -> std::enable_if_t<FunctionVirtualWaveSize <= arch::wavefront::max_size()>
#else
        -> void
#endif
    {
        base_type::exclusive_scan(input, output, storage, reduction, scan_op);
    }

    /// \cond
    template<class BinaryFunction                 = ::rocprim::plus<>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T /*input*/,
                        T& /*output*/,
                        storage_type& /*storage*/,
                        T& /*reduction*/,
                        BinaryFunction /*scan_op*/ = BinaryFunction())
        -> std::enable_if_t<(FunctionVirtualWaveSize > arch::wavefront::max_size())>
    {
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size."
                                 " Aborting warp scan.");
    }
    /// \endcond

    /// \brief Performs inclusive and exclusive scan operations across threads
    /// in a logical warp.
    ///
    /// \tparam BinaryFunction type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input thread input value.
    /// \param [out] inclusive_output reference to a thread inclusive-scan output value.
    /// \param [out] exclusive_output reference to a thread exclusive-scan output value.
    /// \param [in] init initial value used to start the exclusive scan. Should be the same
    /// for all threads in a logical warp.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op binary operation function object that will be used for scan.
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
    /// The examples present min inclusive and exclusive scan operations performed on groups of 32 threads,
    /// each provides one \p float value, result is returned using the same variable as for input.
    /// Hardware warp size is 64. Block (tile) size is 256.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // blockDim.x = 256
    /// {
    ///     // specialize warp_scan for float and logical warp of 32 threads
    ///     using warp_scan_f = rocprim::warp_scan<float, 32>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_scan_float::storage_type temp[8]; // 256/32 = 8
    ///
    ///     int logical_warp_id = threadIdx.x/32;
    ///     float input = ...;
    ///     float ex_output, in_output;
    ///     // execute exclusive min scan
    ///     warp_scan_float().scan(
    ///         input,
    ///         in_output,
    ///         ex_output,
    ///         100.0f, // init
    ///         temp[logical_warp_id],
    ///         rocprim::minimum<float>()
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the initial value is \p 100 and input values across threads in a block/tile are
    /// <tt>{1, -2, 3, -4, ..., 255, -256}</tt>, then \p in_output values in the first logical
    /// warp will be <tt>{1, -2, -2, -4, ..., -32},</tt> in the second:
    /// <tt>{33, -34, -34, -36, ..., -64}</tt> and so forth, \p ex_output values in the first
    /// logical warp will be <tt>{100, 1, -2, -2, -4, ..., -30},</tt> in the second:
    /// <tt>{100, 33, -34, -34, -36, ..., -62}</tt> etc.
    /// \endparblock
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scan(T              input,
              T&             inclusive_output,
              T&             exclusive_output,
              T              init,
              storage_type&  storage,
              BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
            }
        }
        base_type::scan(input, inclusive_output, exclusive_output, init, storage, scan_op);
    }

    /// \brief Performs inclusive and exclusive scan operations across threads
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scan(T, T&, T&, T, storage_type&, BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size "
                                 ". Aborting warp sort.");
        return;
    }

    /// \brief Performs inclusive and exclusive scan operations, and reduction across
    /// threads in a logical warp.
    ///
    /// \tparam BinaryFunction type of binary function used for scan. Default type
    /// is rocprim::plus<T>.
    ///
    /// \param [in] input thread input value.
    /// \param [out] inclusive_output reference to a thread inclusive-scan output value.
    /// \param [out] exclusive_output reference to a thread exclusive-scan output value.
    /// \param [in] init initial value used to start the exclusive scan. Should be the same
    /// for all threads in a logical warp.
    /// \param [out] reduction result of reducing of all \p input values in logical warp.
    /// \p init value is not included in the reduction.
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    /// \param [in] scan_op binary operation function object that will be used for scan.
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
    /// The examples present inclusive and exclusive prefix sum operations performed on groups
    /// of 64 threads, each thread provides one \p int value. Hardware warp size is 64.
    /// Block (tile) size is 256.
    ///
    /// \code{.cpp}
    /// __global__ void example_kernel(...) // blockDim.x = 256
    /// {
    ///     // specialize warp_scan for int and logical warp of 64 threads
    ///     using warp_scan_int = rocprim::warp_scan<int, 64>;
    ///     // allocate storage in shared memory
    ///     __shared__ warp_scan_int::storage_type temp[4]; // 256/64 = 4
    ///
    ///     int logical_warp_id = threadIdx.x/64;
    ///     int input = ...;
    ///     int in_output, ex_output, reduction;
    ///     // inclusive and exclusive prefix sum
    ///     warp_scan_int().scan(
    ///         input,
    ///         in_output,
    ///         ex_output,
    ///         init,
    ///         reduction,
    ///         temp[logical_warp_id]
    ///     );
    ///     ...
    /// }
    /// \endcode
    ///
    /// If the initial value is \p 10 and \p input values across threads in a block/tile are
    /// <tt>{1, 1, ..., 1, 1}</tt>, then \p in_output values in every logical warp will be
    /// <tt>{1, 2, 3, 4, ..., 63, 64}</tt>, and \p ex_output values in every logical warp will
    /// be <tt>{10, 11, 12, 13, ..., 73}</tt>. The \p reduction will be 64.
    /// \endparblock
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scan(T              input,
              T&             inclusive_output,
              T&             exclusive_output,
              T              init,
              T&             reduction,
              storage_type&  storage,
              BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()),
                                void>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
            }
        }
        base_type::scan(input,
                        inclusive_output,
                        exclusive_output,
                        init,
                        reduction,
                        storage,
                        scan_op);
    }

    /// \brief Performs inclusive and exclusive scan operations across threads
    /// Invalid Warp Size
    template<class BinaryFunction                 = ::rocprim::plus<T>,
             unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scan(T, T&, T&, T, T&, storage_type&, BinaryFunction scan_op = BinaryFunction()) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), void>::type
    {
        (void)scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size "
                                 ". Aborting warp sort.");
        return;
    }

    /// \brief Broadcasts value from one thread to all threads in logical warp.
    ///
    /// \param [in] input value to broadcast.
    /// \param [in] src_lane id of the thread whose value should be broadcasted
    /// \param [in] storage reference to a temporary storage object of type storage_type.
    ///
    /// \par Storage reusage
    /// Synchronization barrier should be placed before \p storage is reused
    /// or repurposed: \p __syncthreads() or \p rocprim::syncthreads().
    template<unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto broadcast(T input, const unsigned int src_lane, storage_type& storage) ->
        typename std::enable_if<(FunctionVirtualWaveSize <= arch::wavefront::max_size()), T>::type
    {
        if constexpr(TargetWaveSize == ::rocprim::arch::wavefront::target::dynamic)
        {
            if(VirtualWaveSize > ::rocprim::arch::wavefront::size())
            {
                ROCPRIM_PRINT_ERROR_ONCE(
                    "Specified warp size exceeds current hardware supported warp "
                    "size. Aborting warp sort.");
            }
        }
        return base_type::broadcast(input, src_lane, storage);
    }

    /// \brief Broadcasts value from one thread to all threads in logical warp.
    /// Invalid Warp Size
    template<unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto broadcast(T, const unsigned int, storage_type&) ->
        typename std::enable_if<(FunctionVirtualWaveSize > arch::wavefront::max_size()), T>::type
    {
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp "
                                 "size. Aborting warp sort.");
        return T();
    }

    /// \brief Broadcasts value from one thread to all threads in logical warp.
    /// Invalid Warp Size
    template<unsigned int FunctionVirtualWaveSize = VirtualWaveSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto broadcast(T, const unsigned int, storage_type&) ->
        typename std::enable_if<(!detail::is_power_of_two(FunctionVirtualWaveSize)), T>::type
        = delete;
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_SCAN_HPP_
