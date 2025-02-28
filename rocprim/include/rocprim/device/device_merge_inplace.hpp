// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_HPP_

#include "../config.hpp"
#include "../functional.hpp"

#include "../detail/temp_storage.hpp"
#include "../device/config_types.hpp"
#include "../device/device_merge.hpp"
#include "../device/device_transform.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

/// \brief Parallel merge inplace primitive for device level.
///
/// The `merge_inplace` function performs a device-wide merge in place. It merges two ordered sets
/// of input values based on a comparison function using significantly less temporary storage
/// compared to `merge`.
///
/// \warning This function prioritizes temporary storage over speed. In most cases using `merge`
/// and a device copy is significantly faster.
///
/// \warning This function uses cooperative groups. Please make sure that the device and platform
/// supports cooperative kernel launches.
///
/// \par Overview
/// * The function can write intermediate values to the data array while the algorithm is running.
/// * Returns the required size of `temporary_storage` in `storage_size` if `temporary_storage` is a
/// null pointer.
/// * Accepts a custom `compare_function`.
///
/// \tparam Config Configuration of the primitive, must be `default_config`.
/// \tparam Iterator Random access iterator type for the input and output range. Must meet the
/// requirements of `std::random_access_iterator`.
/// \tparam BinaryFunction Binary function type that is used for the comparison.
///
/// \param [in] temporary_storage Pointer to a device-accessible temporary storage. When a null
/// pointer is passed the required allocation size in bytes is written to `storage_size` and the
/// function returns `hipSuccess` without performing the merge operation.
/// \param [in,out] storage_size Reference to size in bytes of `temporary_storage`.
/// \param [in,out] data Iterator to the first value to merge.
/// \param [in] left_size Number of elements in the first input range.
/// \param [in] right_size Number of elements in the second input range.
/// \param [in] compare_function Binary operation function that will be used for comparison. The
/// signature of the function should be equivalent to the following: `bool f(const T &a, const T &b);`.
/// The signature does not need to have `const &`, but the function object must not modify
/// the objects passed to it. The default value is `BinaryFunction()`.
/// \param [in] stream The HIP stream object. Default is `0` (`hipDefaultStream`).
/// \param [in] debug_synchronous If `true`, forces a device synchronization after every kernel
/// launch in order to check for errors. Default value is `false`.
///
/// \returns `hipSuccess` (`0`) after succesful merge; otherwise a HIP runtime error of type
/// `hipError_t`.
///
/// \par Example
/// \parblock
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// size_t left_size;  // e.g. 4
/// size_t right_size; // e.g. 4
/// int*   data;       // e.g. [1, 3, 5, 7, 0, 2, 4, 6]
/// // output: [0, 1, 2, 3, 4, 5, 6, 7]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
///
/// rocprim::merge_inplace(
///     temporary_storage_ptr,
///     temporary_storage_size_bytes,
///     data,
///     left_size,
///     right_size);
///
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// rocprim::merge_inplace(
///     temporary_storage_ptr,
///     temporary_storage_size_bytes,
///     data,
///     left_size,
///     right_size);
/// \endcode
/// \endparblock
template<class Config = default_config,
         class Iterator,
         class BinaryFunction
         = ::rocprim::less<typename std::iterator_traits<Iterator>::value_type>>
inline hipError_t merge_inplace(void*             temporary_storage,
                                size_t&           storage_size,
                                Iterator          data,
                                size_t            left_size,
                                size_t            right_size,
                                BinaryFunction    compare_function  = BinaryFunction(),
                                const hipStream_t stream            = 0,
                                bool              debug_synchronous = false)
{
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    // Get temporary size of normal merge.
    size_t merge_storage_size = 0;
    ROCPRIM_RETURN_ON_ERROR(::rocprim::merge(nullptr,
                                             merge_storage_size,
                                             data,
                                             data + left_size,
                                             data,
                                             left_size,
                                             right_size,
                                             compare_function,
                                             stream,
                                             debug_synchronous));

    // Create temporary storage for duplicated input and merge.
    void*       merge_storage;
    value_type* copied_input;
    ROCPRIM_RETURN_ON_ERROR(detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&copied_input, left_size + right_size),
            detail::temp_storage::make_partition(&merge_storage, merge_storage_size))));

    // Return required temporary storage if no temporary storage is given.
    if(temporary_storage == nullptr)
    {
        return hipSuccess;
    }

    // Duplicate the input so we can use data as output.
    ROCPRIM_RETURN_ON_ERROR(rocprim::transform(data,
                                               copied_input,
                                               left_size + right_size,
                                               ::rocprim::identity<value_type>{},
                                               stream,
                                               debug_synchronous));

    // Merge from copied input to data.
    ROCPRIM_RETURN_ON_ERROR(rocprim::merge(merge_storage,
                                           merge_storage_size,
                                           copied_input,
                                           copied_input + left_size,
                                           data,
                                           left_size,
                                           right_size,
                                           compare_function,
                                           stream,
                                           debug_synchronous));

    return hipSuccess;
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_HPP_
