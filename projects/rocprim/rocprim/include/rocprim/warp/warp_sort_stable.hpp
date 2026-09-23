// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_WARP_WARP_SORT_STABLE_HPP_
#define ROCPRIM_WARP_WARP_SORT_STABLE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../functional.hpp"
#include "../intrinsics.hpp"

#include "detail/warp_sort_shuffle_stable.hpp"
#include "detail/warp_sort_stable.hpp"

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Available algorithms for warp_sort_stable primitive.
enum class warp_sort_stable_algorithm
{
    /// \brief A merge path based algorithm using Shared Memory (LDS).
    merge_path,

    /// \brief A bitonic-sort based algorithm using Register Shuffles.
    /// This implementation tracks original indices to maintain stability.
    /// It consumes no Shared Memory but may have higher register pressure.
    shuffle
};

/// \brief The warp_sort_stable class provides stable sort methods for items partitioned
/// across a thread warp.
///
/// \tparam Key Data type for parameter Key
/// \tparam BlockSize The number of threads in a block (required for shared memory allocation in some algorithms)
/// \tparam VirtualWaveSize [optional] The number of threads in a warp. Must be power of 2.
/// \tparam ItemsPerThread [optional] The number of items processed by each thread.
/// \tparam Value [optional] Data type for parameter Value. By default, it's empty_type.
/// \tparam Algorithm [optional] Selected sort algorithm.
///
/// \par Overview
/// * \p VirtualWaveSize must be power of two.
/// * \p VirtualWaveSize must be equal to or less than the size of hardware warp.
/// * \p BlockSize must be passed because the default algorithm (merge path) requires block-wide shared memory allocation.
/// * Stable sort preserves the relative order of elements with equivalent keys.
///
/// \par Examples
/// \parblock
/// \code{.cpp}
/// __global__ void example_kernel(...)
/// {
///     // Specializing for int key, float value, 256 threads per block, logical warp of 64, 4 items per thread
///     using wsort_t = rocprim::warp_sort_stable<int, 256, 64, 4, float>;
///
///     __shared__ wsort_t::storage_type storage;
///
///     int keys[4] = ...;
///     float values[4] = ...;
///
///     wsort_t().sort(keys, values, storage);
/// }
/// \endcode
/// \endparblock
template<class Key,
         unsigned int BlockSize,
         unsigned int VirtualWaveSize         = arch::wavefront::min_size(),
         unsigned int ItemsPerThread          = 1,
         class Value                          = empty_type,
         warp_sort_stable_algorithm Algorithm = warp_sort_stable_algorithm::merge_path>
class warp_sort_stable
{
    // Always alias merge_path backend (needed for storage and fallback).
    // Merge path is faster than bitonic shuffle sort on non-full waves, i.e. where
    // 'input_size' is used.
    using merge_path_impl
        = detail::warp_sort_stable<Key, BlockSize, VirtualWaveSize, ItemsPerThread, Value>;

    // Select shuffle backend if requested, otherwise fallback to merge_path type
    // (the fallback type is technically unused due to if constexpr, but must be valid).
    using chosen_impl = std::conditional_t<
        Algorithm == warp_sort_stable_algorithm::shuffle,
        detail::warp_sort_shuffle_stable<Key, BlockSize, VirtualWaveSize, ItemsPerThread, Value>,
        merge_path_impl>;

public:
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// \note Storage is always merge_path storage (even for shuffle Algorithm),
    /// because shuffle backend cannot safely handle partial input without padding.
    using storage_type = typename merge_path_impl::storage_type;

    /// \brief Stable sort for Key only (Per-thread single value).
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key, BinaryFunction compare_function = BinaryFunction())
    {
        chosen_impl().sort(thread_key, compare_function);
    }

    /// \brief Stable sort for Key only with storage (Per-thread single value).
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&           thread_key,
              storage_type&  storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        if constexpr(Algorithm == warp_sort_stable_algorithm::shuffle)
        {
            (void)storage;
            chosen_impl().sort(thread_key, compare_function);
        }
        else
        {
            chosen_impl().sort(thread_key, storage, compare_function);
        }
    }

    /// \brief Stable sort for Key Array (Per-thread array).
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction())
    {
        chosen_impl().sort(thread_keys, compare_function);
    }

    /// \brief Stable sort for Key Array with storage (Per-thread array).
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        if constexpr(Algorithm == warp_sort_stable_algorithm::shuffle)
        {
            (void)storage;
            chosen_impl().sort(thread_keys, compare_function);
        }
        else
        {
            chosen_impl().sort(thread_keys, storage, compare_function);
        }
    }

    /// \brief Stable sort for Key Array with storage and valid input size.
    ///
    /// \param input_size The number of valid items in the warp.
    /// \note This function automatically falls back to the `merge_path` algorithm
    /// regardless of the template `Algorithm` parameter, as `shuffle` does not support partial inputs.
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              storage_type&      storage,
              const unsigned int input_size,
              BinaryFunction     compare_function = BinaryFunction())
    {
        // Shuffle impl ignores input_size, so we must use merge_path for correctness.
        merge_path_impl().sort(thread_keys, storage, input_size, compare_function);
    }

    /// \brief Stable sort for Key-Value pair (Per-thread single value).
    template<class BinaryFunction = ::rocprim::less<Key>, class V = Value>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&           thread_key,
              Value&         thread_value,
              BinaryFunction compare_function = BinaryFunction())
    {
        chosen_impl().sort(thread_key, thread_value, compare_function);
    }

    /// \brief Stable sort for Key-Value pair with storage (Per-thread single value).
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&           thread_key,
              Value&         thread_value,
              storage_type&  storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        if constexpr(Algorithm == warp_sort_stable_algorithm::shuffle)
        {
            (void)storage;
            chosen_impl().sort(thread_key, thread_value, compare_function);
        }
        else
        {
            chosen_impl().sort(thread_key, thread_value, storage, compare_function);
        }
    }

    /// \brief Stable sort for Key-Value pair with storage and valid input size (Per-thread single value).
    ///
    /// \note This function automatically falls back to the `merge_path` algorithm
    /// regardless of the template `Algorithm` parameter, as `shuffle` does not support partial inputs.
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key&               thread_key,
              Value&             thread_value,
              storage_type&      storage,
              const unsigned int input_size,
              BinaryFunction     compare_function = BinaryFunction())
    {
        merge_path_impl().sort(thread_key, thread_value, storage, input_size, compare_function);
    }

    /// \brief Stable sort for Key-Value Arrays (Per-thread array).
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction())
    {
        chosen_impl().sort(thread_keys, thread_values, compare_function);
    }

    /// \brief Stable sort for Key-Value Arrays with storage (Per-thread array).
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        if constexpr(Algorithm == warp_sort_stable_algorithm::shuffle)
        {
            (void)storage;
            chosen_impl().sort(thread_keys, thread_values, compare_function);
        }
        else
        {
            chosen_impl().sort(thread_keys, thread_values, storage, compare_function);
        }
    }

    /// \brief Stable sort for Key-Value Arrays with storage and valid input size.
    ///
    /// \note This function automatically falls back to the `merge_path` algorithm
    /// regardless of the template `Algorithm` parameter, as `shuffle` does not support partial inputs.
    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type&      storage,
              const unsigned int input_size,
              BinaryFunction     compare_function = BinaryFunction())
    {
        merge_path_impl().sort(thread_keys, thread_values, storage, input_size, compare_function);
    }
};

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_SORT_STABLE_HPP_
