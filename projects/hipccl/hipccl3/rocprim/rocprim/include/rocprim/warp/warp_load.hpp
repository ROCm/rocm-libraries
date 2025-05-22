// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_WARP_WARP_LOAD_HPP_
#define ROCPRIM_WARP_WARP_LOAD_HPP_

#include "../block/block_load_func.hpp"
#include "../config.hpp"
#include "../detail/various.hpp"
#include "../intrinsics/arch.hpp"

#include "warp_exchange.hpp"

/// \addtogroup warpmodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief \p warp_load_method enumerates the methods available to load data
/// from continuous memory into a blocked/striped arrangement of items across the warp
enum class warp_load_method
{
    /// Data from continuous memory is loaded into a blocked arrangement of items.
    /// \par Performance Notes:
    /// * Performance decreases with increasing number of items per thread (stride
    /// between reads), because of reduced memory coalescing.
    warp_load_direct,

    /// A striped arrangement of data is read directly from memory.
    warp_load_striped,

    /// Data from continuous memory is loaded into a blocked arrangement of items
    /// using vectorization as an optimization.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, provided that
    /// vectorization requirements are fulfilled. Otherwise, performance will default
    /// to \p warp_load_direct.
    /// \par Requirements:
    /// * The input offset (\p block_input) must be quad-item aligned.
    /// * The following conditions will prevent vectorization and switch to default
    /// \p warp_load_direct:
    ///   * \p ItemsPerThread is odd.
    ///   * The datatype \p T is not a primitive or a HIP vector type (e.g. int2,
    /// int4, etc.
    warp_load_vectorize,

    /// A striped arrangement of data from continuous memory is locally transposed
    /// into a blocked arrangement of items.
    /// \par Performance Notes:
    /// * Performance remains high due to increased memory coalescing, regardless of the
    /// number of items per thread.
    /// * Performance may be better compared to \p warp_load_direct and
    /// \p warp_load_vectorize due to reordering on local memory.
    warp_load_transpose,

    /// Defaults to \p warp_load_direct
    default_method = warp_load_direct
};

/// \brief The \p warp_load class is a warp level parallel primitive which provides methods
/// for loading data from continuous memory into a blocked arrangement of items across a warp.
///
/// \tparam T the input/output type.
/// \tparam ItemsPerThread the number of items to be processed by
/// each thread.
/// \tparam VirtualWaveSize the number of threads in the warp. It must be a divisor of the
/// kernel block size.
/// \tparam Method the method to load data.
/// \tparam TargetWaveSize The hardware wavefront size. It can be used to specialize
/// the targeted wavefront size when compiling to SPIR-V.
///
/// \par Overview
/// * The \p warp_load class has a number of different methods to load data:
///   * [warp_load_direct](\ref warp_load_method::warp_load_direct)
///   * [warp_load_striped](\ref warp_load_method::warp_load_striped)
///   * [warp_load_vectorize](\ref warp_load_method::warp_load_vectorize)
///   * [warp_load_transpose](\ref warp_load_method::warp_load_transpose)
///
/// \par Example:
/// \parblock
/// In the example a load operation is performed on a warp of 8 threads, using type
/// \p int and 4 items per thread.
///
/// \code{.cpp}
/// __global__ void example_kernel(int * input, ...)
/// {
///     constexpr unsigned int threads_per_block = 128;
///     constexpr unsigned int threads_per_warp  =   8;
///     constexpr unsigned int items_per_thread  =   4;
///     constexpr unsigned int warps_per_block   = threads_per_block / threads_per_warp;
///     const unsigned int warp_id = hipThreadIdx_x / threads_per_warp;
///     const int offset = blockIdx.x * threads_per_block * items_per_thread
///         + warp_id * threads_per_warp * items_per_thread;
///     int items[items_per_thread];
///     rocprim::warp_load<int, items_per_thread, threads_per_warp, load_method> warp_load;
///     warp_load.load(input + offset, items);
///     ...
/// }
/// \endcode
/// \endparblock
template<class T,
         unsigned int     ItemsPerThread,
         unsigned int     VirtualWaveSize = ::rocprim::arch::wavefront::min_size(),
         warp_load_method Method          = warp_load_method::warp_load_direct,
         ::rocprim::arch::wavefront::target TargetWaveSize
         = ::rocprim::arch::wavefront::get_target(),
         typename Enabled = void>
class warp_load
{
    static_assert(::rocprim::detail::is_power_of_two(VirtualWaveSize),
                  "Logical warp size must be a power of two.");

private:
    using storage_type_ = typename ::rocprim::detail::empty_storage_type;

public:
    ROCPRIM_INLINE ROCPRIM_HOST_DEVICE warp_load()
    {
        detail::check_virtual_wave_size<VirtualWaveSize>();
    }
    /// \brief Struct used to allocate a temporary memory that is required for thread
    /// communication during operations provided by related parallel primitive.
    ///
    /// Depending on the implemention the operations exposed by parallel primitive may
    /// require a temporary storage for thread communication. The storage should be allocated
    /// using keywords \p __shared__. It can be aliased to
    /// an externally allocated memory, or be a part of a union with other storage types
    /// to increase shared memory reusability.
    #ifndef DOXYGEN_SHOULD_SKIP_THIS // hides storage_type implementation for Doxygen
    using storage_type = typename ::rocprim::detail::empty_storage_type;
    #else
    using storage_type = storage_type_; // only for Doxygen
    #endif

    /// \brief Loads data from continuous memory into an arrangement of items across the
    /// warp.
    ///
    /// \tparam InputIterator [inferred] an iterator type for input (can be a simple
    /// pointer.
    ///
    /// \param [in] input the input iterator to load from.
    /// \param [out] items array that data is loaded to.
    /// \param [in] - temporary storage for inputs.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_blocked(flat_id, input, items);
    }

    /// \brief Loads data from continuous memory into an arrangement of items across the
    /// warp.
    ///
    /// \tparam InputIterator [inferred] an iterator type for input (can be a simple
    /// pointer.
    ///
    /// \param [in] input the input iterator to load from.
    /// \param [out] items array that data is loaded to.
    /// \param [in] valid maximum range of valid numbers to load.
    /// \param [in] - temporary storage for inputs.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_blocked(flat_id, input, items, valid);
    }

    /// \brief Loads data from continuous memory into an arrangement of items across the
    /// warp.
    ///
    /// \tparam InputIterator [inferred] an iterator type for input (can be a simple
    /// pointer.
    ///
    /// \param [in] input the input iterator to load from.
    /// \param [out] items array that data is loaded to.
    /// \param [in] valid maximum range of valid numbers to load.
    /// \param [in] out_of_bounds default value assigned to out-of-bound items.
    /// \param [in] - temporary storage for inputs.
    ///
    /// \par Overview
    /// * The type \p T must be such that an object of type \p InputIterator
    /// can be dereferenced and then implicitly converted to \p T.
    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_blocked(flat_id, input, items, valid,
                                  out_of_bounds);
    }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class T,
         unsigned int     ItemsPerThread,
         unsigned int     VirtualWaveSize,
         warp_load_method Method>
class warp_load<T,
                ItemsPerThread,
                VirtualWaveSize,
                Method,
                ::rocprim::arch::wavefront::target::dynamic>
{
private:
    using warp_load_wave32 = warp_load<T,
                                       ItemsPerThread,
                                       VirtualWaveSize,
                                       Method,
                                       ::rocprim::arch::wavefront::target::size32>;
    using warp_load_wave64 = warp_load<T,
                                       ItemsPerThread,
                                       VirtualWaveSize,
                                       Method,
                                       ::rocprim::arch::wavefront::target::size64>;

    using dispatch = detail::dispatch_wave_size<warp_load_wave32, warp_load_wave64>;

public:
    using storage_type = typename dispatch::storage_type;

    template<typename... Args>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto load(Args&&... args)
    {
        dispatch{}([](auto impl, auto&&... args) { impl.load(args...); }, args...);
    }
};

template<class T,
         unsigned int                       ItemsPerThread,
         unsigned int                       VirtualWaveSize,
         ::rocprim::arch::wavefront::target TargetWaveSize>
class warp_load<T,
                ItemsPerThread,
                VirtualWaveSize,
                warp_load_method::warp_load_striped,
                TargetWaveSize,
                ::rocprim::detail::wave_target_guard_t<TargetWaveSize>>
{
    static_assert(::rocprim::detail::is_power_of_two(VirtualWaveSize),
                  "Logical warp size must be a power of two.");

public:
    ROCPRIM_INLINE ROCPRIM_HOST_DEVICE warp_load()
    {
        detail::check_virtual_wave_size<VirtualWaveSize>();
    }

    using storage_type = typename ::rocprim::detail::empty_storage_type;

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_warp_striped<VirtualWaveSize>(flat_id, input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_warp_striped<VirtualWaveSize>(flat_id, input, items, valid);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_warp_striped<VirtualWaveSize>(flat_id,
                                                        input,
                                                        items,
                                                        valid,
                                                        out_of_bounds);
    }
};

template<class T,
         unsigned int                       ItemsPerThread,
         unsigned int                       VirtualWaveSize,
         ::rocprim::arch::wavefront::target TargetWaveSize>
class warp_load<T,
                ItemsPerThread,
                VirtualWaveSize,
                warp_load_method::warp_load_vectorize,
                TargetWaveSize,
                ::rocprim::detail::wave_target_guard_t<TargetWaveSize>>
{
    static_assert(::rocprim::detail::is_power_of_two(VirtualWaveSize),
                  "Logical warp size must be a power of two.");

public:
    ROCPRIM_INLINE ROCPRIM_HOST_DEVICE warp_load()
    {
        detail::check_virtual_wave_size<VirtualWaveSize>();
    }

    using storage_type = typename ::rocprim::detail::empty_storage_type;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(T* input,
              T (&items)[ItemsPerThread],
              storage_type& /*storage*/)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_blocked_vectorized(flat_id, input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_blocked(flat_id, input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_blocked(flat_id, input, items, valid);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& /*storage*/)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_blocked(flat_id, input, items, valid,
                                  out_of_bounds);
    }
};

template<class T,
         unsigned int                       ItemsPerThread,
         unsigned int                       VirtualWaveSize,
         ::rocprim::arch::wavefront::target TargetWaveSize>
class warp_load<T,
                ItemsPerThread,
                VirtualWaveSize,
                warp_load_method::warp_load_transpose,
                TargetWaveSize,
                ::rocprim::detail::wave_target_guard_t<TargetWaveSize>>
{
    static_assert(::rocprim::detail::is_power_of_two(VirtualWaveSize),
                  "Logical warp size must be a power of two.");

public:
    ROCPRIM_INLINE ROCPRIM_HOST_DEVICE warp_load()
    {
        detail::check_virtual_wave_size<VirtualWaveSize>();
    }

private:
    using exchange_type = ::rocprim::warp_exchange<T, ItemsPerThread, VirtualWaveSize>;

public:
    using storage_type = typename exchange_type::storage_type;

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_warp_striped<VirtualWaveSize>(flat_id, input, items);
        exchange_type().striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_warp_striped<VirtualWaveSize>(flat_id, input, items, valid);
        exchange_type().striped_to_blocked(items, items, storage);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<VirtualWaveSize>();
        block_load_direct_warp_striped<VirtualWaveSize>(flat_id,
                                                        input,
                                                        items,
                                                        valid,
                                                        out_of_bounds);
        exchange_type().striped_to_blocked(items, items, storage);
    }
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

END_ROCPRIM_NAMESPACE

/// @}
// end of group warpmodule

#endif // ROCPRIM_WARP_WARP_LOAD_HPP_
