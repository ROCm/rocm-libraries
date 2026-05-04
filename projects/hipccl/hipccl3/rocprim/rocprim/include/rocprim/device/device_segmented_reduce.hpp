// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_

#include <iostream>
#include <iterator>
#include <type_traits>

#include "../common.hpp"
#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/transform_iterator.hpp"

#include "detail/config/device_segmented_reduce.hpp"
#include "detail/device_segmented_reduce.hpp"
#include "rocprim/type_traits.hpp"

/// \addtogroup devicemodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
namespace detail
{

template<class Config,
         class InputIterator,
         class OutputIterator,
         class OffsetIterator,
         class InitValueType,
         class BinaryFunction>
struct segmented_reduce_impl
{
    constexpr hipError_t operator()(void*          temporary_storage,
                                    size_t&        storage_size,
                                    InputIterator  input,
                                    OutputIterator output,
                                    size_t         segments,
                                    OffsetIterator begin_offsets,
                                    OffsetIterator end_offsets,
                                    BinaryFunction reduce_op,
                                    InitValueType  initial_value,
                                    hipStream_t    stream,
                                    bool           debug_synchronous)
    {
        using input_type  = typename std::iterator_traits<InputIterator>::value_type;
        using result_type = ::rocprim::accumulator_t<BinaryFunction, input_type>;

        using Selector = segmented_reduce_config_selector<result_type>;

        const target current_target(stream);

        const auto params = get_config<Selector>(Config{}, current_target);

        const unsigned int block_size = params.kernel_config.block_size;
        // HIP supports (2^32 - 1) max threads.  We have to ensure block_size * segments
        // doesn't exceed that.  Compute the maximum number of segments:
        const size_t segments_limit = static_cast<size_t>(params.kernel_config.size_limit)
                                      / static_cast<size_t>(block_size);
        const size_t num_launch = ceiling_div(segments, segments_limit);

        if(temporary_storage == nullptr)
        {
            // Make sure user won't try to allocate 0 bytes memory, because
            // hipMalloc will return nullptr when size is zero.
            storage_size = 4;
            return hipSuccess;
        }

        if(segments == size_t{0})
        {
            return hipSuccess;
        }

        if(debug_synchronous)
        {
            std::cout << "----------------------------------\n";
            std::cout << "segments:       " << segments << '\n';
            std::cout << "segments_limit: " << segments_limit << '\n';
            std::cout << "num_launch:     " << num_launch << '\n';
            std::cout << "block_size:     " << block_size << '\n';
            std::cout << "----------------------------------\n";
        }

        for(size_t launch = 0, segments_offset = 0; launch < num_launch;
            ++launch, segments_offset += segments_limit)
        {
            const unsigned int current_segments = static_cast<unsigned int>(
                std::min<size_t>(segments - segments_offset, segments_limit));

            std::chrono::steady_clock::time_point start;
            if(debug_synchronous)
            {
                std::cout << "launch:           " << launch << '\n';
                std::cout << "current_segments: " << current_segments << '\n';
                std::cout << "segments_offset:  " << segments_offset << '\n';

                start = std::chrono::steady_clock::now();
            }
            auto segmented_reduce_kernel = [=](auto target_config)
            {
                segmented_reduce<decltype(target_config)>(input,
                                                          output + segments_offset,
                                                          begin_offsets + segments_offset,
                                                          end_offsets + segments_offset,
                                                          reduce_op,
                                                          static_cast<result_type>(initial_value));
            };

            ROCPRIM_RETURN_ON_ERROR(execute_launch_plan<Config, Selector>(current_target,
                                                                          segmented_reduce_kernel,
                                                                          dim3(current_segments),
                                                                          dim3(block_size),
                                                                          0,
                                                                          stream));
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_reduce",
                                                        current_segments,
                                                        start);
        }
        return hipSuccess;
    }
};

template<typename OffsetT>
struct transform_idx_to_offset_op
{
    size_t       segments;
    unsigned int segment_length;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    OffsetT      operator()(size_t i) const
    {
        if(i < segments)
        {
            return static_cast<OffsetT>(segment_length * i);
        }
        else
        {
            return static_cast<OffsetT>(segment_length * segments);
        }
    }
};

template<class Config,
         class InputIterator,
         class OutputIterator,
         class InitValueType,
         class BinaryFunction>
struct device_segmented_reduce_fixed_size_invoker
{
    using SizeT = size_t;

private:
    template<class OffsetIterator>
    using segmented_reduce_impl_t = segmented_reduce_impl<Config,
                                                          InputIterator,
                                                          OutputIterator,
                                                          OffsetIterator,
                                                          InitValueType,
                                                          BinaryFunction>;

    template<class OffsetT>
    using transform_idx_to_offset_op_t = transform_idx_to_offset_op<OffsetT>;

    template<class ActualSizeT>
    static inline constexpr auto in_range(const SizeT& size)
    {
        using common_t = std::common_type_t<SizeT, ActualSizeT>;
        return static_cast<common_t>(size)
               < static_cast<common_t>(std::numeric_limits<ActualSizeT>::max());
    }

    template<class ArgsPre, class ArgsPost, class OffsetIterator>
    struct invoke_impl
    {
        ArgsPre&       args_pre;
        ArgsPost&      args_post;
        size_t         segments;
        OffsetIterator begin_offsets;
        OffsetIterator end_offsets;

        template<std::size_t... Ip, std::size_t... Iq>
        hipError_t operator()(std::index_sequence<Ip...>, std::index_sequence<Iq...>) const
        {
            return segmented_reduce_impl_t<OffsetIterator>{}(std::get<Ip>(args_pre)...,
                                                             segments,
                                                             begin_offsets,
                                                             end_offsets,
                                                             std::get<Iq>(args_post)...);
        }
    };

public:
    template<class ArgsPre, class ArgsPost>
    static inline hipError_t invoke(const size_t&       segments,
                                    const unsigned int& segment_length,
                                    ArgsPre&&           args_pre,
                                    ArgsPost&&          args_post)
    {
        const SizeT size = segments * segment_length;

        auto run_segmented_reduce = [&](auto transform_op) -> hipError_t
        {
            auto offsets
                = ::rocprim::make_transform_iterator(::rocprim::make_counting_iterator<size_t>(0),
                                                     transform_op);
            using OffsetIterator            = decltype(offsets);
            constexpr unsigned int PreSize  = std::tuple_size<std::decay_t<ArgsPre>>::value;
            constexpr unsigned int PostSize = std::tuple_size<std::decay_t<ArgsPost>>::value;

            auto impl = invoke_impl<std::decay_t<ArgsPre>, std::decay_t<ArgsPost>, OffsetIterator>{
                args_pre,
                args_post,
                segments,
                offsets,
                offsets + 1};

            return impl(std::make_index_sequence<PreSize>{}, std::make_index_sequence<PostSize>{});
        };

        if(in_range<std::uint32_t>(size))
        {
            auto transform_op
                = transform_idx_to_offset_op_t<std::uint32_t>{segments, segment_length};
            return run_segmented_reduce(transform_op);
        }
        else
        {
            auto transform_op
                = transform_idx_to_offset_op_t<std::uint64_t>{segments, segment_length};
            return run_segmented_reduce(transform_op);
        }
    }
};

} // namespace detail

#endif // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Parallel segmented reduction primitive for device level.
///
/// segmented_reduce function performs a device-wide reduction operation across multiple sequences
/// using binary \p reduce_op operator.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input must have at least \p size (<tt>end_offsets[segments-1]</tt>)
/// elements, \p output must have \p segments elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `reduce_config`.
/// \tparam InputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam OffsetIterator random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction type of binary function used for reduction. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
/// \tparam InitValueType type of the initial value.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to reduce.
/// \param [out] output iterator to the first element in the output range.
/// \param [in] segments number of segments in the input range.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// \param [in] initial_value initial value to start the reduction.
/// \param [in] reduce_op binary operation function object that will be used for reduction.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful reduction; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level segmented min-reduction operation is performed on an array of
/// integer values (<tt>short</tt>s are reduced into <tt>int</tt>s) using custom operator.
///
/// The full example is [on GitHub](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocprim/example/rocprim/device/example_device_segmented_reduce.cpp).
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom reduce function
/// auto min_op =
///     [] (int a, int b) -> int
///     {
///         return a < b ? a : b;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int segments;   // e.g., 3
/// short * input;           // e.g., [4, 7, 6, 2, 5, 1, 3, 8]
/// int * output;            // empty array of 3 elements
/// int * offsets;           // e.g. [0, 2, 3, 8]
/// int init_value;          // e.g., 9
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output,
///     segments, offsets, offsets + 1,
///     min_op, init_value
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform segmented reduction
/// rocprim::segmented_reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output,
///     segments, offsets, offsets + 1,
///     min_op, init_value
/// );
/// // output: [4, 6, 1]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class InputIterator,
         class OutputIterator,
         class OffsetIterator,
         class BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>,
         class InitValueType = typename std::iterator_traits<InputIterator>::value_type>
inline hipError_t segmented_reduce(void*          temporary_storage,
                                   size_t&        storage_size,
                                   InputIterator  input,
                                   OutputIterator output,
                                   size_t         segments,
                                   OffsetIterator begin_offsets,
                                   OffsetIterator end_offsets,
                                   BinaryFunction reduce_op         = BinaryFunction(),
                                   InitValueType  initial_value     = InitValueType(),
                                   hipStream_t    stream            = 0,
                                   bool           debug_synchronous = false)
{
    using segmented_reduce_impl_t = detail::segmented_reduce_impl<Config,
                                                                  InputIterator,
                                                                  OutputIterator,
                                                                  OffsetIterator,
                                                                  InitValueType,
                                                                  BinaryFunction>;
    return segmented_reduce_impl_t{}(temporary_storage,
                                     storage_size,
                                     input,
                                     output,
                                     segments,
                                     begin_offsets,
                                     end_offsets,
                                     reduce_op,
                                     initial_value,
                                     stream,
                                     debug_synchronous);
}

/// \brief Parallel segmented reduction primitive for device level.
///
/// segmented_reduce function performs a device-wide reduction operation across multiple sequences
/// of fixed (and equal) length using binary \p reduce_op operator.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Ranges specified by \p input must have at least \p size (<tt>segments * segment_length</tt>)
/// elements, \p output must have \p segments elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `reduce_config`.
/// \tparam InputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam BinaryFunction type of binary function used for reduction. Default type
/// is \p rocprim::plus<T>, where \p T is a \p value_type of \p InputIterator.
/// \tparam InitValueType type of the initial value.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the reduction operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input iterator to the first element in the range to reduce.
/// \param [out] output iterator to the first element in the output range.
/// \param [in] segments number of segments in the input range.
/// \param [in] segment_length fixed length of the segments in the input range.
/// \param [in] initial_value initial value to start the reduction.
/// \param [in] reduce_op binary operation function object that will be used for reduction.
/// The signature of the function should be equivalent to the following:
/// <tt>T f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream [optional] HIP stream object. The default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. The default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful reduction; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // custom reduce function
/// auto min_op =
///     [] (int a, int b) -> int
///     {
///         return a < b ? a : b;
///     };
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// unsigned int segments;       // e.g., 4
/// unsigned int segment_length; // e.g.  2
/// short * input;               // e.g., [4, 7, 6, 2, 5, 1, 3, 8]
/// int * output;                // empty array of 4 elements
/// int init_value;              // e.g., 9
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output,
///     segments, segment_length,
///     min_op, init_value
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform segmented reduction
/// rocprim::segmented_reduce(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output,
///     segments, segment_length,
///     min_op, init_value
/// );
/// // output: [4, 2, 1, 3]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>,
         class InitValueType = typename std::iterator_traits<InputIterator>::value_type>
inline hipError_t segmented_reduce(void*          temporary_storage,
                                   size_t&        storage_size,
                                   InputIterator  input,
                                   OutputIterator output,
                                   size_t         segments,
                                   unsigned int   segment_length,
                                   BinaryFunction reduce_op         = BinaryFunction(),
                                   InitValueType  initial_value     = InitValueType(),
                                   hipStream_t    stream            = 0,
                                   bool           debug_synchronous = false)
{
    return detail::device_segmented_reduce_fixed_size_invoker<
        Config,
        InputIterator,
        OutputIterator,
        InitValueType,
        BinaryFunction>::invoke(segments,
                                segment_length,
                                std::tie(temporary_storage, storage_size, input, output),
                                std::tie(reduce_op, initial_value, stream, debug_synchronous));
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group devicemodule

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
