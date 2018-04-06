// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HC_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../iterator/constant_iterator.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/discard_iterator.hpp"
#include "../iterator/zip_iterator.hpp"

#include "device_reduce_by_key_hc.hpp"
#include "device_select_hc.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule_hc
/// @{

namespace detail
{

#define ROCPRIM_DETAIL_HC_SYNC(name, size, start) \
    { \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            acc_view.wait(); \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

} // end detail namespace

template<
    class InputIterator,
    class UniqueOutputIterator,
    class CountsOutputIterator,
    class RunsCountOutputIterator
>
inline
void run_length_encode(void * temporary_storage,
                       size_t& storage_size,
                       InputIterator input,
                       unsigned int size,
                       UniqueOutputIterator unique_output,
                       CountsOutputIterator counts_output,
                       RunsCountOutputIterator runs_count_output,
                       hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                       bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using count_type = unsigned int;

    ::rocprim::reduce_by_key(
        temporary_storage, storage_size,
        input, make_constant_iterator<count_type>(1), size,
        unique_output, counts_output, runs_count_output,
        ::rocprim::plus<count_type>(), ::rocprim::equal_to<input_type>(),
        acc_view, debug_synchronous
    );
}

template<
    class InputIterator,
    class OffsetsOutputIterator,
    class CountsOutputIterator,
    class RunsCountOutputIterator
>
inline
void run_length_encode_non_trivial_runs(void * temporary_storage,
                                        size_t& storage_size,
                                        InputIterator input,
                                        unsigned int size,
                                        OffsetsOutputIterator offsets_output,
                                        CountsOutputIterator counts_output,
                                        RunsCountOutputIterator runs_count_output,
                                        hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                                        bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using offset_type = unsigned int;
    using count_type = unsigned int;
    using offset_count_pair = typename ::rocprim::tuple<offset_type, count_type>;

    auto reduce_op = [](const offset_count_pair& a, const offset_count_pair& b) [[hc]]
    {
        return offset_count_pair(
            ::rocprim::get<0>(a), // Always use offset of the first item of the run
            ::rocprim::get<1>(a) + ::rocprim::get<1>(b) // Number of items in the run
        );
    };
    auto non_trivial_runs_select_op = [](const offset_count_pair& a) [[hc]]
    {
        return ::rocprim::get<1>(a) > 1;
    };

    offset_type * offsets_tmp = nullptr;
    count_type * counts_tmp = nullptr;
    count_type * all_runs_count_tmp = nullptr;

    // Calculate size of temporary storage for reduce_by_key operation
    size_t reduce_by_key_bytes;
    ::rocprim::reduce_by_key(
        nullptr, reduce_by_key_bytes,
        input,
        ::rocprim::make_zip_iterator(
            ::rocprim::make_tuple(
                ::rocprim::make_counting_iterator<offset_type>(0),
                ::rocprim::make_constant_iterator<count_type>(1)
            )
        ),
        size,
        ::rocprim::make_discard_iterator(),
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_tmp, counts_tmp)),
        all_runs_count_tmp,
        reduce_op, ::rocprim::equal_to<input_type>(),
        acc_view, debug_synchronous
    );
    reduce_by_key_bytes = ::rocprim::detail::align_size(reduce_by_key_bytes);

    // Calculate size of temporary storage for select operation
    size_t select_bytes;
    ::rocprim::select(
        nullptr, select_bytes,
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_tmp, counts_tmp)),
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_output, counts_output)),
        runs_count_output,
        size,
        non_trivial_runs_select_op,
        acc_view, debug_synchronous
    );
    select_bytes = ::rocprim::detail::align_size(select_bytes);

    const size_t offsets_tmp_bytes = ::rocprim::detail::align_size(size * sizeof(offset_type));
    const size_t counts_tmp_bytes = ::rocprim::detail::align_size(size * sizeof(count_type));
    const size_t all_runs_count_tmp_bytes = sizeof(count_type);
    if(temporary_storage == nullptr)
    {
        storage_size = ::rocprim::max(reduce_by_key_bytes, select_bytes) +
            offsets_tmp_bytes + counts_tmp_bytes + all_runs_count_tmp_bytes;
        return;
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    ptr += ::rocprim::max(reduce_by_key_bytes, select_bytes);
    offsets_tmp = reinterpret_cast<offset_type *>(ptr);
    ptr += offsets_tmp_bytes;
    counts_tmp = reinterpret_cast<count_type *>(ptr);
    ptr += counts_tmp_bytes;
    all_runs_count_tmp = reinterpret_cast<count_type *>(ptr);

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    ::rocprim::reduce_by_key(
        temporary_storage, reduce_by_key_bytes,
        input,
        ::rocprim::make_transform_iterator( // Workaround: without transform zip_iterator returns zeros
            ::rocprim::make_zip_iterator(
                ::rocprim::make_tuple(
                    ::rocprim::make_counting_iterator<offset_type>(0),
                    ::rocprim::make_constant_iterator<count_type>(1)
                )
            ),
            [](offset_count_pair a) { return a; }
        ),
        size,
        ::rocprim::make_discard_iterator(), // Ignore unique output
        ::rocprim::make_zip_iterator(rocprim::make_tuple(offsets_tmp, counts_tmp)),
        all_runs_count_tmp,
        reduce_op, ::rocprim::equal_to<input_type>(),
        acc_view, debug_synchronous
    );
    ROCPRIM_DETAIL_HC_SYNC("rocprim::reduce_by_key", size, start)

    // Read count of all runs (including trivial runs)
    count_type all_runs_count;
    hc::copy(hc::array<count_type>(hc::extent<1>(1), acc_view, all_runs_count_tmp), &all_runs_count);

    // Select non-trivial runs
    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    ::rocprim::select(
        temporary_storage, select_bytes,
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_tmp, counts_tmp)),
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_output, counts_output)),
        runs_count_output,
        all_runs_count,
        non_trivial_runs_select_op,
        acc_view, debug_synchronous
    );
    ROCPRIM_DETAIL_HC_SYNC("rocprim::select", all_runs_count, start)
}

#undef ROCPRIM_DETAIL_HC_SYNC

/// @}
// end of group devicemodule_hc

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HC_HPP_
