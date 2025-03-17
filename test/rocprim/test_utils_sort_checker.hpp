// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TEST_UTILS_SORT_CHECKER_HPP_
#define TEST_UTILS_SORT_CHECKER_HPP_

#include <rocprim/detail/various.hpp>

#include "../common_test_header.hpp"

#include "../../common/utils_device_ptr.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/device/device_transform.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

namespace test_utils
{
namespace detail
{
template<class InputType, class BinaryFunction>
ROCPRIM_HOST_DEVICE
inline bool sort_check_compare(const InputType&     i,
                               const InputType&     next,
                               const BinaryFunction binary_function)
{
    auto ret_1 = binary_function(i, next);
    auto ret_2 = binary_function(next, i);
    if(!((ret_1 && ret_2) || (!ret_1 && !ret_2)) && !ret_1)
    {
        return false;
    }
    return true;
}

template<class ValueType>
struct transform_output_placeholder
{

    using value_type        = ValueType;
    using reference         = const value_type&;
    using pointer           = const value_type*;
    using difference_type   = typename std::iterator_traits<pointer>::difference_type;
    using iterator_category = std::output_iterator_tag;

    struct fake_ref
    {
        constexpr fake_ref operator=(const value_type&) const
        {
            return fake_ref{};
        }
    };

    constexpr fake_ref operator[](difference_type) const
    {
        return fake_ref{};
    }
    constexpr transform_output_placeholder operator+(difference_type) const
    {
        return transform_output_placeholder{};
    }
};

template<class InputIterator, class BinaryFunction>
inline bool host_sort_check(const InputIterator  input,
                            const size_t         size,
                            const BinaryFunction binary_function,
                            const hipStream_t    stream = hipStreamDefault)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    if(size < 0)
    {
        return false;
    }
    else if(size <= 1)
    {
        return true;
    }

    std::vector<input_type> host_input(size);
    HIP_CHECK(hipMemcpyAsync(host_input.data(),
                             input,
                             sizeof(input_type) * size,
                             hipMemcpyDeviceToHost,
                             stream));
    for(size_t i = 0; i < size - 1; ++i)
    {
        if(!detail::sort_check_compare(host_input[i], host_input[i + 1], binary_function))
        {
            return false;
        }
    }
    return true;
}

} // namespace detail
template<class InputIterator, class BinaryFunction>
inline bool device_sort_check(const InputIterator  input,
                              const size_t         size,
                              const BinaryFunction binary_function,
                              const hipStream_t    stream            = hipStreamDefault,
                              const bool           debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    if(size < 0)
    {
        return false;
    }
    else if(size <= 1)
    {
        return true;
    }

    common::device_ptr<int>     d_success_flag(std::vector<int>({1}));
    int* const                  d_success_flag_pointer = d_success_flag.get();

    const auto deref_op = [=](const input_type& i) -> input_type
    {
        // Obtaining the address is not suitable for all kinds of iterators
        // but not sure if there is a better way to do this.
        auto        addr = &i;
        const auto& next = *(++addr); // next value
        if(!detail::sort_check_compare(i, next, binary_function))
        {
            *d_success_flag_pointer = 0;
        }
        return i;
    };
    const auto d_iter
        = rocprim::transform_iterator<input_type*, decltype(deref_op)>(input, deref_op);

    HIP_CHECK(rocprim::transform(d_iter,
                                 detail::transform_output_placeholder<input_type>{},
                                 size - 1,
                                 rocprim::identity<input_type>(),
                                 stream,
                                 debug_synchronous));

    return static_cast<bool>(d_success_flag.load()[0]);
}

} // namespace test_utils

#endif // TEST_UTILS_SORT_CHECKER_HPP_
