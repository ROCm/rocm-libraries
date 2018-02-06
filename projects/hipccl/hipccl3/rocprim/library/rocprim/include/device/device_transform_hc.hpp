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

#ifndef ROCPRIM_DEVICE_DEVICE_TRANSFORM_HC_HPP_
#define ROCPRIM_DEVICE_DEVICE_TRANSFORM_HC_HPP_

#include <type_traits>
#include <iterator>

// HC API
#include <hcc/hc.hpp>

#include "../detail/config.hpp"
#include "../detail/various.hpp"

#include "detail/device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE

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

} // end of detail namespace

template<
    class InputIterator,
    class OutputIterator,
    class UnaryFunction
>
void device_transform(InputIterator input,
                      OutputIterator output,
                      const size_t size,
                      UnaryFunction transform_op,
                      hc::accelerator_view acc_view = hc::accelerator().get_default_view(),
                      bool debug_synchronous = false)
{
    // TODO: Those values should depend on type size
    constexpr unsigned int block_size = 256;
    constexpr unsigned int items_per_thread = 4;
    constexpr auto items_per_block = block_size * items_per_thread;

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    auto grid_size = number_of_blocks * block_size;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    hc::parallel_for_each(
        acc_view,
        hc::tiled_extent<1>(grid_size, block_size),
        [=](hc::tiled_index<1>) [[hc]]
        {
            detail::transform_kernel_impl<block_size, items_per_thread>(
                input, size, output, transform_op
            );
        }
    );
    ROCPRIM_DETAIL_HC_SYNC("transform_kernel", size, start)
}

#undef ROCPRIM_DETAIL_HC_SYNC

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_TRANSFORM_HC_HPP_
