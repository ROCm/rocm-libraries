/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <vector>
#include <thread>

namespace miopen {
namespace detail {
struct Partition
{
    int operator()(int work_size)
    {
        auto const thread_count = std::thread::hardware_concurrency();
        if(thread_count < work_size)
            return thread_count;
        return work_size;
    }
};
} // namespace detail

template<class ForwardIt, class ForwardIt2, class OutputIt, class Operation, class PartitionT=detail::Partition>
void par_walk(ForwardIt begin, ForwardIt end, ForwardIt2 begin2, OutputIt output_begin, Operation op)
{
    const auto work_size = end - begin;
    const auto thread_count = PartitionT{}(work_size);
    if(thread_count < 2)
    {
        op(begin, end, begin2, output_begin);
        return;
    }
    const auto group_size = (work_size / thread_count);
    std::vector<std::thread> threads{};

    threads.reserve(thread_count);
    for(int i = 0; i < thread_count; ++i)
    {
        auto chunk_begin = begin + (i * group_size);
        auto chunk_end = chunk_begin + group_size;
        if(i == thread_count - 1 && chunk_end != end)
            chunk_end = end;
        auto chunk_begin2 = begin2 + (i * group_size);
        auto chunk_output_begin = output_begin + (i * group_size);
        threads.emplace_back([&op, chunk_begin, chunk_end, chunk_begin2, chunk_output_begin, end]()
        {
            op(chunk_begin, (chunk_end > end ? end : chunk_end), chunk_begin2, chunk_output_begin);
        });
    }

    for(auto& t : threads)
    {
        if(t.joinable())
            t.join();
    }
}

template<class ForwardIt, class Operation, class PartitionT=detail::Partition>
void par_walk(ForwardIt begin, ForwardIt end, Operation op)
{
    par_walk(begin, end, begin, begin, [&op](auto chunk_begin, auto chunk_end, [[maybe_unused]] auto chunk_begin2, [[maybe_unused]] auto chunk_output_begin)
    {
        op(chunk_begin, chunk_end);
    });
}
} // namespace miopen
