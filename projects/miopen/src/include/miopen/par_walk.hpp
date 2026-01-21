// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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
