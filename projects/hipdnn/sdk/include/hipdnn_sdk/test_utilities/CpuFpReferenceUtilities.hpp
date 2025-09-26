// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <thread>
#include <tuple>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

// Parallel execution utilities for CPU reference implementations

struct JoinableThread : std::thread
{
    template <typename... Xs>
    JoinableThread(Xs&&... xs)
        : std::thread(std::forward<Xs>(xs)...)
    {
    }

    JoinableThread(JoinableThread&&) = default;
    JoinableThread& operator=(JoinableThread&&) = default;

    ~JoinableThread()
    {
        if(this->joinable())
        {
            this->join();
        }
    }
};

template <typename F, typename T, std::size_t... Is>
static auto
    callFuncUnpackArgsImpl(F f, T args, [[maybe_unused]] std::index_sequence<Is...> sequence)
{
    return f(std::get<Is>(args)...);
}

template <typename F, typename T>
static auto callFuncUnpackArgs(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};
    return callFuncUnpackArgsImpl(f, args, std::make_index_sequence<N>{});
}

template <typename F, typename... Xs>
struct ParallelTensorFunctor
{
    F _func;
    static constexpr std::size_t NDIM = sizeof...(Xs);
    std::array<std::size_t, NDIM> _lengths;
    std::array<std::size_t, NDIM> _strides;
    std::size_t _totalElements;

    ParallelTensorFunctor(F f, Xs... xs)
        : _func(f)
        , _lengths({static_cast<std::size_t>(xs)...})
    {
        _strides.back() = 1;
        std::partial_sum(_lengths.rbegin(),
                         _lengths.rend() - 1,
                         _strides.rbegin() + 1,
                         std::multiplies<std::size_t>());
        _totalElements = _strides[0] * _lengths[0];
    }

    std::array<std::size_t, NDIM> getNdIndices(std::size_t i) const
    {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0; idim < NDIM; ++idim)
        {
            indices[idim] = i / _strides[idim];
            i -= indices[idim] * _strides[idim];
        }

        return indices;
    }

    void operator()(std::size_t numThreads = 1) const
    {
        if(numThreads == 0 || _totalElements == 0)
        {
            return;
        }

        std::size_t workPerThread = (_totalElements + numThreads - 1) / numThreads;

        std::vector<JoinableThread> threads(numThreads);

        for(std::size_t threadIdx = 0; threadIdx < numThreads; ++threadIdx)
        {
            std::size_t workBegin = threadIdx * workPerThread;
            std::size_t workEnd = std::min((threadIdx + 1) * workPerThread, _totalElements);

            auto threadFunc = [=, *this] {
                for(std::size_t workIdx = workBegin; workIdx < workEnd; ++workIdx)
                {
                    callFuncUnpackArgs(_func, getNdIndices(workIdx));
                }
            };
            threads[threadIdx] = JoinableThread(threadFunc);
        }
    }
};

template <typename F, typename... Xs>
static auto makeParallelTensorFunctor(F f, Xs... xs)
{
    return ParallelTensorFunctor<F, Xs...>(f, xs...);
}

} // namespace test_utilities
} // namespace hipdnn_sdk
