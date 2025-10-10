// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <numeric>
#include <thread>
#include <tuple>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

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

template <typename F>
struct ParallelTensorFunctorDynamic
{
    F func;
    std::vector<std::size_t> lengths;
    std::vector<std::size_t> strides;
    std::size_t totalElements{1};

    ParallelTensorFunctorDynamic(F f, const std::vector<int64_t>& dimensions)
        : func(f)
        , lengths(dimensions.begin(), dimensions.end())
        , strides(dimensions.size())
    {
        if(lengths.empty())
        {
            totalElements = 0;
            return;
        }

        auto generatedStrides = hipdnn_sdk::utilities::generateStrides(dimensions);
        strides.assign(generatedStrides.begin(), generatedStrides.end());
        totalElements = strides[0] * lengths[0];
    }

    std::vector<int64_t> getNdIndices(std::size_t i) const
    {
        std::vector<int64_t> indices(lengths.size());

        for(std::size_t idim = 0; idim < lengths.size(); ++idim)
        {
            indices[idim] = static_cast<int64_t>(i / strides[idim]);
            i -= static_cast<std::size_t>(indices[idim]) * strides[idim];
        }

        return indices;
    }

    void operator()(std::size_t numThreads = 1) const
    {
        if(numThreads == 0 || totalElements == 0)
        {
            return;
        }

        std::size_t workPerThread = (totalElements + numThreads - 1) / numThreads;

        std::vector<JoinableThread> threads(numThreads);

        for(std::size_t threadIdx = 0; threadIdx < numThreads; ++threadIdx)
        {
            std::size_t workBegin = threadIdx * workPerThread;
            std::size_t workEnd = std::min((threadIdx + 1) * workPerThread, totalElements);

            auto threadFunc = [=, *this] {
                for(std::size_t workIdx = workBegin; workIdx < workEnd; ++workIdx)
                {
                    func(getNdIndices(workIdx));
                }
            };
            threads[threadIdx] = JoinableThread(threadFunc);
        }
    }
};

template <typename F>
static auto makeParallelTensorFunctor(F f, const std::vector<int64_t>& dimensions)
{
    return ParallelTensorFunctorDynamic<F>(f, dimensions);
}

} // namespace test_utilities
} // namespace hipdnn_sdk
