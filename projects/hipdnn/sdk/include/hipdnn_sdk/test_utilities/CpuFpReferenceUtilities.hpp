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
    F _func;
    std::vector<std::size_t> _lengths;
    std::vector<std::size_t> _strides;
    std::size_t _totalElements{1};

    ParallelTensorFunctorDynamic(F f, const std::vector<int64_t>& dimensions)
        : _func(f)
        , _lengths(dimensions.begin(), dimensions.end())
        , _strides(dimensions.size())
    {
        if(_lengths.empty())
        {
            _totalElements = 0;
            return;
        }

        auto generatedStrides = hipdnn_sdk::utilities::generateStrides(dimensions);
        _strides.assign(generatedStrides.begin(), generatedStrides.end());
        _totalElements = _strides[0] * _lengths[0];
    }

    std::vector<int64_t> getNdIndices(std::size_t i) const
    {
        std::vector<int64_t> indices(_lengths.size());

        for(std::size_t idim = 0; idim < _lengths.size(); ++idim)
        {
            indices[idim] = static_cast<int64_t>(i / _strides[idim]);
            i -= static_cast<std::size_t>(indices[idim]) * _strides[idim];
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
                    _func(getNdIndices(workIdx));
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

// Iterates the elements along each of the dimensions specified in dims and calls func for each unique index
// Formally, we are iterating over a cartesian product of the ranges [0, dims[0]), [0, dims[1]), ..., [0, dims[n - 1]) for n dimensions
template <typename F>
static void iterateAlongDimensions(const std::vector<int64_t>& dims, F&& func)
{
    if(dims.empty())
    {
        func({});
        return;
    }

    int64_t totalElements = 1;
    for(auto dim : dims)
    {
        totalElements *= dim;
    }

    std::vector<int64_t> indices(dims.size(), 0);

    // Iterate over each unique position
    for(int64_t iter = 0; iter < totalElements; ++iter)
    {
        func(indices);

        for(int dim = static_cast<int>(dims.size()) - 1; dim >= 0; --dim)
        {
            auto dimIdx = static_cast<size_t>(dim);
            indices[dimIdx]++;

            if(indices[dimIdx] < dims[dimIdx])
            {
                break;
            }

            indices[dimIdx] = 0;
        }
    }
}

// Constructs a full tensor indices vector from batch, channel, and spatial components. spatialOffset allows
// skipping initial elements in the spatialIndices vector for convenience.
static inline std::vector<int64_t> buildTensorIndices(int64_t batchIdx,
                                                      int64_t channelIdx,
                                                      const std::vector<int64_t>& spatialIndices,
                                                      size_t spatialOffset = 0)
{
    std::vector<int64_t> fullIndices = {batchIdx, channelIdx};
    fullIndices.insert(fullIndices.end(),
                       spatialIndices.begin() + static_cast<std::ptrdiff_t>(spatialOffset),
                       spatialIndices.end());
    return fullIndices;
}

} // namespace test_utilities
} // namespace hipdnn_sdk
