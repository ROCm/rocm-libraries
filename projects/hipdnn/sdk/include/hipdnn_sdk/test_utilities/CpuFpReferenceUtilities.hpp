// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <numeric>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

// Type trait to validate tensor types (arithmetic types + half + hip_bfloat16)
template <typename T>
constexpr bool IS_VALID_TENSOR_TYPE_V = std::
    disjunction_v<std::is_arithmetic<T>, std::is_same<T, half>, std::is_same<T, hip_bfloat16>>;

/**
 * @brief Safely convert between types while avoiding implicit precision loss warnings
 * 
 * This function handles type conversions that may trigger compiler warnings about
 * implicit precision loss, particularly when converting from double to hip_bfloat16
 * or half. It makes the conversion path explicit to eliminate warnings.
 * 
 * @tparam TargetType The type to convert to
 * @tparam SourceType The type to convert from
 * @param value The value to convert
 * @return The converted value
 */
template <typename TargetType, typename SourceType>
inline TargetType safeConvert(const SourceType& value)
{
    if constexpr(std::is_same_v<TargetType, hip_bfloat16>)
    {
        // For hip_bfloat16, explicitly convert through float to avoid precision warnings
        // hip_bfloat16 lacks direct constructor from double, only from float
        return static_cast<TargetType>(static_cast<float>(value));
    }
    else if constexpr(std::is_same_v<TargetType, half>)
    {
        // For half, explicitly convert through float to avoid precision warnings
        // half lacks direct constructor from double, only from float
        return static_cast<TargetType>(static_cast<float>(value));
    }
    else
    {
        // For all other types, direct cast is fine
        return static_cast<TargetType>(value);
    }
}

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
                    if constexpr(std::is_invocable_r_v<bool, F, std::vector<int64_t>>)
                    {
                        if(!func(getNdIndices(workIdx)))
                        {
                            return;
                        }
                    }
                    else
                    {
                        func(getNdIndices(workIdx));
                    }
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
