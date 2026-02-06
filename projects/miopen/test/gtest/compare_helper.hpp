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

#pragma once

#include <limits>
#include <type_traits>
#include <utility>

#include <iostream>

#include "verify.hpp"
#include "../tensor_holder.hpp"

namespace test_helpers {
template <typename T>
constexpr bool is_floating_point_tensor =
    (std::is_same_v<typename T::value_type, half_float::half> ||
     std::is_floating_point_v<typename T::value_type>);

template <class... Ts>
constexpr bool any_float_tensors()
{
    return (is_floating_point_tensor<Ts> || ...);
}

template <class T>
auto const& GetResultData(std::vector<T> const& result)
{
    return result;
}

template <class T>
std::vector<T> const& GetResultData(tensor<T> const& result)
{
    return result.data;
}

template <class T>
constexpr double GetResultDataErrorMargin(std::vector<T> const& result, double tolerance)
{
    return (std::numeric_limits<T>::epsilon() * tolerance);
}

template <class T>
constexpr double GetResultDataErrorMargin(tensor<T> const& result, double tolerance)
{
    return (std::numeric_limits<T>::epsilon() * tolerance);
}

template <class LeftTuple, class RightTuple, std::size_t... I>
bool CompareFloatTensorTuples(LeftTuple const& left,
                              RightTuple const& right,
                              double tolerance,
                              std::index_sequence<I...>)
{
    return (
        (miopen::rms_range(GetResultData(std::get<I>(left)), GetResultData(std::get<I>(right))) <=
         (GetResultDataErrorMargin(std::get<I>(left), tolerance))) &&
        ...);
}

template <class... CpuT, class... GpuT>
std::enable_if_t<!(any_float_tensors<CpuT...>() || any_float_tensors<GpuT...>()), bool>
Compare(std::tuple<CpuT...> const& cpu_result, std::tuple<GpuT...> const& gpu_result, double)
{
    return cpu_result == gpu_result;
}

template <class... CpuT, class... GpuT>
std::enable_if_t<(any_float_tensors<CpuT...>() || any_float_tensors<GpuT...>()), bool> Compare(
    std::tuple<CpuT...> const& cpu_result, std::tuple<GpuT...> const& gpu_result, double tolerance)
{
    return CompareFloatTensorTuples(
        cpu_result, gpu_result, tolerance, std::index_sequence_for<CpuT...>{});
}

template <typename T>
bool Compare(tensor<T> cpu_result, tensor<T> gpu_result, double tolerance)
{
    double threshold = std::numeric_limits<T>::epsilon() * tolerance;
    return (miopen::rms_range(cpu_result.data, gpu_result.data) <= threshold);
}

/// Generic comparison helper for structs using cpu() and gpu() methods
template <class VerifyT>
auto CompareResults(VerifyT&& verifier, double tolerance = 80.f)
    -> std::pair<decltype(verifier.cpu()), decltype(verifier.gpu())>
{
    const auto cpu_result = verifier.cpu();
    const auto gpu_result = verifier.gpu();
    if(!Compare(cpu_result, gpu_result, tolerance))
    {
        verifier.fail();
    }

    return std::make_pair(cpu_result, gpu_result);
}
} // namespace test_helpers
