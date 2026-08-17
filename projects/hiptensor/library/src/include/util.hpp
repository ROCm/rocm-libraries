/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <ck/utility/data_type.hpp>
#include <ck/utility/tuple.hpp>

#include "data_types.hpp"
#include "hiptensor/internal/types.hpp"
#include "logger.hpp"
#include "platform.hpp"

namespace hiptensor
{
    template <typename intT1,
              class = typename std::enable_if<std::is_integral<intT1>::value>::type,
              typename intT2,
              class = typename std::enable_if<std::is_integral<intT2>::value>::type>
    static constexpr intT1 ceilDiv(const intT1 numerator, const intT2 divisor)
    {
        return (numerator + divisor - 1) / divisor;
    }

    template <typename T>
    static inline std::vector<T> stridesFromLengths(std::vector<T> const& lengths,
                                                    bool                  col_major = true)
    {
        if(lengths.empty())
        {
            return lengths;
        }

        // Re-construct strides from lengths, assuming packed.
        std::vector<T> strides(lengths.size(), 1);
        if(!col_major)
        {
            strides.back() = 1;
            std::partial_sum(
                lengths.rbegin(), lengths.rend() - 1, strides.rbegin() + 1, std::multiplies<T>());
        }
        else
        {
            strides.front() = 1;
            std::partial_sum(
                lengths.begin(), lengths.end() - 1, strides.begin() + 1, std::multiplies<T>());
        }

        return strides;
    }

    // Get count of element of a tensor. Note that the count is 1 if the rank of tensor is 0.
    template <typename T>
    static inline T elementsFromLengths(std::vector<T> const& lengths)
    {
        return std::accumulate(lengths.begin(), lengths.end(), T{1}, std::multiplies<T>());
    }

    inline std::vector<std::size_t> getTensorLengths(const hiptensorTensorDescriptor_t desc)
    {
        return desc ? desc->mLengths : std::vector<std::size_t>{};
    }

    inline std::vector<std::size_t> getTensorStrides(const hiptensorTensorDescriptor_t desc)
    {
        return desc ? desc->mStrides : std::vector<std::size_t>{};
    }

    // Get the rank of a tensor based on its strides
    // Ex: a_ms_ks_lengths = [5, 6, 3, 4]
    //     a_ms_ks_strides = [1, 5, 30, 90] (col major)
    //                     = [72, 12, 4, 1] (row major)
    //                rank = 2
    //
    // Ex: a_ms_ks_lengths = [5, 6, 3, 1, 4, 3, 4, 2]
    //     a_ms_ks_strides = [1, 5, 30, 90, 90, 180, 540, 2160] (col major)
    //                     = [1728, 288, 96, 96, 24, 8, 2, 1]   (row major)
    //                rank = 4
    static inline uint32_t getRank(std::vector<std::size_t> const& a_ms_ks_strides)
    {
        return a_ms_ks_strides.size() / 2;
    }

    static std::vector<int32_t> findIndices(const std::vector<int32_t>& haystack,
                                            const std::vector<int32_t>& needles)
    {
        std::vector<int32_t> indices;

        for(auto needle : needles)
        {
            auto it = std::find(haystack.begin(), haystack.end(), needle);
            if(it != haystack.end())
            {
                indices.push_back(std::distance(haystack.begin(), it));
            }
        }
        return indices;
    }

    inline void printErrorMessage(hiptensor::Logger& logger,
                                  hiptensorStatus_t  errorCode,
                                  const std::string& paramName)
    {
        char msg[512];
        snprintf(msg,
                 sizeof(msg),
                 "Initialization Error : %s = nullptr (%s)",
                 paramName.c_str(),
                 hiptensorGetErrorString(errorCode));
        logger.logError("hiptensorPermute", msg);
    };

    // Read and environment variable and check if it's value evaluates to true or false.
    // "ON", "on" and "1" evaluates to true, any other value or the absense of the value
    // evaluates to false.
    inline bool checkEnvironmentVariableEnabled(const char* name)
    {
        auto var = getEnvironmentVariable(name);
        if(!var.has_value())
        {
            return false;
        }

        std::string upper = var.value();
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        if((upper.compare("ON") == 0) || (upper.compare("1") == 0))
        {
            return true;
        }

        return false;
    }

    /** @name static_for
     *  @{
     */
    template <size_t N, typename Func, size_t... I>
    constexpr void static_for_impl(Func&& func, std::index_sequence<I...>)
    {
        (func(std::integral_constant<size_t, I>{}), ...);
    }

    template <size_t N, typename Func>
    constexpr void static_for(Func&& func)
    {
        static_for_impl<N>(std::forward<Func>(func), std::make_index_sequence<N>{});
    }
    /** @} */

    template <typename T, typename U, size_t N>
    void convertVectorToCkArray(std::vector<T> const& v, std::array<U, N>& a)
    {
        std::copy_n(v.begin(), N, a.begin());
    }

    template <typename T>
    hiptensorStatus_t getSortingIndices(const T*          referenceOrder,
                                        size_t            referenceSize,
                                        const T*          elementsToSort,
                                        size_t            elementsSize,
                                        std::vector<int>& sortingIndices)
    {
        // Create a position map for elements in the reference order
        std::unordered_map<T, int> elementPosition;
        for(size_t i = 0; i < referenceSize; ++i)
        {
            elementPosition[referenceOrder[i]] = static_cast<int>(i);
        }

        // Verify all elements in elementsToSort exist in referenceOrder
        for(size_t i = 0; i < elementsSize; ++i)
        {
            if(elementPosition.find(elementsToSort[i]) == elementPosition.end())
            {
                return HIPTENSOR_STATUS_INVALID_VALUE;
            }
        }

        // Resize the output vector to match input size
        sortingIndices.resize(elementsSize);

        // Create indices vector [0, 1, 2, ..., elementsSize-1]
        std::iota(sortingIndices.begin(), sortingIndices.end(), 0);

        // Sort indices based on the order of corresponding elements in referenceOrder
        std::sort(sortingIndices.begin(),
                  sortingIndices.end(),
                  [elementsToSort, &elementPosition](int i, int j) {
                      return elementPosition[elementsToSort[i]]
                             < elementPosition[elementsToSort[j]];
                  });

        return HIPTENSOR_STATUS_SUCCESS;
    }

    template <typename T>
    hiptensorStatus_t applySortingIndices(const std::vector<int>& sortingIndices,
                                          std::vector<T>&         vector)
    {
        // Check if sizes match
        if(sortingIndices.size() != vector.size())
        {
            return HIPTENSOR_STATUS_INVALID_VALUE;
        }

        // Resize output vector
        std::vector<T> sortedVector = std::vector<T>(vector.size());

        // Apply sorting indices
        for(size_t i = 0; i < sortingIndices.size(); ++i)
        {
            auto index = sortingIndices[i];
            if(index < 0 || index >= static_cast<int>(vector.size()))
            {
                return HIPTENSOR_STATUS_INVALID_VALUE;
            }
            sortedVector[i] = vector[index];
        }
        vector = sortedVector;

        return HIPTENSOR_STATUS_SUCCESS;
    }

    /** @name CK type tuple to hiptensor type tuple transform functions
     *  @{
     *
     *  ck::Tuple<ck::bhalf, float> => ck::Tuple<bfloat16_t, float>
     */
    // Primary template for the type transformer
    template <typename T>
    struct CkTypeTupleToHiptensorTypeTupleTransformer
    {
        using type = T;
    };

    // Specialization to transform ck::bhalf_t to bf16
    template <>
    struct CkTypeTupleToHiptensorTypeTupleTransformer<ck::bhalf_t>
    {
        using type = bfloat16_t;
    };

    // Recursive template to transform each tuple type
    template <typename Tuple, typename = std::make_index_sequence<Tuple::Size()>>
    struct TupleCkTypeTupleToHiptensorTypeTupleTransformer;

    template <typename Tuple, size_t... Indices>
    struct TupleCkTypeTupleToHiptensorTypeTupleTransformer<Tuple, std::index_sequence<Indices...>>
    {
        using type = ck::Tuple<typename CkTypeTupleToHiptensorTypeTupleTransformer<
            ck::tuple_element_t<Indices, Tuple>>::type...>;
    };

    // Convenient type alias for easier usage
    template <typename Tuple>
    using tuple_ck_type_tuple_to_hiptensor_type_tuple_t =
        typename TupleCkTypeTupleToHiptensorTypeTupleTransformer<Tuple>::type;
    /** @} */

// define a macro since it can convert `paramName` to a string
#define CheckApiParams(checkResult, logger, errorCode, paramName) \
    if(!paramName)                                                \
    {                                                             \
        printErrorMessage(logger, errorCode, #paramName);         \
        checkResult = errorCode;                                  \
    }

} // namespace hiptensor
