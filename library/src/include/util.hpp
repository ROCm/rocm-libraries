/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_SRC_UTIL_HPP
#define HIPTENSOR_SRC_UTIL_HPP

#include <ck/utility/data_type.hpp>
#include <ck/utility/tuple.hpp>
#include <hiptensor/hiptensor.hpp>
#include <logger.hpp>
#include <type_traits>
#include <vector>

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
        logger.logError("hiptensorPermutation", msg);
    };

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

#endif // HIPTENSOR_SRC_UTIL_HPP
