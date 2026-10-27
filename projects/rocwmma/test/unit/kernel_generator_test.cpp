/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "kernel_generator.hpp"

#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

namespace
{
    using namespace rocwmma;
    template <int N>
    using Tag = std::integral_constant<int, N>;

    // Cover flattening, decay, duplicate parameters, and Cartesian-product order.
    static_assert(std::is_same_v<Concat<const int, std::tuple<float, double>>::Result,
                                 std::tuple<int, float, double>>);
    static_assert(std::is_same_v<Concat<std::tuple<int&>, float>::Result, std::tuple<int&, float>>);
    static_assert(std::is_same_v<Concat<std::tuple<int>, std::tuple<float>, double>::Result,
                                 std::tuple<int, float, double>>);
    static_assert(
        std::is_same_v<CombineOne<Tag<1>, std::tuple<>>::Result, std::tuple<std::tuple<Tag<1>>>>);
    using Product  = CombineLists<std::tuple<Tag<1>, Tag<2>>,
                                  std::tuple<Tag<3>, Tag<4>>,
                                  std::tuple<Tag<5>, Tag<6>>>::Result;
    using Expected = std::tuple<std::tuple<Tag<1>, Tag<3>, Tag<5>>,
                                std::tuple<Tag<1>, Tag<3>, Tag<6>>,
                                std::tuple<Tag<1>, Tag<4>, Tag<5>>,
                                std::tuple<Tag<1>, Tag<4>, Tag<6>>,
                                std::tuple<Tag<2>, Tag<3>, Tag<5>>,
                                std::tuple<Tag<2>, Tag<3>, Tag<6>>,
                                std::tuple<Tag<2>, Tag<4>, Tag<5>>,
                                std::tuple<Tag<2>, Tag<4>, Tag<6>>>;
    static_assert(std::is_same_v<Product, Expected>);

    struct Generator
    {
        using ResultT = int;
        template <int N>
        static int generate(std::tuple<Tag<N>>)
        {
            return N;
        }
    };

    template <size_t... I>
    auto generateLarge(std::index_sequence<I...>)
    {
        using Params = std::tuple<std::tuple<Tag<static_cast<int>(I)>>...>;
        return KernelGenerator<Params, Generator>::generate();
    }
}

TEST(KernelGenerator, PreservesOrderDuplicatesAndAppend)
{
    using Params  = std::tuple<std::tuple<Tag<3>>, std::tuple<Tag<1>>, std::tuple<Tag<3>>>;
    using Kernels = rocwmma::KernelGenerator<Params, Generator>;
    EXPECT_EQ(Kernels::generate(), (std::vector<int>{3, 1, 3}));
    std::vector<int> result{9};
    Kernels::generate(result);
    EXPECT_EQ(result, (std::vector<int>{9, 3, 1, 3}));
}

TEST(KernelGenerator, LargeParameterList)
{
    auto result = generateLarge(std::make_index_sequence<512>{});
    ASSERT_EQ(result.size(), 512u);
    for(size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_EQ(result[i], static_cast<int>(i));
    }
}
