/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#pragma once

//
// Value "suites" for arithemtic and expression tests
//

#include <cstdint>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace TestValues
{

    inline std::vector<int> shiftValues = {0, 1, 2, 4, 7, 23, 31};

    inline std::vector<uint32_t> uint32Values
        = {0u,    1u,    2u,    4u,     5u,     7u,       8u,          12u,        16u,  17u,
           19u,   32u,   33u,   63u,    64u,    128u,     256u,        512u,       906u, 1033u,
           3017u, 4096u, 8000u, 12344u, 40221u, 3333222u, 1141374976u, 2215116800u};

    inline std::vector<int> int32Values
        = {-50002, -146, -100, -10,  -4,   -1,   0,    1,     2,     4,       5,
           7,      8,    12,   16,   17,   19,   32,   33,    63,    64,      128,
           256,    512,  906,  1033, 3017, 4096, 8000, 12344, 40221, 3333222, 1141374976};

    inline std::vector<int64_t> int64Values = {
        -1098879408657145920l,
        -18030891251015744l,
        -50002,
        -146,
        -1,
        0,
        1,
        2,
        4,
        5,
        7,
        8,
        12,
        16,
        19,
        32,
        33,
        63,
        64,
        128,
        256,
        512,
        906,
        3017,
        4096,
        8000,
        12344,
        40221,
        18030891251015744l,
        1098879408657145920l,
        1l << 30,
        1l << 40,
    };

    inline std::vector<uint64_t> uint64Values = {
        0,
        1,
        2,
        4,
        5,
        7,
        8,
        12,
        16,
        19,
        32,
        33,
        63,
        64,
        128,
        256,
        512,
        906,
        3017,
        4096,
        8000,
        12344,
        40221,
        18030891251015744l,
        1098879408657145920l,
        1ul << 30,
        1ul << 40,
        1ul << 63,
        std::numeric_limits<uint64_t>::max(),
    };

    inline std::vector<float> floatValues
        = {-50002.0f, -14610.0f, -146.0f,  -104.6f, -73.5f,   -1.5f,     -1.0f, -0.5f,
           0.0f,      1.0f,      2.0f,     2.5f,    16.0f,    17.25f,    41.4f, 192.0f,
           1024.0f,   12344.0f,  12981.0f, 42e5f,   3.14159f, 2.7182818f};

    inline std::vector<double> doubleValues = {-50002.0,
                                               -14610.0,
                                               -146.0,
                                               -1.5,
                                               -1.0,
                                               0.0,
                                               1.0,
                                               2.5,
                                               16.0,
                                               17.25,
                                               192.0,
                                               12344.0,
                                               12981.0,
                                               42e5};

    template <typename T>
    struct ByType
    {
    };

    template <>
    struct ByType<uint32_t>
    {
        inline const static auto values = uint32Values;
    };

    template <>
    struct ByType<int32_t>
    {
        inline const static auto values = int32Values;
    };

    template <>
    struct ByType<uint64_t>
    {
        inline const static auto values = uint64Values;
    };

    template <>
    struct ByType<int64_t>
    {
        inline const static auto values = int64Values;
    };

    template <>
    struct ByType<float>
    {
        inline const static auto values = floatValues;
    };

    template <>
    struct ByType<double>
    {
        inline const static auto values = doubleValues;
    };

    template <typename T>
    concept CHasTestValues = requires()
    {
        {
            ByType<T>::values
            } -> std::convertible_to<std::vector<T>>;
    };

    inline std::vector<rocRoller::CommandArgumentValue> byType(rocRoller::DataType type)
    {
        auto exp = rocRoller::Expression::literal(0, type);

        auto singleValue = std::get<rocRoller::CommandArgumentValue>(*exp);

        auto visitor = [type](auto value) -> std::vector<rocRoller::CommandArgumentValue> {
            using T = std::decay_t<decltype(value)>;
            if constexpr(CHasTestValues<T>)
            {
                std::vector<rocRoller::CommandArgumentValue> rv{ByType<T>::values.begin(),
                                                                ByType<T>::values.end()};

                // ByType<T>::values;
                return rv;
            }
            else
            {
                rocRoller::Throw<rocRoller::FatalError>("No test values for ", ShowValue(type));
            }
        };

        return std::visit(visitor, singleValue);
    }

}
