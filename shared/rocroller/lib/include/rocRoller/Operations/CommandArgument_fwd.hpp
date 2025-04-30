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

#include <memory>
#include <span>
#include <variant>

#include <rocRoller/DataTypes/DataTypes.hpp>

namespace rocRoller
{
    class CommandArgument;
    using CommandArgumentPtr = std::shared_ptr<CommandArgument>;

    using CommandArgumentValue = std::variant<
        // int16_t,
        int32_t,
        int64_t,
        // uint16_t,
        uint32_t,
        uint64_t,
        float,
        double,
        Half,
        BFloat16,
        FP8,
        BF8,
        FP6,
        BF6,
        FP4,
        bool,
        // int16_t*,
        int32_t*,
        int64_t*,
        // uint16_t*,
        uint8_t*,
        uint32_t*,
        uint64_t*,
        float*,
        double*,
        Half*,
        BFloat16*,
        FP8*,
        BF8*,
        FP6*,
        BF6*,
        FP4*>;

    template <typename T>
    concept CCommandArgumentValue = requires(T& val)
    {
        {CommandArgumentValue(val)};
    };

    static_assert(!CCommandArgumentValue<bool*>);

    using RuntimeArguments = std::span<uint8_t const>;
}
