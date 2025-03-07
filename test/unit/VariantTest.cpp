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

#include <sstream>
#include <variant>

#include <rocRoller/Utilities/Generator.hpp>

#include "SimpleFixture.hpp"

using namespace rocRoller;

using Foo = std::variant<int, float, std::string>;

class VariantTest : public SimpleFixture
{
};

struct ConvertToString
{
    template <typename T>
    std::string operator()(T const& val) const
    {
        std::ostringstream msg;
        msg << val;
        return msg.str();
    }
};

struct YieldCharacters
{
    template <typename T>
    Generator<char> operator()(T const& val) const
    {
        std::ostringstream msg;
        msg << val;
        auto str = msg.str();
        for(auto const& c : str)
            co_yield c;
    }
};

struct PrintIt
{
    void operator()(int val) const
    {
        std::cout << val;
    }

    void operator()(float val) const
    {
        std::cout << val;
    }

    void operator()(std::string const& val) const
    {
        std::cout << val;
    }
};

TEST_F(VariantTest, Basic)
{
    Foo f("four");

    EXPECT_EQ("four", std::get<std::string>(f));

    Foo     g(2);
    PrintIt printer;
    std::visit(printer, g);

    // std::visit([](auto v){std::cout << v;}, g);

    EXPECT_EQ("2", std::visit(ConvertToString(), g));

    std::string str;
    for(auto c : std::visit(YieldCharacters(), g))
    {
        // cppcheck-suppress useStlAlgorithm
        str += c;
    }

    EXPECT_EQ("2", str);

    auto coroutine = std::visit(YieldCharacters(), f);
    str            = std::string(coroutine.begin(), coroutine.end());

    EXPECT_EQ("four", str);
}

struct AddIt
{
    void operator()(int& val) const
    {
        val++;
    }

    void operator()(float& val) const
    {
        val += 0.5f;
    }

    void operator()(std::string& val) const
    {
        val = val + " point five";
    }
};

TEST_F(VariantTest, Reference)
{
    AddIt adder;

    Foo val = 1;
    std::visit(adder, val);
    EXPECT_EQ("2", std::visit(ConvertToString(), val));

    val = 2.0f;
    std::visit(adder, val);
    EXPECT_EQ("2.5", std::visit(ConvertToString(), val));

    val = "three";
    std::visit(adder, val);
    EXPECT_EQ("three point five", std::visit(ConvertToString(), val));
}
