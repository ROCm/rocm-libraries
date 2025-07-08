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

#include <rocRoller/Utilities/Component.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <iostream>
#include <string>
#include <vector>

using namespace rocRoller::Component;

struct TestArgument
{
    bool classA = false;
};

struct Printer
{
    using Argument = std::shared_ptr<TestArgument>;

    static const std::string Basename;

    virtual void print() = 0;
};

RegisterComponentBase(Printer);

static_assert(ComponentBase<Printer>);

struct APrinter : public Printer
{
    using Base = Printer;
    static const std::string Name;

    static bool Match(Argument arg)
    {
        return arg->classA;
    }

    static std::shared_ptr<Printer> Build(Argument arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<APrinter>();
    }

    virtual void print() override
    {
        std::cout << "A" << std::endl;
    }
};

struct BPrinter : public Printer
{
    using Base = Printer;
    static const std::string Name;

    static bool Match(Argument arg)
    {
        return !arg->classA;
    }

    static std::shared_ptr<Printer> Build(Argument arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<BPrinter>();
    }

    virtual void print() override
    {
        std::cout << "B" << std::endl;
    }
};

static_assert(Component<APrinter>);
static_assert(Component<BPrinter>);

RegisterComponent(APrinter);
RegisterComponent(BPrinter);

using myarr = std::array<int, 4>;

struct asdf
{
    myarr fdsa;

    asdf(std::initializer_list<int> fd)
        : fdsa{}
    {
        using namespace rocRoller;
        AssertFatal(fd.size() <= fdsa.size(), ShowValue(fd.size()), ShowValue(fdsa.size()));
        std::copy(fd.begin(), fd.end(), fdsa.begin());
    }
};

int main(int argc, const char* argv[])
{
    auto argA = std::make_shared<TestArgument>();
    auto argB = std::make_shared<TestArgument>();

    argA->classA = true;
    argB->classA = false;

    auto instA = Get<Printer>(argA);
    instA->print();

    auto instB = Get<Printer>(argB);
    instB->print();

    asdf foo({1, 3, 4, 5});

    for(int i = 0; i < 4; i++)
        std::cout << foo.fdsa[i] << std::endl;

    return 0;
}
