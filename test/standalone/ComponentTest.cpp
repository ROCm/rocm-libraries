

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

    static const std::string Name;

    virtual void print() = 0;
};

RegisterComponentBase(Printer);

static_assert(ComponentBase<Printer>);

struct APrinter : public Printer
{
    using Base = Printer;
    static const std::string Name;
    static const std::string Basename;

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
    static const std::string Basename;

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
        assert(fd.size() <= fdsa.size());
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
