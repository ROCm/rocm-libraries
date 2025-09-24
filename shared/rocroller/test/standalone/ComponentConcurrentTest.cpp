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

#include <fmt/format.h>

#include <thread>

using namespace rocRoller::Component;

using TestArgument = unsigned;

struct TestComponentBase
{
    using Argument = std::shared_ptr<TestArgument>;
    static const std::string Basename;
    virtual unsigned         getValue() = 0;
};

const std::string TestComponentBase::Basename = "TestComponentBase";

static_assert(ComponentBase<TestComponentBase>);

template <unsigned ID>
struct TestComponent : public TestComponentBase
{
    using Base = TestComponentBase;
    static const std::string Name;

    static bool Match(Argument arg)
    {
        return *arg == ID;
    }

    static std::shared_ptr<TestComponentBase> Build(Argument arg)
    {
        if(!Match(arg))
            return nullptr;
        return std::make_shared<TestComponent<ID>>();
    }

    virtual unsigned getValue() override
    {
        return ID;
    }
};

template <unsigned ID>
const std::string TestComponent<ID>::Name = fmt::format("TestComponent{}", ID);

static_assert(Component<TestComponent<0>>);

static const unsigned THREAD_COUNT    = 16;
static const unsigned COMPONENT_COUNT = 1000;

struct Thread
{
    bool result = false;

    void run()
    {
        const unsigned expectedResult = (COMPONENT_COUNT - 1) * COMPONENT_COUNT / 2;
        unsigned       actualResult   = 0;

        for(unsigned i = 0; i < COMPONENT_COUNT; ++i)
        {
            auto arg     = std::make_shared<unsigned>(i);
            auto comp    = Get<TestComponentBase>(arg);
            actualResult = actualResult + comp->getValue();
        }

        result = expectedResult == actualResult;
    }
};

int main(int argc, char const* argv[])
{
    (void)argc;
    (void)argv;

    fmt::println("Component Concurrent Test");

    std::vector<Thread>      threadContexts;
    std::vector<std::thread> threadRunners;
    threadContexts.reserve(THREAD_COUNT);
    threadRunners.reserve(THREAD_COUNT);

    for(unsigned i = 0; i < THREAD_COUNT; ++i)
    {
        threadContexts.push_back({});
    }
    for(unsigned i = 0; i < THREAD_COUNT; ++i)
    {
        threadRunners.emplace_back(&Thread::run, &threadContexts[i]);
    }

    for(auto&& thread : threadRunners)
    {
        thread.join();
    }

    auto result = std::ranges::all_of(threadContexts, [](auto&& thread) { return thread.result; });

    fmt::println("Checksum result : {}", result);

    return 0;
}

template <unsigned ID>
struct RegisterTestComponent
{
    template <typename Factory>
    void operator()(Factory&& factory)
    {
        factory.template registerComponent<TestComponent<ID>>();
        RegisterTestComponent<ID - 1>()(factory);
    }
};

template <>
struct RegisterTestComponent<0>
{
    template <typename Factory>
    void operator()(Factory&& factory)
    {
        factory.template registerComponent<TestComponent<0>>();
    }
};

template <>
void ComponentFactory<TestComponentBase>::registerImplementations()
{
    using Factory = ComponentFactory<TestComponentBase>;
    RegisterTestComponent<COMPONENT_COUNT>()(*this);
}
