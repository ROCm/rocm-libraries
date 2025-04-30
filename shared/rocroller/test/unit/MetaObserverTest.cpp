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

#include <rocRoller/Context.hpp>
#include <rocRoller/Scheduling/MetaObserver.hpp>
#include <rocRoller/Scheduling/Observers/AllocatingObserver.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>

#include "SimpleFixture.hpp"

using namespace rocRoller;

class MetaObserverTest : public SimpleFixture
{
};

class TestFalseObserver
{
public:
    TestFalseObserver() {}

    TestFalseObserver(ContextPtr context)
        : m_context(context){

        };

    Scheduling::InstructionStatus peek(Instruction const& inst) const
    {
        return Scheduling::InstructionStatus();
    };

    void modify(Instruction& inst) const {}

    void observe(Instruction const& inst) {}

    constexpr static bool required(GPUArchitectureTarget const& target)
    {
        return false;
    }

private:
    std::weak_ptr<Context> m_context;
};

class TestTrueObserver
{
public:
    TestTrueObserver() {}

    TestTrueObserver(ContextPtr context)
        : m_context(context){

        };

    Scheduling::InstructionStatus peek(Instruction const& inst) const
    {
        return Scheduling::InstructionStatus();
    };

    void modify(Instruction& inst) const {}

    void observe(Instruction const& inst) {}

    constexpr static bool required(GPUArchitectureTarget const& target)
    {
        return true;
    }

private:
    std::weak_ptr<Context> m_context;
};

static_assert(Scheduling::CObserver<TestTrueObserver>);
static_assert(Scheduling::CObserver<TestFalseObserver>);

TEST_F(MetaObserverTest, MultipleObserverTest)
{
    rocRoller::ContextPtr m_context = std::make_shared<Context>();

    std::tuple<Scheduling::AllocatingObserver,
               Scheduling::WaitcntObserver,
               Scheduling::AllocatingObserver,
               Scheduling::WaitcntObserver>
        constructedObservers = {Scheduling::AllocatingObserver(m_context),
                                Scheduling::WaitcntObserver(m_context),
                                Scheduling::AllocatingObserver(m_context),
                                Scheduling::WaitcntObserver(m_context)};

    using MyObserver      = Scheduling::MetaObserver<Scheduling::AllocatingObserver,
                                                Scheduling::WaitcntObserver,
                                                Scheduling::AllocatingObserver,
                                                Scheduling::WaitcntObserver>;
    m_context->observer() = std::make_shared<MyObserver>(constructedObservers);
}
