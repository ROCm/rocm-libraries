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

#include <iostream>
#include <rocRoller/Utilities/Component.hpp>
#include <string>

#include "SimpleFixture.hpp"

namespace rocRollerTest
{
    class ComponentTest : public SimpleFixture
    {
    };

    using namespace rocRoller::Component;

    struct TestArgument
    {
        bool classA = false;
    };

    struct Interface
    {
        using Argument = std::shared_ptr<TestArgument>;

        static const std::string Basename;

        virtual std::string name() = 0;
    };

    RegisterComponentBase(Interface);

    static_assert(ComponentBase<Interface>);

    struct AImpl : public Interface
    {
        using Base = Interface;
        static const std::string Name;

        static bool Match(Argument arg)
        {
            return arg->classA;
        }

        static std::shared_ptr<Interface> Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<AImpl>();
        }

        virtual std::string name() override
        {
            return Name;
        }
    };

    struct BImpl : public Interface
    {
        using Base = Interface;
        static const std::string Name;

        static bool Match(Argument arg)
        {
            return !arg->classA;
        }

        static std::shared_ptr<Interface> Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<BImpl>();
        }

        virtual std::string name() override
        {
            return Name;
        }
    };

    static_assert(Component<AImpl>);
    static_assert(Component<BImpl>);

    RegisterComponent(AImpl);
    RegisterComponent(BImpl);

    TEST_F(ComponentTest, Basic)
    {
        auto argA = std::make_shared<TestArgument>();
        auto argB = std::make_shared<TestArgument>();

        argA->classA = true;
        argB->classA = false;

        auto instA = Get<Interface>(argA);
        EXPECT_EQ("AImpl", instA->name());

        EXPECT_NE(nullptr, std::dynamic_pointer_cast<AImpl>(instA));

        auto instB = Get<Interface>(argB);
        EXPECT_EQ("BImpl", instB->name());

        EXPECT_NE(nullptr, std::dynamic_pointer_cast<BImpl>(instB));

        // Get() returns a cached instance associated with the argument.
        auto instA2 = Get<Interface>(argA);
        EXPECT_EQ("AImpl", instA2->name());
        EXPECT_EQ(instA, instA2);

        // GetNew() returns a new instance
        auto instA3 = GetNew<Interface>(argA);
        EXPECT_EQ("AImpl", instA3->name());
        EXPECT_NE(nullptr, std::dynamic_pointer_cast<AImpl>(instA3));
        EXPECT_NE(instA, instA3);

        // Get() returns a cached instance associated with the argument.
        auto instB2 = Get<Interface>(argB);
        EXPECT_EQ("BImpl", instB2->name());
        EXPECT_EQ(instB, instB2);

        auto argB2    = std::make_shared<TestArgument>();
        argB2->classA = false;

        // New argument, new instance object
        auto instB3 = Get<Interface>(argB2);
        EXPECT_EQ("BImpl", instB3->name());
        EXPECT_NE(instB, instB3);
    }

}
