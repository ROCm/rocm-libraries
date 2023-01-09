
#include <gtest/gtest.h>

#include <iostream>
#include <rocRoller/Utilities/Component.hpp>
#include <string>

namespace rocRollerTest
{
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

    TEST(ComponentTest, Basic)
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
