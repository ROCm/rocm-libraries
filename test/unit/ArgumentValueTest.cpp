
#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GenericContextFixture.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class ArgumentValueTest : public GenericContextFixture
    {
    };

    TEST_F(ArgumentValueTest, Basic)
    {
        auto five      = std::make_shared<Expression::Expression>(5.0f);
        auto two       = std::make_shared<Expression::Expression>(2.0f);
        auto valueExpr = (two * five) * (two * five) + two - five;
        EXPECT_EQ(97.0, std::get<float>(Expression::evaluate(valueExpr)));

        m_context->kernel()->addArgument(
            {"value", {DataType::Float}, DataDirection::ReadOnly, valueExpr, -1, -1});

        KernelArguments argsActual;

        for(auto& arg : m_context->kernel()->arguments())
        {
            auto value = Expression::evaluate(arg.expression);
            EXPECT_EQ(arg.variableType, variableType(value));
            argsActual.append(arg.name, value);
        }

        KernelArguments argsExpected;
        argsExpected.append("value", 97.0f);

        EXPECT_EQ(argsExpected.dataVector(), argsActual.dataVector());
    }
}
