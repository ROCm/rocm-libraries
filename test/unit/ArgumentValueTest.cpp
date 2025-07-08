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
