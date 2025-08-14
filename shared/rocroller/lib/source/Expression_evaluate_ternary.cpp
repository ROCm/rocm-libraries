/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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

#include <rocRoller/Expression.hpp>
#include <rocRoller/Expression_evaluate_detail.hpp>

namespace rocRoller::Expression::EvaluateDetail
{
    template <>
    struct OperationEvaluatorVisitor<AddShiftL>
    {
        CommandArgumentValue call(CommandArgumentValue const& lhs,
                                  CommandArgumentValue const& r1hs,
                                  CommandArgumentValue const& r2hs) const
        {
            auto sum = evaluateOp(Add{}, lhs, r1hs);
            return evaluateOp(ShiftL{}, sum, r2hs);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<ShiftLAdd>
    {
        CommandArgumentValue call(CommandArgumentValue const& lhs,
                                  CommandArgumentValue const& r1hs,
                                  CommandArgumentValue const& r2hs) const
        {
            auto temp = evaluateOp(ShiftL{}, lhs, r1hs);
            return evaluateOp(Add{}, temp, r2hs);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<MultiplyAdd>
    {
        CommandArgumentValue call(CommandArgumentValue const& lhs,
                                  CommandArgumentValue const& r1hs,
                                  CommandArgumentValue const& r2hs) const
        {
            auto product = evaluateOp(Multiply{}, lhs, r1hs);
            return evaluateOp(Add{}, product, r2hs);
        }
    };

    struct ToBoolVisitor
    {
        template <typename T>
        constexpr bool operator()(T const& val)
        {
            Throw<FatalError>("Invalid bool type: ", typeName<T>());
            return 0;
        }

        template <std::integral T>
        constexpr bool operator()(T const& val)
        {
            return val != 0;
        }

        bool call(CommandArgumentValue const& val)
        {
            return std::visit(*this, val);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Conditional>
    {

        CommandArgumentValue call(CommandArgumentValue const& lhs,
                                  CommandArgumentValue const& r1hs,
                                  CommandArgumentValue const& r2hs) const
        {
            AssertFatal(r1hs.index() == r2hs.index(),
                        "Types of each conditional option must match!",
                        ShowValue(r1hs),
                        ShowValue(r2hs));

            if(ToBoolVisitor().call(lhs))
            {
                return r1hs;
            }
            else
            {
                return r2hs;
            }
        }
    };

    template <>
    struct OperationEvaluatorVisitor<MatrixMultiply>
    {
        CommandArgumentValue call(CommandArgumentValue const& lhs,
                                  CommandArgumentValue const& r1hs,
                                  CommandArgumentValue const& r2hs) const
        {
            Throw<FatalError>("Matrix multiply present in runtime expression.");
        }
    };

    template <CTernary T>
    __attribute__((noinline)) CommandArgumentValue evaluateOp(T const&                    op,
                                                              CommandArgumentValue const& lhs,
                                                              CommandArgumentValue const& r1hs,
                                                              CommandArgumentValue const& r2hs)
    {
        auto visitor = OperationEvaluatorVisitor<T>();
        return visitor.call(lhs, r1hs, r2hs);
    }

#define INSTANTIATE_TERNARY_OP(Op)                                                 \
    template CommandArgumentValue evaluateOp<Op>(Op const&                   op,   \
                                                 CommandArgumentValue const& lhs,  \
                                                 CommandArgumentValue const& r1hs, \
                                                 CommandArgumentValue const& r2hs)

    INSTANTIATE_TERNARY_OP(AddShiftL);
    INSTANTIATE_TERNARY_OP(Conditional);
    INSTANTIATE_TERNARY_OP(MatrixMultiply);
    INSTANTIATE_TERNARY_OP(MultiplyAdd);
    INSTANTIATE_TERNARY_OP(ShiftLAdd);
}
