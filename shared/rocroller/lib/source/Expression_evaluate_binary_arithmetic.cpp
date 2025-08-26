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
#include <rocRoller/Expression_evaluate_detail_binary.hpp>

namespace rocRoller::Expression::EvaluateDetail
{
    CAN_OPERATE_CONCEPT(Add, +);
    static_assert(CCanAdd<int, int>);
    static_assert(CCanAdd<float*, int>);
    static_assert(!CCanAdd<float, int*>);

    /**
     * Custom declared so that we can check for null pointers where appropriate.
     */
    template <>
    struct OperationEvaluatorVisitor<Add> : public BinaryEvaluatorVisitor<Add>
    {
        template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>
        requires CCanAdd<LHS, RHS>
        auto evaluate(LHS const& lhs, RHS const& rhs) const
        {
            assertNonNullPointer(lhs);
            assertNonNullPointer(rhs);

            return lhs + rhs;
        }
    };

    CAN_OPERATE_CONCEPT(Subtract, -);

    /**
     * Custom declared so that we can check for null pointers where appropriate.
     */
    template <>
    struct OperationEvaluatorVisitor<Subtract> : public BinaryEvaluatorVisitor<Subtract>
    {
        template <CCommandArgumentValue LHS, CCommandArgumentValue RHS>
        requires CCanSubtract<LHS, RHS>
        auto evaluate(LHS const& lhs, RHS const& rhs) const
        {
            assertNonNullPointer(lhs);
            assertNonNullPointer(rhs);

            return lhs - rhs;
        }
    };

    SIMPLE_BINARY_OP(Multiply, *);
    SIMPLE_BINARY_OP(Divide, /);
    SIMPLE_BINARY_OP(Modulo, %);

    template <>
    struct OperationEvaluatorVisitor<MultiplyHigh> : public BinaryEvaluatorVisitor<MultiplyHigh>
    {
        int evaluate(int const& lhs, int const& rhs) const
        {
            assertNonNullPointer(lhs);
            assertNonNullPointer(rhs);

            return (lhs * (int64_t)rhs) >> 32;
        }

        int evaluate(int const& lhs, unsigned int const& rhs) const
        {
            assertNonNullPointer(lhs);
            assertNonNullPointer(rhs);

            return (lhs * (int64_t)rhs) >> 32;
        }

        int evaluate(unsigned int const& lhs, int const& rhs) const
        {
            assertNonNullPointer(lhs);
            assertNonNullPointer(rhs);

            return (lhs * (int64_t)rhs) >> 32;
        }

        unsigned int evaluate(unsigned int const& lhs, unsigned int const& rhs) const
        {
            assertNonNullPointer(lhs);
            assertNonNullPointer(rhs);

            return (lhs * (uint64_t)rhs) >> 32;
        }

        int64_t evaluate(int const& lhs, int64_t const& rhs) const
        {
            return evaluate((int64_t)lhs, rhs);
        }

        int64_t evaluate(int64_t const& lhs, int64_t const& rhs) const
        {
            assertNonNullPointer(lhs);
            assertNonNullPointer(rhs);

            return ((__int128_t)lhs * (__int128_t)rhs) >> 64;
        }

        uint64_t evaluate(uint64_t const& lhs, uint64_t const& rhs) const
        {
            assertNonNullPointer(lhs);
            assertNonNullPointer(rhs);

            return ((__uint128_t)lhs * (__uint128_t)rhs) >> 64;
        }
    };

    INSTANTIATE_BINARY_OP(Add);
    INSTANTIATE_BINARY_OP(Subtract);
    INSTANTIATE_BINARY_OP(Multiply);
    INSTANTIATE_BINARY_OP(MultiplyHigh);
    INSTANTIATE_BINARY_OP(Divide);
    INSTANTIATE_BINARY_OP(Modulo);

    static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Add>, double, double>);
    static_assert(!CCanEvaluateBinary<OperationEvaluatorVisitor<Add>, double, int*>);
    static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Add>, float*, int>);

    static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Subtract>, double, double>);
    static_assert(!CCanEvaluateBinary<OperationEvaluatorVisitor<Subtract>, double, int*>);
    static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Subtract>, float*, int>);

    static_assert(CCanEvaluateBinary<OperationEvaluatorVisitor<Multiply>, double, double>);
    static_assert(!CCanEvaluateBinary<OperationEvaluatorVisitor<Multiply>, double, int*>);
    static_assert(!CCanEvaluateBinary<OperationEvaluatorVisitor<Multiply>, float*, int>);
}
