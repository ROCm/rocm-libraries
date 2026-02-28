// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Expression.hpp>

template <typename T>
constexpr auto cast_to_unsigned(T val)
{
    return static_cast<typename std::make_unsigned<T>::type>(val);
}

namespace rocRoller
{
    namespace Expression
    {
        /**
         * Simplify trivial arithmetic expressions involving translation time constants.
         *
         * Simplifications:
         * - Add two integers, or add 0
         * - Multiply two integers, multiply by 0, multiply by 1
         * - Divide by 1
         * - Modulo by 1
         * - Shift by 0
         */

        template <typename T>
        concept CIntegral = std::integral<T> && !std::same_as<bool, T>;

        template <typename T>
        concept CBoolean = std::same_as<bool, T>;

        template <typename T>
        struct SimplifyByConstant
        {
            VariableType resultVarType;

            ExpressionPtr call(ExpressionPtr lhs, CommandArgumentValue rhs)
            {
                return nullptr;
            }
        };

        template <typename T>
        struct SimplifyByConstantLHS
        {
            VariableType resultVarType;

            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs)
            {
                return nullptr;
            }
        };

        template <>
        struct SimplifyByConstant<Add>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<Subtract>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return lhs;

                if constexpr(std::signed_integral<RHS>)
                {
                    if(rhs < 0)
                        return lhs + literal(-rhs);
                }

                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <CShift ShiftType>
        struct SimplifyByConstant<ShiftType>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return lhs;

                if(CIsAnyOf<ShiftType, ShiftL, LogicalShiftR>)
                {
                    auto const elementSize = resultVarType.getElementSize();
                    //
                    // Literal 0 can only accept Int32/UInt32/Int64/UInt64/Half/Float/Double
                    //
                    if((elementSize == 4u or elementSize == 8u) and (rhs >= elementSize * 8u))
                    {
                        return literal(0, resultVarType);
                    }
                }

                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<BitwiseAnd>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return literal(0, resultVarType);
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<LogicalAnd>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == false)
                    return literal(false);
                if(rhs == true)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<LogicalOr>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs = nullptr;

            template <typename RHS>
            requires(CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == true)
                    return literal(true);
                if(rhs == false)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CBoolean<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<Multiply>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 0)
                    return literal(0, resultVarType);
                if(rhs == 1)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstant<Divide>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 1)
                    return lhs;
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstantLHS<Divide>
        {
            VariableType  resultVarType;
            ExpressionPtr rhs;

            template <typename LHS>
            requires(CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                if(lhs == 0)
                    return literal(0, resultVarType);
                return nullptr;
            }

            template <typename LHS>
            requires(!CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                return nullptr;
            }

            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs_)
            {
                rhs = rhs_;
                return visit(*this, lhs);
            }
        };

        template <>
        struct SimplifyByConstant<Modulo>
        {
            VariableType  resultVarType;
            ExpressionPtr lhs;

            template <typename RHS>
            requires(CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                if(rhs == 1)
                    return literal(0, resultVarType);
                return nullptr;
            }

            template <typename RHS>
            requires(!CIntegral<RHS>) ExpressionPtr operator()(RHS rhs)
            {
                return nullptr;
            }

            ExpressionPtr call(ExpressionPtr lhs_, CommandArgumentValue rhs)
            {
                lhs = lhs_;
                return visit(*this, rhs);
            }
        };

        template <>
        struct SimplifyByConstantLHS<Modulo>
        {
            VariableType  resultVarType;
            ExpressionPtr rhs;

            template <typename LHS>
            requires(CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                if(lhs == 0)
                    return literal(0, resultVarType);
                return nullptr;
            }

            template <typename LHS>
            requires(!CIntegral<LHS>) ExpressionPtr operator()(LHS lhs)
            {
                return nullptr;
            }

            ExpressionPtr call(CommandArgumentValue lhs, ExpressionPtr rhs_)
            {
                rhs = rhs_;
                return visit(*this, lhs);
            }
        };

        struct ConcatenatePartialSimplifyVisitor
        {
            ConcatenatePartialSimplifyVisitor(Concatenate const& concatenate)
                : m_concatenate(concatenate)
            {
            }

            ExpressionPtr operator()(CommandArgumentValue const& expr1,
                                     CommandArgumentValue const& expr2) const
            {
                // TODO: Support partial 32 bit literal merging into 64 bit literal, codegen (copier) has problems
                // Full merging is enabled because a concatenate of a single operand is simplified to the operand
                // Partial merging would also require a Raw64 type
                if(m_concatenate.operands.size() != 2
                   || m_concatenate.destinationType != DataType::UInt64)
                    return {};

                return std::visit(
                    [](auto const& val1, auto const& val2) -> ExpressionPtr {
                        using T1 = std::decay_t<decltype(val1)>;
                        using T2 = std::decay_t<decltype(val2)>;
                        if constexpr((std::is_same_v<T1, uint32_t> || std::is_same_v<T1, Raw32>)&&(
                                         std::is_same_v<T2, uint32_t> || std::is_same_v<T2, Raw32>))
                        {
                            auto get_value = [](auto const& val) -> uint32_t {
                                if constexpr(std::is_same_v<std::decay_t<decltype(val)>, Raw32>)
                                    return val.value;
                                else
                                    return val;
                            };
                            uint64_t result = (static_cast<uint64_t>(get_value(val2)) << 32)
                                              | static_cast<uint64_t>(get_value(val1));
                            return literal(result);
                        }
                        return {};
                    },
                    expr1,
                    expr2);
            }

            ExpressionPtr operator()(BitFieldExtract const& expr1,
                                     BitFieldExtract const& expr2) const
            {
                if(identical(expr1.arg, expr2.arg)
                   && resultVariableType(expr1.arg).getElementSize() == 8 && expr1.offset == 0
                   && expr1.width == 32 && expr2.offset == 32 && expr2.width == 32)
                {
                    return expr1.arg;
                }

                return {};
            }

            template <typename ARG1, typename ARG2>
            ExpressionPtr operator()(ARG1 const& expr1, ARG2 const& expr2) const
            {
                return {};
            }

            ExpressionPtr call(ExpressionPtr expr1, ExpressionPtr expr2) const
            {
                return std::visit(*this, *expr1, *expr2);
            }

        private:
            Concatenate m_concatenate;
        };

        /**
         * Merges consecutive Concatenate operands where possible.
         * Checks if a pair of operands are two 32-bit BitFieldExtracts from the same 64-bit source
         */
        Concatenate concatenatePartialSimplify(Concatenate const& expr)
        {
            Concatenate                cpy = expr;
            std::vector<ExpressionPtr> operands;
            auto const                 visitor = ConcatenatePartialSimplifyVisitor(cpy);

            for(size_t i = 0; i < expr.operands.size(); ++i)
            {
                if(i + 1 < expr.operands.size())
                {
                    if(auto simplified = visitor.call(expr.operands[i], expr.operands[i + 1]))
                    {
                        operands.push_back(simplified);
                        ++i;
                        continue;
                    }
                }
                operands.push_back(expr.operands[i]);
            }

            cpy.operands = std::move(operands);
            return cpy;
        }

        struct BitfieldCombineSinkVisitor
        {
            BitfieldCombineSinkVisitor(ExpressionPtr src,
                                       uint32_t      srcOffset,
                                       uint32_t      dstOffset,
                                       uint32_t      width)
                : m_src(src)
                , m_srcOffset(srcOffset)
                , m_dstOffset(dstOffset)
                , m_width(width)
                , m_srcIsZero(std::nullopt)
                , m_dstIsZero(std::nullopt)
                , m_sunk(false)
            {
            }

            ExpressionPtr operator()(BitfieldCombine const& expr) const
            {
                uint32_t combineStartBit = expr.dstOffset;
                uint32_t combineEndBit   = expr.dstOffset + expr.width - 1;
                uint32_t startBit        = m_dstOffset;
                uint32_t endBit          = m_dstOffset + m_width - 1;

                // If expr BitfieldCombine is fully contained the the sinking BitfieldCombine, remove it
                if(startBit <= combineStartBit && combineEndBit <= endBit)
                    return call(expr.rhs);

                // Give up if BitfieldCombines overlap
                if(combineStartBit <= endBit && combineEndBit >= startBit)
                    return std::make_shared<Expression>(expr);

                BitfieldCombine cpy = expr;
                // Recusively sink into nested BitfieldCombines
                cpy.rhs = call(expr.rhs);

                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(CommandArgumentValue const& expr) const
            {
                auto sunkBitfieldCombine = tryEvaluate(bfc(m_src,
                                                           literal(expr),
                                                           m_srcOffset,
                                                           m_dstOffset,
                                                           m_width,
                                                           m_srcIsZero,
                                                           m_dstIsZero));

                if(!sunkBitfieldCombine.has_value())
                    return std::make_shared<Expression>(expr);

                m_sunk = true;
                return literal(sunkBitfieldCombine.value());
            }

            template <typename Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

            bool sunk() const
            {
                return m_sunk;
            }

        private:
            ExpressionPtr       m_src;
            uint32_t            m_srcOffset;
            uint32_t            m_dstOffset;
            uint32_t            m_width;
            std::optional<bool> m_srcIsZero;
            std::optional<bool> m_dstIsZero;
            mutable bool        m_sunk;
        };

        /**
         * Tries to sink a BitfieldCombine into its destination if its source is a literal.
         * Only done in nested BitfieldCombines where the destination is another BitfieldCombine.
         * Also removes BitfieldCombines that are fully overridden by BitfieldCombine argument
         */
        ExpressionPtr sinkBitfieldCombine(BitfieldCombine const& expr)
        {
            BitfieldCombine cpy = expr;

            auto const visitor
                = BitfieldCombineSinkVisitor{cpy.lhs, cpy.srcOffset, cpy.dstOffset, cpy.width};

            auto result = visitor.call(cpy.rhs);
            // BitfieldCombine was folded away
            if(visitor.sunk())
                return result;

            return bfc(cpy.lhs,
                       result,
                       cpy.srcOffset,
                       cpy.dstOffset,
                       cpy.width,
                       cpy.srcIsZero,
                       cpy.dstIsZero);
        }

        struct DeepBitfieldExtractVisitor
        {
            DeepBitfieldExtractVisitor(uint32_t offset, uint32_t width, DataType outputDataType)
                : m_offset(offset)
                , m_width(width)
                , m_outputDataType(outputDataType)
            {
            }

            ExpressionPtr operator()(BitfieldCombine const& expr) const
            {
                uint32_t combineStartBit = expr.dstOffset;
                uint32_t combineEndBit   = expr.dstOffset + expr.width - 1;
                uint32_t endBit          = m_offset + m_width - 1;
                // No overlap with this dword
                if(combineStartBit > endBit || combineEndBit < m_offset)
                {
                    return call(expr.rhs);
                }
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr operator()(Concatenate const& expr) const
            {
                uint32_t operandStartBit = 0;
                uint32_t operandEndBit   = 0;
                uint32_t startBit        = m_offset;
                uint32_t endBit          = m_offset + m_width - 1;

                std::vector<ExpressionPtr> overlapOperands;
                uint32_t                   firstOperandStartBit = 0;

                for(int i = 0; i < expr.operands.size(); ++i)
                {
                    operandStartBit = operandEndBit;
                    operandEndBit += resultVariableType(expr.operands[i]).getElementSize() * 8;

                    // bitfield overlaps with this operand
                    if(operandStartBit <= endBit && startBit <= (operandEndBit - 1))
                    {
                        if(overlapOperands.empty())
                            firstOperandStartBit = operandStartBit;

                        overlapOperands.push_back(expr.operands[i]);
                    }
                }

                // bitfield is fully contained within this operand
                if(overlapOperands.size() == 1)
                {
                    this->m_offset -= firstOperandStartBit;
                    return call(overlapOperands[0]);
                }

                uint32_t operandsSize = 0;
                for(auto const& operand : overlapOperands)
                    operandsSize += resultVariableType(operand).getElementSize();

                // The overlapping operands compose a single operand of type that matches the BitFieldExtract output type
                if(operandsSize == DataTypeInfo::Get(m_outputDataType).elementBytes)
                {
                    this->m_offset -= firstOperandStartBit;
                    return concat(overlapOperands, m_outputDataType);
                }

                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr operator()(BitFieldExtract const& expr) const
            {
                uint32_t endBit     = m_offset + m_width - 1;
                uint32_t exprEndBit = expr.width - 1;

                // Bitfield is fully contained within this BitFieldExtract
                if(endBit <= exprEndBit)
                {
                    m_offset += expr.offset;
                    return call(expr.arg);
                }

                return std::make_shared<Expression>(expr);
            }

            template <typename Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                return std::visit(*this, *expr);
            }

            uint32_t get_offset() const
            {
                return m_offset;
            }

        private:
            mutable uint32_t m_offset;
            uint32_t         m_width;
            DataType         m_outputDataType;
        };

        /**
         * Returns a BitFieldExtract expression that extracts the specified bitfield from the given expression.
         * Looks through Concatenate expressions to find the corresponding operand to extract.
         * Looks through BitfieldCombine expressions to extract from its destination operand if the BitfieldCombine and BitFieldExtract do not overlap.
         * Looks through nested BitFieldExtract expressions, if the extraction is fully contained within the inner BitFieldExtract.
         */
        BitFieldExtract deepBitFieldExtract(BitFieldExtract expr)
        {
            auto visitor = DeepBitfieldExtractVisitor(expr.offset, expr.width, expr.outputDataType);
            auto extracted  = visitor.call(expr.arg);
            uint32_t offset = visitor.get_offset();

            return BitFieldExtract{{extracted}, expr.outputDataType, offset, expr.width};
        }

        struct SimplifyExpressionVisitor
        {
            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                if constexpr(Expr::Type == Category::Conversion)
                {
                    if(expr.arg)
                    {
                        if(resultVariableType(expr) == resultVariableType(expr.arg))
                            return call(expr.arg);
                    }
                }

                Expr cpy = expr;
                cpy.arg  = call(expr.arg);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Reinterpret const& expr) const
            {
                if(resultVariableType(expr.arg).dataType == expr.destinationType)
                    return call(expr.arg);

                if(expr.arg && std::holds_alternative<Reinterpret>(*expr.arg))
                {
                    // Collapse: reinterpret(B, reinterpret(A, x)) -> reinterpret(B, x)
                    auto const& innerReinterpret = std::get<Reinterpret>(*expr.arg);
                    return call(reinterpret(expr.destinationType, innerReinterpret.arg));
                }

                Reinterpret cpy = expr;
                cpy.arg         = call(expr.arg);
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto resultVarType = resultVariableType(expr);

                auto lhs = call(expr.lhs);
                auto rhs = call(expr.rhs);

                bool eval_lhs = evaluationTimes(lhs)[EvaluationTime::Translate];
                bool eval_rhs = evaluationTimes(rhs)[EvaluationTime::Translate];

                auto simplifier = SimplifyByConstant<Expr>{resultVarType};

                ExpressionPtr rv = nullptr;

                if(eval_lhs && eval_rhs)
                {
                    rv = literal(evaluate(Expr({lhs, rhs})));
                }
                else if(CCommutativeBinary<Expr> && eval_lhs)
                {
                    rv = simplifier.call(rhs, evaluate(lhs));
                }
                else if(eval_rhs)
                {
                    rv = simplifier.call(lhs, evaluate(rhs));
                }
                else if(!CCommutativeBinary<Expr> && eval_lhs)
                {
                    auto simplifierLHS = SimplifyByConstantLHS<Expr>{resultVarType};

                    rv = simplifierLHS.call(evaluate(lhs), rhs);
                }

                if(rv != nullptr)
                {
                    if(resultVariableType(rv) != resultVarType)
                    {
                        AssertFatal(!resultVarType.isPointer(),
                                    ShowValue(expr),
                                    ShowValue(rv),
                                    ShowValue(resultVarType));
                        rv = convert(resultVarType.dataType, rv);
                    }

                    copyComment(rv, expr);
                    return rv;
                }

                return std::make_shared<Expression>(Expr({lhs, rhs, expr.comment}));
            }

            ExpressionPtr operator()(BitfieldCombine const& expr) const
            {
                auto cpy        = expr;
                auto rhsVarType = resultVariableType(cpy.rhs);

                // Nothing to combine
                if(cpy.width == 0)
                    return call(cpy.rhs);
                // Src overrides entire dst
                else if(cpy.dstOffset == 0 && cpy.width == rhsVarType.getElementSize() * 8)
                    return call(bfe(rhsVarType.dataType, cpy.lhs, cpy.srcOffset, cpy.width));

                cpy.lhs = call(expr.lhs);
                cpy.rhs = call(expr.rhs);

                return sinkBitfieldCombine(cpy);
            }

            ExpressionPtr operator()(BitFieldExtract const& expr) const
            {
                AssertFatal(expr.width > 0, "BitfieldExtract with width 0");

                BitFieldExtract cpy = deepBitFieldExtract(expr);

                // Evaluation might be possitble after deep extraction due to Concatenate expressions
                auto result = tryEvaluate(cpy);
                if(result.has_value())
                    return literal(result.value());

                // Extracting the entire arg with no offset
                auto argVarType = resultVariableType(cpy.arg);
                if(cpy.offset == 0 && cpy.width == argVarType.getElementSize() * 8
                   && argVarType.getElementSize()
                          == DataTypeInfo::Get(cpy.outputDataType).elementBytes)
                    return call(reinterpret(cpy.outputDataType, cpy.arg));

                cpy.arg = call(cpy.arg);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Concatenate const& expr) const
            {
                auto cpy = expr;
                AssertFatal(!cpy.operands.empty(), "Concatenate with no operands");
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });

                cpy = concatenatePartialSimplify(cpy);

                if(cpy.operands.size() == 1)
                    return cpy.operands[0];

                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                cpy.matA                 = call(expr.matA);
                cpy.matB                 = call(expr.matB);
                cpy.matC                 = call(expr.matC);
                cpy.scaleA               = call(expr.scaleA);
                cpy.scaleB               = call(expr.scaleB);
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Conditional const& expr) const
            {
                // Check if the condition can be evaluated at Translate time or not.
                bool const eval_lhs = evaluationTimes(expr.lhs)[EvaluationTime::Translate];
                if(eval_lhs)
                {
                    bool const condFalse = std::visit(
                        [](auto&& arg) {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr(std::is_pointer_v<T>)
                                return arg == nullptr;
                            else
                                return arg == T();
                        },
                        evaluate(expr.lhs));

                    if(condFalse)
                        return call(expr.r2hs);
                    else
                        return call(expr.r1hs);
                }

                auto cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.r1hs = call(expr.r1hs);
                cpy.r2hs = call(expr.r2hs);
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                cpy.lhs  = call(expr.lhs);
                cpy.r1hs = call(expr.r1hs);
                cpy.r2hs = call(expr.r2hs);
                return std::make_shared<Expression>(cpy);
            }

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) { op = call(op); });
                return std::make_shared<Expression>(std::move(cpy));
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return expr;

                auto evalTimes = evaluationTimes(expr);
                if(evalTimes[EvaluationTime::Translate])
                {
                    auto rv = literal(evaluate(expr));
                    AssertFatal(resultVariableType(rv) == resultVariableType(expr),
                                ShowValue(rv),
                                ShowValue(expr));
                    return rv;
                }

                auto rv = std::visit(*this, *expr);
                return rv;
            }
        };

        ExpressionPtr simplify(ExpressionPtr expr)
        {
            auto visitor = SimplifyExpressionVisitor();
            return visitor.call(expr);
        }

    }
}
