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

#include <rocRoller/CommonSubexpressionElim.hpp>
#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct SubstituteValueVisitor
        {
            SubstituteValueVisitor(Register::ValuePtr target, Register::ValuePtr replacement)
                : m_target(target)
                , m_replacement(replacement)
            {
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = call(expr.rhs);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                if(expr.matA)
                {
                    cpy.matA = call(expr.matA);
                }
                if(expr.matB)
                {
                    cpy.matB = call(expr.matB);
                }
                if(expr.matC)
                {
                    cpy.matC = call(expr.matC);
                }
                if(expr.scaleA)
                {
                    cpy.scaleA = call(expr.scaleA);
                }
                if(expr.scaleB)
                {
                    cpy.scaleB = call(expr.scaleB);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = call(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = call(expr.r2hs);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Register::ValuePtr const& value) const
            {
                return equivalent(value->expression(), m_target->expression())
                           ? m_replacement->expression()
                           : value->expression();
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return {};

                return std::visit(*this, *expr);
            }

        private:
            Register::ValuePtr m_target, m_replacement;
        };

        struct ReplaceValueWithExprVisitor
        {
            ReplaceValueWithExprVisitor(Register::ValuePtr target, ExpressionPtr replacement)
                : m_target(target)
                , m_replacement(replacement)
            {
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = call(expr.rhs);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                if(expr.matA)
                {
                    cpy.matA = call(expr.matA);
                }
                if(expr.matB)
                {
                    cpy.matB = call(expr.matB);
                }
                if(expr.matC)
                {
                    cpy.matC = call(expr.matC);
                }
                if(expr.scaleA)
                {
                    cpy.scaleA = call(expr.scaleA);
                }
                if(expr.scaleB)
                {
                    cpy.scaleB = call(expr.scaleB);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = call(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = call(expr.r2hs);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(Register::ValuePtr const& value) const
            {
                return equivalent(value->expression(), m_target->expression())
                           ? m_replacement
                           : value->expression();
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return {};

                return std::visit(*this, *expr);
            }

        private:
            Register::ValuePtr m_target;
            ExpressionPtr      m_replacement;
        };

        /**
         * @brief Replaces a register value with another in an expression
         *
         * @param expr Expression to find replacements for
         * @param target The target register to replace
         * @param replacement The new register to replace with
         * @return ExpressionPtr
         */
        ExpressionPtr substituteValue(ExpressionPtr      expr,
                                      Register::ValuePtr target,
                                      Register::ValuePtr replacement)
        {
            auto visitor = SubstituteValueVisitor(target, replacement);
            return visitor.call(expr);
        }

        struct ConsolidateSubExpressionVisitor
        {
            ConsolidateSubExpressionVisitor(ContextPtr context)
                : m_context(context)
            {
            }

            ExpressionTree operator()(Convert const& expr) const
            {
                auto tree = callUnary(expr);
                if(tree.empty())
                {
                    // Bail
                    return {};
                }

                AssertFatal(tree.back().deps.size() == 1);
                for(auto dep : tree.back().deps)
                {
                    if(tree.at(dep).reg->variableType() == expr.destinationType)
                    {
                        // Simplify
                        tree.pop_back();
                        tree.back().reg = nullptr;
                        return tree;
                    }
                }
                return tree;
            }

            template <CUnary Expr>
            ExpressionTree operator()(Expr const& expr) const
            {
                return callUnary(expr);
            }

            template <CUnary Expr>
            ExpressionTree callUnary(Expr const& expr) const
            {
                ExpressionTree tree;
                std::set<int>  deps;
                int            loc;

                tree = generateTree(expr.arg);
                if(tree.empty())
                    return {};

                loc = tree.size() - 1;
                deps.insert(loc);

                tree.push_back(ExpressionNode{
                    nullptr,
                    std::make_shared<Expression>(Expr{
                        canUseAsReg(tree.back()) ? tree.back().reg->expression() : tree.back().expr,
                        getComment(expr),
                    }
                                                     .copyParams(expr)),
                    deps,
                    tree.back().consolidationCount,
                });

                return tree;
            }

            template <CBinary Expr>
            ExpressionTree operator()(Expr const& expr) const
            {
                ExpressionTree rhsTree, tree;
                std::set<int>  deps;
                int            lhsLoc, rhsLoc;
                int            consolidationCount = 0;

                tree = generateTree(expr.lhs);
                if(tree.empty())
                    return {};

                lhsLoc = tree.size() - 1;
                deps.insert(lhsLoc);
                int lhsSize = tree.size();

                rhsTree = generateTree(expr.rhs);
                if(rhsTree.empty())
                    return {};

                auto combo = combine(tree, rhsTree);
                tree       = std::get<0>(combo);
                rhsLoc     = std::get<1>(combo);

                deps.insert(rhsLoc);

                consolidationCount += tree.at(lhsLoc).consolidationCount;
                consolidationCount += tree.at(rhsLoc).consolidationCount;
                consolidationCount += std::get<2>(combo);

                tree.push_back(ExpressionNode{
                    nullptr,
                    std::make_shared<Expression>(Expr{
                        canUseAsReg(tree.at(lhsLoc)) ? tree.at(lhsLoc).reg->expression()
                                                     : tree.at(lhsLoc).expr,
                        canUseAsReg(tree.at(rhsLoc)) ? tree.at(rhsLoc).reg->expression()
                                                     : tree.at(rhsLoc).expr,
                        getComment(expr),
                    }
                                                     .copyParams(expr)),
                    deps,
                    consolidationCount,
                });

                return tree;
            }

            template <CTernary Expr>
            ExpressionTree operator()(Expr const& expr) const
            {
                ExpressionTree r1hsTree, r2hsTree, tree;
                std::set<int>  deps;
                int            lhsLoc, r1hsLoc, r2hsLoc;
                int            consolidationCount = 0;

                tree = generateTree(expr.lhs);
                if(tree.empty())
                    return {};

                lhsLoc = tree.size() - 1;
                deps.insert(lhsLoc);
                int lhsSize = tree.size();

                r1hsTree = generateTree(expr.r1hs);
                if(r1hsTree.empty())
                    return {};

                r2hsTree = generateTree(expr.r2hs);
                if(r2hsTree.empty())
                    return {};

                auto combo1 = combine(tree, r1hsTree);
                tree        = std::get<0>(combo1);
                r1hsLoc     = std::get<1>(combo1);

                deps.insert(r1hsLoc);

                auto combo2 = combine(tree, r2hsTree);
                tree        = std::get<0>(combo2);
                r2hsLoc     = std::get<1>(combo2);

                deps.insert(r2hsLoc);

                consolidationCount += tree.at(lhsLoc).consolidationCount;
                consolidationCount += tree.at(r1hsLoc).consolidationCount;
                consolidationCount += tree.at(r2hsLoc).consolidationCount;
                consolidationCount += std::get<2>(combo1);
                consolidationCount += std::get<2>(combo2);

                tree.push_back(ExpressionNode{
                    nullptr,
                    std::make_shared<Expression>(Expr{
                        canUseAsReg(tree.at(lhsLoc)) ? tree.at(lhsLoc).reg->expression()
                                                     : tree.at(lhsLoc).expr,
                        canUseAsReg(tree.at(r1hsLoc)) ? tree.at(r1hsLoc).reg->expression()
                                                      : tree.at(r1hsLoc).expr,
                        canUseAsReg(tree.at(r2hsLoc)) ? tree.at(r2hsLoc).reg->expression()
                                                      : tree.at(r2hsLoc).expr,
                    }
                                                     .copyParams(expr)),
                    deps,
                    consolidationCount,
                });

                setComment(tree.back().expr, getComment(expr));

                return tree;
            }

            ExpressionTree operator()(ScaledMatrixMultiply const& expr) const
            {
                return {};
            }

            ExpressionTree operator()(WaveTilePtr const& value) const
            {
                return {};
            }

            ExpressionTree operator()(DataFlowTag const& value) const
            {
                return {};
            }
            ExpressionTree operator()(Register::ValuePtr const& value) const
            {
                return {{value, value->expression(), {}, 0}};
            }

            template <CValue Value>
            ExpressionTree operator()(Value const& expr) const
            {
                return {{nullptr, std::make_shared<Expression>(expr), {}, 0}};
            }

            ExpressionTree call(ExpressionPtr expr) const
            {
                if(!expr)
                    return {};
                return std::visit(*this, *expr);
            }

            ExpressionTree call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

        private:
            ContextPtr m_context;

            Register::ValuePtr resultPlaceholder(ResultType const& resType,
                                                 bool              allowSpecial = true,
                                                 int               valueCount   = 1) const
            {
                if(Register::IsSpecial(resType.regType) && resType.varType == DataType::Bool)
                {
                    if(allowSpecial)
                        return m_context->getSCC();
                    else
                        return Register::Value::Placeholder(
                            m_context, Register::Type::Scalar, resType.varType, valueCount);
                }
                return Register::Value::Placeholder(
                    m_context, resType.regType, resType.varType, valueCount);
            }

            /**
             * @brief Generate a tree from an expression's argument. Creates placeholder.
             *
             * @param expr Expression to decompose
             * @return ExpressionTree
             */
            ExpressionTree generateTree(ExpressionPtr expr) const
            {
                ExpressionTree tree = {};

                if(expr)
                {
                    tree = call(expr);
                    if(tree.empty())
                    {
                        // Bail
                        return {};
                    }
                    if(tree.back().reg == nullptr)
                    {
                        tree.back().reg = resultPlaceholder(resultType(expr), sccAvailable(tree));
                    }
                }

                return tree;
            }

            /**
             * @brief Determine whether we can use the register of the node, or if we should use the expression instead
             *
             * @param target Node to analyze
             */
            bool canUseAsReg(ExpressionNode const& target) const
            {
                return !(target.reg && target.reg->regType() == Register::Type::Literal)
                       && !(target.expr
                            && (std::holds_alternative<CommandArgumentPtr>(*target.expr)
                                || std::holds_alternative<AssemblyKernelArgumentPtr>(*target.expr)
                                || std::holds_alternative<WaveTilePtr>(*target.expr)));
            }

            /**
             * @brief Determine if we should note this node in consolidation count tracking
             *
             * @param target Node to analyze
             */
            bool shouldTrack(ExpressionNode const& target) const
            {
                return !(target.reg && target.reg->regType() == Register::Type::Literal)
                       && !(target.expr
                            && (std::holds_alternative<Register::ValuePtr>(*target.expr)
                                || std::holds_alternative<CommandArgumentPtr>(*target.expr)
                                || std::holds_alternative<AssemblyKernelArgumentPtr>(*target.expr)
                                || std::holds_alternative<WaveTilePtr>(*target.expr)));
            }

            /**
             * @brief Check if the SCC register is available to use
             *
             * @param tree The expression tree to check
             * @return False if SCC is not to be used
             */
            bool sccAvailable(ExpressionTree const& tree) const
            {
                return false;
            }

            /**
             * @brief Combine two ExpressionTrees
             *
             * @param tree1
             * @param tree2
             * @return std::tuple<ExpressionTree, int> Combined tree, the index of tree2's root in resulting tree
             */
            std::tuple<ExpressionTree, int, int> combine(ExpressionTree const& tree1,
                                                         ExpressionTree const& tree2) const
            {
                auto tree    = tree1; // Tree to graft onto
                auto rhsTree = tree2; // Tree to merge in
                int  lhsSize = tree.size();

                // Replace the RHS dependencies with what they would be if the trees were combined without elimination
                for(auto& rhsNode : rhsTree)
                {
                    std::set<int> newDeps;
                    for(auto dep : rhsNode.deps)
                    {
                        newDeps.insert(dep + lhsSize);
                    }
                    rhsNode.deps = newDeps;
                }

                // The location of the root of the right tree after combining
                int rhsLoc = -1; // Initialize with invalid value

                // Track how many nodes were skipped
                int tracker = 0;

                auto sccExpr = m_context->getSCC()->expression();
                auto lhsHasSCC
                    = std::find_if(tree.begin(),
                                   tree.end(),
                                   [sccExpr](auto origNode) {
                                       return equivalent(origNode.reg->expression(), sccExpr);
                                   })
                      != tree.end();

                // Loop through the RHS
                for(auto rhsNodeIt = rhsTree.begin(); rhsNodeIt != rhsTree.end(); rhsNodeIt++)
                {
                    // Find a node that has an equivalent expression in the growing tree
                    auto it = std::find_if(tree.begin(), tree.end(), [&rhsNodeIt](auto origNode) {
                        return equivalent(rhsNodeIt->expr, origNode.expr);
                    });

                    bool rhsIsSCC = equivalent(rhsNodeIt->reg->expression(),
                                               m_context->getSCC()->expression());

                    if(it != tree.end())
                    {
                        // If we found a valid expression to eliminate
                        auto rhsIndex = rhsNodeIt - rhsTree.begin();
                        auto newIndex = it - tree.begin();

                        if(rhsTree.size() - 1 == rhsIndex)
                        {
                            // If we're on the last RHS node, the root is no longer the last element
                            rhsLoc = newIndex;
                        }

                        // Loop through remaining RHS nodes
                        for(auto scanIt = rhsNodeIt + 1; scanIt != rhsTree.end(); scanIt++)
                        {
                            // Find relevant dependencies
                            if(scanIt->deps.contains(rhsIndex + lhsSize))
                            {
                                // Replace expression
                                scanIt->expr
                                    = substituteValue(scanIt->expr, rhsNodeIt->reg, it->reg);
                                // Replace dependency
                                scanIt->deps.erase(rhsIndex + lhsSize);
                                scanIt->deps.insert(newIndex);
                            }
                        }
                        if(shouldTrack(*rhsNodeIt))
                        {
                            tracker++;
                        }

                        rhsNodeIt->reg = nullptr;
                        continue;
                    }

                    if(rhsIsSCC && lhsHasSCC)
                    {
                        // Found 2 SCCs! Fixup
                        rhsNodeIt->reg = resultPlaceholder(resultType(rhsNodeIt->expr), false);
                    }

                    // Node is good to add to tree
                    tree.push_back(*rhsNodeIt);

                    auto rhsIndex = rhsNodeIt - rhsTree.begin();

                    // Update following RHS nodes if necessary
                    for(auto scanIt = rhsNodeIt + 1; scanIt != rhsTree.end(); scanIt++)
                    {
                        if(rhsIndex + lhsSize != tree.size() - 1
                           && scanIt->deps.contains(rhsIndex + lhsSize))
                        {
                            scanIt->deps.erase(rhsIndex + lhsSize);
                            scanIt->deps.insert(tree.size() - 1);
                        }
                    }
                }

                // The last element is the RHS root, unless it was otherwise updated
                if(rhsLoc == -1)
                    rhsLoc = tree.size() - 1;

                return {tree, rhsLoc, tracker};
            }
        };

        ExpressionTree consolidateSubExpressions(ExpressionPtr expr, ContextPtr context)
        {
            if(!expr)
                return {};
            auto visitor = ConsolidateSubExpressionVisitor(context);
            return visitor.call(*expr);
        }

        int getConsolidationCount(ExpressionTree const& tree)
        {
            return tree.back().consolidationCount;
        }

        ExpressionPtr rebuildExpressionHelper(ExpressionTree const& tree, int index)
        {
            auto root = tree.at(index);
            if(root.deps.empty())
            {
                return root.expr;
            }

            ExpressionPtr modifiedExpr = root.expr;
            for(auto dep : root.deps)
            {
                auto                        replacementExpr = rebuildExpressionHelper(tree, dep);
                ReplaceValueWithExprVisitor v(tree.at(dep).reg, replacementExpr);
                modifiedExpr = v.call(modifiedExpr);
            }
            return modifiedExpr;
        }

        ExpressionPtr rebuildExpression(ExpressionTree const& tree)
        {
            if(tree.empty())
            {
                return nullptr;
            }
            return rebuildExpressionHelper(tree, tree.size() - 1);
        }
    }
}
