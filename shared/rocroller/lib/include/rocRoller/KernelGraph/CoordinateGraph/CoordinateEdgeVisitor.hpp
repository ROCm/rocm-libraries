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

#pragma once

#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdge.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        template <typename T>
        concept CTUndefinedEdge = std::is_same<ConstructMacroTile, T>::value || std::
            is_same<DestructMacroTile, T>::value || std::is_same<Forget, T>::value;

        struct BaseEdgeVisitor
        {
            // index expressions for the dimensions
            std::vector<Expression::ExpressionPtr> indexes;
            std::vector<Dimension>                 srcs, dsts;
            std::vector<int>                       srcTags, dstTags;

            inline void setLocation(std::vector<Expression::ExpressionPtr> _indexes,
                                    std::vector<Dimension> const&          _srcs,
                                    std::vector<Dimension> const&          _dsts,
                                    std::vector<int> const&                _srcTags,
                                    std::vector<int> const&                _dstTags)
            {
                indexes = _indexes;
                srcs    = _srcs;
                dsts    = _dsts;
                srcTags = _srcTags;
                dstTags = _dstTags;
            }
        };

        struct ForwardEdgeVisitor : public BaseEdgeVisitor
        {
            std::vector<Expression::ExpressionPtr> operator()(Flatten const& e)
            {
                auto result = indexes[0];
                for(uint d = 1; d < srcs.size(); ++d)
                    result = result * getSize(srcs[d]) + indexes[d];
                setComment(result, "Flatten");
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Join const& e)
            {
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));
                auto result = indexes[0] * getStride(srcs[0]);
                for(uint d = 1; d < srcs.size(); ++d)
                    result = result + indexes[d] * getStride(srcs[d]);
                setComment(result, "Join");
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(PiecewiseAffineJoin const& e)
            {
                AssertFatal(srcs.size() == e.strides.first.size(),
                            ShowValue(e.strides.first.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));
                auto branchTrue  = indexes[0] * e.strides.first[0];
                auto branchFalse = indexes[0] * e.strides.second[0];
                for(uint d = 1; d < srcs.size(); ++d)
                {
                    branchTrue  = branchTrue + indexes[d] * e.strides.first[d];
                    branchFalse = branchFalse + indexes[d] * e.strides.second[d];
                }
                if(e.initialValues.first)
                    branchTrue = branchTrue + e.initialValues.first;
                if(e.initialValues.second)
                    branchFalse = branchFalse + e.initialValues.second;

                auto condition = positionalArgumentPropagation(e.condition, indexes);

                auto result = std::make_shared<Expression::Expression>(
                    Expression::Conditional{condition, branchTrue, branchFalse});
                setComment(result, "PiecewiseAffineJoin");
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Sunder const& e)
            {
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));
                AssertFatal(srcs.size() > 1, ShowValue(srcs.size()));

                int index = getUnsignedInt(evaluate(indexes.back()));
                AssertFatal(index >= 0 && index < (srcs.size() - 1));

                Expression::ExpressionPtr offset = nullptr;

                for(int i = 0; i < index; i++)
                {
                    auto mySize = getSize(srcs[i]);
                    offset      = offset ? offset + mySize : mySize;
                }

                auto result = indexes[index];
                if(offset != nullptr)
                    result = result + offset;

                setComment(result, "Sunder");
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Tile const& e)
            {
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                std::vector<Expression::ExpressionPtr> rv(dsts.size());

                auto input = indexes[0];
                for(int i = dsts.size() - 1; i > 0; i--)
                {
                    auto size = getSize(dsts[i]);
                    rv[i]     = input % size;
                    input     = input / size;
                }
                rv[0] = input;

                for(int i = 0; i < rv.size(); i++)
                {
                    setComment(rv[i], concatenate("Tile[", i, "]"));
                }

                return rv;
            }

            template <CTUndefinedEdge T>
            std::vector<Expression::ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Edge transform not defined.");
            }

            template <typename T>
            std::vector<Expression::ExpressionPtr> operator()(Split const& e)
            {
                Throw<FatalError>("Split edge found in forward transform.");
            }

            template <typename T>
            std::vector<Expression::ExpressionPtr> operator()(T const& e)
            {
                return indexes;
            }

            std::vector<Expression::ExpressionPtr> call(Edge const& e)
            {
                return std::visit(
                    [&](Edge const& edge) {
                        return std::visit(
                            [&](auto const& subEdge) { return std::visit(*this, subEdge); }, edge);
                    },
                    e);
            }
        };

        struct ReverseEdgeVisitor : public BaseEdgeVisitor
        {
            std::vector<Expression::ExpressionPtr> operator()(Flatten const& e)
            {
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));
                if(srcs.size() == 1)
                    return indexes;

                std::vector<Expression::ExpressionPtr> rv(srcs.size());

                auto input = indexes[0];
                for(int i = srcs.size() - 1; i > 0; i--)
                {
                    auto size = getSize(srcs[i]);
                    rv[i]     = input % size;
                    input     = input / size;
                }
                rv[0] = input;

                for(int i = 0; i < rv.size(); i++)
                {
                    setComment(rv[i], concatenate("Flatten[", i, "]"));
                }

                return rv;
            }

            std::vector<Expression::ExpressionPtr> operator()(Split const& e)
            {
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                auto result = indexes[0] * getStride(dsts[0]);
                for(uint d = 1; d < dsts.size(); ++d)
                    result = result + indexes[d] * getStride(dsts[d]);

                setComment(result, "Split");
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(PiecewiseAffineJoin const& e)
            {
                AssertFatal(dsts.size() == e.strides.first.size(),
                            ShowValue(e.strides.first.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                auto branchTrue  = indexes[0] * e.strides.first[0];
                auto branchFalse = indexes[0] * e.strides.second[0];
                for(uint d = 1; d < dsts.size(); ++d)
                {
                    branchTrue  = branchTrue + indexes[d] * e.strides.first[d];
                    branchFalse = branchFalse + indexes[d] * e.strides.second[d];
                }
                if(e.initialValues.first)
                    branchTrue = branchTrue + e.initialValues.first;
                if(e.initialValues.second)
                    branchFalse = branchFalse + e.initialValues.second;

                auto condition = positionalArgumentPropagation(e.condition, indexes);

                auto result = std::make_shared<Expression::Expression>(
                    Expression::Conditional{condition, branchTrue, branchFalse});
                setComment(result, "PiecewiseAffineJoin");
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Sunder const& e)
            {
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                AssertFatal(dsts.size() > 1, ShowValue(dsts.size()));

                int index = getUnsignedInt(evaluate(indexes.back()));
                AssertFatal(index >= 0 && index < (dsts.size() - 1));

                Expression::ExpressionPtr offset = nullptr;

                for(int i = 0; i < index; i++)
                {
                    auto mySize = getSize(dsts[i]);
                    offset      = offset ? offset + mySize : mySize;
                }

                auto result = indexes[index];
                if(offset != nullptr)
                    result = result + offset;

                setComment(result, "Sunder");
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Tile const& e)
            {
                auto result = indexes[0];
                for(uint d = 1; d < dsts.size(); ++d)
                    result = result * getSize(dsts[d]) + indexes[d];
                setComment(result, "Tile");
                return {result};
            }

            template <CTUndefinedEdge T>
            std::vector<Expression::ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Edge transform not defined.");
            }

            template <typename T>
            std::vector<Expression::ExpressionPtr> operator()(T const& e)
            {
                return indexes;
            }

            std::vector<Expression::ExpressionPtr> call(Edge const& e)
            {
                return std::visit(
                    [&](Edge const& edge) {
                        return std::visit(
                            [&](auto const& subEdge) { return std::visit(*this, subEdge); }, edge);
                    },
                    e);
            }
        };

        /*
         * Diff edge visitors.
         */
        struct BaseEdgeDiffVisitor : public BaseEdgeVisitor
        {
            Expression::ExpressionPtr zero;

            std::map<int, Expression::ExpressionPtr> deltas;

            BaseEdgeDiffVisitor() = delete;
            BaseEdgeDiffVisitor(int x, Expression::ExpressionPtr dx)
            {
                deltas.emplace(x, dx);
                zero = Expression::literal(0);
            }

            //
            // Get delta associated with Dimension thus far.
            //
            Expression::ExpressionPtr getDelta(int tag) const
            {
                if(deltas.count(tag) > 0)
                {
                    auto expr = deltas.at(tag);
                    if(evaluationTimes(expr)[Expression::EvaluationTime::Translate])
                    {
                        return Expression::literal(evaluate(deltas.at(tag)));
                    }
                    return simplify(expr);
                }
                return zero;
            }
        };

        struct ForwardEdgeDiffVisitor : public BaseEdgeDiffVisitor
        {
            using BaseEdgeDiffVisitor::BaseEdgeDiffVisitor;

            std::vector<Expression::ExpressionPtr> operator()(Flatten const& e)
            {
                AssertFatal(srcs.size() > 0 && srcs.size() == indexes.size(),
                            ShowValue(srcs.size()),
                            ShowValue(indexes.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));

                std::vector<Expression::ExpressionPtr> strides(srcs.size());
                strides[srcs.size() - 1] = Expression::literal(1);
                for(size_t i = srcs.size() - 1; i > 0; --i)
                {
                    strides[i - 1] = strides[i] * getSize(srcs[i]);
                }

                auto index = indexes[0] * strides[0];
                auto delta = getDelta(srcTags[0]) * strides[0];
                for(uint d = 1; d < srcs.size(); ++d)
                {
                    index = index + indexes[d] * strides[d];
                    delta = delta + getDelta(srcTags[d]) * strides[d];
                }
                deltas.emplace(dstTags[0], delta);
                setComment(index, "DFlatten");
                return {index};
            }

            std::vector<Expression::ExpressionPtr> operator()(Join const& e)
            {
                AssertFatal(srcs.size() > 0 && srcs.size() == indexes.size(),
                            ShowValue(srcs.size()),
                            ShowValue(indexes.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));

                auto index = indexes[0] * getStride(srcs[0]);
                auto delta = getDelta(srcTags[0]) * getStride(srcs[0]);
                for(uint d = 1; d < srcs.size(); ++d)
                {
                    index = index + indexes[d] * getStride(srcs[d]);
                    delta = delta + getDelta(srcTags[d]) * getStride(srcs[d]);
                }
                deltas.emplace(dstTags[0], delta);
                setComment(index, "DJoin");
                return {index};
            }

            std::vector<Expression::ExpressionPtr> operator()(PiecewiseAffineJoin const& e)
            {
                AssertFatal(srcs.size() == e.strides.first.size(),
                            ShowValue(e.strides.first.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));
                auto branchTrue  = indexes[0] * e.strides.first[0];
                auto branchFalse = indexes[0] * e.strides.second[0];
                for(uint d = 1; d < srcs.size(); ++d)
                {
                    branchTrue  = branchTrue + indexes[d] * e.strides.first[d];
                    branchFalse = branchFalse + indexes[d] * e.strides.second[d];
                }
                if(e.initialValues.first)
                    branchTrue = branchTrue + e.initialValues.first;
                if(e.initialValues.second)
                    branchFalse = branchFalse + e.initialValues.second;

                auto condition = positionalArgumentPropagation(e.condition, indexes);

                auto result = std::make_shared<Expression::Expression>(
                    Expression::Conditional{condition, branchTrue, branchFalse});
                setComment(result, "PiecewiseAffineJoin");

                // TODO: This isn't right; but currently only used
                // within Workgroup remappings, and these don't
                // contribute to strides.
                auto delta = getDelta(srcTags[0]);
                deltas.emplace(dstTags[0], delta);
                setComment(result, "DPiecewiseAffineJoin");

                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Sunder const& e)
            {
                AssertFatal(srcs.size() > 1 && srcs.size() == indexes.size(),
                            ShowValue(srcs.size()),
                            ShowValue(indexes.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));

                int index = getUnsignedInt(evaluate(indexes.back()));
                AssertFatal(index >= 0 && index < (srcs.size() - 1));

                Expression::ExpressionPtr offset = nullptr;

                for(int i = 0; i < index; i++)
                {
                    auto mySize = getSize(srcs[i]);
                    offset      = offset ? offset + mySize : mySize;
                }

                auto result = indexes[index];
                if(offset != nullptr)
                    result = result + offset;

                auto delta = getDelta(srcTags[index]);
                deltas.emplace(dstTags[0], delta);
                setComment(result, "DSunder");
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Tile const& e)
            {
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                auto delta = getDelta(srcTags[0]);
                auto input = indexes[0];

                std::vector<Expression::ExpressionPtr> rv(dsts.size());
                for(int i = dsts.size() - 1; i > 0; i--)
                {
                    auto size = getSize(dsts[i]);
                    rv[i]     = input % size;
                    input     = input / size;
                    deltas.emplace(dstTags[i], delta % size);
                    delta = delta / size;
                }
                deltas.emplace(dstTags[0], delta);
                rv[0] = input;

                for(int i = 0; i < rv.size(); i++)
                    setComment(rv[i], concatenate("DTile[", i, "]"));
                return rv;
            }

            std::vector<Expression::ExpressionPtr> operator()(PassThrough const& e)
            {
                AssertFatal(srcs.size() == 1 && srcs.size() == indexes.size(),
                            ShowValue(srcs.size()),
                            ShowValue(indexes.size()));
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));

                auto delta = getDelta(srcTags[0]);
                deltas.emplace(dstTags[0], delta);
                return {indexes};
            }

            template <CTUndefinedEdge T>
            std::vector<Expression::ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Edge transform not defined.");
            }

            template <typename T>
            std::vector<Expression::ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Forward derivative not implemented yet for: ",
                                  ShowValue(e.toString()));
            }

            std::vector<Expression::ExpressionPtr> call(Edge const& e)
            {
                return std::visit(
                    [&](Edge const& edge) {
                        return std::visit(
                            [&](auto const& subEdge) { return std::visit(*this, subEdge); }, edge);
                    },
                    e);
            }
        };

        struct ReverseEdgeDiffVisitor : public BaseEdgeDiffVisitor
        {
            using BaseEdgeDiffVisitor::BaseEdgeDiffVisitor;

            std::vector<Expression::ExpressionPtr> operator()(Split const& e)
            {
                AssertFatal(dsts.size() > 0 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));

                auto index = indexes[0] * getStride(dsts[0]);
                auto delta = getDelta(dstTags[0]) * getStride(dsts[0]);
                for(uint d = 1; d < dsts.size(); ++d)
                {
                    index = index + indexes[d] * getStride(dsts[d]);
                    delta = delta + getDelta(dstTags[d]) * getStride(dsts[d]);
                }
                deltas.emplace(srcTags[0], delta);
                setComment(index, "DSplit");
                return {index};
            }

            std::vector<Expression::ExpressionPtr> operator()(PiecewiseAffineJoin const& e)
            {
                AssertFatal(dsts.size() == e.strides.first.size(),
                            ShowValue(e.strides.first.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                auto branchTrue  = indexes[0] * e.strides.first[0];
                auto branchFalse = indexes[0] * e.strides.second[0];
                for(uint d = 1; d < dsts.size(); ++d)
                {
                    branchTrue  = branchTrue + indexes[d] * e.strides.first[d];
                    branchFalse = branchFalse + indexes[d] * e.strides.second[d];
                }
                if(e.initialValues.first)
                    branchTrue = branchTrue + e.initialValues.first;
                if(e.initialValues.second)
                    branchFalse = branchFalse + e.initialValues.second;

                auto condition = positionalArgumentPropagation(e.condition, indexes);

                auto result = std::make_shared<Expression::Expression>(
                    Expression::Conditional{condition, branchTrue, branchFalse});
                setComment(result, "PiecewiseAffineJoin");

                // TODO: This isn't right; but currently only used
                // within Workgroup remappings, and these don't
                // contribute to strides.
                auto delta = getDelta(dstTags[0]);
                deltas.emplace(srcTags[0], delta);
                setComment(result, "DPiecewiseAffineJoin");

                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Sunder const& e)
            {
                AssertFatal(dsts.size() > 1 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));

                int index = getUnsignedInt(evaluate(indexes.back()));
                AssertFatal(index >= 0 && index < (dsts.size() - 1));

                Expression::ExpressionPtr offset = nullptr;

                for(int i = 0; i < index; i++)
                {
                    auto mySize = getSize(dsts[i]);
                    offset      = offset ? offset + mySize : mySize;
                }

                auto result = indexes[index];
                if(offset != nullptr)
                    result = result + offset;

                auto delta = getDelta(dstTags[index]);
                deltas.emplace(srcTags[0], delta);
                setComment(result, "DSunder");
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Tile const& e)
            {
                AssertFatal(dsts.size() > 0 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));

                auto index = indexes[0];
                auto delta = getDelta(dstTags[0]);
                for(uint d = 1; d < dsts.size(); ++d)
                {
                    index = index * getSize(dsts[d]) + indexes[d];
                    delta = delta * getSize(dsts[d]) + getDelta(dstTags[d]);
                }
                deltas.emplace(srcTags[0], delta);
                setComment(index, "DTile");
                return {index};
            }

            std::vector<Expression::ExpressionPtr> operator()(Flatten const& e)
            {
                AssertFatal(dsts.size() == 1 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() > 1, ShowValue(srcs.size()));

                auto delta = getDelta(dstTags[0]);
                auto input = indexes[0];

                std::vector<Expression::ExpressionPtr> rv(srcs.size());
                for(int i = srcs.size() - 1; i > 0; i--)
                {
                    auto size = getSize(srcs[i]);
                    rv[i]     = input % size;
                    input     = input / size;
                    deltas.emplace(srcTags[i], delta % size);
                    delta = delta / size;
                }
                deltas.emplace(srcTags[0], delta);
                rv[0] = input;
                for(int i = 0; i < rv.size(); i++)
                    setComment(rv[i], concatenate("DFlatten[", i, "]"));
                return rv;
            }

            std::vector<Expression::ExpressionPtr> operator()(PassThrough const& e)
            {
                AssertFatal(dsts.size() == 1 && dsts.size() == indexes.size(),
                            ShowValue(dsts.size()),
                            ShowValue(indexes.size()));
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));

                auto delta = getDelta(dstTags[0]);
                deltas.emplace(srcTags[0], delta);
                return {indexes};
            }

            template <CTUndefinedEdge T>
            std::vector<Expression::ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Edge transform not defined.");
            }

            template <typename T>
            std::vector<Expression::ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Reverse derivative not implemented yet for: ",
                                  ShowValue(e.toString()));
            }

            std::vector<Expression::ExpressionPtr> call(Edge const& e)
            {
                return std::visit(
                    [&](Edge const& edge) {
                        return std::visit(
                            [&](auto const& subEdge) { return std::visit(*this, subEdge); }, edge);
                    },
                    e);
            }
        };
    }
}
