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

#include <string>

#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdge_fwd.hpp>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/StructUtils.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateGraph
    {
        /**
         * EdgeType - type of edge in the coordinate transfrom graph.
         *
         * NOTE: The order of nodes matter!
         *
         * Traversal routines can be limited to traversing only a
         * particular type of edge.
         */
        enum class EdgeType : int
        {
            CoordinateTransform,
            DataFlow,
            Any,
            Count,
            None = Count
        };

        inline std::string toString(EdgeType const& e)
        {
            switch(e)
            {
            case EdgeType::CoordinateTransform:
                return "CoordinateTransform";
            case EdgeType::DataFlow:
                return "DataFlow";
            case EdgeType::Any:
                return "Any";
            default:
                break;
            }
            Throw<FatalError>("Invalid EdgeType");
        }

        inline std::ostream& operator<<(std::ostream& stream, EdgeType e)
        {
            return stream << toString(e);
        }

        /*
         * Coordinate transform edges
         *
         * Represents the arithmetic from going to one set of coordinates
         * to another.
         */

        /**
         * DataFlow - used to denote data flow through storage locations.
         */
        RR_EMPTY_STRUCT_WITH_NAME(DataFlow);

        /**
         * Alias - denotes that the source will borrow the register
         * allocation from the dest.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Alias);

        /**
         * Index - denotes that the source will index the register
         * allocation from the dest.
         */
        struct Index
        {
            int index = -1;

            Index() = default;

            Index(int const index)
                : index(index)
            {
            }

            std::string toString() const
            {
                return name();
            }

            std::string name() const
            {
                return "Index";
            }
        };

        struct Segment
        {
            int index = -1;

            Segment() = default;

            Segment(int const index)
                : index(index)
            {
            }

            std::string toString() const
            {
                return name();
            }

            std::string name() const
            {
                return "Segment";
            }
        };

        /**
         * Buffer - denotes SRD for MUBUF instructions
         */
        RR_EMPTY_STRUCT_WITH_NAME(Buffer);

        /**
         * Offset - denotes offset between target/increment
         * dimensions.
         *
         * See ComputeIndex.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Offset);

        /**
         * Stride - denotes stride between target/increment
         * dimensions.
         *
         * See ComputeIndex.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Stride);

        /**
         * View - denotes a virtual node with is a view into another
         * node.
         */
        RR_EMPTY_STRUCT_WITH_NAME(View);

        /**
         * Construct MacroTile.
         *
         * Joins SubDimensions to MacroTile during translation and
         * lowering.  Should not persist.
         */
        RR_EMPTY_STRUCT_WITH_NAME(ConstructMacroTile);

        /**
         * Destruct MacroTile.
         *
         * Splits MacroTile into SubDimensions during translation and
         * lowering.  Should not persist.
         */
        RR_EMPTY_STRUCT_WITH_NAME(DestructMacroTile);

        /**
         * Flatten dimensions (row-major, contiguous storage).
         *
         * For example, with input dimensions
         *
         *   I = Dimension(size=n_i, stride=s_i)
         *   J = Dimension(size=n_j, stride=s_j)
         *
         * and output dimension
         *
         *   O = Dimension()
         *
         * the coordinate transform
         *
         *   Flatten(I, J; O)(i, j) = i * n_j + j
         *
         * and the inverse coordinate transform is
         *
         *   Flatten'(O; I, J)(o) = { o / n_j, o % n_j }.
         *
         */
        RR_EMPTY_STRUCT_WITH_NAME(Flatten);

        /**
         * Forget (drop) dimensions.
         *
         * Used to express a transform where coordinates are "lost"
         * and can't be reconstructed or transformed further.  For
         * example, a scalar VGPR doesn't have a coordinate.
         *
         * Forward and reverse transforms aren't defined.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Forget);

        /**
         * Inherit dimension(s).
         *
         * Used to express a transform where coordinates are inherited
         * from other coordinates.  For example, a scalar VGPR doesn't
         * have a coordinate, but may conceptually inherit coordinates
         * from other dimensions (eg, workgroup and wavefront).
         *
         * The forward coordinate transform returns the destination
         * indexes.  The reverse coordinate transform returns the
         * source indexes.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Inherit);

        /**
         * Join dimensions (forward version of Split).
         *
         * For example, with input dimensions
         *
         *   I = Dimension(size=n_i, stride=s_i)
         *   J = Dimension(size=n_j, stride=s_j)
         *
         * and output dimensions
         *
         *   F = Dimension()
         *
         * the coordinate transform is
         *
         *   Join(I, J; F)(i, j) = i * s_i + j * s_j
         *
         */
        RR_EMPTY_STRUCT_WITH_NAME(Join);

        /**
         * Make output (subsequent dims should be tagged as output).
         */
        RR_EMPTY_STRUCT_WITH_NAME(MakeOutput);

        /**
         * PassThrough (identity).
         *
         * Forward and reverse transforms are the identity.
         */
        RR_EMPTY_STRUCT_WITH_NAME(PassThrough);

        /**
         * Join dimensions using conditional strides and initial values.
         *
         * The strides and initial values are passed in.  They are not
         * queried from the graph.
         *
         * For example, with input dimensions
         *
         *   I = Dimension()
         *   J = Dimension()
         *
         * and output dimensions
         *
         *   F = Dimension()
         *
         * the coordinate transform is
         *
         *   selector = Condition(I, J) ? 0 : 1
         *   Join(I, J; F)(i, j) = i * strides[selector]_i + j * strides[selector]_j + initial[selector]
         *
         */
        struct PiecewiseAffineJoin
        {
            using ExpressionPtr     = Expression::ExpressionPtr;
            using ExpressionPtrPair = std::pair<ExpressionPtr, ExpressionPtr>;
            using ExpressionPtrVectorPair
                = std::pair<std::vector<ExpressionPtr>, std::vector<ExpressionPtr>>;

            ExpressionPtr           condition;
            ExpressionPtrVectorPair strides;
            ExpressionPtrPair       initialValues;

            PiecewiseAffineJoin() = default;

            PiecewiseAffineJoin(ExpressionPtr           condition,
                                ExpressionPtrVectorPair strides,
                                ExpressionPtrPair       initialValues)
                : condition(std::move(condition))
                , strides(std::move(strides))
                , initialValues(std::move(initialValues))
            {
                AssertFatal(strides.first.size() == strides.second.size());
            }

            std::string toString() const
            {
                return name();
            }

            std::string name() const
            {
                return "PiecewiseAffineJoin";
            }
        };

        /**
         * Duplicate (identity) -- THIS IS TEMPORARY.
         *
         * Forward and reverse transforms are the identity.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Duplicate);

        /**
         * Split a tensor into subdimensions.
         *
         * For example, with input dimensions
         *
         *   F = Dimension()
         *
         * and output dimensions
         *
         *   I = Dimension(size=n_i, stride=s_i)
         *   J = Dimension(size=n_j, stride=s_j)
         *
         * the inverse coordinate transform is
         *
         *   Split'(I, J; F)(i, j) = i * s_i + j * s_j
         *
         */
        RR_EMPTY_STRUCT_WITH_NAME(Split);

        /**
         * Unevenly split (asunder) a dimension into subdimensions, and select the active
         * dimension based on the last indexed input dimension.
         *
         * For example, with input dimensions
         *
         *   F = Dimension()
         *
         * and output dimensions
         *
         *   I = Dimension(size=n_i, stride=1)
         *   J = Dimension(size=n_j, stride=1)
         *   K = Dimension(size=2, stride=1)
         *
         * the inverse coordinate transform is
         *
         *   Sunder'(I, J, K; F)(i, j, k) = i     | k=0
         *                                  j+n_i | k=1
         *
         */
        RR_EMPTY_STRUCT_WITH_NAME(Sunder);

        /**
         * Tile a dimension.
         *
         * For example, with an input dimension of
         *
         *   I = Dimension(size=n_i, stride=s_i)
         *
         * and output dimensions of
         *
         *   B = Dimension(size=null)
         *   T = Dimension(size=64)
         *
         * the coordinate transform of Tile(I; B, T) is
         *
         *   Tile(I; B, T)(i) = { i / 64, i % 64 }
         *
         * and the reverse coordinate transform Tile'(B, T; I) is
         *
         *   Tile'(B, T; I)(b, t) = b * 64 + t
         */
        RR_EMPTY_STRUCT_WITH_NAME(Tile);

        /*
         * Helpers
         */

        inline std::string toString(const CoordinateTransformEdge& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }

        template <CConcreteCoordinateTransformEdge T>
        inline std::string name(T const& x)
        {
            return x.toString();
        }

        inline std::string name(const CoordinateTransformEdge& x)
        {
            return toString(x);
        }

        inline std::string toString(const DataFlowEdge& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }

        template <CConcreteDataFlowEdge T>
        inline std::string name(T const& x)
        {
            return x.toString();
        }

        inline std::string name(const DataFlowEdge& x)
        {
            return toString(x);
        }

        inline std::string toString(const Edge& x)
        {
            return std::visit([](const auto& a) { return toString(a); }, x);
        }

        inline std::string name(Edge const& x)
        {
            return std::visit(
                rocRoller::overloaded{[](const CoordinateTransformEdge&) { return "Transform"; },
                                      [](const DataFlowEdge&) { return "DataFlow"; }},
                x);
        }

        template <typename T>
        requires(
            std::constructible_from<
                CoordinateTransformEdge,
                T> || std::constructible_from<DataFlowEdge, T>) inline bool isEdge(const Edge& x)
        {
            if constexpr(std::is_same_v<DataFlowEdge, T>)
            {
                if(std::holds_alternative<DataFlowEdge>(x))
                {
                    return true;
                }
            }
            else if constexpr(std::constructible_from<DataFlowEdge, T>)
            {
                if(std::holds_alternative<DataFlowEdge>(x))
                {
                    if(std::holds_alternative<T>(std::get<DataFlowEdge>(x)))
                        return true;
                }
            }
            else if constexpr(std::is_same_v<CoordinateTransformEdge, T>)
            {
                if(std::holds_alternative<CoordinateTransformEdge>(x))
                {
                    return true;
                }
            }
            else if constexpr(std::constructible_from<CoordinateTransformEdge, T>)
            {
                if(std::holds_alternative<CoordinateTransformEdge>(x))
                {
                    if(std::holds_alternative<T>(std::get<CoordinateTransformEdge>(x)))
                        return true;
                }
            }
            return false;
        }
    }
}
