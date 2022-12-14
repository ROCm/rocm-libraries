#pragma once

#include <string>

#include "Edge_fwd.hpp"

namespace rocRoller
{
    namespace KernelGraph::CoordGraph
    {
        /**
         * EdgeType - type of edge in the coordinate transfrom graph.
         * Traversal routines can be limited to only traversing a
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

        /*
         * Coordinate transform edges
         */

        /**
         * DataFlow - used to denote data flow.
         */
        struct DataFlow
        {
            virtual std::string toString() const
            {
                return "DataFlow";
            }
        };

        /**
         * Buffer - denotes SRD for MUBUF instructions
         */
        struct Buffer
        {
            virtual std::string toString() const
            {
                return "Buffer";
            }
        };
        /**
         * Offset - denotes offset between target/increment
         * dimensions.
         *
         * See ComputeIndex.
         */
        struct Offset
        {
            virtual std::string toString() const
            {
                return "Offset";
            }
        };

        /**
         * Stride - denotes stride between target/increment
         * dimensions.
         *
         * See ComputeIndex.
         */
        struct Stride
        {
            virtual std::string toString() const
            {
                return "Stride";
            }
        };

        /**
         * Construct MacroTile.
         *
         * Joins SubDimensions to MacroTile during translation and
         * lowering.  Should not persist.
         */
        struct ConstructMacroTile
        {
            std::string toString() const
            {
                return "ConstructTensorTile";
            }
        };

        /**
         * Destruct MacroTile.
         *
         * Splits MacroTile into SubDimensions during translation and
         * lowering.  Should not persist.
         */
        struct DestructMacroTile
        {
            std::string toString() const
            {
                return "DestructTensorTile";
            }
        };

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
        struct Flatten
        {
            std::string toString() const
            {
                return "Flatten";
            }
        };

        /**
         * Forget (drop) dimensions.
         *
         * Used to express a transform where coordinates are "lost"
         * and can't be reconstructed or transformed further.  For
         * example, a scalar VGPR doesn't have a coordinate.
         *
         * Forward and reverse transforms aren't defined.
         */
        struct Forget
        {
            std::string toString() const
            {
                return "Forget";
            }
        };

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
        struct Inherit
        {
            std::string toString() const
            {
                return "Inherit";
            }
        };

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
        struct Join
        {
            std::string toString() const
            {
                return "Join";
            }
        };

        /**
         * Make output (subsequent dims should be tagged as output).
         */
        struct MakeOutput
        {
            std::string toString() const
            {
                return "MakeOutput";
            }
        };

        /**
         * PassThrough (identity).
         *
         * Forward and reverse transforms are the identity.
         */
        struct PassThrough
        {
            std::string toString() const
            {
                return "PassThrough";
            }
        };

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
        struct Split
        {
            std::string toString() const
            {
                return "Split";
            }
        };

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
        struct Tile
        {
            std::string toString() const
            {
                return "Tile";
            }
        };

        /*
         * Helpers
         */

        inline std::string toString(const CoordinateTransformEdge& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }

        inline std::string toString(const DataFlowEdge& x)
        {
            return std::visit([](const auto& a) { return a.toString(); }, x);
        }

        inline std::string toString(const Edge& x)
        {
            return std::visit([](const auto& a) { return toString(a); }, x);
        }

        template <typename T>
        requires(
            std::constructible_from<
                CoordinateTransformEdge,
                T> || std::constructible_from<DataFlowEdge, T>) inline bool isEdge(const Edge& x)
        {
            if constexpr(std::constructible_from<DataFlowEdge, T>)
            {
                if(std::holds_alternative<DataFlowEdge>(x))
                {
                    if(std::holds_alternative<T>(std::get<DataFlowEdge>(x)))
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
