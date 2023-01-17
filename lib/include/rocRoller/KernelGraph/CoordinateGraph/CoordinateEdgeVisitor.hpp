
#pragma once

#include <vector>

#include <rocRoller/Expression_fwd.hpp>
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
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Join const& e)
            {
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));
                auto result = indexes[0] * getStride(srcs[0]);
                for(uint d = 1; d < srcs.size(); ++d)
                    result = result + indexes[d] * getStride(srcs[d]);
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Tile const& e)
            {
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                std::vector<Expression::ExpressionPtr> rv(dsts.size());

                // TODO: Audit/test this for > 2 destinations
                auto input = indexes[0];
                for(size_t i = 1; i < dsts.size(); i++)
                {
                    rv[i - 1] = input / getSize(dsts[i]);
                    input     = input % getSize(dsts[i]);
                }
                rv.back() = input;

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
                return rv;
            }

            std::vector<Expression::ExpressionPtr> operator()(Split const& e)
            {
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                auto result = indexes[0] * getStride(dsts[0]);
                for(uint d = 1; d < dsts.size(); ++d)
                    result = result + indexes[d] * getStride(dsts[d]);
                return {result};
            }

            std::vector<Expression::ExpressionPtr> operator()(Tile const& e)
            {
                auto result = indexes[0];
                for(uint d = 1; d < dsts.size(); ++d)
                    result = result * getSize(dsts[d]) + indexes[d];
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
    }
}
