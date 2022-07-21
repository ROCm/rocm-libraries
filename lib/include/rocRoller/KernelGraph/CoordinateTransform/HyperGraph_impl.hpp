#pragma once

namespace rocRoller
{
    namespace KernelGraph::CoordinateTransform
    {
        /*
         * Traverse
         */

        using ExpressionMapType = std::map<TagType, Expression::ExpressionPtr>;

        inline void update(ExpressionMapType&                            exprs,
                           std::vector<Expression::ExpressionPtr> const& es,
                           std::vector<Dimension> const&                 ds)
        {
            AssertFatal(es.size() == ds.size(), ShowValue(es), ShowValue(ds));
            for(uint i = 0; i < es.size(); ++i)
                exprs.emplace(getTag(ds[i]), es[i]);
        }

        //
        // Traverse path from srcs to dsts, building coordinate
        // transform expressions along the way.
        //
        // Each edge in the graph is a coordinate transform.  Before
        // evaluating a coordinate transform (edge), we build argument
        // arrays based on the edges src nodes.
        //
        template <typename Visitor>
        inline std::vector<Expression::ExpressionPtr>
            HyperGraph::traverse(std::vector<Expression::ExpressionPtr> const& indexes,
                                 std::vector<Dimension> const&                 srcs,
                                 std::vector<Dimension> const&                 dsts,
                                 bool                                          forward,
                                 Visitor&                                      visitor,
                                 Expression::ExpressionTransducer              transducer) const
        {
            ExpressionMapType exprs;

            auto edges = path(tags(srcs), tags(dsts), EdgeType::CoordinateTransform, forward);
            AssertRecoverable(!edges.empty(), "Path not found");

            update(exprs, indexes, forward ? srcs : dsts);
            for(auto const& e : edges)
            {
                // build arguments for edge
                std::vector<Expression::ExpressionPtr> einds;
                std::vector<Dimension>                 esrcs, edsts;
                for(auto const& tag : e.stags)
                {
                    if(forward)
                    {
                        einds.push_back(exprs[tag]);
                    }
                    esrcs.emplace_back(m_nodes.at(tag));
                }
                for(auto const& tag : e.dtags)
                {
                    if(!forward)
                    {
                        einds.push_back(exprs[tag]);
                    }
                    edsts.emplace_back(m_nodes.at(tag));
                }

                // get edge and evaluate
                auto const& edge = m_edges.at(e);
                visitor.setLocation(einds, esrcs, edsts);
                update(exprs, visitor(edge), forward ? edsts : esrcs);
            }

            std::vector<Expression::ExpressionPtr> r;
            for(auto& tag : tags(forward ? dsts : srcs))
                r.push_back(transducer ? transducer(exprs.at(tag)) : exprs.at(tag));
            return r;
        }
    }
}
