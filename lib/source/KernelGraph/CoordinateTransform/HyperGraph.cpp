#include <set>
#include <string>

#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/EdgeVisitor.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/HyperGraph.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph::CoordinateTransform
    {
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /*
         * Edge visitors (coordinate transforms).
         */

        struct ForwardEdgeVisitor : public BaseEdgeVisitor
        {
            std::vector<ExpressionPtr> operator()(Flatten const& e)
            {
                auto result = indexes[0];
                for(uint d = 1; d < srcs.size(); ++d)
                    result = result * getSize(srcs[d]) + indexes[d];
                return {result};
            }

            std::vector<ExpressionPtr> operator()(Join const& e)
            {
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));
                auto result = indexes[0] * getStride(srcs[0]);
                for(uint d = 1; d < srcs.size(); ++d)
                    result = result + indexes[d] * getStride(srcs[d]);
                return {result};
            }

            std::vector<ExpressionPtr> operator()(Tile const& e)
            {
                std::vector<ExpressionPtr> rv(dsts.size());

                auto input = indexes[0] / getStride(srcs[0]);

                for(int i = 1; i < dsts.size(); i++)
                {
                    rv[i - 1] = input / getSize(dsts[i]);
                    input     = input % getSize(dsts[i]);
                }
                rv.back() = input;

                return rv;
            }

            template <CTUndefinedEdge T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Edge transform not defined.");
            }

            template <CTEdgePassthrough T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                return std::visit(*this, e);
            }

            template <typename T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                return indexes;
            }
        };

        struct ReverseEdgeVisitor : public BaseEdgeVisitor
        {
            std::vector<ExpressionPtr> operator()(Flatten const& e)
            {
                AssertFatal(dsts.size() == 1, ShowValue(dsts.size()));
                if(srcs.size() == 1)
                    return indexes;

                std::vector<ExpressionPtr> rv(srcs.size());

                auto input = indexes[0];

                for(int i = srcs.size() - 1; i >= 0; i--)
                {
                    auto mysrc = srcs[i];
                    auto size  = getSize(mysrc);
                    rv[i]      = input % size;
                    input      = input / size;
                }
                return rv;
            }

            std::vector<ExpressionPtr> operator()(Split const& e)
            {
                AssertFatal(srcs.size() == 1, ShowValue(srcs.size()));
                auto result = indexes[0] * getStride(dsts[0]);
                for(uint d = 1; d < dsts.size(); ++d)
                    result = result + indexes[d] * getStride(dsts[d]);
                return {result};
            }

            std::vector<ExpressionPtr> operator()(Tile const& e)
            {
                auto result = indexes[0];
                for(uint d = 1; d < dsts.size(); ++d)
                    result = result * getSize(dsts[d]) + indexes[d];
                return {result};
            }

            template <CTUndefinedEdge T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                Throw<FatalError>("Edge transform not defined.");
            }

            template <CTEdgePassthrough T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                return std::visit(*this, e);
            }

            template <typename T>
            std::vector<ExpressionPtr> operator()(T const& e)
            {
                return indexes;
            }
        };

        /*
         * HyperGraph methods
         */

        std::string HyperGraph::toString(bool ordered) const
        {
            return toDOT(ordered);
        }

        std::string HyperGraph::toDOT(bool ordered) const
        {
            std::stringstream ss;

            std::vector<EdgeKeyType> edges;
            if(ordered)
            {
                edges = topographicalSort(EdgeType::Any);
            }
            else
            {
                for(auto const& kv : m_edges)
                    edges.push_back(kv.first);
            }

            for(auto const& edge : edges)
            {
                ss << " { ";
                auto stags = edge.stags;
                std::sort(stags.begin(), stags.end());
                for(int i = 0; i < stags.size() - 1; ++i)
                {
                    ss << "\"" + CoordinateTransform::toString(m_nodes.at(stags[i])) + "\", ";
                }
                ss << "\"" + CoordinateTransform::toString(m_nodes.at(stags.back())) + "\"";
                ss << " } -> { ";
                auto dtags = edge.dtags;
                std::sort(dtags.begin(), dtags.end());
                for(int i = 0; i < dtags.size() - 1; ++i)
                {
                    ss << "\"" + CoordinateTransform::toString(m_nodes.at(dtags[i])) + "\", ";
                }
                ss << "\"" + CoordinateTransform::toString(m_nodes.at(dtags.back())) + "\"";
                ss << " } [";
                if(edge.type == EdgeType::CoordinateTransform)
                {
                    ss << "color=blue ";
                }
                else if(edge.type == EdgeType::DataFlow)
                {
                    ss << "color=red ";
                }
                ss << "label=\"" + CoordinateTransform::toString(m_edges.at(edge)) + "\"";
                ss << "]" << std::endl;
            }
            return ss.str();
        }

        std::vector<TagType> HyperGraph::addDimensions(std::vector<Dimension>& dims)
        {
            std::vector<TagType> tags;
            tags.reserve(dims.size());

            for(auto& d : dims)
            {
                tags.push_back(addDimension(d));
            }

            return tags;
        }

        TagType HyperGraph::addDimension(Dimension& dim)
        {
            auto tag = getTag(dim);
            if(tag.ctag < 0)
                tag = allocateTag(dim);
            else
                m_nextTag = std::max(tag.ctag + 1, m_nextTag);

            m_nodes.emplace(tag, dim);

            return tag;
        }

        int HyperGraph::nextTag() const
        {
            return m_nextTag;
        }

        TagType HyperGraph::allocateTag(Dimension& dim)
        {
            TagType tag = getTag(dim);

            AssertFatal(tag.ctag < 0, "Tag already allocated: ", ShowValue(dim));

            tag.ctag = m_nextTag;
            m_nextTag++;

            std::visit([tag](auto& d) { d.tag = tag.ctag; }, dim);

            return tag;
        }

        int HyperGraph::allocateTag()
        {
            return m_nextTag++;
        }

        void HyperGraph::resetDimension(Dimension dim)
        {
            AssertFatal(m_nodes.count(getTag(dim)) > 0,
                        "Dimension not present in graph.",
                        ShowValue(getTag(dim)));
            m_nodes.insert_or_assign(getTag(dim), dim);
        }

        Dimension HyperGraph::getDimension(TagType tag) const
        {
            AssertRecoverable(m_nodes.count(tag) > 0, "No dimension with tag: " + tag.toString());
            return m_nodes.at(tag);
        }

        std::vector<Dimension> HyperGraph::getDimensions(std::vector<TagType> tags) const
        {
            std::vector<Dimension> result;
            for(auto const& tag : tags)
                result.push_back(m_nodes.at(tag));
            return result;
        }

        std::vector<Dimension> HyperGraph::getDimensions() const
        {
            std::vector<Dimension> result;
            for(auto const& kv : m_nodes)
                result.push_back(kv.second);
            return result;
        }

        std::vector<Dimension> HyperGraph::getOutputs(TagType tag, EdgeType edgeType) const
        {
            std::set<TagType> otags;

            for(auto const& pair : m_edges)
            {
                if(edgeType == EdgeType::Any || pair.first.type == edgeType)
                {
                    if(std::find(pair.first.stags.begin(), pair.first.stags.end(), tag)
                       != pair.first.stags.end())
                    {
                        otags.insert(pair.first.dtags.begin(), pair.first.dtags.end());
                    }
                }
            }

            std::vector<TagType> tags(otags.begin(), otags.end());

            return getDimensions(std::move(tags));
        }

        std::vector<Dimension> HyperGraph::getLinearDimensions(std::unordered_set<int> ndtags) const
        {
            std::vector<Dimension> result;
            for(auto const& ndtag : ndtags)
            {
                // TODO make this more robust
                auto linear = getTag(Dimension(Linear(ndtag)));
                if(m_nodes.count(linear) > 0)
                {
                    result.push_back(m_nodes.at(linear));
                    continue;
                }
                auto vgpr = getTag(Dimension(VGPR(ndtag)));
                if(m_nodes.count(vgpr) > 0)
                {
                    result.push_back(m_nodes.at(vgpr));
                    continue;
                }
                auto tiled = getTag(Dimension(MacroTile(ndtag, 2)));
                if(m_nodes.count(tiled) > 0)
                {
                    result.push_back(m_nodes.at(tiled));
                    continue;
                }
                return {};
            }
            return result;
        }

        void HyperGraph::addEdge(std::vector<Dimension>  srcs,
                                 std::vector<Dimension>  dsts,
                                 CoordinateTransformEdge edge)
        {
            // TODO: Add strict mode
            AssertFatal(!srcs.empty(), "Can not add edge with no tail.");
            AssertFatal(!dsts.empty(), "Can not add edge with no head.");
            this->addDimensions(srcs);
            this->addDimensions(dsts);
            this->m_edges.emplace(
                EdgeKeyType{tags(srcs), tags(dsts), EdgeType::CoordinateTransform}, edge);
        }

        void HyperGraph::addEdge(std::vector<Dimension> srcs,
                                 std::vector<Dimension> dsts,
                                 DataFlow               edge)
        {
            // TODO: Add strict mode
            AssertFatal(!srcs.empty(), "Can not add edge with no tail.");
            AssertFatal(!dsts.empty(), "Can not add edge with no head.");
            this->addDimensions(srcs);
            this->addDimensions(dsts);
            this->m_edges.emplace(EdgeKeyType{tags(srcs), tags(dsts), EdgeType::DataFlow}, edge);
        }

        std::vector<EdgeKeyType> HyperGraph::getEdges() const
        {
            std::vector<EdgeKeyType> r;
            for(auto const& kv : m_edges)
                r.push_back(kv.first);
            return r;
        }

        Edge HyperGraph::getEdge(EdgeKeyType e) const
        {
            return m_edges.at(e);
        }

        void HyperGraph::removeEdge(EdgeKeyType ekey)
        {
            m_edges.erase(ekey);
        }

        std::vector<ExpressionPtr> HyperGraph::forward(std::vector<ExpressionPtr> const& sdims,
                                                       std::vector<Dimension> const&     srcs,
                                                       std::vector<Dimension> const&     dsts,
                                                       ExpressionTransducer transducer) const
        {
            auto visitor = ForwardEdgeVisitor();
            return this->traverse(sdims, srcs, dsts, true, visitor, transducer);
        }

        std::vector<ExpressionPtr> HyperGraph::reverse(std::vector<ExpressionPtr> const& sdims,
                                                       std::vector<Dimension> const&     srcs,
                                                       std::vector<Dimension> const&     dsts,
                                                       ExpressionTransducer transducer) const
        {
            auto visitor = ReverseEdgeVisitor();
            return this->traverse(sdims, srcs, dsts, false, visitor, transducer);
        }

        std::vector<Dimension> HyperGraph::bottom(EdgeType type) const
        {
            std::set<TagType> pointed_to;

            for(auto const& kv : m_edges)
            {
                if(!(type == EdgeType::Any || kv.first.type == type))
                    continue;
                for(auto const& dtag : kv.first.dtags)
                    pointed_to.insert(dtag);
            }

            std::vector<Dimension> results;
            for(auto const& kv : m_nodes)
                if(pointed_to.count(kv.first) <= 0)
                    results.push_back(kv.second);
            return results;
        }

        std::vector<Dimension> HyperGraph::top(EdgeType type) const
        {
            std::set<TagType> pointed_from;

            for(auto const& kv : m_edges)
            {
                if(!(type == EdgeType::Any || kv.first.type == type))
                    continue;
                for(auto const& dtag : kv.first.stags)
                    pointed_from.insert(dtag);
            }

            std::vector<Dimension> results;
            for(auto const& kv : m_nodes)
                if(pointed_from.count(kv.first) <= 0)
                    results.push_back(kv.second);
            return results;
        }

        std::vector<EdgeKeyType> HyperGraph::topographicalSort(EdgeType type) const
        {
            return path(tags(bottom(type)), tags(top(type)), type);
        }

        inline bool have_all(std::map<TagType, int> const& visitedCount,
                             std::map<TagType, int> const& incidentCount,
                             std::vector<TagType> const&   tags)
        {
            for(auto& tag : tags)
            {
                if(visitedCount.count(tag) <= 0)
                    return false;
                if(incidentCount.count(tag) <= 0)
                    continue;
                if(visitedCount.at(tag) < incidentCount.at(tag))
                    return false;
            }
            return true;
        }

        inline std::optional<EdgeKeyType> front(std::map<TagType, int> const& visitedCount,
                                                std::map<TagType, int> const& incidentCount,
                                                std::set<EdgeKeyType> const&  visitedEdges,
                                                EdgeMapType const&            edges,
                                                EdgeType                      type,
                                                bool                          forward)
        {
            for(auto const& kv : edges)
            {
                if(type != EdgeType::Any && kv.first.type != type)
                    continue;
                if(visitedEdges.count(kv.first) > 0)
                    continue;

                auto stags = forward ? kv.first.stags : kv.first.dtags;
                auto dtags = forward ? kv.first.dtags : kv.first.stags;
                if(have_all(visitedCount, incidentCount, stags)
                   && !have_all(visitedCount, incidentCount, dtags))
                    return kv.first;
            }
            return {};
        }

        //
        // To compute a path
        //
        // First:
        //   - For each node in srcs; set 'visited' count to zero
        //   - For each edge in graph; mark as not traversed
        //   - For each node in graph; count number of incident edges
        //
        // Then, iterate below until all nodes in dsts have been visited:
        //   - Pick an edge that hasn't been traversed.  If we are
        //     tracing data flow, we also require that all the edge's
        //     src nodes are fully visited; that is, all src nodes
        //     should have a visited count greater or equal to their
        //     indicent count.  If we are only performing a coordinate
        //     transform, having an index for each src is sufficent.
        //   - Mark the edge as traversed.
        //   - Increment all it's dst nodes visited counts.
        //
        // While we are doing this, at each node we store the edges
        // that had to be traversed in order to reach it.
        //
        std::vector<EdgeKeyType> HyperGraph::path(std::vector<TagType> const& srcs,
                                                  std::vector<TagType> const& dsts,
                                                  EdgeType                    type,
                                                  bool                        forward) const
        {
            std::map<TagType, std::vector<EdgeKeyType>> paths;
            std::map<TagType, int>                      visitedCount, incidentCount;
            std::set<EdgeKeyType>                       visitedEdges;

            for(auto const& tag : forward ? srcs : dsts)
                visitedCount[tag]++;

            if(type != EdgeType::CoordinateTransform)
            {
                // when finding paths for, eg, dataflow we can't
                // proceed until all dependencies are met
                for(auto const& kv : m_edges)
                    if(type == EdgeType::Any || kv.first.type == type)
                        for(auto const& tag : kv.first.dtags)
                            incidentCount[tag]++;
            }
            else
            {
                // when finding paths for cooridnate transforms, we
                // can proceed once we get an index; we don't have to
                // track data dependencies
                for(auto const& kv : m_edges)
                    if(type == EdgeType::Any || kv.first.type == type)
                        for(auto const& tag : kv.first.stags)
                            incidentCount[tag] = 1;
            }

            // iterate until we have reached all dsts
            while(!have_all(visitedCount, incidentCount, forward ? dsts : srcs))
            {
                // find an edge that's ready
                auto const& found_edge
                    = front(visitedCount, incidentCount, visitedEdges, m_edges, type, forward);
                if(!found_edge)
                {
                    std::stringstream ss;
                    ss << "Unable to find available edge:" << std::endl;
                    ss << "  - forward: " << forward << std::endl;
                    for(auto const& tag : srcs)
                        ss << "  - src: " << CoordinateTransform::toString(m_nodes.at(tag))
                           << std::endl;
                    for(auto const& tag : dsts)
                        ss << "  - dst: " << CoordinateTransform::toString(m_nodes.at(tag))
                           << std::endl;
                    for(auto const& kv : visitedCount)
                        ss << "  - visited count "
                           << CoordinateTransform::toString(m_nodes.at(kv.first)) << ": "
                           << kv.second << std::endl;
                    for(auto const& kv : incidentCount)
                        ss << "  - incident count "
                           << CoordinateTransform::toString(m_nodes.at(kv.first)) << ": "
                           << kv.second << std::endl;
                    ss << "Graph: " << std::endl << toDOT() << std::endl;
                    {
                        std::ofstream badGraph("badGraph.dot");
                        badGraph << toDOT() << std::endl;
                    }
                    AssertRecoverable(found_edge, ss.str());
                }
                auto const& edge = *found_edge;

                // build the new path by combining all src node paths
                std::vector<EdgeKeyType> path;
                for(auto tags : {edge.stags, edge.dtags})
                {
                    std::sort(tags.begin(), tags.end());
                    for(auto const& tag : tags)
                    {
                        if(paths.count(tag) > 0)
                        {
                            for(auto const& e : paths.at(tag))
                            {
                                if(find(path.cbegin(), path.cend(), e) == path.cend())
                                    path.push_back(e);
                            }
                        }
                    }
                }
                path.push_back(edge);

                // each edge dst node gets the new path
                for(auto const& tag : forward ? edge.dtags : edge.stags)
                {
                    if(paths.count(tag) > 0)
                        paths.erase(tag);
                    paths.emplace(tag, path);
                }

                // mark this edge as visited, and bump dst visited counts
                visitedEdges.insert(edge);
                for(auto const& tag : forward ? edge.dtags : edge.stags)
                    visitedCount[tag]++;
            }

            // combine paths required to reach all dsts into a single path
            std::vector<EdgeKeyType> path;
            for(auto const& tag : forward ? dsts : srcs)
            {
                if(paths.count(tag) > 0)
                {
                    for(auto const& e : paths.at(tag))
                    {
                        if(find(path.cbegin(), path.cend(), e) == path.cend())
                            path.push_back(e);
                    }
                }
            }

            return path;
        }
    }
}
