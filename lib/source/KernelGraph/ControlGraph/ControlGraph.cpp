#include <set>
#include <string>

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/TagType.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        void ControlGraph::toDOT(std::ostream& ss, std::string prefix) const
        {
            std::map<TagType, std::string> dotNodes;

            // nodes
            for(auto const& kv : m_nodes)
            {
                auto const&        op = kv.second;
                std::ostringstream addr;
                addr << &op;
                auto name = "\"" + prefix
                            + toString(op)
                            //+ "_" + addr.str()
                            + "\"";
                ss << name << "[label=\"" << toString(op) << "\"];" << std::endl;
                dotNodes.emplace(getTag(op), name);
            }

            // edges
            for(auto const& edge : m_edges)
            {
                ss << dotNodes[edge.first.first] << " -> " << dotNodes[edge.first.second]
                   << "[label=\"" << toString(edge.second) << "\"];" << std::endl;
            }
        }

        std::ostream& operator<<(std::ostream& stream, ControlGraph const& graph)
        {
            graph.toDOT(stream, "krn");

            return stream;
        }

        void ControlGraph::addEdge(std::vector<Operation> const& srcs,
                                   std::vector<Operation> const& dsts,
                                   ControlEdge const&            edge)
        {
            for(auto const& src : srcs)
            {
                auto tag = getTag(src);
                if(m_nodes.count(tag) <= 0)
                {
                    recognizeTag(tag);
                    m_nodes.emplace(tag, src);
                }
            }

            for(auto const& dst : dsts)
            {
                auto tag = getTag(dst);
                if(m_nodes.count(tag) <= 0)
                {
                    recognizeTag(tag);
                    m_nodes.emplace(tag, dst);
                }
            }

            for(auto const& src : srcs)
                for(auto const& dst : dsts)
                    m_edges[{getTag(src), getTag(dst)}] = edge;
        }

        void ControlGraph::addEdge(std::vector<CoordinateTransform::Dimension> const& srcs,
                                   std::vector<Operation> const&                      dsts)
        {
            addEdge(srcs, dsts, Sequence());
        }

        void ControlGraph::addEdge(std::vector<CoordinateTransform::Dimension> const& dims,
                                   std::vector<Operation> const&                      dsts,
                                   ControlEdge const&                                 edge)
        {
            std::vector<Operation> srcs;

            for(auto const& dim : dims)
            {
                auto const& dtag = getTag(dim);
                recognizeTag(dtag);
                for(auto const& kv : m_nodes)
                    if(dtag.ctag == kv.first.ctag)
                        srcs.push_back(kv.second);
            }

            addEdge(srcs, dsts, edge);
        }

        Edge ControlGraph::removeEdge(Operation const& src, Operation const& dst)
        {
            auto edgeIter = m_edges.find({getTag(src), getTag(dst)});

            AssertFatal(edgeIter != m_edges.end());

            auto result = *edgeIter;
            m_edges.erase(edgeIter);

            return result;
        }

        void ControlGraph::recognizeTag(TagType tag)
        {
            m_nextTag = std::max(m_nextTag, tag.ctag + 1);
        }

        int ControlGraph::nextTag() const
        {
            return m_nextTag;
        }

        TagType ControlGraph::allocateTag(Operation& oper)
        {
            auto tag = getTag(oper);
            if(tag.ctag < 0)
                tag.ctag = m_nextTag;

            std::visit([tag](auto& op) { op.tag = tag.ctag; }, oper);

            m_nextTag++;

            return tag;
        }

        int ControlGraph::allocateTag()
        {
            return m_nextTag++;
        }

        Operation ControlGraph::getOperation(TagType tag) const
        {
            return m_nodes.at(tag);
        }

        std::vector<Edge> ControlGraph::getOperationEdges() const
        {
            // XXX rename and do topo sort by default
            std::vector<Edge> rv(m_edges.begin(), m_edges.end());
            return rv;
        }

        ControlEdge ControlGraph::getEdge(EdgeKey const& key) const
        {
            AssertFatal(m_edges.count(key) == 1, ShowValue(key));
            return m_edges.at(key);
        }

        Operation ControlGraph::getRootOperation() const
        {
            // Root node should be of type Kernel.
            Operation rv;
            bool      found = false;
            for(auto const& pair : m_nodes)
            {
                if(std::holds_alternative<Kernel>(pair.second))
                {
                    rv    = pair.second;
                    found = true;
                    break;
                }
            }

            AssertFatal(found, "There should be exactly one Kernel node.");

            // Root node should be the only one with no incoming edges.
            std::set<TagType> tags;
            for(auto const& pair : m_nodes)
                tags.insert(pair.first);

            for(auto const& pair : m_edges)
                tags.erase(pair.first.second);

            if(tags.size() != 1 || *tags.begin() != getTag(rv))
            {
                std::ostringstream msg;
                streamJoin(msg, tags, ", ");
                Throw<FatalError>("There should be exactly one root node. [",
                                  msg.str(),
                                  "], ",
                                  rv,
                                  "/",
                                  getTag(rv));
            }

            return rv;
        }

        std::vector<TagType> ControlGraph::getInputTags(TagType const& dst) const
        {
            std::vector<TagType> rv;

            for(auto const& kv : m_edges)
                if(kv.first.second == dst)
                    rv.push_back(kv.first.first);

            return rv;
        }

        std::vector<Operation> ControlGraph::getInputs(TagType const& dst) const
        {
            return getOperations(getInputTags(dst));
        }

        std::vector<TagType> ControlGraph::getInputTags(TagType const& dst, ControlEdge edge) const
        {
            return std::visit(
                [this, &dst](auto const& edge) {
                    using T = std::decay_t<decltype(edge)>;
                    return getInputTags<T>(dst);
                },
                edge);
        }

        std::vector<Operation> ControlGraph::getInputs(TagType const& dst, ControlEdge edge) const
        {
            return std::visit(
                [this, &dst](auto const& edge) {
                    using T = std::decay_t<decltype(edge)>;
                    return getInputs<T>(dst);
                },
                edge);
        }

        std::vector<TagType> ControlGraph::getOutputTags(TagType const& src) const
        {
            std::vector<TagType> rv;
            for(auto const& kv : m_edges)
                if(kv.first.first == src)
                    rv.push_back(kv.first.second);
            return rv;
        }

        std::vector<Operation> ControlGraph::getOutputs(TagType const& src) const
        {
            return getOperations(getOutputTags(src));
        }

        std::vector<TagType> ControlGraph::getOutputTags(TagType const& src, ControlEdge edge) const
        {
            return std::visit(
                [this, &src](auto const& edge) {
                    using T = std::decay_t<decltype(edge)>;
                    return getOutputTags<T>(src);
                },
                edge);
        }

        std::vector<Operation> ControlGraph::getOutputs(TagType const& src, ControlEdge edge) const
        {
            return std::visit(
                [this, &src](auto const& edge) {
                    using T = std::decay_t<decltype(edge)>;
                    return getOutputs<T>(src);
                },
                edge);
        }

        std::vector<Operation> ControlGraph::getOperations() const
        {
            std::vector<Operation> rv;
            for(auto const& kv : m_nodes)
                rv.push_back(kv.second);
            return rv;
        }

        std::vector<Operation> ControlGraph::getOperations(std::vector<TagType> const& tags) const
        {
            std::vector<Operation> rv;
            rv.reserve(tags.size());
            for(auto const& tag : tags)
                rv.push_back(m_nodes.at(tag));
            return rv;
        }

        void ControlGraph::resetOperation(Operation const& op)
        {
            AssertFatal(m_nodes.count(getTag(op)) > 0,
                        "Opertaion not present in graph.",
                        ShowValue(getTag(op).toString()));
            m_nodes.insert_or_assign(getTag(op), op);
        }

        void ControlGraph::removeOperation(TagType const& tag, bool fullyCompletely)
        {
            if(fullyCompletely)
            {
                auto op = getOperation(tag);
                for(auto const& x : getOutputs(tag))
                    removeEdge(op, x);
                for(auto const& x : getInputs(tag))
                    removeEdge(x, op);
            }
            m_nodes.erase(tag);
        }
    }
}
