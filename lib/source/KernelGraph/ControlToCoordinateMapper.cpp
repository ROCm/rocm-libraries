
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>

namespace rocRoller::KernelGraph
{

    void ControlToCoordinateMapper::connect(ControlToCoordinateMapper::Connection const& connection)
    {
        connect(connection.control, connection.coordinate, connection.connection);
    }

    void ControlToCoordinateMapper::connect(int                         control,
                                            int                         coordinate,
                                            Connections::ConnectionSpec conn)
    {
        auto key = key_type{control, conn};
        m_map.insert_or_assign(key, coordinate);
    }

    void ControlToCoordinateMapper::connect(int control, int coordinate, NaryArgument arg)
    {
        connect(control, coordinate, Connections::JustNaryArgument{arg});
    }

    void ControlToCoordinateMapper::disconnect(int                         control,
                                               int                         coordinate,
                                               Connections::ConnectionSpec conn)
    {
        auto key = key_type{control, conn};
        m_map.erase(key);
    }

    std::vector<ControlToCoordinateMapper::Connection>
        ControlToCoordinateMapper::getConnections(int control) const
    {
        std::vector<Connection> rv;
        for(auto const& kv : m_map)
        {
            if(std::get<0>(kv.first) == control)
            {
                rv.push_back({std::get<0>(kv.first), kv.second, std::get<1>(kv.first)});
            }
        }
        return rv;
    }

    std::vector<ControlToCoordinateMapper::Connection>
        ControlToCoordinateMapper::getConnections() const
    {
        std::vector<Connection> rv;
        rv.reserve(m_map.size());

        for(auto const& kv : m_map)
        {
            //cppcheck-suppress useStlAlgorithm
            rv.push_back({std::get<0>(kv.first), kv.second, std::get<1>(kv.first)});
        }

        return rv;
    }

    std::vector<ControlToCoordinateMapper::Connection>
        ControlToCoordinateMapper::getCoordinateConnections(int coordinate) const
    {
        std::vector<Connection> rv;
        for(auto const& kv : m_map)
        {
            if(kv.second == coordinate)
            {
                rv.push_back({std::get<0>(kv.first), kv.second, std::get<1>(kv.first)});
            }
        }
        return rv;
    }

    void ControlToCoordinateMapper::purge(int control)
    {
        std::vector<ControlToCoordinateMapper::key_type> purge;
        for(auto const& kv : m_map)
        {
            if(std::get<0>(kv.first) == control)
            {
                purge.push_back(kv.first);
            }
        }
        for(auto const& k : purge)
        {
            m_map.erase(k);
        }
    }

    void ControlToCoordinateMapper::purgeMappingsTo(int coordinate)
    {
        std::vector<ControlToCoordinateMapper::key_type> toPurge;
        for(auto const& kv : m_map)
        {
            if(kv.second == coordinate)
            {
                toPurge.push_back(kv.first);
            }
        }
        for(auto const& k : toPurge)
        {
            m_map.erase(k);
        }
    }

    std::string ControlToCoordinateMapper::toDOT(std::string const& coord,
                                                 std::string const& cntrl,
                                                 bool               addLabels) const
    {
        std::stringstream ss;

        // Sort the map keys so that dot output is consistent.
        // Currently only sorting based on the control index (first value of key tuple) and
        // coordinate index (the value in the map). If we ever output additional information
        // we'll need to take it into account when sorting as well.
        std::vector<key_type> keys;
        for(auto kv : m_map)
        {
            keys.insert(std::upper_bound(keys.begin(),
                                         keys.end(),
                                         kv.first,
                                         [&](key_type a, key_type b) {
                                             return (std::get<0>(a) < std::get<0>(b))
                                                    || (std::get<0>(a) == std::get<0>(b)
                                                        && m_map.at(a) < m_map.at(b));
                                         }),
                        kv.first);
        }

        for(auto key : keys)
        {
            ss << "\"" << coord << m_map.at(key) << "\" -> \"" << cntrl << std::get<0>(key)
               << "\" [style=dotted,weight=0,arrowsize=0";
            if(addLabels)
                ss << ",label=\"" << std::get<1>(key) << "\"";
            ss << "]\n";
        }
        return ss.str();
    }

    std::vector<int> ControlToCoordinateMapper::getControls() const
    {
        std::set<int> retval;
        for(auto it = m_map.begin(); it != m_map.end(); ++it)
        {
            retval.insert(std::get<0>(it->first));
        }
        return std::vector<int>(retval.begin(), retval.end());
    }

    namespace Connections
    {
        std::string name(ConnectionSpec const& cs)
        {
            return std::visit(
                rocRoller::overloaded{
                    [](std::monostate const&) { return "none"; },
                    [](JustNaryArgument const&) { return "NaryArgument"; },
                    [](ComputeIndex const&) { return "ComputeIndex"; },
                    [](TypeAndSubDimension const&) { return "TypeAndSubDimension"; },
                    [](TypeAndNaryArgument const&) { return "TypeAndNaryArgument"; },
                    [](LDSTypeAndSubDimension const&) { return "LDSTypeAndSubDimension"; }},
                cs);
        }

        std::string toString(ComputeIndexArgument cia)
        {
            switch(cia)
            {
            case ComputeIndexArgument::TARGET:
                return "TARGET";
            case ComputeIndexArgument::INCREMENT:
                return "INCREMENT";
            case ComputeIndexArgument::BASE:
                return "BASE";
            case ComputeIndexArgument::OFFSET:
                return "OFFSET";
            case ComputeIndexArgument::STRIDE:
                return "STRIDE";
            case ComputeIndexArgument::ZERO:
                return "ZERO";
            case ComputeIndexArgument::BUFFER:
                return "BUFFER";
            default:
                return "Invalid";
            }
        }

        std::string toString(LDSLoadStore ld)
        {
            switch(ld)
            {
            case LDSLoadStore::LOAD_FROM_GLOBAL:
                return "LOAD_FROM_GLOBAL";
            case LDSLoadStore::STORE_INTO_LDS:
                return "STORE_INTO_LDS";
            case LDSLoadStore::LOAD_FROM_LDS:
                return "LOAD_FROM_LDS";
            case LDSLoadStore::STORE_INTO_GLOBAL:
                return "STORE_INTO_GLOBAL";
            default:
                return "Invalid";
            }
        }

        struct CSToStringVisitor
        {
            std::string operator()(std::monostate const&) const
            {
                return "EMPTY";
            }

            std::string operator()(JustNaryArgument const& n) const
            {
                return toString(n.argument);
            }

            std::string operator()(ComputeIndex const& ci) const
            {
                return concatenate(ci.argument, ": (", ci.index, ")");
            }

            std::string operator()(TypeAndSubDimension const& ci) const
            {
                return concatenate(ci.id, ": (", ci.subdimension, ")");
            }

            std::string operator()(TypeAndNaryArgument const& ci) const
            {
                return concatenate(ci.id, ": (", ci.argument, ")");
            }

            std::string operator()(LDSTypeAndSubDimension const& ci) const
            {
                return concatenate(
                    "LDS: ", ci.id, ": (", ci.subdimension, ", ", toString(ci.direction), ")");
            }
        };

        std::string toString(ConnectionSpec const& cs)
        {
            return std::visit(CSToStringVisitor(), cs);
        }

        std::ostream& operator<<(std::ostream& stream, ConnectionSpec const& cs)
        {
            return stream << toString(cs);
        }
    }
}
