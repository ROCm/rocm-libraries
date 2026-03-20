// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
        auto iter = m_map.find(control);
        if(iter == m_map.end())
            m_map[control].emplace(conn, coordinate);
        else
            iter->second.insert_or_assign(conn, coordinate);

        m_coordToControl[coordinate].insert(control);
    }

    void ControlToCoordinateMapper::connect(int control, int coordinate, NaryArgument arg)
    {
        connect(control, coordinate, Connections::JustNaryArgument{arg});
    }

    void ControlToCoordinateMapper::disconnect(int                         control,
                                               int                         coordinate,
                                               Connections::ConnectionSpec conn)
    {
        auto iter = m_map.find(control);
        if(iter != m_map.end())
        {
            auto coord = iter->second.at(conn);
            iter->second.erase(conn);

            m_coordToControl.at(coord).erase(control);
        }
    }

    std::vector<ControlToCoordinateMapper::Connection>
        ControlToCoordinateMapper::getConnections(int control) const
    {
        std::vector<Connection> rv;
        auto                    iter = m_map.find(control);
        if(iter != m_map.end())
        {
            for(auto const& [conn, coord] : iter->second)
            {
                rv.push_back({control, coord, conn});
            }
        }

        return rv;
    }

    std::vector<ControlToCoordinateMapper::Connection>
        ControlToCoordinateMapper::getConnections() const
    {
        std::vector<Connection> rv;
        rv.reserve(m_map.size());

        for(auto const& [control, conn_coord] : m_map)
            for(auto const& [conn, coord] : conn_coord)
                rv.push_back({control, coord, conn});

        return rv;
    }

    std::vector<ControlToCoordinateMapper::Connection>
        ControlToCoordinateMapper::getCoordinateConnections(int requested_coordinate) const
    {
        std::vector<Connection> rv;
        auto                    iter = m_coordToControl.find(requested_coordinate);
        if(iter != m_coordToControl.end())
        {
            for(auto const& control : iter->second)
            {
                // TODO: consider making m_coordToControl store {control, conn} to
                //       speed up search
                for(auto const& [conn, coord] : m_map.at(control))
                {
                    if(coord == requested_coordinate)
                    {
                        rv.push_back({control, requested_coordinate, conn});
                        break;
                    }
                }
            }
        }

        return rv;
    }

    int ControlToCoordinateMapper::getConnectionSubdimension(int control, int coordinate) const
    {
        auto iter = m_map.find(control);
        AssertFatal(iter != m_map.end());

        for(auto const& [conn, coord] : iter->second)
        {
            if(coord == coordinate)
            {
                if(std::holds_alternative<Connections::TypeAndSubDimension>(conn))
                {
                    auto curConnection = std::get<Connections::TypeAndSubDimension>(conn);
                    return curConnection.subdimension;
                }
                else
                {
                    return -1;
                }
            }
        }
        return -1;
    }

    void ControlToCoordinateMapper::purge(int control)
    {
        auto iter = m_map.find(control);
        if(iter != m_map.end())
        {
            for(auto const& [conn, coord] : m_map.at(control))
            {
                m_coordToControl.at(coord).erase(control);
            }
            m_map.erase(iter);
        }
    }

    void ControlToCoordinateMapper::purgeMappingsTo(int requested_coordinate)
    {
        auto iter = m_coordToControl.find(requested_coordinate);
        if(iter != m_coordToControl.end())
        {
            for(auto const& control : iter->second)
            {
                for(auto const& [conn, coord] : m_map.at(control))
                {
                    if(coord == requested_coordinate)
                    {
                        m_map.at(control).erase(conn);
                        break;
                    }
                }
            }

            m_coordToControl.erase(iter);
        }
    }

    std::string ControlToCoordinateMapper::toDOT(std::string const& coord,
                                                 std::string const& cntrl,
                                                 bool               addLabels) const
    {
        std::stringstream ss;

        // Sort the map keys so that dot output is consistent.  Currently only
        // sorting based on the control index (first value of key tuple) and
        // coordinate index. If we ever output additional information we'll
        // need to take it into account when sorting as well.
        std::vector<std::pair<key_type, int>> key_coords;
        key_coords.reserve(m_map.size()); // just a rough estimate
        for(auto const& [control, conn_coord] : m_map)
        {
            for(auto const& [conn, coordinate] : conn_coord)
            {
                key_coords.emplace_back(std::make_tuple(control, conn), coordinate);
            }
        }
        std::sort(key_coords.begin(), key_coords.end(), [](auto const& a, auto const& b) {
            return (std::get<0>(a.first) < std::get<0>(b.first))
                   || (std::get<0>(a.first) == std::get<0>(b.first) && a.second < b.second);
        });

        for(auto const& [key, coordNode] : key_coords)
        {
            auto const& [ctrlNode, spec] = key;

            auto coordKey = concatenate(coord, coordNode);
            auto ctrlKey  = concatenate(cntrl, ctrlNode);

            auto coordTag = concatenate("to_", cntrl, "_", coordNode, "_", ctrlNode);
            auto ctrlTag  = concatenate("to_", coord, "_", ctrlNode, "_", coordNode);

            auto label = concatenate(ctrlNode, "->", coordNode, ": ", spec);

            ss << '"' << coordKey << "\" -> \"" << coordTag << "\"" << std::endl;
            ss << '"' << coordTag << '"' << "[label=\"" << label << "\", shape=cds]" << std::endl;

            ss << '"' << ctrlKey << "\" -> \"" << ctrlTag << "\"" << std::endl;
            ss << '"' << ctrlTag << '"' << "[label=\"" << label << "\", shape=cds]" << std::endl;
        }
        return ss.str();
    }

    std::vector<int> ControlToCoordinateMapper::getControls() const
    {
        std::set<int> retval;
        for(auto it = m_map.begin(); it != m_map.end(); ++it)
        {
            retval.insert(it->first);
        }
        return std::vector<int>(retval.begin(), retval.end());
    }

    namespace Connections
    {
        std::string name(ConnectionSpec const& cs)
        {
            return std::visit(rocRoller::overloaded{
                                  [](std::monostate const&) { return "none"; },
                                  [](JustNaryArgument const&) { return "NaryArgument"; },
                                  [](TypeAndSubDimension const&) { return "TypeAndSubDimension"; },
                                  [](TypeAndNaryArgument const&) { return "TypeAndNaryArgument"; },
                                  [](UnrollStride const&) { return "UnrollStride"; },
                                  [](BaseOffset const&) { return "BaseOffset"; },
                              },
                              cs);
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

            std::string operator()(TypeAndSubDimension const& ci) const
            {
                return concatenate(ci.id, ": (", ci.subdimension, ")");
            }

            std::string operator()(UnrollStride const& ci) const
            {
                return concatenate(ci.unrollStride, ": (", ci.unrollDimension, ")");
            }

            std::string operator()(BaseOffset const& ci) const
            {
                return concatenate(ci.base, ": (", ci.subdimension, ")");
            }

            std::string operator()(TypeAndNaryArgument const& ci) const
            {
                return concatenate(ci.id, ": (", ci.argument, ")");
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

        NaryArgument getNaryArgument(Connections::ConnectionSpec const& conn)
        {
            auto visitor
                = rocRoller::overloaded{[](JustNaryArgument const& arg) { return arg.argument; },
                                        [](TypeAndNaryArgument const& arg) { return arg.argument; },
                                        [](auto const& other) { return NaryArgument::None; }};

            return std::visit(visitor, conn);
        }
    }

    std::string toString(ControlToCoordinateMapper::Connection const& conn)
    {
        return fmt::format("{{control: {}, coord: {}, {}}}",
                           conn.control,
                           conn.coordinate,
                           toString(conn.connection));
    }

    NaryArgument getNaryArgument(ControlToCoordinateMapper::Connection const& conn)
    {
        return getNaryArgument(conn.connection);
    }

}
