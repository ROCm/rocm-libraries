
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>

namespace rocRoller::KernelGraph
{

    void ControlToCoordinateMapper::connect(int control, int coordinate, ConnectionSpec conn)
    {
        auto key = key_type{control, conn};
        m_map.insert_or_assign(key, coordinate);
    }

    void ControlToCoordinateMapper::disconnect(int control, int coordinate, ConnectionSpec conn)
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

    std::string ControlToCoordinateMapper::toDOT(std::string const& coord,
                                                 std::string const& cntrl) const
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
               << "\" [style=dotted,weight=0,arrowsize=0]\n";
        }
        return ss.str();
    }
}
