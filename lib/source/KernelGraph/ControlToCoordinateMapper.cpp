
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>

namespace rocRoller::KernelGraph
{

    void ControlToCoordinateMapper::connect(int             control,
                                            int             coordinate,
                                            std::type_index tindex,
                                            int             subDimension)
    {
        auto key = key_type{control, tindex, subDimension};
        m_map.emplace(key, coordinate);
    }

    std::vector<ControlToCoordinateMapper::Connection>
        ControlToCoordinateMapper::getConnections(int control) const
    {
        std::vector<Connection> rv;
        for(auto const& kv : m_map)
        {
            if(std::get<0>(kv.first) == control)
            {
                rv.push_back({std::get<0>(kv.first),
                              std::get<1>(kv.first),
                              std::get<2>(kv.first),
                              kv.second});
            }
        }
        return rv;
    }

    std::string ControlToCoordinateMapper::toDOT(std::string coord, std::string cntrl) const
    {
        std::stringstream ss;
        for(auto kv : m_map)
        {
            ss << "\"" << coord << kv.second << "\" -> \"" << cntrl << std::get<0>(kv.first)
               << "\" [style=dotted,weight=0,arrowsize=0]\n";
        }
        return ss.str();
    }
}
