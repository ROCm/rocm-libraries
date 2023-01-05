#pragma once

#include "ControlToCoordinateMapper.hpp"
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{

    template <typename T>
    inline void ControlToCoordinateMapper::connect(int control, int coordinate, int subDimension)
    {
        connect(control, coordinate, std::type_index(typeid(T)), subDimension);
    }

    template <typename T>
    inline void ControlToCoordinateMapper::disconnect(int control, int coordinate, int subDimension)
    {
        disconnect(control, coordinate, std::type_index(typeid(T)), subDimension);
    }

    template <typename T>
    inline int ControlToCoordinateMapper::get(int control, int subDimension) const
    {
        auto key = key_type{control, std::type_index(typeid(T)), subDimension};
        auto it  = m_map.find(key);
        if(it == m_map.end())
            return -1;
        return it->second;
    }

}
