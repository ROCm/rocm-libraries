#pragma once

#include "ControlToCoordinateMapper.hpp"

namespace rocRoller::KernelGraph
{

    template <typename T>
    inline void ControlToCoordinateMapper::connect(int control, int coordinate, int subDimension)
    {
        connect(control, coordinate, std::type_index(typeid(T)), subDimension);
    }

    template <typename T>
    inline int ControlToCoordinateMapper::get(int control, int subDimension) const
    {
        auto key = key_type{control, std::type_index(typeid(T)), subDimension};
        return m_map.at(key);
    }

}
