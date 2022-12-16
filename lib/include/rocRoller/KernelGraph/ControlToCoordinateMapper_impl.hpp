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
        AssertFatal(
            m_map.count(key) > 0, "Connection from operation not found.", ShowValue(control));
        return m_map.at(key);
    }

}
