// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Parameters/Solution/StreamK.hpp>

#include <string>

namespace rocRoller
{
    std::string toString(StreamKMode mode)
    {
        switch(mode)
        {
        case StreamKMode::None:
            return "None";
        case StreamKMode::Standard:
            return "Standard";
        case StreamKMode::TwoTile:
            return "TwoTile";
        case StreamKMode::TwoTileDPFirst:
            return "TwoTileDPFirst";
        default:
            break;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, StreamKMode const& mode)
    {
        return stream << toString(mode);
    }

    std::string toString(StreamKConfig const& config)
    {
        return "StreamK(" + toString(config.mode) + ")";
    }

    std::ostream& operator<<(std::ostream& stream, StreamKConfig const& config)
    {
        return stream << toString(config);
    }
} // namespace rocRoller
