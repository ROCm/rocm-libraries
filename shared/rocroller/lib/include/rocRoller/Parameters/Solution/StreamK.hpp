// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace rocRoller
{
    enum class StreamKMode
    {
        None,
        Standard,
        TwoTile,
        TwoTileDPFirst,
        Count,
    };

    struct StreamKConfig
    {
        StreamKMode mode = StreamKMode::None;

        auto operator<=>(const StreamKConfig&) const = default;

        explicit operator bool() const
        {
            return mode == StreamKMode::Standard or mode == StreamKMode::TwoTile
                   or mode == StreamKMode::TwoTileDPFirst;
        }

        bool operator==(StreamKMode rhs) const
        {
            return mode == rhs;
        }

        StreamKConfig& operator=(StreamKMode newMode)
        {
            mode = newMode;
            return *this;
        }

        bool isTwoTileMode() const
        {
            return mode == StreamKMode::TwoTile or mode == StreamKMode::TwoTileDPFirst;
        }
    };

    std::string   toString(StreamKMode mode);
    std::ostream& operator<<(std::ostream& stream, StreamKMode const& mode);

    std::string   toString(StreamKConfig const& config);
    std::ostream& operator<<(std::ostream& stream, StreamKConfig const& config);
} // namespace rocRoller
