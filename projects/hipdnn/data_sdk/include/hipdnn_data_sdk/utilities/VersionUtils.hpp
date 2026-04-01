// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace hipdnn_data_sdk::utilities
{
struct Version
{
    int major, minor, patch;

    Version()
        : major(0)
        , minor(0)
        , patch(0)
    {
    }

    Version(int inMajor, int inMinor, int inPatch)
        : major(inMajor)
        , minor(inMinor)
        , patch(inPatch)
    {
    }

    Version(const std::string& versionStr)
    {
        char dot1;
        char dot2;
        std::istringstream iss(versionStr);
        if(!(iss >> major >> dot1 >> minor >> dot2 >> patch) || dot1 != '.' || dot2 != '.')
        {
            throw std::invalid_argument("Version string does not match required format. String = "
                                        + std::string(versionStr)
                                        + " Required format: \"MAJOR.MINOR.PATCH\"");
        }
    }

    Version(std::string_view versionStr)
        : Version(std::string(versionStr))
    {
    }

    std::string str() const
    {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }
};

inline bool operator==(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           == std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator!=(const Version& lhs, const Version& rhs)
{
    return !(lhs == rhs);
}

inline bool operator<(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           < std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator>(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           > std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator<=(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           <= std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}

inline bool operator>=(const Version& lhs, const Version& rhs)
{
    return std::make_tuple(lhs.major, lhs.minor, lhs.patch)
           >= std::make_tuple(rhs.major, rhs.minor, rhs.patch);
}
}
