// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/ops/pooling_bwd.hpp"

namespace ck_tile {

struct PoolBwdKernelTraits
{
    std::string pooling_dim; // "2d" or "3d"
    bool has_overlap;        // true => atomic-add path, false => set path

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << pooling_dim << "_" << (has_overlap ? "overlap" : "nooverlap");
        return oss.str();
    }
};

inline PoolBwdKernelTraits extract_pooling_bwd_traits_from_name(const std::string& name)
{
    PoolBwdKernelTraits traits;
    traits.pooling_dim = (name.find("3d") != std::string::npos) ? "3d" : "2d";
    traits.has_overlap = (name.find("overlap") != std::string::npos) &&
                         (name.find("nooverlap") == std::string::npos);
    return traits;
}

} // namespace ck_tile
