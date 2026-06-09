// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <string>

namespace ck_tile {
namespace dispatcher {

struct HstuKernelKey
{
    std::string name;
    std::string data_type;
    bool use_causal   = true;
    bool use_softmax  = false;
    bool has_bias     = false;
    int max_k         = 128;
    int mtile         = 64;
    bool use_splitkv  = false;

    [[nodiscard]] std::string encode_identifier() const
    {
        std::ostringstream oss;
        oss << "hstu_" << data_type << "_causal" << (use_causal ? 1 : 0) << "_softmax"
            << (use_softmax ? 1 : 0) << "_bias" << (has_bias ? 1 : 0) << "_maxk" << max_k
            << "_mtile" << mtile << "_splitkv" << (use_splitkv ? 1 : 0);
        if(!name.empty())
            oss << "_" << name;
        return oss.str();
    }
};

} // namespace dispatcher
} // namespace ck_tile
