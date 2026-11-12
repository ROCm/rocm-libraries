// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_bench
{

/// The graph files a `--graph-list` names: one path per line, in order, blank lines skipped.
/// A path is taken as written (it may contain spaces); only a line's trailing carriage return,
/// from a list written on Windows, is dropped.
/// @throws std::invalid_argument when the list names no graph.
inline std::vector<std::string> readGraphList(std::istream& list)
{
    std::vector<std::string> paths;
    std::string line;
    while(std::getline(list, line))
    {
        if(!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if(line.find_first_not_of(" \t") != std::string::npos)
        {
            paths.push_back(line);
        }
    }
    if(paths.empty())
    {
        throw std::invalid_argument("--graph-list names no graph");
    }
    return paths;
}

} // namespace hipdnn_bench
