// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/IUhdAdapter.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

/// Score @p rows, written as one vector per candidate, through IUhdAdapter::scoreBatch, which
/// takes one row-major buffer as UhdKernelHeuristic::rankWith passes it. Every row must have
/// the same width.
inline hipdnn_plugin_sdk::uhd::BatchScores
    scoreRows(const hipdnn_plugin_sdk::uhd::IUhdAdapter& adapter,
              const std::vector<std::vector<double>>& rows)
{
    const size_t width = rows.empty() ? 0 : rows.front().size();
    std::vector<double> values;
    values.reserve(rows.size() * width);
    for(const auto& row : rows)
    {
        if(row.size() != width)
        {
            throw std::invalid_argument("scoreRows: every row must have the same width");
        }
        values.insert(values.end(), row.begin(), row.end());
    }
    return adapter.scoreBatch({values.data(), rows.size(), width});
}

} // namespace hipdnn_test_sdk::utilities

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
