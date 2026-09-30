// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace hipdnn_backend::logging
{

class GraphLogger
{
public:
    /// Logs the graph as JSON, named by the graph's ID. Only a finalized graph carries an ID;
    /// an unidentified or malformed buffer is not logged. Logging a graph whose ID already has
    /// a file in the output directory is a no-op, so re-finalized and replayed graphs are
    /// written once; a graph rebuilt from scratch has a new ID and is written separately.
    /// @param serializedGraph pointer to flatbuffer binary
    /// @param size size of the flatbuffer binary
    static void logGraph(const uint8_t* serializedGraph, size_t size);

    /// Checks if graph logging is enabled by the environment variable HIPDNN_LOG_GRAPH_DIR
    /// being non-empty.
    static bool isEnabled();

    /// Resets the cached output directory so it will be re-read from the environment
    /// on the next call to getOutputDirectory().
    static void resetCache();

private:
    /// Returns the output directory for graph JSON files.
    /// Uses the directory from HIPDNN_LOG_GRAPH_DIR. Returns an empty path if unset.
    static std::filesystem::path getOutputDirectory();
};

} // namespace hipdnn_backend::logging
