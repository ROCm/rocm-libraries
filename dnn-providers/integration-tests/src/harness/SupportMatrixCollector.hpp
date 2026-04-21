// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <fstream>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_integration_tests
{

struct GraphSupportRecord
{
    std::string graphName;
    std::string graphDescription;
    std::string testName;
    std::set<std::string> supportingEngines;
    std::string note;
};

// Singleton that collects graph-support information during test execution
// and generates a markdown support matrix when requested.
class SupportMatrixCollector
{
public:
    static SupportMatrixCollector& get()
    {
        static SupportMatrixCollector s_instance;
        return s_instance;
    }

    SupportMatrixCollector(const SupportMatrixCollector&) = delete;
    SupportMatrixCollector& operator=(const SupportMatrixCollector&) = delete;
    SupportMatrixCollector(SupportMatrixCollector&&) = delete;
    SupportMatrixCollector& operator=(SupportMatrixCollector&&) = delete;

    void setEnabled(bool enabled)
    {
        _enabled = enabled;
    }

    bool isEnabled() const
    {
        return _enabled;
    }

    void setOutputPath(std::string path)
    {
        _outputPath = std::move(path);
    }

    const std::string& getOutputPath() const
    {
        return _outputPath;
    }

    // Record support information for a graph.
    // Thread-safe: protected by mutex for parallel GTest execution.
    void recordGraphSupport(const std::string& graphName,
                            const std::string& graphDescription,
                            const std::string& testName,
                            const std::vector<int64_t>& supportingEngineIds,
                            const std::string& note = {})
    {
        if(!_enabled)
        {
            return;
        }

        std::set<std::string> engineNames;
        for(auto id : supportingEngineIds)
        {
            try
            {
                engineNames.emplace(hipdnn_data_sdk::utilities::getEngineNameFromId(id));
            }
            catch(const std::out_of_range&)
            {
                engineNames.emplace("Unknown(" + std::to_string(id) + ")");
            }
        }

        std::lock_guard<std::mutex> lock(_mutex);
        _records.push_back({graphName, graphDescription, testName, std::move(engineNames), note});
    }

    const std::vector<GraphSupportRecord>& getRecords() const
    {
        return _records;
    }

    // Generate the markdown output and write to file.
    // allEngineNames: the engine columns to include in the table.
    void writeMarkdown(const std::vector<std::string>& allEngineNames) const
    {
        // Group records by (graphDescription, note), union the engine support sets.
        // Use ordered map so output is deterministic.
        using GroupKey = std::pair<std::string, std::string>;
        struct AggregatedEntry
        {
            std::string graphName;
            std::set<std::string> supportingEngines;
        };
        std::map<GroupKey, AggregatedEntry> grouped;

        for(const auto& record : _records)
        {
            GroupKey key{record.graphDescription, record.note};
            auto& entry = grouped[key];
            if(entry.graphName.empty())
            {
                entry.graphName = record.graphName;
            }
            entry.supportingEngines.insert(record.supportingEngines.begin(),
                                           record.supportingEngines.end());
        }

        std::ofstream out(_outputPath);
        if(!out.is_open())
        {
            std::cerr << "Error: Could not open " << _outputPath << " for writing\n";
            return;
        }

        out << "# Engine Support Matrix\n\n";

        // Table header
        out << "| Operations | Notes |";
        for(const auto& engine : allEngineNames)
        {
            out << " " << engine << " |";
        }
        out << "\n";

        // Separator
        out << "|------------|-------|";
        for(const auto& engine : allEngineNames)
        {
            out << std::string(engine.size() + 2, '-') << "|";
        }
        out << "\n";

        // Rows
        for(const auto& [key, entry] : grouped)
        {
            const auto& [description, note] = key;
            out << "| " << description << " | " << note << " |";
            for(const auto& engine : allEngineNames)
            {
                bool supported = entry.supportingEngines.count(engine) > 0;
                out << " " << (supported ? "\xe2\x9c\x85" : "-") << " |";
            }
            out << "\n";
        }

        out.close();
        std::cout << "Support matrix written to: " << _outputPath << "\n";
    }

private:
    SupportMatrixCollector() = default;

    mutable std::mutex _mutex;
    std::vector<GraphSupportRecord> _records;
    bool _enabled = false;
    std::string _outputPath = "support_matrix.md";
};

} // namespace hipdnn_integration_tests
