// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

namespace hipdnn_integration_tests
{

struct GraphSupportRecord
{
    std::string graphName;
    std::string graphDescription;
    std::string testName;
    std::set<std::string> supportingEngines;
    std::string note;
    std::string layout;
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
                            const std::string& note = {},
                            const std::string& layout = {})
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
        _records.push_back(
            {graphName, graphDescription, testName, std::move(engineNames), note, layout});
    }

    // Thread-safe: returns a copy of the records under mutex protection.
    std::vector<GraphSupportRecord> getRecords() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _records;
    }

    // Reset collector to initial state.
    // Not thread-safe: call only when no parallel recording is in progress.
    void reset()
    {
        _records.clear();
        _enabled = false;
        _outputPath = "support_matrix.md";
    }

    // Generate the markdown output and write to file.
    // Thread-safe: takes a snapshot of records under mutex before writing.
    //
    // allEngineNames: the engine columns to include in the table.
    //
    // Example output:
    //
    //   # Engine Support Matrix
    //
    //   ## Convolution
    //
    //   | Operations | Notes | EngineA | EngineB |
    //   |------------|-------|---------|---------|
    //   | Convolution:FWD fp32 |  | checkmark NHWC | - |
    //   | Convolution:BWD fp16 | unstable | checkmark NHWC, NCHW | checkmark NHWC |
    void writeMarkdown(const std::vector<std::string>& allEngineNames) const
    {
        // Snapshot records under lock, then release before doing file I/O.
        std::vector<GraphSupportRecord> records;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            records = _records;
        }

        // Group records by (graphDescription, note), union the engine support sets.
        // Use ordered map so output is deterministic.
        using GroupKey = std::pair<std::string, std::string>;
        struct AggregatedEntry
        {
            std::string graphName;
            // Maps engine name -> set of layout names that engine supports.
            std::map<std::string, std::set<std::string>> engineLayouts;
        };
        std::map<GroupKey, AggregatedEntry> grouped;

        for(const auto& record : records)
        {
            GroupKey key{record.graphDescription, record.note};
            auto& entry = grouped[key];
            if(entry.graphName.empty())
            {
                entry.graphName = record.graphName;
            }
            for(const auto& engine : record.supportingEngines)
            {
                auto& layouts = entry.engineLayouts[engine];
                if(!record.layout.empty())
                {
                    layouts.insert(record.layout);
                }
            }
        }

        std::ofstream out(_outputPath);
        if(!out.is_open())
        {
            std::cerr << "Error: Could not open " << _outputPath << " for writing\n";
            return;
        }

        out << "# Engine Support Matrix\n\n";

        // Group entries by the first operation name (text before the first space).
        // Preserve ordering from the sorted grouped map.
        std::vector<
            std::pair<std::string, std::vector<std::pair<GroupKey, const AggregatedEntry*>>>>
            sections;
        std::string currentSection;

        for(const auto& [key, entry] : grouped)
        {
            const auto& description = key.first;
            auto spacePos = description.find(' ');
            std::string firstOp
                = (spacePos != std::string::npos) ? description.substr(0, spacePos) : description;

            // Strip attribute suffix (e.g., ":AVG", ":RELU_FWD") to group
            // operations of the same base type under one section heading.
            auto colonPos = firstOp.find(':');
            std::string sectionKey
                = (colonPos != std::string::npos) ? firstOp.substr(0, colonPos) : firstOp;

            if(sectionKey != currentSection)
            {
                currentSection = sectionKey;
                sections.push_back({sectionKey, {}});
            }
            sections.back().second.push_back({key, &entry});
        }

        auto writeTableHeader = [&]() {
            out << "| Operations | Notes |";
            for(const auto& engine : allEngineNames)
            {
                out << " " << engine << " |";
            }
            out << "\n";

            out << "|------------|-------|";
            for(const auto& engine : allEngineNames)
            {
                out << std::string(engine.size() + 2, '-') << "|";
            }
            out << "\n";
        };

        for(const auto& [sectionName, entries] : sections)
        {
            out << "## " << sectionName << "\n\n";
            writeTableHeader();

            for(const auto& [key, entryPtr] : entries)
            {
                const auto& [description, note] = key;
                out << "| " << description << " | " << note << " |";
                for(const auto& engine : allEngineNames)
                {
                    auto it = entryPtr->engineLayouts.find(engine);
                    if(it != entryPtr->engineLayouts.end())
                    {
                        out << " \xe2\x9c\x85";
                        if(!it->second.empty())
                        {
                            bool first = true;
                            out << " ";
                            for(const auto& layout : it->second)
                            {
                                if(!first)
                                {
                                    out << ", ";
                                }
                                first = false;
                                out << layout;
                            }
                        }
                        out << " |";
                    }
                    else
                    {
                        out << " - |";
                    }
                }
                out << "\n";
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
