// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

/// @file OperationDirectory.hpp
/// @brief Loading a directory of operation declarations.
///
/// Split out of MetadataCorpus.hpp, which links the frontend for its engine-backed oracle and
/// so could not be reached by a unit test. Loading is pure filesystem work and is exactly where
/// a silent skip is dangerous: an operation whose declaration fails to parse and is quietly
/// dropped leaves a hole in the corpus that looks identical to an operation nobody declared.
namespace hipdnn_corpus_gen
{

/// Every `*.opmeta.json` in a directory, parsed. Files that fail validation are reported
/// rather than skipped: a declaration that does not load is a hole in the corpus.
struct MetadataSet
{
    std::vector<std::pair<std::string, OperationMetadata>> operations;
    std::vector<std::string> errors;
};

inline MetadataSet loadOperationDirectory(const std::filesystem::path& directory)
{
    MetadataSet set;
    if(!std::filesystem::is_directory(directory))
    {
        set.errors.push_back("not a directory: " + directory.string());
        return set;
    }

    // Sorted, so a corpus generated twice visits its operations in one order. Directory
    // iteration order is unspecified, and an unordered visit would reshuffle which operations
    // a maxCombinations bound reaches.
    std::vector<std::filesystem::path> files;
    for(const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if(entry.path().extension() == ".json"
           && entry.path().string().find(".opmeta.") != std::string::npos)
        {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for(const auto& file : files)
    {
        std::ifstream stream(file);
        if(!stream)
        {
            set.errors.push_back("cannot read " + file.string());
            continue;
        }

        try
        {
            auto parsed = parseOperationMetadata(nlohmann::json::parse(stream));
            if(!parsed.ok())
            {
                for(const auto& error : parsed.errors)
                {
                    set.errors.push_back(file.filename().string() + ": " + error);
                }
                continue;
            }
            set.operations.emplace_back(file.string(), std::move(*parsed.metadata));
        }
        catch(const std::exception& error)
        {
            // Named with its file: one malformed declaration among twenty is otherwise a
            // parser message with no indication of which file produced it.
            set.errors.push_back(file.filename().string() + ": " + error.what());
        }
    }
    return set;
}

} // namespace hipdnn_corpus_gen
