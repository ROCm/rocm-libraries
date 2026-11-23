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
/// @brief Loading a directory of operation declarations; kept free of the frontend so unit
/// tests can link it.
namespace hipdnn_corpus_gen
{

/// Every `*.opmeta.json` in a directory, parsed. Files that fail to load are reported in
/// `errors`, never silently skipped.
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

    // Sorted: directory iteration order is unspecified, and the visit order decides which
    // operations a maxCombinations bound reaches.
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
            // Prefix with the file name; the parser's message does not say which file.
            set.errors.push_back(file.filename().string() + ": " + error.what());
        }
    }
    return set;
}

} // namespace hipdnn_corpus_gen
