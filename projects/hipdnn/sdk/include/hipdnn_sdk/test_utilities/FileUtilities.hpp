// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <vector>

namespace hipdnn_sdk::test_utilities
{

class ScopedDirectory
{
    std::filesystem::path _path;

public:
    ScopedDirectory(std::filesystem::path path)
    {
        if(std::filesystem::create_directory(path))
        {
            _path = std::move(path);
        }
        else
        {
            throw std::runtime_error("ScopedDirectory: Directory already exists");
        }
    }
    const std::filesystem::path& path() const
    {
        return _path;
    }

    ScopedDirectory(const ScopedDirectory&) = delete;
    ScopedDirectory& operator=(const ScopedDirectory&) = delete;
    ScopedDirectory(ScopedDirectory&&) = default;
    ScopedDirectory& operator=(ScopedDirectory&&) = default;
    ~ScopedDirectory()
    {
        if(!_path.empty())
        {
            std::filesystem::remove_all(_path);
        }
    }
};

inline std::vector<std::filesystem::path> filesInDirectoryWithExt(const std::filesystem::path& path,
                                                                  const std::string& ext)
{
    std::vector<std::filesystem::path> paths;
    std::copy_if(std::filesystem::directory_iterator(path),
                 std::filesystem::directory_iterator(),
                 std::back_inserter(paths),
                 [ext](const std::filesystem::path& p) { return p.extension() == ext; });

    return paths;
}

// Temporary helper function
inline std::vector<std::filesystem::path>
    filesInDirectoryWithExtReturnEmptyPathOnThrow(const std::filesystem::path& path,
                                                  const std::string& ext)
{
    try
    {
        return filesInDirectoryWithExt(path, ext);
    }
    catch(...)
    {
        return {""};
    }
}

}
