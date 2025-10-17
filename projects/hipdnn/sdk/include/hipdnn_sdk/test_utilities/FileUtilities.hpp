#pragma once

#include <algorithm>
#include <filesystem>
#include <vector>

namespace hipdnn_sdk::test_utilities
{

class TempDirectory
{
    std::filesystem::path _path;

public:
    TempDirectory(std::filesystem::path path)
    {
        if(std::filesystem::create_directory(path))
        {
            _path = std::move(path);
        }
        else
        {
            throw std::runtime_error("TempDirectory: Directory already exists");
        }
    }
    const std::filesystem::path& path() const
    {
        return _path;
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    TempDirectory(TempDirectory&&) = default;
    TempDirectory& operator=(TempDirectory&&) = default;
    ~TempDirectory()
    {
        if(!_path.empty())
        {
            std::filesystem::remove_all(_path);
        }
    }
};

inline std::vector<std::filesystem::path> filesInDirectoryWithExt(std::filesystem::path const& path,
                                                                  std::string const& ext)
{
    std::vector<std::filesystem::path> paths;
    std::copy_if(std::filesystem::directory_iterator(path),
                 std::filesystem::directory_iterator(),
                 std::back_inserter(paths),
                 [ext](std::filesystem::path const& p) { return p.extension() == ext; });

    return paths;
}

// Temporary helper function
inline std::vector<std::filesystem::path>
    filesInDirectoryWithExtReturnEmptyPathOnThrow(std::filesystem::path const& path,
                                                  std::string const& ext)
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
