#pragma once

#include <filesystem>

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

}
