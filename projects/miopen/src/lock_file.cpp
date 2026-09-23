// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/errors.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/logger.hpp>
#include <miopen/md5.hpp>
#include <miopen/stringutils.hpp>

namespace miopen {

inline void LogFsError(const fs::filesystem_error& ex, const std::string_view from)
{
    // clang-format off
    MIOPEN_LOG_E_FROM(from, "File system operation error in LockFile. "
                            "Error code: " << ex.code() << ". "
                            "Description: '" << ex.what() << "'");
    // clang-format on
}

fs::path LockFilePath(const fs::path& filename_)
{
    try
    {
        const auto directory = fs::temp_directory_path() / "miopen-lockfiles";

        if(!fs::exists(directory))
        {
            fs::create_directories(directory);
            fs::permissions(directory, fs::perms::all);
        }
        const auto hash = md5(filename_.parent_path().string());
        const auto file = directory / (hash + "_" + filename_.filename() + ".lock");

        return file;
    }
    catch(const fs::filesystem_error& ex)
    {
        LogFsError(ex, MIOPEN_GET_FN_NAME);
        throw;
    }
}

LockFile::LockFile(const fs::path& path_, PassKey) : path(path_)
{
    try
    {
        if(!fs::exists(path))
        {
            if(!std::ofstream{path})
                MIOPEN_THROW("Error creating file <" + path + "> for locking.");
            fs::permissions(path, fs::perms::all);
        }

        flock = decltype(flock)(path.string());
    }
    catch(const fs::filesystem_error& ex)
    {
        LogFsError(ex, MIOPEN_GET_FN_NAME);
        throw;
    }
    catch(const std::exception& ex)
    {
        LogFlockError(ex, "lock initialization", MIOPEN_GET_FN_NAME);
        throw;
    }
}

LockFile& LockFile::Get(const fs::path& file)
{
#ifdef _WIN32
    // The character ':' is reserved on Windows and cannot be used for constructing
    // a path except when it follows a drive letter.
    fs::path path{ReplaceString(file.string(), ":memory:", "memory_")};
#else
#define path file
#endif
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    { // To guarantee that construction won't be called if not required.
        auto found = LockFiles().find(path);

        if(found != LockFiles().end())
            return found->second;
    }

    auto emplaced = LockFiles().emplace(std::piecewise_construct,
                                        std::forward_as_tuple(path),
                                        std::forward_as_tuple(path, PassKey{}));
    return emplaced.first->second;
}
} // namespace miopen
