// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <vector>

// Owner-private directories, an inter-process lock and crash-safe file
// replacement. Failures are Status::Code::Failed with the path in the message.
namespace hipblaslt_jit::files
{
    // Creates path with mode 0700 when it is missing (its parents with the
    // default mode), then requires a directory that is not a symbolic link, is
    // owned by the effective user and is not writable by group or others.
    // Windows requires a directory that is not a reparse point.
    Status privateDirectory(const std::filesystem::path& path);

    // An exclusive lock between processes and between threads of one process.
    // The lock file is created when missing and never deleted.
    class FileLock
    {
    public:
        FileLock() = default;
        FileLock(const FileLock&)            = delete;
        FileLock& operator=(const FileLock&) = delete;
        ~FileLock();

        static Status acquire(const std::filesystem::path& file,
                              std::chrono::milliseconds    timeout,
                              FileLock&                    lock);
        void          release() noexcept;

    private:
#ifdef _WIN32
        void* m_handle = nullptr;
#else
        int m_descriptor = -1;
#endif
    };

    // Writes bytes to a new file in staging, flushes it to storage and renames
    // it over target, which must be on the same file system. Readers see the old
    // or the new content, never a mixture.
    Status writeAtomically(const std::filesystem::path& staging,
                           const std::filesystem::path& target,
                           const std::vector<uint8_t>&  bytes);

    // Reads a whole file of at most limit bytes.
    Status readFile(const std::filesystem::path& path, size_t limit, std::vector<uint8_t>& bytes);

    // Changes whenever a file is replaced or rewritten.
    struct FileIdentity
    {
        bool     exists = false;
        uint64_t device = 0, inode = 0, size = 0;
        int64_t  modified = 0; // nanoseconds

        bool operator==(const FileIdentity& other) const noexcept
        {
            return exists == other.exists && device == other.device && inode == other.inode
                   && size == other.size && modified == other.modified;
        }
        bool operator!=(const FileIdentity& other) const noexcept
        {
            return !(*this == other);
        }
    };
    FileIdentity identify(const std::filesystem::path& path) noexcept;
}
