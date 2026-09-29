// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-fs.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hipblaslt_jit::files
{
    namespace
    {
        namespace fs = std::filesystem;

        Status failure(const fs::path& path, const std::string& reason)
        {
            return {Status::Code::Failed, Stage::Configure, path.u8string() + ": " + reason};
        }

#ifndef _WIN32
        Status systemFailure(const fs::path& path, const char* operation, int error)
        {
            return failure(path, std::string(operation) + ": " + std::strerror(error));
        }
#endif

        // Distinct between threads through the counter and between live
        // processes through the process id.
        std::string temporaryName(const fs::path& target)
        {
            static std::atomic<uint64_t> counter{0};
            static const uint64_t        salt = std::random_device{}();
#ifdef _WIN32
            const auto process = _getpid();
#else
            const auto process = getpid();
#endif
            return "." + target.filename().u8string() + "." + std::to_string(process) + "."
                   + std::to_string(salt + counter++) + ".tmp";
        }

        fs::path withoutTrailingSeparator(const fs::path& path)
        {
            return path.has_filename() ? path : path.parent_path();
        }
    }

#ifdef _WIN32
    Status privateDirectory(const fs::path& requested)
    {
        const auto      path = withoutTrailingSeparator(requested);
        std::error_code error;
        fs::create_directories(path, error);
        const auto attributes = GetFileAttributesW(path.c_str());
        if(attributes == INVALID_FILE_ATTRIBUTES)
            return failure(path, "cannot be created or read");
        if(attributes & FILE_ATTRIBUTE_REPARSE_POINT)
            return failure(path, "is a reparse point");
        if(!(attributes & FILE_ATTRIBUTE_DIRECTORY))
            return failure(path, "is not a directory");
        return {};
    }

    FileLock::~FileLock()
    {
        release();
    }

    Status FileLock::acquire(const fs::path&           file,
                             std::chrono::milliseconds timeout,
                             FileLock&                 lock)
    {
        lock.release();
        const auto handle = CreateFileW(file.c_str(),
                                        GENERIC_READ | GENERIC_WRITE,
                                        FILE_SHARE_READ | FILE_SHARE_WRITE,
                                        nullptr,
                                        OPEN_ALWAYS,
                                        FILE_ATTRIBUTE_NORMAL,
                                        nullptr);
        if(handle == INVALID_HANDLE_VALUE)
            return failure(file, "cannot be opened");
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for(;;)
        {
            OVERLAPPED region{};
            if(LockFileEx(handle,
                          LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                          0,
                          1,
                          0,
                          &region))
                break;
            if(std::chrono::steady_clock::now() >= deadline)
            {
                CloseHandle(handle);
                return failure(file, "timed out waiting for the lock");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        lock.m_handle = handle;
        return {};
    }

    void FileLock::release() noexcept
    {
        if(!m_handle)
            return;
        OVERLAPPED region{};
        UnlockFileEx(m_handle, 0, 1, 0, &region);
        CloseHandle(m_handle);
        m_handle = nullptr;
    }

    Status writeAtomically(const fs::path&             staging,
                           const fs::path&             target,
                           const std::vector<uint8_t>& bytes)
    {
        const auto temporary = staging / temporaryName(target);
        const auto handle    = CreateFileW(temporary.c_str(),
                                        GENERIC_WRITE,
                                        0,
                                        nullptr,
                                        CREATE_ALWAYS,
                                        FILE_ATTRIBUTE_NORMAL,
                                        nullptr);
        if(handle == INVALID_HANDLE_VALUE)
            return failure(temporary, "cannot be created");
        size_t written = 0;
        bool   ok      = true;
        while(ok && written < bytes.size())
        {
            DWORD      count = 0;
            const auto chunk = static_cast<DWORD>(std::min<size_t>(bytes.size() - written, 1u << 30));
            ok               = WriteFile(handle, bytes.data() + written, chunk, &count, nullptr);
            written += count;
        }
        ok = ok && FlushFileBuffers(handle);
        CloseHandle(handle);
        if(!ok)
        {
            DeleteFileW(temporary.c_str());
            return failure(temporary, "cannot be written");
        }
        // The stock loader reads through std::ifstream, which does not share
        // delete access, so a replacement can fail while a reader holds the file.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while(!MoveFileExW(temporary.c_str(),
                           target.c_str(),
                           MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        {
            const auto error = GetLastError();
            if((error != ERROR_SHARING_VIOLATION && error != ERROR_ACCESS_DENIED)
               || std::chrono::steady_clock::now() >= deadline)
            {
                DeleteFileW(temporary.c_str());
                return failure(target, "cannot be replaced");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return {};
    }

    FileIdentity identify(const fs::path& path) noexcept
    {
        FileIdentity    identity;
        std::error_code error;
        const auto      size = fs::file_size(path, error);
        if(error)
            return identity;
        const auto modified = fs::last_write_time(path, error);
        if(error)
            return identity;
        identity.exists   = true;
        identity.size     = size;
        identity.modified = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                modified.time_since_epoch())
                                .count();
        return identity;
    }
#else
    Status privateDirectory(const fs::path& requested)
    {
        const auto path = withoutTrailingSeparator(requested);
        if(path.has_parent_path() && path.parent_path() != path)
        {
            std::error_code ignored; // mkdir below reports a missing parent
            fs::create_directories(path.parent_path(), ignored);
        }
        if(::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST)
            return systemFailure(path, "mkdir", errno);
        struct stat info{};
        if(::lstat(path.c_str(), &info) != 0)
            return systemFailure(path, "lstat", errno);
        if(S_ISLNK(info.st_mode))
            return failure(path, "is a symbolic link");
        if(!S_ISDIR(info.st_mode))
            return failure(path, "is not a directory");
        if(info.st_uid != ::geteuid())
            return failure(path, "is not owned by the current user");
        if(info.st_mode & (S_IWGRP | S_IWOTH))
            return failure(path, "is writable by group or others");
        return {};
    }

    FileLock::~FileLock()
    {
        release();
    }

    Status FileLock::acquire(const fs::path&           file,
                             std::chrono::milliseconds timeout,
                             FileLock&                 lock)
    {
        lock.release();
        const int descriptor = ::open(file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if(descriptor < 0)
            return systemFailure(file, "open", errno);
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        // flock locks belong to the open file description, so a second
        // descriptor in the same process waits like another process does.
        while(::flock(descriptor, LOCK_EX | LOCK_NB) != 0)
        {
            const int error = errno;
            if(error != EWOULDBLOCK && error != EINTR)
            {
                ::close(descriptor);
                return systemFailure(file, "flock", error);
            }
            if(std::chrono::steady_clock::now() >= deadline)
            {
                ::close(descriptor);
                return failure(file, "timed out waiting for the lock");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        lock.m_descriptor = descriptor;
        return {};
    }

    void FileLock::release() noexcept
    {
        if(m_descriptor < 0)
            return;
        ::flock(m_descriptor, LOCK_UN);
        ::close(m_descriptor);
        m_descriptor = -1;
    }

    Status writeAtomically(const fs::path&             staging,
                           const fs::path&             target,
                           const std::vector<uint8_t>& bytes)
    {
        const auto temporary = staging / temporaryName(target);
        const int  descriptor
            = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
        if(descriptor < 0)
            return systemFailure(temporary, "open", errno);
        const auto abandon = [&](const char* operation) {
            const int error = errno;
            ::close(descriptor);
            ::unlink(temporary.c_str());
            return systemFailure(temporary, operation, error);
        };
        for(size_t written = 0; written < bytes.size();)
        {
            const auto count = ::write(descriptor, bytes.data() + written, bytes.size() - written);
            if(count < 0 && errno == EINTR)
                continue;
            if(count < 0)
                return abandon("write");
            written += static_cast<size_t>(count);
        }
        if(::fsync(descriptor) != 0)
            return abandon("fsync");
        if(::close(descriptor) != 0)
        {
            const int error = errno;
            ::unlink(temporary.c_str());
            return systemFailure(temporary, "close", error);
        }
        if(::rename(temporary.c_str(), target.c_str()) != 0)
        {
            const int error = errno;
            ::unlink(temporary.c_str());
            return systemFailure(target, "rename", error);
        }
        const auto parent    = target.parent_path();
        const int  directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if(directory < 0)
            return systemFailure(parent, "open", errno);
        const bool synced = ::fsync(directory) == 0;
        const int  error  = errno;
        ::close(directory);
        return synced ? Status{} : systemFailure(parent, "fsync", error);
    }

    FileIdentity identify(const fs::path& path) noexcept
    {
        FileIdentity identity;
        struct stat  info{};
        if(::stat(path.c_str(), &info) != 0)
            return identity;
        identity.exists   = true;
        identity.device   = static_cast<uint64_t>(info.st_dev);
        identity.inode    = static_cast<uint64_t>(info.st_ino);
        identity.size     = static_cast<uint64_t>(info.st_size);
        identity.modified = static_cast<int64_t>(info.st_mtim.tv_sec) * 1000000000LL
                            + info.st_mtim.tv_nsec;
        return identity;
    }
#endif

    Status readFile(const fs::path& path, size_t limit, std::vector<uint8_t>& bytes)
    {
        bytes.clear();
        std::ifstream input(path, std::ios::binary);
        if(!input)
            return failure(path, "cannot be opened");
        input.seekg(0, std::ios::end);
        const auto size = static_cast<std::streamoff>(input.tellg());
        if(size < 0)
            return failure(path, "cannot be read");
        if(static_cast<uint64_t>(size) > limit)
            return failure(path, "is larger than " + std::to_string(limit) + " bytes");
        input.seekg(0);
        bytes.resize(static_cast<size_t>(size));
        if(size && !input.read(reinterpret_cast<char*>(bytes.data()), size))
            return failure(path, "cannot be read");
        return {};
    }
}
