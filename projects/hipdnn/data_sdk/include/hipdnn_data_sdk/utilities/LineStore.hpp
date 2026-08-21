// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// LineStore is a generic, record-format-agnostic append-only shard file: a bare version
// string on the first line, followed by caller-defined record lines, one per line. It
// treats every line after the first as opaque text and hands parsing/formatting to the
// caller via a callback -- this header must never depend on nlohmann/json or any other
// JSON type; JSON-record callers supply their own encode/decode.
//
// Concurrency: each shard has its own advisory lock, held for the duration of an append.
// Locking is per-file, not per-line -- concurrent writers append serially, never
// interleaved or torn.
//
// Failure handling is fail-soft: open/lock/read failures and version mismatches are
// reported through the return value, never an exception. A line the caller's parse
// callback rejects is skipped without affecting any other line. Resolving duplicate-keyed
// records (e.g. last-line-wins) is the caller's job.

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace hipdnn_data_sdk::utilities
{

/// Outcome of opening a shard file or acquiring its lock. Any non-OK value is a
/// caller-visible decline -- fall back to in-memory behavior, don't treat it as fatal.
enum class LineStoreStatus
{
    OK,
    OPEN_FAILED,
    LOCK_FAILED,
    IO_ERROR,
    VERSION_MISMATCH,
};

namespace detail
{

class LineStoreAccess;

#if defined(_WIN32)
using NativeLineStoreHandle = HANDLE;
// Not constexpr: INVALID_HANDLE_VALUE casts an integer to a pointer, which is not a valid
// constant expression in standard C++, even though every Win32 SDK defines it that way.
inline const NativeLineStoreHandle INVALID_LINE_STORE_HANDLE = INVALID_HANDLE_VALUE;
inline bool isValidLineStoreHandle(NativeLineStoreHandle handle) noexcept
{
    return handle != INVALID_HANDLE_VALUE;
}
#else
using NativeLineStoreHandle = int;
constexpr NativeLineStoreHandle INVALID_LINE_STORE_HANDLE = -1;
inline bool isValidLineStoreHandle(NativeLineStoreHandle handle) noexcept
{
    return handle >= 0;
}
#endif

// Forward-declared so LineStoreShard's destructor/move-assignment below can release a
// held lock without reordering the raw-file helpers ahead of the type they operate on.
void releaseLineStoreLock(NativeLineStoreHandle handle) noexcept;
void closeLineStoreHandle(NativeLineStoreHandle handle) noexcept;

} // namespace detail

/// A handle to one open, version-checked shard file, obtained only via openLineStore() --
/// callers never see a half-open or version-mismatched shard through this type.
class LineStoreShard
{
public:
    LineStoreShard() = delete;
    LineStoreShard(const LineStoreShard&) = delete;
    LineStoreShard& operator=(const LineStoreShard&) = delete;

    LineStoreShard(LineStoreShard&& other) noexcept
        : _path(std::move(other._path))
        , _handle(other._handle)
        , _locked(other._locked)
    {
        other._handle = detail::INVALID_LINE_STORE_HANDLE;
        other._locked = false;
    }

    LineStoreShard& operator=(LineStoreShard&& other) noexcept
    {
        if(this != &other)
        {
            closeIfOpen();
            _path = std::move(other._path);
            _handle = other._handle;
            _locked = other._locked;
            other._handle = detail::INVALID_LINE_STORE_HANDLE;
            other._locked = false;
        }
        return *this;
    }

    ~LineStoreShard()
    {
        closeIfOpen();
    }

    /// True while this handle holds the shard's advisory lock.
    bool isLocked() const noexcept
    {
        return _locked;
    }

private:
    friend class detail::LineStoreAccess;

    explicit LineStoreShard(std::filesystem::path path)
        : _path(std::move(path))
    {
    }

    void closeIfOpen() noexcept
    {
        if(detail::isValidLineStoreHandle(_handle))
        {
            if(_locked)
            {
                detail::releaseLineStoreLock(_handle);
                _locked = false;
            }
            detail::closeLineStoreHandle(_handle);
            _handle = detail::INVALID_LINE_STORE_HANDLE;
        }
    }

    std::filesystem::path _path;
    detail::NativeLineStoreHandle _handle = detail::INVALID_LINE_STORE_HANDLE;
    bool _locked = false;
};

namespace detail
{

// Acquires a whole-file exclusive advisory lock, blocking until held. POSIX: fcntl()
// F_SETLKW over the whole file. Win32: LockFileEx() over the same range, blocking.
inline LineStoreStatus acquireLineStoreLock(NativeLineStoreHandle handle) noexcept
{
#if defined(_WIN32)
    OVERLAPPED overlapped{};
    if(LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &overlapped) == 0)
    {
        return LineStoreStatus::LOCK_FAILED;
    }
#else
    struct flock fl
    {
    };
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0; // whole file
    if(::fcntl(handle, F_SETLKW, &fl) == -1)
    {
        return LineStoreStatus::LOCK_FAILED;
    }
#endif
    return LineStoreStatus::OK;
}

inline void releaseLineStoreLock(NativeLineStoreHandle handle) noexcept
{
#if defined(_WIN32)
    OVERLAPPED overlapped{};
    UnlockFileEx(handle, 0, MAXDWORD, MAXDWORD, &overlapped);
#else
    struct flock fl
    {
    };
    fl.l_type = F_UNLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;
    ::fcntl(handle, F_SETLK, &fl);
#endif
}

inline void closeLineStoreHandle(NativeLineStoreHandle handle) noexcept
{
#if defined(_WIN32)
    CloseHandle(handle);
#else
    ::close(handle);
#endif
}

// Appends @p line plus a trailing '\n' as a single OS-level write, relying on the
// handle's atomic-append mode (O_APPEND on POSIX, FILE_APPEND_DATA on Win32) so a
// concurrent lock-free reader never observes a torn line.
inline bool appendRawLineStoreLine(NativeLineStoreHandle handle, std::string_view line) noexcept
{
    std::string buffer;
    buffer.reserve(line.size() + 1);
    buffer.append(line);
    buffer.push_back('\n');

    size_t written = 0;
#if defined(_WIN32)
    while(written < buffer.size())
    {
        DWORD chunk = 0;
        if(WriteFile(handle,
                     buffer.data() + written,
                     static_cast<DWORD>(buffer.size() - written),
                     &chunk,
                     nullptr)
               == 0
           || chunk == 0)
        {
            return false;
        }
        written += chunk;
    }
#else
    while(written < buffer.size())
    {
        const ssize_t chunk = ::write(handle, buffer.data() + written, buffer.size() - written);
        if(chunk <= 0)
        {
            return false;
        }
        written += static_cast<size_t>(chunk);
    }
#endif
    return true;
}

// Reads the entire contents of @p handle from offset zero; used for both the version-line
// check and readAllLines(), since shards are small enough that a full read is simplest.
inline std::optional<std::string> readAllLineStoreBytes(NativeLineStoreHandle handle) noexcept
{
    std::string content;
    std::array<char, 65536> buffer{};

#if defined(_WIN32)
    LARGE_INTEGER origin{};
    if(SetFilePointerEx(handle, origin, nullptr, FILE_BEGIN) == 0)
    {
        return std::nullopt;
    }
    for(;;)
    {
        DWORD count = 0;
        if(ReadFile(handle, buffer.data(), static_cast<DWORD>(buffer.size()), &count, nullptr) == 0)
        {
            return std::nullopt;
        }
        if(count == 0)
        {
            break;
        }
        content.append(buffer.data(), count);
    }
#else
    if(::lseek(handle, 0, SEEK_SET) == static_cast<off_t>(-1))
    {
        return std::nullopt;
    }
    for(;;)
    {
        const ssize_t count = ::read(handle, buffer.data(), buffer.size());
        if(count < 0)
        {
            return std::nullopt;
        }
        if(count == 0)
        {
            break;
        }
        content.append(buffer.data(), static_cast<size_t>(count));
    }
#endif
    return content;
}

// Splits raw content into whole lines on '\n'. A trailing chunk with no newline is an
// incomplete write (a reader racing a not-yet-flushed append) and is dropped.
inline std::vector<std::string> splitLineStoreLines(const std::string& content)
{
    std::vector<std::string> lines;
    size_t start = 0;
    while(start < content.size())
    {
        const size_t newlinePos = content.find('\n', start);
        if(newlinePos == std::string::npos)
        {
            break;
        }
        lines.emplace_back(content, start, newlinePos - start);
        start = newlinePos + 1;
    }
    return lines;
}

// Grants the free functions below access to LineStoreShard's private state (a template
// friend can't carry readAllLines()'s default Record argument).
class LineStoreAccess
{
public:
    static LineStoreShard make(std::filesystem::path path)
    {
        return LineStoreShard(std::move(path));
    }

    static NativeLineStoreHandle handle(const LineStoreShard& shard) noexcept
    {
        return shard._handle;
    }

    static void setHandle(LineStoreShard& shard, NativeLineStoreHandle handle) noexcept
    {
        shard._handle = handle;
    }

    static bool locked(const LineStoreShard& shard) noexcept
    {
        return shard._locked;
    }

    static void setLocked(LineStoreShard& shard, bool locked) noexcept
    {
        shard._locked = locked;
    }
};

} // namespace detail

/// Opens the shard file at @p path, creating it (and writing @p expectedVersion as its
/// first line) if absent. An existing file whose first line doesn't match
/// @p expectedVersion returns VERSION_MISMATCH rather than throwing, so a version bump
/// never crashes an older reader of the same cache directory.
///
/// @return The open shard and OK on success; nullopt and a non-OK status otherwise.
inline std::pair<std::optional<LineStoreShard>, LineStoreStatus>
    openLineStore(const std::filesystem::path& path, std::string_view expectedVersion)
{
#if defined(_WIN32)
    const HANDLE nativeHandle = CreateFileW(path.wstring().c_str(),
                                            FILE_GENERIC_READ | FILE_APPEND_DATA,
                                            FILE_SHARE_READ | FILE_SHARE_WRITE,
                                            nullptr,
                                            OPEN_ALWAYS,
                                            FILE_ATTRIBUTE_NORMAL,
                                            nullptr);
    if(nativeHandle == INVALID_HANDLE_VALUE)
    {
        return {std::nullopt, LineStoreStatus::OPEN_FAILED};
    }
#else
    const int nativeHandle = ::open(path.c_str(), O_RDWR | O_CREAT | O_APPEND, 0644);
    if(nativeHandle < 0)
    {
        return {std::nullopt, LineStoreStatus::OPEN_FAILED};
    }
#endif

    LineStoreShard shard = detail::LineStoreAccess::make(path);
    detail::LineStoreAccess::setHandle(shard, nativeHandle);

    // Locked only long enough to check/write the version line, so two racing creators
    // never both write one. The returned shard starts unlocked.
    if(detail::acquireLineStoreLock(nativeHandle) != LineStoreStatus::OK)
    {
        return {std::nullopt, LineStoreStatus::LOCK_FAILED};
    }

    const auto content = detail::readAllLineStoreBytes(nativeHandle);
    if(!content)
    {
        detail::releaseLineStoreLock(nativeHandle);
        return {std::nullopt, LineStoreStatus::IO_ERROR};
    }

    if(content->empty())
    {
        if(!detail::appendRawLineStoreLine(nativeHandle, expectedVersion))
        {
            detail::releaseLineStoreLock(nativeHandle);
            return {std::nullopt, LineStoreStatus::IO_ERROR};
        }
    }
    else
    {
        const auto lines = detail::splitLineStoreLines(*content);
        if(lines.empty() || lines.front() != expectedVersion)
        {
            detail::releaseLineStoreLock(nativeHandle);
            return {std::nullopt, LineStoreStatus::VERSION_MISMATCH};
        }
    }

    detail::releaseLineStoreLock(nativeHandle);
    return {std::optional<LineStoreShard>(std::move(shard)), LineStoreStatus::OK};
}

/// Acquires @p shard's advisory lock, blocking until held or failed. A failure (e.g. an
/// incompatible external lock holder) reports LOCK_FAILED rather than throwing.
inline LineStoreStatus lockLineStore(LineStoreShard& shard)
{
    const auto status = detail::acquireLineStoreLock(detail::LineStoreAccess::handle(shard));
    if(status == LineStoreStatus::OK)
    {
        detail::LineStoreAccess::setLocked(shard, true);
    }
    return status;
}

/// Releases a lock previously acquired by lockLineStore(). A no-op, not an error, if
/// @p shard is not currently locked.
inline void unlockLineStore(LineStoreShard& shard) noexcept
{
    if(!detail::LineStoreAccess::locked(shard))
    {
        return;
    }
    detail::releaseLineStoreLock(detail::LineStoreAccess::handle(shard));
    detail::LineStoreAccess::setLocked(shard, false);
}

/// Appends one caller-formatted record line to @p shard, which must already hold the lock
/// (see lockLineStore()); appendLine() does not acquire it itself. @p line must not
/// contain a newline.
///
/// @return OK on success; IO_ERROR on any write failure. Never throws.
inline LineStoreStatus appendLine(LineStoreShard& shard, std::string_view line)
{
    if(!detail::appendRawLineStoreLine(detail::LineStoreAccess::handle(shard), line))
    {
        return LineStoreStatus::IO_ERROR;
    }
    return LineStoreStatus::OK;
}

/// Reads every record line from @p shard (everything after the version line), handing
/// each to @p parseLine in file order; a std::nullopt result skips that line without
/// affecting any other -- a corrupt or forward-incompatible record never poisons the rest
/// of an otherwise-good shard. Resolving duplicate-keyed records is left to the caller.
///
/// @tparam ParseLine Caller callback type, deduced from the argument (std::function is a
///     non-deduced context that would force naming Record explicitly).
/// @param shard Does not require the lock: a concurrent append is observed as the old or
///     new file state, never a torn line.
/// @return Parsed records in file order and OK; an empty vector and non-OK status if the
///     shard could not be read. Never throws.
template <typename ParseLine,
          typename Record = typename std::invoke_result_t<ParseLine, std::string_view>::value_type>
std::pair<std::vector<Record>, LineStoreStatus> readAllLines(const LineStoreShard& shard,
                                                             ParseLine parseLine)
{
    const auto content = detail::readAllLineStoreBytes(detail::LineStoreAccess::handle(shard));
    if(!content)
    {
        return {{}, LineStoreStatus::IO_ERROR};
    }

    const auto lines = detail::splitLineStoreLines(*content);
    std::vector<Record> records;
    // Line 0 is the version line, already validated; records start at index 1.
    for(size_t i = 1; i < lines.size(); ++i)
    {
        if(auto parsed = parseLine(std::string_view(lines[i])))
        {
            records.push_back(std::move(*parsed));
        }
    }
    return {std::move(records), LineStoreStatus::OK};
}

} // namespace hipdnn_data_sdk::utilities
