// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// LineStore is the generic, record-format-agnostic append-only shard file abstraction that
// layer 1 exposes to every on-disk cache built on top of it (winner cache, autotune cache,
// and future consumers alike). A "shard" is a single file: a bare version string on its
// first line, followed by zero or more caller-defined record lines, one record per line.
//
// LineStore knows nothing about what a record means. It treats every line after the first
// as an opaque std::string and hands parsing/formatting to the caller via a callback or
// template parameter. In particular, this header MUST NOT include or reference
// nlohmann/json or any other JSON type -- data_sdk has no JSON dependency today and must
// not gain one through this file. Callers that want JSON-Lines records (as layer 3 does)
// supply their own encode/decode callbacks from a layer that already depends on JSON.
//
// Concurrency: each shard has its own advisory lock, acquired for the duration of an
// append and released afterward. Locking is per-file, not per-line -- two writers append
// to the same shard serially, never interleaved or torn.
//
// Failure handling is fail-soft throughout: a shard that cannot be opened, locked, or read
// is reported through the return value, never through an exception. A version mismatch on
// read is a caller-visible decline of the whole file, not a throw and not a crash. A single
// line that the caller's parse callback rejects is skipped and does not affect any other
// line in the file -- it never poisons the read of an otherwise-good shard.
//
// Last-line-wins resolution over duplicate-keyed records (e.g. the same logical key
// appended twice because two processes both missed the same cache lookup) is the caller's
// job: LineStore guarantees line order and append atomicity, nothing about record identity.

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

/// Outcome of opening (or creating) a shard file, and of acquiring its lock.
///
/// Every value other than Ok is a caller-visible decline: the caller should treat the
/// shard as unavailable for this operation and fall back to whatever behavior it would
/// have used with no on-disk cache at all, not treat it as fatal.
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
// constant expression in standard C++ even though every Win32 SDK defines the macro that
// way.
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

// Forward-declared here (defined further below, alongside acquireLineStoreLock and the
// rest of the raw-file helpers) so LineStoreShard's destructor/move-assignment, defined
// inline just below, can release a held lock and close the handle without reordering the
// full set of raw-file helpers ahead of the type they operate on.
void releaseLineStoreLock(NativeLineStoreHandle handle) noexcept;
void closeLineStoreHandle(NativeLineStoreHandle handle) noexcept;

} // namespace detail

/// A handle to one open, version-checked shard file.
///
/// Obtained only via openLineStore(); a LineStoreShard is either a live handle to a
/// shard whose version line matched, or it does not exist -- callers never see a
/// half-open or version-mismatched shard through this type.
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

// Acquires a whole-file exclusive advisory lock on @p handle, blocking until it is held.
// POSIX: fcntl() byte-range lock covering the whole file (l_len == 0). Win32: LockFileEx()
// over the same [0, MAXDWORD*2) range, without LOCKFILE_FAIL_IMMEDIATELY so the call
// blocks like fcntl(F_SETLKW) does.
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
// handle's atomic-append file mode (O_APPEND on POSIX, FILE_APPEND_DATA on Win32) so a
// concurrent lock-free reader never observes a torn line -- one local-filesystem write
// call is atomic with respect to concurrent reads of the same file.
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

// Reads the entire current contents of @p handle from offset zero. Used both for the
// version-line check at open time and for readAllLines() -- LineStore shards are sized
// for a modest number of records per (engine, arch) pair, so reading the whole file is
// simpler and safer than incremental parsing.
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

// Splits raw file content into whole lines on '\n'. A trailing chunk with no terminating
// newline is an incomplete write (e.g. a reader racing a not-yet-flushed append) and is
// dropped rather than handed to a caller as a well-formed line.
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

// Grants the free functions below controlled access to LineStoreShard's private state.
// A single friend class, rather than friending each free function individually, sidesteps
// the rule that a template friend declaration may not carry a default template argument
// (needed by readAllLines()'s deduced Record parameter) while keeping LineStoreShard's
// constructor and native handle private to this header.
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

/// Opens the shard file at @p path for record access, creating it (and writing
/// @p expectedVersion as its bare first line) if it does not yet exist.
///
/// On an existing file, the first line is compared against @p expectedVersion. A mismatch
/// is reported as LineStoreStatus::VERSION_MISMATCH and no shard is returned -- this is a
/// decline, not a throw, since a version bump from a newer build must never crash an older
/// one reading the same cache directory.
///
/// @param path Path to the shard file; parent directories are assumed to already exist.
/// @param expectedVersion Bare version string written to (or checked against) the first
///     line. Carries no format requirement beyond being one line of text.
/// @return The open shard and LineStoreStatus::OK on success; std::nullopt and a non-Ok
///     status describing the failure otherwise. Never throws.
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

    // Locked only long enough to check (or write) the version line, so two processes
    // racing to create the same shard never both see an empty file and both write a
    // version line. The shard this function returns starts out unlocked -- the caller
    // must call lockLineStore() itself before appendLine().
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

/// Acquires @p shard's per-shard advisory lock, blocking until it is held or acquisition
/// fails. Never throws; a failure to lock (e.g. an incompatible external lock holder, or a
/// filesystem that does not support the underlying primitive) is reported as
/// LineStoreStatus::LOCK_FAILED so the caller can decline gracefully.
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

/// Appends one caller-formatted record line to @p shard.
///
/// @p shard must already hold the lock (see lockLineStore()) -- appendLine() does not
/// acquire it implicitly, so a caller that needs to re-read the file under the lock before
/// deciding what to append (e.g. to detect a concurrent duplicate write) can do so between
/// locking and appending. @p line must not itself contain a newline; the caller is
/// responsible for producing a single logical line (embedding its own delimiters, escaping,
/// or encoding as needed -- LineStore does not inspect or validate line content beyond
/// writing it followed by exactly one newline).
///
/// @return LineStoreStatus::OK on success; LineStoreStatus::IO_ERROR on any write failure.
/// Never throws.
inline LineStoreStatus appendLine(LineStoreShard& shard, std::string_view line)
{
    if(!detail::appendRawLineStoreLine(detail::LineStoreAccess::handle(shard), line))
    {
        return LineStoreStatus::IO_ERROR;
    }
    return LineStoreStatus::OK;
}

/// Reads every record line from @p shard (i.e. every line after the version line) and
/// hands each one to @p parseLine in file order.
///
/// A line for which @p parseLine returns std::nullopt is skipped: it is not returned in
/// the result, and it does not stop later lines from being read or cause the call to
/// report failure. This is the skip-malformed-line contract -- a single corrupt or
/// forward-incompatible record must never poison the rest of an otherwise-good shard.
/// Resolving duplicate-keyed records (last-line-wins or otherwise) is left entirely to the
/// caller; this function preserves file order and nothing else.
///
/// @tparam ParseLine The caller's callback type, deduced from the argument. Declared as a
///     plain callable rather than a `std::function` parameter: `std::function` is a
///     non-deduced context, so a `std::function<std::optional<Record>(std::string_view)>`
///     parameter forces every caller to name Record explicitly or wrap its lambda. Taking
///     the callable directly also avoids `std::function`'s type erasure and its potential
///     allocation on a path that runs once per shard read.
/// @tparam Record The caller's record type, deduced from @p parseLine's
///     `std::optional<Record>` return type.
/// @param shard The shard to read. Does not require the lock to be held: a snapshot read
///     racing a concurrent append is expected to observe either the old or the new state
///     of the file, never a torn line, because appendLine() only ever adds whole lines.
/// @param parseLine Caller-supplied callback converting one raw line into
///     std::optional<Record>; std::nullopt skips the line.
/// @return Every successfully parsed record, in file order, and LineStoreStatus::OK; or an
///     empty vector and a non-Ok status if the shard itself could not be read (a status
///     level failure, distinct from a per-line parse rejection). Never throws.
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
    // Line 0 is the version line, already validated by openLineStore(); record lines
    // start at index 1.
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
