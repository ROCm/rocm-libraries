// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// LineStore is a generic, record-format-agnostic append-only shard file: a bare version
// string on the first line, followed by caller-defined record lines, one per line. It
// treats every line after the first as opaque text and hands parsing/formatting to the
// caller via a callback -- this header must never depend on nlohmann/json or any other
// JSON type; JSON-record callers supply their own encode/decode.
//
// Concurrency. A shard is guarded by two locks that are always taken in the same order:
// an in-process mutex first, then the OS file lock. Both are owned by one registry entry
// per shard, identified by (device, inode) rather than by path, so two names for one file
// -- a symlink, a bind mount, an NFS alias -- share a single entry.
//
// The registry exists because POSIX fcntl locks are a property of the (process, inode)
// pair, not of a descriptor: a second descriptor on the same file does not block against
// its own process, and closing ANY descriptor for that file, or releasing through it,
// drops every lock the process holds on it. Handing every caller its own descriptor
// therefore lets one thread silently strip a lock another thread believes it is holding.
// The registry keeps exactly one descriptor per shard and never closes it, and the mutex
// makes the threads of a process queue before they reach the file lock at all. Win32
// LockFileEx is per-HANDLE and mandatory rather than advisory, so it does not have the
// POSIX failure mode, but the same structure is used on both platforms so that the two
// mean the same thing.
//
// The in-process mutex is held exclusively even for a read. The threads of one process
// share the entry's single descriptor, and with it one file offset, so two concurrent
// reads would interleave their seek-then-read pairs and each get missing or duplicated
// bytes. A POSIX read lock is per (process, inode) as well, so the first reader to
// finish would release the lock every other reader still relies on. Reads are rare and a
// shard is small, so the mutex serializes them instead of counting them.
//
// One shard at a time. Each registry entry owns an independent mutex, so two threads
// that take shards A then B and B then A deadlock. No in-tree path holds one shard while
// opening or locking another; a caller that needs two at once must impose its own order.
//
// Nesting is handled by the registry rather than by caller discipline. Both lock modes
// cover the whole file, so a shared acquisition nested inside an exclusive one is a
// POSIX lock *conversion* -- the exclusive lock is replaced, and the inner release then
// drops it outright, mid-critical-section -- and on Win32 a legal same-handle overlap
// whose first release frees the exclusive lock. Either way the outer writer would lose
// exclusivity. The registry therefore tracks each THREAD's nesting depth and makes an
// inner shared acquisition a no-op, so readAllLines() is safe to call whether or not the
// caller already holds the shard exclusively.
//
// Adapted from miopen::LockFile (projects/miopen/src/include/miopen/lock_file.hpp and
// file_lock.hpp), which solves the same problem, with four deliberate differences: it
// keys its registry on the path rather than on (device, inode); it never states the
// never-close rule, getting it only because its map is never erased; its lock() takes the
// mutex and the file lock simultaneously via std::lock rather than in a fixed order; and
// its timed acquisitions leak the mutex on timeout. This header also reports failure
// through status codes instead of exceptions, and needs the nesting rule above, which
// MIOpen never hits because nothing re-enters LockFile under its own lock.
//
// Failure handling is fail-soft: open/lock/read failures and version mismatches are
// reported through the return value, never an exception. A line the caller's parse
// callback rejects is skipped without affecting any other line. Resolving duplicate-keyed
// records (e.g. last-line-wins) is the caller's job.

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <new>
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
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
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
    /// The shard does not exist. Only openExistingLineStore() reports this; openLineStore()
    /// creates what is missing. An ordinary miss, not a failure.
    NOT_FOUND,
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

// Acquires a whole-file lock, blocking until held. POSIX: fcntl() F_SETLKW with F_WRLCK
// or F_RDLCK. Win32: LockFileEx() over the same range, with LOCKFILE_EXCLUSIVE_LOCK only
// for an exclusive acquisition.
inline LineStoreStatus acquireNativeLineStoreLock(NativeLineStoreHandle handle,
                                                  bool exclusive) noexcept
{
#if defined(_WIN32)
    OVERLAPPED overlapped{};
    const DWORD flags = exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0u;
    if(LockFileEx(handle, flags, 0, MAXDWORD, MAXDWORD, &overlapped) == 0)
    {
        return LineStoreStatus::LOCK_FAILED;
    }
#else
    struct flock fl
    {
    };
    fl.l_type = exclusive ? F_WRLCK : F_RDLCK;
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

inline void releaseNativeLineStoreLock(NativeLineStoreHandle handle) noexcept
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

/// Identifies a shard by what the filesystem considers it to be, not by the name used to
/// reach it. On POSIX that is (st_dev, st_ino); on Win32 the volume serial plus the
/// 128-bit file id from GetFileInformationByHandleEx(FileIdInfo) -- the older
/// GetFileInformationByHandle's 64-bit index is not unique on ReFS or on some remote
/// shares.
struct LineStoreFileId
{
    uint64_t volume = 0;
    uint64_t high = 0;
    uint64_t low = 0;

    bool operator<(const LineStoreFileId& other) const noexcept
    {
        if(volume != other.volume)
        {
            return volume < other.volume;
        }
        if(high != other.high)
        {
            return high < other.high;
        }
        return low < other.low;
    }
};

inline std::optional<LineStoreFileId> queryLineStoreFileId(NativeLineStoreHandle handle) noexcept
{
    LineStoreFileId id;
#if defined(_WIN32)
    FILE_ID_INFO info{};
    if(GetFileInformationByHandleEx(handle, FileIdInfo, &info, sizeof(info)) == 0)
    {
        return std::nullopt;
    }
    id.volume = info.VolumeSerialNumber;
    uint64_t high = 0;
    uint64_t low = 0;
    for(size_t i = 0; i < 8; ++i)
    {
        low |= static_cast<uint64_t>(info.FileId.Identifier[i]) << (8 * i);
        high |= static_cast<uint64_t>(info.FileId.Identifier[i + 8]) << (8 * i);
    }
    id.high = high;
    id.low = low;
#else
    struct stat info
    {
    };
    if(::fstat(handle, &info) != 0)
    {
        return std::nullopt;
    }
    id.volume = static_cast<uint64_t>(info.st_dev);
    id.low = static_cast<uint64_t>(info.st_ino);
#endif
    return id;
}

/// One shard's descriptor and the two locks guarding it. Owned by the registry for the
/// life of the process; see the never-close rule in the header comment.
struct LineStoreRegistryEntry
{
    NativeLineStoreHandle handle = INVALID_LINE_STORE_HANDLE;
    /// Serializes this process's threads ahead of the OS file lock. Always taken first,
    /// and always exclusively -- see the concurrency note at the top of this header for
    /// why a shared read cannot share this entry's descriptor.
    std::mutex accessMutex;
};

/// Per-thread nesting depth for one shard.
///
/// Nesting is a property of the calling THREAD, not of the process: a second thread
/// reaching a shard another thread holds must queue on accessMutex, not be mistaken for
/// a re-entrant acquisition and waved through. The registry entry is shared, so the depth
/// counter cannot live there; it is keyed per (thread, entry) here.
inline size_t& lineStoreThreadDepth(const LineStoreRegistryEntry* entry)
{
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static thread_local std::map<const LineStoreRegistryEntry*, size_t> s_depths;
    return s_depths[entry];
}

/// The process's shard registry. Entries are created on first use and never erased: an
/// entry's descriptor must outlive every lock taken through it, and erasing one would
/// close a descriptor that another thread is relying on -- the exact failure the registry
/// exists to prevent.
inline std::map<LineStoreFileId, std::unique_ptr<LineStoreRegistryEntry>>& lineStoreRegistry()
{
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static std::map<LineStoreFileId, std::unique_ptr<LineStoreRegistryEntry>> s_entries;
    return s_entries;
}

inline std::mutex& lineStoreRegistryMutex()
{
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex s_mutex;
    return s_mutex;
}

/// Identifies the file at @p path WITHOUT opening it, so a shard already known to this
/// process is never opened a second time.
///
/// Closing a descriptor is what makes a duplicate open dangerous: on POSIX, closing any
/// descriptor for a file releases every lock the process holds on that file, so even
/// opening a redundant descriptor and immediately closing it would strip a lock another
/// thread is holding. The only safe duplicate is the one that never happens.
///
/// @return nullopt if @p path does not exist yet or cannot be identified.
inline std::optional<LineStoreFileId>
    peekLineStoreFileId(const std::filesystem::path& path) noexcept
{
#if defined(_WIN32)
    const HANDLE probe = CreateFileW(path.wstring().c_str(),
                                     FILE_READ_ATTRIBUTES,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                     nullptr,
                                     OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL,
                                     nullptr);
    if(probe == INVALID_HANDLE_VALUE)
    {
        return std::nullopt;
    }
    const auto id = queryLineStoreFileId(probe);
    // Safe to close: Win32 locks are per-HANDLE, and this handle never took one.
    CloseHandle(probe);
    return id;
#else
    struct stat info
    {
    };
    if(::stat(path.c_str(), &info) != 0)
    {
        return std::nullopt;
    }
    LineStoreFileId id;
    id.volume = static_cast<uint64_t>(info.st_dev);
    id.low = static_cast<uint64_t>(info.st_ino);
    return id;
#endif
}

/// Opens the shard file at @p path, creating it if absent.
///
/// Win32 asks for write access as well as append access, which POSIX gets from O_RDWR.
/// Truncating a torn first write (see openLineStore()) has to go through THIS handle: it
/// is the one holding the shard's lock, and a second handle would be subject to that lock
/// on Win32 and would drop it outright on POSIX. The cost is that a Win32 write now lands
/// at the file pointer rather than atomically at end-of-file, so appendRawLineStoreLine()
/// seeks first. Nothing relies on lock-free append atomicity: every writer holds the
/// exclusive lock, and every reader holds the shared one.
inline NativeLineStoreHandle openLineStoreHandle(const std::filesystem::path& path)
{
#if defined(_WIN32)
    return CreateFileW(path.wstring().c_str(),
                       FILE_GENERIC_READ | FILE_GENERIC_WRITE,
                       FILE_SHARE_READ | FILE_SHARE_WRITE,
                       nullptr,
                       OPEN_ALWAYS,
                       FILE_ATTRIBUTE_NORMAL,
                       nullptr);
#else
    return ::open(path.c_str(), O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
#endif
}

/// The registry entry for the shard at @p path, opening and registering the file on its
/// first use in this process.
///
/// Identify, open and register run as ONE critical section under the registry mutex.
/// Split, two threads racing the same first use both open the file; the loser then closes
/// its redundant descriptor, and on POSIX closing any descriptor for a file drops every
/// lock the process holds on it -- including the one the winner is inside. Serialized,
/// the loser finds the winner's entry before it opens anything.
///
/// @return nullptr if the file could not be opened or identified, in which case the
///     caller must decline.
inline LineStoreRegistryEntry* openOrFindLineStoreEntry(const std::filesystem::path& path) noexcept
{
    NativeLineStoreHandle handle = INVALID_LINE_STORE_HANDLE;
    try
    {
        const std::lock_guard<std::mutex> guard(lineStoreRegistryMutex());
        auto& entries = lineStoreRegistry();

        if(const auto known = peekLineStoreFileId(path))
        {
            const auto found = entries.find(*known);
            if(found != entries.end())
            {
                return found->second.get();
            }
        }

        handle = openLineStoreHandle(path);
        if(!isValidLineStoreHandle(handle))
        {
            return nullptr;
        }

        const auto id = queryLineStoreFileId(handle);
        if(!id)
        {
            // Nothing has locked through this descriptor, and no registered descriptor
            // for its file exists -- the registry mutex is held and the lookup above
            // found none -- so closing it drops no lock.
            closeLineStoreHandle(handle);
            return nullptr;
        }

        auto entry = std::make_unique<LineStoreRegistryEntry>();
        entry->handle = handle;
        const auto [position, inserted] = entries.emplace(*id, std::move(entry));
        if(!inserted)
        {
            // peekLineStoreFileId() failed transiently for a file that IS registered, so
            // the descriptor just opened is redundant. It stays open deliberately: a
            // registered descriptor for this inode already exists, and on POSIX closing
            // any descriptor for a file drops every lock the process holds on it --
            // including one another thread is inside. Retaining it costs one descriptor
            // per transient stat failure, which is the cheaper of the two defects.
            return position->second.get();
        }
        // Owned by the registry from here on, which never closes a descriptor. Blanked
        // only now: emplace() can throw, and until it returns this is still the only
        // reference to the descriptor the catch block below needs to close.
        handle = INVALID_LINE_STORE_HANDLE;
        return position->second.get();
    }
    catch(...)
    {
        if(isValidLineStoreHandle(handle))
        {
            closeLineStoreHandle(handle);
        }
        return nullptr;
    }
}

/// Opens @p path for reading without creating it, in the same open handle used to
/// identify it.
///
/// The read-only counterpart of openLineStoreHandle().
///
/// O_RDWR, not O_RDONLY: a shared fcntl lock needs a readable descriptor, and the
/// registry may later hand this same descriptor to a writer for the inode. Omitting
/// O_CREAT is the point.
inline NativeLineStoreHandle openExistingLineStoreHandle(const std::filesystem::path& path)
{
#if defined(_WIN32)
    return CreateFileW(path.wstring().c_str(),
                       FILE_GENERIC_READ | FILE_GENERIC_WRITE,
                       FILE_SHARE_READ | FILE_SHARE_WRITE,
                       nullptr,
                       OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL,
                       nullptr);
#else
    return ::open(path.c_str(), O_RDWR | O_APPEND | O_CLOEXEC);
#endif
}

/// The registry entry for an existing shard at @p path, registering the file on its first
/// use in this process but never creating it.
///
/// The read-only counterpart of openOrFindLineStoreEntry(), and serialized the same way
/// and for the same reason: identify, open and register are one critical section under
/// the registry mutex, so two threads racing the same first use cannot both open the
/// file. See that function for why a redundant descriptor cannot simply be closed.
///
/// @param[out] absent Set when the shard does not exist, distinguishing an ordinary miss
///     from a genuine open failure. Untouched on success.
/// @return nullptr if the file is absent, could not be opened, or could not be
///     identified, in which case the caller must decline.
inline LineStoreRegistryEntry* findExistingLineStoreEntry(const std::filesystem::path& path,
                                                          bool& absent) noexcept
{
    NativeLineStoreHandle handle = INVALID_LINE_STORE_HANDLE;
    try
    {
        const std::lock_guard<std::mutex> guard(lineStoreRegistryMutex());
        auto& entries = lineStoreRegistry();

        if(const auto known = peekLineStoreFileId(path))
        {
            const auto found = entries.find(*known);
            if(found != entries.end())
            {
                return found->second.get();
            }
        }

        handle = openExistingLineStoreHandle(path);
        if(!isValidLineStoreHandle(handle))
        {
#if defined(_WIN32)
            absent
                = GetLastError() == ERROR_FILE_NOT_FOUND || GetLastError() == ERROR_PATH_NOT_FOUND;
#else
            absent = errno == ENOENT;
#endif
            return nullptr;
        }

        const auto id = queryLineStoreFileId(handle);
        if(!id)
        {
            // Safe to close: the registry mutex is held, the lookup above found no
            // registered descriptor for this file, and nothing has locked through this
            // one, so closing it drops no lock.
            closeLineStoreHandle(handle);
            return nullptr;
        }

        auto entry = std::make_unique<LineStoreRegistryEntry>();
        entry->handle = handle;
        const auto [position, inserted] = entries.emplace(*id, std::move(entry));
        if(!inserted)
        {
            // See openOrFindLineStoreEntry(): the redundant descriptor stays open
            // deliberately, because closing it would drop locks held through the
            // registered one.
            return position->second.get();
        }
        handle = INVALID_LINE_STORE_HANDLE;
        return position->second.get();
    }
    catch(...)
    {
        if(isValidLineStoreHandle(handle))
        {
            closeLineStoreHandle(handle);
        }
        return nullptr;
    }
}

} // namespace detail

/// A handle to one open, version-checked shard file, obtained only via openLineStore() --
/// callers never see a half-open or version-mismatched shard through this type.
///
/// Non-owning: the descriptor belongs to the process-wide registry and outlives every
/// shard object. Destroying a shard releases any lock it still holds but never closes the
/// descriptor, because closing one descriptor for a file drops every fcntl lock the
/// process holds on that file.
///
/// Thread affinity: a LOCKED shard belongs to the thread that locked it. Nesting depth is
/// tracked per thread, so moving a locked shard to another thread and releasing it there
/// finds depth 0, returns early, and leaves both the mutex and the OS lock held for the
/// life of the process. Move a shard only while it is unlocked, or lock and release it on
/// one thread. An unlocked shard moves freely.
class LineStoreShard
{
public:
    LineStoreShard(const LineStoreShard&) = delete;
    LineStoreShard& operator=(const LineStoreShard&) = delete;

    LineStoreShard(LineStoreShard&& other) noexcept
        : _entry(other._entry)
        , _locked(other._locked)
    {
        other._entry = nullptr;
        other._locked = false;
    }

    LineStoreShard& operator=(LineStoreShard&& other) noexcept
    {
        if(this != &other)
        {
            releaseIfLocked();
            _entry = other._entry;
            _locked = other._locked;
            other._entry = nullptr;
            other._locked = false;
        }
        return *this;
    }

    ~LineStoreShard()
    {
        releaseIfLocked();
    }

    /// True while this handle holds the shard's exclusive lock.
    bool isLocked() const noexcept
    {
        return _locked;
    }

private:
    friend class detail::LineStoreAccess;

    LineStoreShard() = default;

    void releaseIfLocked() noexcept;

    detail::LineStoreRegistryEntry* _entry = nullptr;
    bool _locked = false;
};

namespace detail
{

// Appends @p line plus a trailing '\n'.
//
// POSIX writes through an O_APPEND descriptor, which places every write at end-of-file
// atomically. Win32 has no equivalent for a handle that also carries write access (see
// openLineStoreHandle()), so the file pointer is moved to the end first; the caller's
// exclusive lock is what makes that pair safe against another writer.
//
// Not noexcept: the staging buffer allocates, and a shard is allowed to grow without
// bound, so a failed allocation must surface as IO_ERROR rather than terminate the host.
inline bool appendRawLineStoreLine(NativeLineStoreHandle handle, std::string_view line)
{
    std::string buffer;
    buffer.reserve(line.size() + 1);
    buffer.append(line);
    buffer.push_back('\n');

    size_t written = 0;
#if defined(_WIN32)
    LARGE_INTEGER end{};
    if(SetFilePointerEx(handle, end, nullptr, FILE_END) == 0)
    {
        return false;
    }
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
        if(chunk < 0)
        {
            if(errno == EINTR)
            {
                continue;
            }
            return false;
        }
        if(chunk == 0)
        {
            return false;
        }
        written += static_cast<size_t>(chunk);
    }
#endif
    return true;
}

// Discards a shard's entire contents through @p handle, which must already hold the
// shard's exclusive lock. Truncating by path instead would need a second handle: on Win32
// that handle is subject to the lock this one holds, and on POSIX opening and closing one
// drops every lock this process holds on the file.
inline bool truncateLineStoreToEmpty(NativeLineStoreHandle handle) noexcept
{
#if defined(_WIN32)
    LARGE_INTEGER origin{};
    if(SetFilePointerEx(handle, origin, nullptr, FILE_BEGIN) == 0)
    {
        return false;
    }
    return SetEndOfFile(handle) != 0;
#else
    return ::ftruncate(handle, 0) == 0;
#endif
}

// Reads the entire contents of @p handle from offset zero; used for both the version-line
// check and readAllLines(), since shards are small enough that a full read is simplest.
//
// Not noexcept: the accumulating string allocates, and the shard has no size bound.
inline std::optional<std::string> readAllLineStoreBytes(NativeLineStoreHandle handle)
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
            if(errno == EINTR)
            {
                continue;
            }
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
// incomplete write (a reader racing a not-yet-flushed append) and is dropped. A '\r'
// immediately before the newline is dropped too, so a shard that has been through a
// text-mode copy still reads.
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
        size_t end = newlinePos;
        if(end > start && content[end - 1] == '\r')
        {
            --end;
        }
        lines.emplace_back(content, start, end - start);
        start = newlinePos + 1;
    }
    return lines;
}

// Grants the free functions below access to LineStoreShard's private state (a template
// friend can't carry readAllLines()'s default Record argument).
class LineStoreAccess
{
public:
    static LineStoreShard make()
    {
        return {};
    }

    static LineStoreRegistryEntry* entry(const LineStoreShard& shard) noexcept
    {
        return shard._entry;
    }

    static void setEntry(LineStoreShard& shard, LineStoreRegistryEntry* entry) noexcept
    {
        shard._entry = entry;
    }

    static NativeLineStoreHandle handle(const LineStoreShard& shard) noexcept
    {
        return shard._entry == nullptr ? INVALID_LINE_STORE_HANDLE : shard._entry->handle;
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

/// Takes the shard's in-process mutex and then its file lock, in that order, and records
/// this thread's nesting depth.
///
/// A shared request made by a thread that already holds the shard exclusively is
/// satisfied by the exclusive lock and recorded as extra depth: re-acquiring at the OS
/// level would convert the whole-file lock down to shared on POSIX, and the matching
/// release would then drop it entirely, stripping exclusivity from the outer critical
/// section. A request from any OTHER thread is not nesting and blocks normally.
inline LineStoreStatus acquireLineStoreLock(LineStoreRegistryEntry& entry, bool exclusive) noexcept
{
    try
    {
        size_t& depth = lineStoreThreadDepth(&entry);
        if(depth > 0)
        {
            if(!exclusive)
            {
                ++depth;
                return LineStoreStatus::OK;
            }
            // An exclusive request nested inside a lock this thread already holds would
            // deadlock against accessMutex; no call path does it, and declining is safer
            // than hanging.
            return LineStoreStatus::LOCK_FAILED;
        }
    }
    catch(...)
    {
        return LineStoreStatus::LOCK_FAILED;
    }

    // Exclusive for both modes; see accessMutex's declaration. The native lock still
    // distinguishes them, so other PROCESSES may read this shard concurrently.
    try
    {
        entry.accessMutex.lock();
    }
    catch(...)
    {
        // A failed acquisition surfaces as std::system_error; this function is noexcept,
        // so letting it out would terminate the host over a cache lock.
        return LineStoreStatus::LOCK_FAILED;
    }

    const auto status = acquireNativeLineStoreLock(entry.handle, exclusive);
    if(status != LineStoreStatus::OK)
    {
        // Release the mutex on the failure path; leaving it held would wedge every later
        // acquisition on this shard.
        entry.accessMutex.unlock();
        return status;
    }

    lineStoreThreadDepth(&entry) = 1;
    return LineStoreStatus::OK;
}

/// Unwinds one acquisition, releasing the OS lock and the mutex only when this thread's
/// outermost one is released.
inline void releaseLineStoreLock(LineStoreRegistryEntry& entry) noexcept
{
    size_t* depth = nullptr;
    try
    {
        depth = &lineStoreThreadDepth(&entry);
    }
    catch(...)
    {
        return;
    }

    if(*depth == 0)
    {
        return;
    }
    if(--*depth > 0)
    {
        return;
    }

    releaseNativeLineStoreLock(entry.handle);
    entry.accessMutex.unlock();
}

/// Releases one acquisition when it goes out of scope, however the scope is left.
///
/// readAllLines() hands each line to a caller-supplied callback, which may throw anything
/// at all -- including a type no catch clause here names. A leaked shard lock is not a
/// local failure: the in-process mutex and the OS file lock both stay held for the life
/// of the process, wedging every later reader and writer, in this process and others.
class LineStoreLockScope
{
public:
    explicit LineStoreLockScope(LineStoreRegistryEntry& entry) noexcept
        : _entry(&entry)
    {
    }
    LineStoreLockScope(const LineStoreLockScope&) = delete;
    LineStoreLockScope& operator=(const LineStoreLockScope&) = delete;
    LineStoreLockScope(LineStoreLockScope&&) = delete;
    LineStoreLockScope& operator=(LineStoreLockScope&&) = delete;

    ~LineStoreLockScope()
    {
        releaseLineStoreLock(*_entry);
    }

private:
    LineStoreRegistryEntry* _entry;
};

} // namespace detail

inline void LineStoreShard::releaseIfLocked() noexcept
{
    if(_locked && _entry != nullptr)
    {
        detail::releaseLineStoreLock(*_entry);
        _locked = false;
    }
}

/// Opens an existing shard at @p path without creating it, for callers that only read.
///
/// Prefer this to openLineStore() on any read-only path. openLineStore() is O_CREAT and
/// stamps a version line, so a lookup that misses materializes a shard nothing has
/// written, and the descriptor registry holds that fd for the life of the process. Where
/// the key space is unbounded, that costs one empty file and one open descriptor per key
/// ever looked up. Creating state belongs to the write path.
///
/// @return The open shard and OK; nullopt and NOT_FOUND when the shard does not exist;
///     nullopt and a non-OK status on any other failure, including VERSION_MISMATCH.
inline std::pair<std::optional<LineStoreShard>, LineStoreStatus>
    openExistingLineStore(const std::filesystem::path& path, std::string_view expectedVersion)
{
    bool absent = false;
    detail::LineStoreRegistryEntry* const entry = detail::findExistingLineStoreEntry(path, absent);
    if(entry == nullptr)
    {
        return {std::nullopt, absent ? LineStoreStatus::NOT_FOUND : LineStoreStatus::OPEN_FAILED};
    }

    LineStoreShard shard = detail::LineStoreAccess::make();
    detail::LineStoreAccess::setEntry(shard, entry);

    // Shared, not exclusive: this call never writes, so concurrent readers need not
    // serialize against each other.
    if(detail::acquireLineStoreLock(*entry, false) != LineStoreStatus::OK)
    {
        return {std::nullopt, LineStoreStatus::LOCK_FAILED};
    }
    const detail::LineStoreLockScope held(*entry);

    // Allocates against a shard with no size bound, so it is wrapped: an escaping
    // bad_alloc would abandon the open with the lock still held.
    try
    {
        const auto content = detail::readAllLineStoreBytes(entry->handle);
        if(!content)
        {
            return {std::nullopt, LineStoreStatus::IO_ERROR};
        }

        const auto lines = detail::splitLineStoreLines(*content);
        if(lines.empty())
        {
            // Present but holding no complete line: a torn first write. openLineStore()
            // repairs it; a reader has nothing to read and must not repair anything.
            return {std::nullopt, LineStoreStatus::NOT_FOUND};
        }
        if(lines.front() != expectedVersion)
        {
            return {std::nullopt, LineStoreStatus::VERSION_MISMATCH};
        }
    }
    catch(const std::exception&)
    {
        return {std::nullopt, LineStoreStatus::IO_ERROR};
    }

    return {std::optional<LineStoreShard>(std::move(shard)), LineStoreStatus::OK};
}

/// Opens the shard file at @p path, creating it (and writing @p expectedVersion as its
/// first line) if absent. An existing file whose first line doesn't match
/// @p expectedVersion returns VERSION_MISMATCH rather than throwing, so a version bump
/// never crashes an older reader of the same cache directory.
///
/// A file that is non-empty but holds no complete line is a first write that was
/// interrupted. The fragment is truncated away before the version line is stamped, since
/// stamping after it would leave the fragment as line 0 and report VERSION_MISMATCH
/// forever with nothing able to repair it. A truncation the platform refuses reports
/// IO_ERROR rather than producing that unrepairable shard.
///
/// @return The open shard and OK on success; nullopt and a non-OK status otherwise.
inline std::pair<std::optional<LineStoreShard>, LineStoreStatus>
    openLineStore(const std::filesystem::path& path, std::string_view expectedVersion)
{
    detail::LineStoreRegistryEntry* const entry = detail::openOrFindLineStoreEntry(path);
    if(entry == nullptr)
    {
        return {std::nullopt, LineStoreStatus::OPEN_FAILED};
    }

    LineStoreShard shard = detail::LineStoreAccess::make();
    detail::LineStoreAccess::setEntry(shard, entry);

    // Locked only long enough to check/write the version line, so two racing creators
    // never both write one. The returned shard starts unlocked.
    if(detail::acquireLineStoreLock(*entry, true) != LineStoreStatus::OK)
    {
        return {std::nullopt, LineStoreStatus::LOCK_FAILED};
    }
    const detail::LineStoreLockScope held(*entry);

    // Everything below allocates against a shard with no size bound, so it is wrapped: an
    // escaping bad_alloc would abandon the open, and the exclusive lock with it.
    try
    {
        const auto content = detail::readAllLineStoreBytes(entry->handle);
        if(!content)
        {
            return {std::nullopt, LineStoreStatus::IO_ERROR};
        }

        const auto lines = detail::splitLineStoreLines(*content);
        if(lines.empty())
        {
            // Empty, or a torn first write that never reached its newline. Clear the
            // fragment before stamping, through the same handle that holds the lock.
            if(!content->empty() && !detail::truncateLineStoreToEmpty(entry->handle))
            {
                return {std::nullopt, LineStoreStatus::IO_ERROR};
            }
            if(!detail::appendRawLineStoreLine(entry->handle, expectedVersion))
            {
                return {std::nullopt, LineStoreStatus::IO_ERROR};
            }
        }
        else if(lines.front() != expectedVersion)
        {
            return {std::nullopt, LineStoreStatus::VERSION_MISMATCH};
        }
    }
    catch(...)
    {
        return {std::nullopt, LineStoreStatus::IO_ERROR};
    }

    return {std::optional<LineStoreShard>(std::move(shard)), LineStoreStatus::OK};
}

/// Acquires @p shard's exclusive lock, blocking until held or failed. A failure (e.g. an
/// incompatible external lock holder) reports LOCK_FAILED rather than throwing.
inline LineStoreStatus lockLineStore(LineStoreShard& shard)
{
    auto* entry = detail::LineStoreAccess::entry(shard);
    if(entry == nullptr)
    {
        return LineStoreStatus::LOCK_FAILED;
    }
    const auto status = detail::acquireLineStoreLock(*entry, true);
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
    auto* entry = detail::LineStoreAccess::entry(shard);
    if(entry != nullptr)
    {
        detail::releaseLineStoreLock(*entry);
    }
    detail::LineStoreAccess::setLocked(shard, false);
}

/// Appends one caller-formatted record line to @p shard, which must already hold the lock
/// (see lockLineStore()); appendLine() does not acquire it itself. @p line must not
/// contain a newline: a record carrying one would be read back as two, so callers whose
/// encoding can emit newlines must escape them.
///
/// @return OK on success; IO_ERROR on any write failure, including a failed allocation.
///     Never throws.
inline LineStoreStatus appendLine(LineStoreShard& shard, std::string_view line)
{
    if(line.find('\n') != std::string_view::npos)
    {
        return LineStoreStatus::IO_ERROR;
    }

    try
    {
        if(!detail::appendRawLineStoreLine(detail::LineStoreAccess::handle(shard), line))
        {
            return LineStoreStatus::IO_ERROR;
        }
    }
    catch(const std::exception&)
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
/// @param shard Takes the shard's shared FILE lock for the duration of the read, which is
///     what makes the read legal on Win32, where the file lock is mandatory rather than
///     advisory. Other processes may read concurrently; the threads of this one may not,
///     since they share the descriptor. Safe to call while already holding the shard
///     exclusively: the acquisition is recognised as nested and the outer lock is left
///     intact.
/// @return Parsed records in file order and OK; an empty vector and non-OK status if the
///     shard could not be read. Never throws.
template <typename ParseLine,
          typename Record = typename std::invoke_result_t<ParseLine, std::string_view>::value_type>
std::pair<std::vector<Record>, LineStoreStatus> readAllLines(const LineStoreShard& shard,
                                                             ParseLine parseLine)
{
    auto* entry = detail::LineStoreAccess::entry(shard);
    if(entry == nullptr)
    {
        return {{}, LineStoreStatus::IO_ERROR};
    }

    if(detail::acquireLineStoreLock(*entry, false) != LineStoreStatus::OK)
    {
        return {{}, LineStoreStatus::LOCK_FAILED};
    }
    const detail::LineStoreLockScope held(*entry);

    try
    {
        const auto content = detail::readAllLineStoreBytes(entry->handle);
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
    catch(...)
    {
        // A shard may grow without bound, so reading one can exhaust memory, and
        // @p parseLine is caller code that may throw anything. Both are declines like any
        // other, not a reason to terminate the host process.
        return {{}, LineStoreStatus::IO_ERROR};
    }
}

} // namespace hipdnn_data_sdk::utilities
