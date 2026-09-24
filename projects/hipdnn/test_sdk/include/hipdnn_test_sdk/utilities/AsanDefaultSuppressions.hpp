// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Backing logic for __asan_default_suppressions(), which src/AsanDefaultSuppressions.cpp defines.
//
// It lives in a header so it is reachable from a test translation unit, and is deliberately not
// guarded by ADDRESS_SANITIZER so the tests run in every configuration rather than only under ASAN.
//
// Everything here runs inside the ASan runtime's start-up, before InitializeAsanInterceptors(), so
// it must not call libc. An intercepted function's real-function pointer is still null that early
// and calling one jumps to address 0. That rules out getenv(), strlen(), strcmp() and memcmp(), and
// equally memcpy()/memmove(), so the environment is read through raw syscalls, compared by hand,
// and never copied.
namespace hipdnn_test_sdk::utilities::asan
{

// Upstream rocBLAS/Tensile data race on the lazy placeholder-library load: a solution matching
// table is read while an std::async loader thread deserializes into it and reallocates the backing
// storage. AIBTINFRA-48, ROCm/rocm-libraries#8869.
inline constexpr const char* K_DEFAULT_SUPPRESSIONS = "interceptor_via_fun:*findBestKeyMatch*\n";

inline constexpr const char* K_DISABLE_VARIABLE = "HIPDNN_ASAN_NO_DEFAULT_SUPPRESSIONS";

// Looks for an assignment to `name` in a NUL-separated environment block that arrives in pieces.
//
// The block is "A=1\0B=2\0", and a piece may end anywhere in it, including inside the name being
// searched for. The match state carries across pieces, so where they divide is not something the
// matcher can observe.
//
// That state is also what anchors a match to an entry boundary: `_candidate` stays true only while
// the bytes since the last NUL are still a prefix of `name`, so the name appearing inside a value --
// `SOMEVAR=HIPDNN_ASAN_NO_DEFAULT_SUPPRESSIONS=1` -- does not count as the variable being set.
class EnvironFlagScanner
{
public:
    explicit EnvironFlagScanner(const char* name)
        : _name(name)
    {
    }

    void feed(const char* chunk, long length)
    {
        if(chunk == nullptr || _name == nullptr || length <= 0 || _found)
        {
            return;
        }

        for(long i = 0; i < length; ++i)
        {
            const char c = chunk[i];

            if(c == '\0')
            {
                // Entry ended without matching; the next byte begins a fresh one.
                _matched = 0;
                _candidate = true;
            }
            else if(!_candidate)
            {
                continue; // This entry is already ruled out; skip to its terminator.
            }
            else if(_name[_matched] == '\0')
            {
                // The whole name has matched. Require the '=' so a longer variable sharing this
                // name as a prefix does not count.
                if(c == '=')
                {
                    _found = true;
                    return;
                }
                _candidate = false;
            }
            else if(c == _name[_matched])
            {
                ++_matched;
            }
            else
            {
                _candidate = false;
            }
        }
    }

    bool found() const
    {
        return _found;
    }

private:
    const char* _name;
    long _matched = 0;
    bool _candidate = true;
    bool _found = false;
};

// Searches a complete NUL-separated environment block for an assignment to `name`.
//
// `buffer` needs no terminator of its own -- every read is bounded by `length` -- so a truncated
// block is scanned safely rather than running off the end.
inline bool environBufferHasFlag(const char* buffer, long length, const char* name)
{
    EnvironFlagScanner scanner(name);
    scanner.feed(buffer, length);
    return scanner.found();
}

#if defined(__linux__)

// Read granularity. The environment is consumed in blocks until the kernel reports end of file, so
// this bounds stack use rather than how much can be seen.
inline constexpr std::size_t K_ENVIRON_BLOCK_SIZE = 1024;

// Backstop so the loop ends even if read() never reports end of file. It must sit above any
// environment the kernel will accept, or it becomes a silent truncation of its own: the ceiling is
// RLIMIT_STACK/4, which is 2 MiB at the usual 8 MiB stack but rises with it. At 1 KiB per block
// this allows 64 MiB, well past a raised stack limit.
inline constexpr long K_MAX_ENVIRON_BLOCKS = 65536;

// Reads the environment as the kernel recorded it at exec.
//
// That snapshot is what /proc/self/environ exposes, so a variable introduced later with setenv()
// does not appear here. The override therefore has to be set before the process starts.
inline bool environmentFlagSet(const char* name)
{
    EnvironFlagScanner scanner(name);

    const long fd = syscall(SYS_openat, AT_FDCWD, "/proc/self/environ", O_RDONLY, 0);
    if(fd < 0)
    {
        return false;
    }

    // Deliberately left uninitialized: value-initializing it would emit a memset call, and memset
    // is one of the interceptors that is not yet wired up when this runs.
    std::array<char, K_ENVIRON_BLOCK_SIZE> block; // NOLINT(cppcoreguidelines-pro-type-member-init)

    for(long blocks = 0; blocks < K_MAX_ENVIRON_BLOCKS; ++blocks)
    {
        // A short read is not end of file, so this loops on the count rather than comparing it to
        // the block size. Zero is end of file; negative is an error, and abandoning the search is
        // the same outcome as not finding the variable.
        const long length = syscall(SYS_read, fd, block.data(), block.size());
        if(length <= 0)
        {
            break;
        }

        scanner.feed(block.data(), length);
        if(scanner.found())
        {
            break;
        }
    }

    syscall(SYS_close, fd);
    return scanner.found();
}

#endif // __linux__

// The suppression text the ASan hook hands back.
//
// The HIPDNN_ASAN_NO_DEFAULT_SUPPRESSIONS override is Linux-only. Reading the environment this
// early needs a platform-specific route that cannot be exercised without a Windows machine, and
// getting it wrong costs every ASAN binary a silent start-up crash, so elsewhere the suppressions
// are unconditional and disabling them needs a rebuild.
inline const char* defaultSuppressions()
{
#if defined(__linux__)
    if(environmentFlagSet(K_DISABLE_VARIABLE))
    {
        return "";
    }
#endif
    return K_DEFAULT_SUPPRESSIONS;
}

} // namespace hipdnn_test_sdk::utilities::asan
