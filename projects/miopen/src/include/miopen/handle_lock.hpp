// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef GUARD_MIOPEN_HANDLE_LOCK_HPP
#define GUARD_MIOPEN_HANDLE_LOCK_HPP

#include <miopen/config.h>
#include <miopen/filesystem.hpp>
#include <miopen/file_lock.hpp>
#include <miopen/logger.hpp>
#include <miopen/unique_path.hpp>

#include <chrono>
#include <fstream>
#include <mutex>

namespace miopen {

#define MIOPEN_DECLARE_HANDLE_MUTEX(x)                                    \
    struct x                                                              \
    {                                                                     \
        static const char* env::value() { return ".miopen-" #x ".lock"; } \
    };

#if MIOPEN_GPU_SYNC
MIOPEN_DECLARE_HANDLE_MUTEX(gpu_handle_mutex)
#define MIOPEN_HANDLE_LOCK                                    \
    auto MIOPEN_PP_CAT(miopen_handle_lock_guard_, __LINE__) = \
        miopen::get_handle_lock(miopen::gpu_handle_mutex{});
#else
#define MIOPEN_HANDLE_LOCK
#endif

inline fs::path get_handle_lock_path(const char* name)
{
    const auto p = fs::current_path() / name;
    if(!fs::exists(p))
    {
        const auto tmp = fs::current_path() / miopen::unique_path();
        std::ofstream{tmp}; // NOLINT(bugprone-unused-raii)
        fs::rename(tmp, p);
    }
    return p;
}

struct handle_mutex
{
    std::recursive_timed_mutex m;
    miopen::file_lock flock;

    handle_mutex(const std::string& name) : flock(name) {}

    bool try_lock() { return std::try_lock(m, flock) != 0; }

    void lock() { std::lock(m, flock); }

    template <class Duration>
    bool try_lock_for(Duration d)
    {
        return m.try_lock_for(d) && flock.timed_lock(std::chrono::steady_clock::now() +
                                                     std::chrono::duration_cast<Duration>(d));
    }

    template <class Point>
    bool try_lock_until(Point p)
    {
        return m.try_lock_for(p - std::chrono::system_clock::now());
    }

    void unlock()
    {
        flock.unlock();
        m.unlock();
    }
};

template <class T>
inline std::unique_lock<handle_mutex> get_handle_lock(T, int timeout = 120)
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static handle_mutex m{get_handle_lock_path(T::value())};
    return {m, std::chrono::seconds{timeout}};
}

} // namespace miopen

#endif // GUARD_MIOPEN_HANDLE_LOCK_HPP
