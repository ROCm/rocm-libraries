/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef GUARD_MLOPEN_BINARY_CACHE_HPP
#define GUARD_MLOPEN_BINARY_CACHE_HPP

#include <miopen/config.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/filesystem.hpp>

#include <string>
#include <vector>

namespace miopen {

MIOPEN_INTERNALS_EXPORT bool IsCacheDisabled();

MIOPEN_INTERNALS_EXPORT fs::path
GetCacheFile(const std::string& device, const fs::path& name, const std::string& args);

MIOPEN_INTERNALS_EXPORT fs::path GetCachePath(bool is_system);

#if !MIOPEN_ENABLE_SQLITE_KERN_CACHE
MIOPEN_INTERNALS_EXPORT fs::path LoadBinary(const TargetProperties& target,
                                            std::size_t num_cu,
                                            const fs::path& name,
                                            const std::string& args);

fs::path SaveBinary(const fs::path& binary_path,
                    const TargetProperties& target,
                    const fs::path& name,
                    const std::string& args);
#else
MIOPEN_INTERNALS_EXPORT std::vector<char> LoadBinary(const TargetProperties& target,
                                                     std::size_t num_cu,
                                                     const fs::path& name,
                                                     const std::string& args);

MIOPEN_INTERNALS_EXPORT void SaveBinary(const std::vector<char>& hsaco,
                                        const TargetProperties& target,
                                        std::size_t num_cu,
                                        const fs::path& name,
                                        const std::string& args);
#endif

#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
/// Groups the kernel binaries stored by SaveBinary into bounded SQLite transactions for as long
/// as the object lives, instead of committing every binary on its own.
///
/// Callers of SaveBinary do not change: while any KernelCacheBatch is alive in the process, each
/// SaveBinary call, from any thread, prepares its record and appends it to the pending buffer of
/// the user kernel database, then returns. The buffer is committed in one transaction whenever it
/// reaches the KernDb::BatchLimits bounds (record count, bytes, age), and everything still pending
/// is committed when a batch ends. Pending binaries are visible to LoadBinary in this process
/// immediately; other processes see them once committed. See KernDb for the failure contract.
///
/// Keep batches around bounded units of work (one tuning search, one precompilation pass).
class MIOPEN_INTERNALS_EXPORT KernelCacheBatch
{
public:
    KernelCacheBatch();
    /// Ends the batch if Commit() was not called, e.g. during stack unwinding. Commit failures
    /// are logged as errors since a destructor cannot report them.
    ~KernelCacheBatch();
    KernelCacheBatch(const KernelCacheBatch&)            = delete;
    KernelCacheBatch& operator=(const KernelCacheBatch&) = delete;

    /// Ends the batch and commits all pending binaries. Throws if a commit fails.
    void Commit();
    /// Commits pending binaries that have waited longer than the age bound. Lets a long-running
    /// owner keep the bound while no new binaries are stored. Throws if a commit fails.
    void FlushExpired() const;

private:
    bool active = true;
};
#else
class KernelCacheBatch
{
public:
    void Commit() {}
    void FlushExpired() const {}
};
#endif

#ifdef MIOPEN_BUILD_TESTING
namespace testing {
/// Reset cached paths for testing purposes
/// This allows tests to reinitialize paths with different mocks/env vars
MIOPEN_INTERNALS_EXPORT void ResetCachedPaths();
} // namespace testing
#endif

} // namespace miopen

#endif
