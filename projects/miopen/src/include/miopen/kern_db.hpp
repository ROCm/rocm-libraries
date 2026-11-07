/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_KERN_DB_HPP_
#define GUARD_MIOPEN_KERN_DB_HPP_

#include <miopen/config.h>

#if MIOPEN_ENABLE_SQLITE

#include <miopen/sqlite_db.hpp>
#include <miopen/bz2.hpp>
#include <miopen/md5.hpp>

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace miopen {
struct KernelConfig
{
    static std::string table_name() { return "kern_db"; }
    fs::path kernel_name;
    std::string kernel_args;
    std::vector<char> kernel_blob;
    static std::vector<std::string> FieldNames()
    {
        return {"kernel_name", "kernel_args", "kernel_blob"};
    }
    static std::string CreateQuery()
    {
        std::ostringstream ss;
        ss << "CREATE TABLE IF NOT EXISTS `" << KernelConfig::table_name() << "` ("
           << "`id` INTEGER PRIMARY KEY ASC" << ",`kernel_name` TEXT NOT NULL"
           << ",`kernel_args` TEXT NOT NULL" << ",`kernel_blob` BLOB NOT NULL"
           << ",`kernel_hash` TEXT NOT NULL" << ",`uncompressed_size` INT NOT NULL" << ");"
           << "CREATE UNIQUE INDEX IF NOT EXISTS " << "`idx_" << KernelConfig::table_name() << "` "
           << "ON " << KernelConfig::table_name() << "(kernel_name, kernel_args);";
        return ss.str();
    }
    std::string Where() const
    {
        std::ostringstream ss;
        ss << "(kernel_name = '" << kernel_name.string() << "')" << " AND (kernel_args = '"
           << kernel_args << "')";
        return ss.str();
    }
};

/// Kernel binary database.
///
/// Write batching: while a kernel cache batch is active (see KernelCacheBatch in
/// binary_cache.hpp), StoreRecord does not commit each record on its own. The record is prepared
/// (compressed and hashed) outside any lock and transaction, then appended to a pending buffer
/// owned by this instance. The buffer is committed as one BEGIN IMMEDIATE ... COMMIT transaction
/// when any BatchLimits bound is reached, when any batch ends, or when the instance is destroyed.
/// Only the INSERTs run inside the transaction, so it is never held across kernel compilation.
/// Outside a batch every StoreRecord commits immediately, as before.
///
/// Visibility: pending records are visible to FindRecord on the same instance (the cached
/// instance shared by the whole process) as soon as StoreRecord returns, with the same
/// last-write-wins semantics as INSERT OR REPLACE; RemoveRecord also discards a pending record.
/// Other connections and processes see a record only after its batch commits.
///
/// Failure: a batch is committed atomically or not at all. When a commit fails, the transaction is
/// rolled back, the batch is discarded (its kernels are recompiled the next time they are needed,
/// as for any cache miss) and the failure is thrown to the caller that triggered the commit.
/// Records committed by earlier batches are not affected.
class KernDb : public SQLiteBase<KernDb>
{
public:
    /// Bounds on the records an instance may keep pending while a batch is active.
    struct BatchLimits
    {
        /// Commit once this many records are pending. 1 commits every record individually.
        std::size_t max_records;
        /// Commit once the pending blobs (as stored, i.e. compressed) reach this many bytes.
        std::size_t max_bytes;
        /// Commit once the oldest pending record has waited this long. Checked on every store
        /// and by KernelCacheBatch::FlushExpired().
        std::chrono::milliseconds max_age;
    };

    MIOPEN_INTERNALS_EXPORT KernDb(DbKinds db_kind, const fs::path& filename_, bool is_system);
    // This constructor is only intended for testing
    MIOPEN_INTERNALS_EXPORT
    KernDb(DbKinds db_kind,
           const fs::path& filename_,
           bool is_system_,
           std::function<std::vector<char>(const std::vector<char>&, bool*)> compress_fn_,
           std::function<std::vector<char>(const std::vector<char>&, unsigned int)> decompress_fn_);
    /// Commits pending records; failures are logged since a destructor cannot report them.
    MIOPEN_INTERNALS_EXPORT ~KernDb();
    KernDb(const KernDb&)            = delete;
    KernDb& operator=(const KernDb&) = delete;

    MIOPEN_INTERNALS_EXPORT bool RemoveRecordUnsafe(const KernelConfig& problem_config);
    MIOPEN_INTERNALS_EXPORT std::optional<std::vector<char>>
    FindRecordUnsafe(const KernelConfig& problem_config);
    MIOPEN_INTERNALS_EXPORT bool StoreRecordUnsafe(const KernelConfig& problem_config);

    MIOPEN_INTERNALS_EXPORT std::size_t PendingCount() const;
    MIOPEN_INTERNALS_EXPORT void SetBatchLimits(const BatchLimits& limits);
    /// Limits applied to new instances; overridable through MIOPEN_DEBUG_KERNEL_CACHE_BATCH_*.
    MIOPEN_INTERNALS_EXPORT static BatchLimits DefaultBatchLimits();

    /// Process-wide batch control used by KernelCacheBatch. Batches nest; stores are batched while
    /// at least one is active.
    MIOPEN_INTERNALS_EXPORT static void BeginBatch();
    MIOPEN_INTERNALS_EXPORT static void EndBatch();
    MIOPEN_INTERNALS_EXPORT static bool IsBatchActive();
    /// Commits the pending records of every live instance (only those whose oldest pending record
    /// exceeded max_age if only_expired is set). Every instance is attempted; the first failure is
    /// rethrown afterwards.
    MIOPEN_INTERNALS_EXPORT static void FlushAll(bool only_expired = false);

private:
    /// A kern_db row ready for insertion: blob compressed (if that succeeded) and hashed.
    struct PendingRecord
    {
        std::string kernel_name;
        std::string kernel_args;
        std::vector<char> blob;
        std::string md5;
        int64_t uncompressed_size;
    };

    std::function<std::vector<char>(const std::vector<char>&, bool*)> compress_fn;
    std::function<std::vector<char>(const std::vector<char>&, unsigned int)> decompress_fn;

    // Guarded by instance_mutex.
    BatchLimits batch_limits;
    std::vector<PendingRecord> pending;
    std::size_t pending_bytes = 0;
    std::chrono::steady_clock::time_point pending_since;

    PendingRecord PrepareRecord(const KernelConfig& problem_config) const;
    std::vector<PendingRecord>::iterator FindPendingUnsafe(const KernelConfig& problem_config);
    void ErasePendingUnsafe(std::vector<PendingRecord>::iterator it);
    bool IsExpiredUnsafe(std::chrono::steady_clock::time_point now) const;
    void InsertUnsafe(const PendingRecord& record);
    void FlushUnsafe();
};
} // namespace miopen
#endif
#endif // GUARD_MIOPEN_KERN_DB_HPP_
