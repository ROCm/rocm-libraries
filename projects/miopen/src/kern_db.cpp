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
#include "miopen/bz2.hpp"
#include <miopen/env.hpp>
#include <miopen/kern_db.hpp>
#include <miopen/logger.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <exception>
#include <mutex>
#include <set>

MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_RECORDS, 64)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_BYTES, 32ULL << 20)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_AGE_MS, 2000)

namespace miopen {

namespace {

/// Live KernDb instances, so that ending a batch can commit every pending buffer without knowing
/// which databases the batch wrote to. Lock order: Registry::mutex before any instance_mutex; no
/// code path acquires Registry::mutex while holding an instance_mutex.
struct Registry
{
    std::mutex mutex;
    std::set<KernDb*> instances;
};

Registry& GetRegistry()
{
    // Intentionally leaked: cached KernDb instances live in static storage that may be destroyed
    // after any function-local static, and their destructors still unregister themselves.
    static auto* const registry = new Registry{};
    return *registry;
}

std::atomic<int>& BatchDepth()
{
    static std::atomic<int> depth{0};
    return depth;
}

} // namespace

KernDb::KernDb(DbKinds db_kind, const fs::path& filename_, bool is_system_)
    : KernDb(db_kind, filename_, is_system_, compress, decompress)
{
}

KernDb::KernDb(
    DbKinds db_kind,
    const fs::path& filename_,
    bool is_system_,
    std::function<std::vector<char>(const std::vector<char>&, bool*)> compress_fn_,
    std::function<std::vector<char>(const std::vector<char>&, unsigned int)> decompress_fn_)
    : SQLiteBase(db_kind, filename_, is_system_),
      compress_fn(compress_fn_),
      decompress_fn(decompress_fn_),
      batch_limits(DefaultBatchLimits())
{
    // Register last: a constructor that throws must not leave a dangling registry entry.
    const auto register_instance = [this]() {
        auto& registry = GetRegistry();
        const std::lock_guard<std::mutex> lock{registry.mutex};
        registry.instances.insert(this);
    };

    if(disable_file_io)
    {
        register_instance();
        return;
    }

    if(dbInvalid)
    {
        if(filename.empty())
            MIOPEN_LOG_I("database not present");
        else
            MIOPEN_LOG_I(filename << " database invalid");
        register_instance();
        return;
    }
    if(!is_system)
    {
        const std::string create_table = KernelConfig::CreateQuery();
        sql.Exec(create_table);
        MIOPEN_LOG_I2("Database created successfully");
        if(miopen::IsLogging(LoggingLevel::Info2))
        {
            auto journal     = sql.Exec("PRAGMA journal_mode;");
            auto synchronous = sql.Exec("PRAGMA synchronous;");
            MIOPEN_LOG_I2(std::dec << filename << ": journal_mode="
                                   << (journal.empty() ? "?" : journal[0]["journal_mode"])
                                   << " synchronous="
                                   << (synchronous.empty() ? "?" : synchronous[0]["synchronous"])
                                   << " batch max_records=" << batch_limits.max_records
                                   << " max_bytes=" << batch_limits.max_bytes
                                   << " max_age_ms=" << batch_limits.max_age.count());
        }
    }
    if(!CheckTableColumns(KernelConfig::table_name(), KernelConfig::FieldNames()))
    {
        std::ostringstream ss;
        ss << "Invalid fields in table: " << KernelConfig::table_name() << " disabling access to "
           << filename;
        MIOPEN_LOG_W(ss.str());
        dbInvalid = true;
    }
    register_instance();
}

KernDb::~KernDb()
{
    {
        auto& registry = GetRegistry();
        const std::lock_guard<std::mutex> lock{registry.mutex};
        registry.instances.erase(this);
    }

    const std::lock_guard<std::recursive_mutex> lock{instance_mutex};
    if(pending.empty())
        return;
    try
    {
        FlushUnsafe();
    }
    catch(const std::exception& ex)
    {
        MIOPEN_LOG_E(ex.what());
    }
}

KernDb::BatchLimits KernDb::DefaultBatchLimits()
{
    return {
        static_cast<std::size_t>(env::value(MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_RECORDS)),
        static_cast<std::size_t>(env::value(MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_BYTES)),
        std::chrono::milliseconds{static_cast<std::chrono::milliseconds::rep>(
            env::value(MIOPEN_DEBUG_KERNEL_CACHE_BATCH_MAX_AGE_MS))},
    };
}

void KernDb::SetBatchLimits(const BatchLimits& limits)
{
    const std::lock_guard<std::recursive_mutex> lock{instance_mutex};
    batch_limits = limits;
}

std::size_t KernDb::PendingCount() const
{
    const std::lock_guard<std::recursive_mutex> lock{instance_mutex};
    return pending.size();
}

void KernDb::BeginBatch() { ++BatchDepth(); }

void KernDb::EndBatch()
{
    [[maybe_unused]] const auto previous = BatchDepth()--;
    assert(previous > 0);
}

bool KernDb::IsBatchActive() { return BatchDepth() > 0; }

void KernDb::FlushAll(bool only_expired)
{
    auto& registry = GetRegistry();
    const std::lock_guard<std::mutex> lock{registry.mutex};
    const auto now = std::chrono::steady_clock::now();
    std::exception_ptr first_error;
    for(auto* db : registry.instances)
    {
        try
        {
            const std::lock_guard<std::recursive_mutex> db_lock{db->instance_mutex};
            if(!only_expired || db->IsExpiredUnsafe(now))
                db->FlushUnsafe();
        }
        catch(...)
        {
            if(!first_error)
                first_error = std::current_exception();
        }
    }
    if(first_error)
        std::rethrow_exception(first_error);
}

KernDb::PendingRecord KernDb::PrepareRecord(const KernelConfig& problem_config) const
{
    PendingRecord record;
    record.kernel_name = problem_config.kernel_name.string();
    record.kernel_args = problem_config.kernel_args;
    record.md5         = md5(problem_config.kernel_blob);
    bool success       = false;
    auto compressed    = compress_fn(problem_config.kernel_blob, &success);
    if(success)
    {
        record.blob              = std::move(compressed);
        record.uncompressed_size = static_cast<int64_t>(problem_config.kernel_blob.size());
    }
    else
    {
        record.blob              = problem_config.kernel_blob;
        record.uncompressed_size = 0;
    }
    return record;
}

std::vector<KernDb::PendingRecord>::iterator
KernDb::FindPendingUnsafe(const KernelConfig& problem_config)
{
    const auto name = problem_config.kernel_name.string();
    return std::find_if(pending.begin(), pending.end(), [&](const PendingRecord& record) {
        return record.kernel_name == name && record.kernel_args == problem_config.kernel_args;
    });
}

void KernDb::ErasePendingUnsafe(std::vector<PendingRecord>::iterator it)
{
    pending_bytes -= it->blob.size();
    pending.erase(it);
}

bool KernDb::IsExpiredUnsafe(std::chrono::steady_clock::time_point now) const
{
    return !pending.empty() && now - pending_since >= batch_limits.max_age;
}

void KernDb::InsertUnsafe(const PendingRecord& record)
{
    auto insert_query = "INSERT OR REPLACE INTO " + KernelConfig::table_name() +
                        "(kernel_name, kernel_args, kernel_blob, kernel_hash, "
                        "uncompressed_size) VALUES(?, ?, ?, ?, ?);";
    auto stmt = SQLite::Statement{sql, insert_query};
    stmt.BindText(1, record.kernel_name);
    stmt.BindText(2, record.kernel_args);
    stmt.BindBlob(3, record.blob);
    stmt.BindText(4, record.md5);
    stmt.BindInt64(5, record.uncompressed_size);

    auto rc = stmt.Step(sql);
    if(rc != SQLITE_DONE)
        MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
}

void KernDb::FlushUnsafe()
{
    if(pending.empty())
        return;

    // The batch leaves the buffer before the commit is attempted: a failed batch is discarded
    // rather than retried, so a persistent failure (e.g. a full disk) cannot grow the buffer
    // past its bounds or turn every later store into a retry of the same records.
    auto records       = std::move(pending);
    const auto bytes   = pending_bytes;
    pending            = {};
    pending_bytes      = 0;
    const auto started = std::chrono::steady_clock::now();
    try
    {
        SQLite::Transaction transaction{sql};
        for(const auto& record : records)
            InsertUnsafe(record);
        transaction.Commit();
    }
    catch(const std::exception& ex)
    {
        MIOPEN_THROW(
            miopenStatusInternalError,
            "Failed to commit " + std::to_string(records.size()) + " kernel binaries to " +
                filename.string() +
                "; the batch was rolled back and will be recompiled when needed: " + ex.what());
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started);
    MIOPEN_LOG_I2(std::dec << "Committed " << records.size() << " kernel binaries (" << bytes
                           << " bytes) to " << filename << " in " << elapsed.count() << " us");
}

bool KernDb::RemoveRecordUnsafe(const KernelConfig& problem_config)
{
    const std::lock_guard<std::recursive_mutex> lock{instance_mutex};
    if(filename.empty())
        return true;

    // Drop a pending copy too, otherwise a later commit would resurrect the record.
    const auto it = FindPendingUnsafe(problem_config);
    if(it != pending.end())
        ErasePendingUnsafe(it);

    auto del_query =
        "DELETE FROM " + KernelConfig::table_name() + " WHERE " + problem_config.Where() + ";";
    auto stmt = SQLite::Statement{sql, del_query};
    auto rc   = stmt.Step(sql);
    if(rc == SQLITE_DONE)
    {
        return true;
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
        return false;
    }
}

std::optional<std::vector<char>> KernDb::FindRecordUnsafe(const KernelConfig& problem_config)
{
    if(filename.empty())
        return {};

    // Lock only around the lookup so that bz2 decompression and MD5
    // checking can run concurrently across threads.
    std::vector<char> compressed_blob;
    std::string md5_hash;
    int64_t uncompressed_size = 0;
    {
        const std::lock_guard<std::recursive_mutex> lock{instance_mutex};
        // A pending record is newer than anything committed under the same key.
        const auto it = FindPendingUnsafe(problem_config);
        if(it != pending.end())
        {
            compressed_blob   = it->blob;
            md5_hash          = it->md5;
            uncompressed_size = it->uncompressed_size;
        }
        else
        {
            // Where clause with inserted values defeats the purpose of a prepared statement
            auto select_query = "SELECT kernel_blob, kernel_hash, uncompressed_size FROM " +
                                KernelConfig::table_name() + " WHERE " + problem_config.Where() +
                                ";";
            auto stmt = SQLite::Statement{sql, select_query};
            auto rc   = stmt.Step(sql);
            if(rc == SQLITE_ROW)
            {
                compressed_blob   = stmt.ColumnBlob(0);
                md5_hash          = stmt.ColumnText(1);
                uncompressed_size = stmt.ColumnInt64(2);
            }
            else if(rc == SQLITE_DONE)
            {
                return {};
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
                return {};
            }
        }
    }

    // Decompression and MD5 verification run outside the lock.
    std::vector<char>& decompressed_blob = compressed_blob;
    if(uncompressed_size != 0)
    {
        decompressed_blob = decompress_fn(compressed_blob, uncompressed_size);
    }
    auto new_md5 = md5(decompressed_blob);
    if(new_md5 != md5_hash)
        MIOPEN_THROW(miopenStatusInternalError, "Possible database corruption");
    return decompressed_blob;
}

bool KernDb::StoreRecordUnsafe(const KernelConfig& problem_config)
{
    if(filename.empty())
        return false;

    // Compression and MD5 run outside the lock (and outside any transaction).
    auto record = PrepareRecord(problem_config);

    const std::lock_guard<std::recursive_mutex> lock{instance_mutex};
    const auto now = std::chrono::steady_clock::now();
    // INSERT OR REPLACE semantics: a newer store of the same key replaces the pending one.
    const auto it = FindPendingUnsafe(problem_config);
    if(it != pending.end())
        ErasePendingUnsafe(it);
    if(pending.empty())
        pending_since = now;
    pending_bytes += record.blob.size();
    pending.push_back(std::move(record));

    // Outside a batch this commits the record right away, exactly like an autocommit INSERT.
    if(!IsBatchActive() || pending.size() >= batch_limits.max_records ||
       pending_bytes >= batch_limits.max_bytes || IsExpiredUnsafe(now))
        FlushUnsafe();
    return true;
}

} // namespace miopen
