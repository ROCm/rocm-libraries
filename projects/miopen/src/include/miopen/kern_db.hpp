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

#include <functional>
#include <optional>
#include <string>

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
};

/// Kernel binary database.
///
/// Statements: lookups, inserts and deletes bind the record key as parameters of fixed SQL, so key
/// values are never interpreted as SQL. Each statement is prepared on first use, reused for the
/// lifetime of the instance (that is, of its connection; see SQLiteBase::GetCached), and only
/// bound and stepped under instance_mutex. Every use resets its statement before releasing the
/// lock, so no statement holds a read transaction, a row or bindings between uses.
class KernDb : public SQLiteBase<KernDb>
{
    std::function<std::vector<char>(const std::vector<char>&, bool*)> compress_fn;
    std::function<std::vector<char>(const std::vector<char>&, unsigned int)> decompress_fn;

    // Guarded by instance_mutex. As members of the derived class they are finalized before
    // SQLiteBase closes the connection they belong to.
    std::optional<SQLite::Statement> select_stmt;
    std::optional<SQLite::Statement> insert_stmt;
    std::optional<SQLite::Statement> delete_stmt;

    SQLite::Statement& PreparedUnsafe(std::optional<SQLite::Statement>& stmt,
                                      const std::string& query)
    {
        if(!stmt)
            stmt.emplace(sql, query);
        return *stmt;
    }

public:
    MIOPEN_INTERNALS_EXPORT KernDb(DbKinds db_kind, const fs::path& filename_, bool is_system);
    // This constructor is only intended for testing
    MIOPEN_INTERNALS_EXPORT
    KernDb(DbKinds db_kind,
           const fs::path& filename_,
           bool is_system_,
           std::function<std::vector<char>(const std::vector<char>&, bool*)> compress_fn_,
           std::function<std::vector<char>(const std::vector<char>&, unsigned int)> decompress_fn_);

    bool RemoveRecordUnsafe(const KernelConfig& problem_config)
    {
        const std::lock_guard<std::recursive_mutex> lock{instance_mutex};
        if(filename.empty())
            return true;
        auto& stmt = PreparedUnsafe(delete_stmt,
                                    "DELETE FROM " + KernelConfig::table_name() +
                                        " WHERE kernel_name = ? AND kernel_args = ?;");
        const SQLite::Statement::ScopedReset reset{stmt};
        stmt.BindPath(1, problem_config.kernel_name);
        stmt.BindText(2, problem_config.kernel_args);
        auto rc = stmt.Step(sql);
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

    std::optional<std::vector<char>> FindRecordUnsafe(const KernelConfig& problem_config)
    {
        if(filename.empty())
            return {};

        // Lock only around the SQLite query so that bz2 decompression and MD5
        // checking can run concurrently across threads.
        std::vector<char> compressed_blob;
        std::string md5_hash;
        int64_t uncompressed_size = 0;
        {
            const std::lock_guard<std::recursive_mutex> lock{instance_mutex};
            auto& stmt = PreparedUnsafe(select_stmt,
                                        "SELECT kernel_blob, kernel_hash, uncompressed_size FROM " +
                                            KernelConfig::table_name() +
                                            " WHERE kernel_name = ? AND kernel_args = ?;");
            const SQLite::Statement::ScopedReset reset{stmt};
            stmt.BindPath(1, problem_config.kernel_name);
            stmt.BindText(2, problem_config.kernel_args);
            auto rc = stmt.Step(sql);
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

    bool StoreRecordUnsafe(const KernelConfig& problem_config)
    {
        if(filename.empty())
            return false;

        // Compression and MD5 run outside the lock.
        auto md5_sum           = md5(problem_config.kernel_blob);
        auto uncompressed_size = problem_config.kernel_blob.size();
        bool success           = false;
        auto compressed_blob   = compress_fn(problem_config.kernel_blob, &success);

        // Lock only around the SQLite insert.
        const std::lock_guard<std::recursive_mutex> lock{instance_mutex};
        auto& stmt = PreparedUnsafe(insert_stmt,
                                    "INSERT OR REPLACE INTO " + KernelConfig::table_name() +
                                        "(kernel_name, kernel_args, kernel_blob, kernel_hash, "
                                        "uncompressed_size) VALUES(?, ?, ?, ?, ?);");
        const SQLite::Statement::ScopedReset reset{stmt};
        stmt.BindPath(1, problem_config.kernel_name);
        stmt.BindText(2, problem_config.kernel_args);
        if(!success)
        {
            stmt.BindBlob(3, problem_config.kernel_blob);
            stmt.BindInt64(5, 0);
        }
        else
        {
            stmt.BindBlob(3, compressed_blob);
            stmt.BindInt64(5, uncompressed_size);
        }
        stmt.BindText(4, md5_sum);

        auto rc = stmt.Step(sql);
        if(rc != SQLITE_DONE)
            MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
        return true;
    }
};
} // namespace miopen
#endif
#endif // GUARD_MIOPEN_KERN_DB_HPP_
