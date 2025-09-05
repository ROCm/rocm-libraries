// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../shared/environment.h"

#include "library_path.h"
#include "logging.h"
#include "rtc_cache.h"
#include "rtc_compile.h"
#include "rtc_subprocess.h"
#include "sqlite3.h"

#include <chrono>
#include <hip/hip_version.h>
#include <hip/hiprtc.h>
#include <mutex>
#include <optional>

namespace fs = std::filesystem;

std::unique_ptr<RTCCache> RTCCache::single;

static const char* default_cache_filename = "rocfft_kernel_cache.db";

// Lock for in-process compilation - due to limits in ROCclr, we
// can do at most one compilation in a process before we have to
// delegate to a subprocess.  But we should at least do one
// in-process instead of making everything go to subprocess.
static std::mutex compile_lock;

// Get paths to system RTC cache, in decreasing order of preference.
static std::vector<fs::path> rtccache_db_sys_paths()
{
    // if env var is set, use that directly
    std::vector<fs::path> paths;
    auto                  env_path = rocfft_getenv("ROCFFT_RTC_SYS_CACHE_PATH");

    if(!env_path.empty())
    {
        paths.push_back(env_path);
    }
    else
    {
        auto lib_path = get_library_path();
        if(!lib_path.empty())
        {
            // try next to the library, and in rocfft subdir
            fs::path library_parent_path = lib_path.parent_path();
            paths.push_back(library_parent_path / default_cache_filename);
            paths.push_back(library_parent_path / "rocfft" / default_cache_filename);
        }
    }
    return paths;
}

// Get list of candidate paths to RTC user cache DB, in decreasing
// order of preference.
static std::vector<fs::path> rtccache_db_user_paths()
{
    // use user-defined cache path if present
    std::vector<fs::path> paths;
    auto                  env_path = rocfft_getenv("ROCFFT_RTC_CACHE_PATH");

    if(!env_path.empty())
        paths.push_back(env_path);

    // finally, fall back to in-memory db if all else fails
    paths.push_back({});
    return paths;
}

static sqlite3_stmt_ptr prepare_stmt(sqlite3_ptr& db, const char* sql)
{
    sqlite3_stmt* stmt = nullptr;
    if(sqlite3_prepare_v2(db.get(), sql, -1, &stmt, nullptr) == SQLITE_OK)
        return sqlite3_stmt_ptr(stmt);
    throw std::runtime_error(std::string("sqlite_prepare_v2 failed: ") + sqlite3_errmsg(db.get()));
}

sqlite3_ptr RTCCache::connect_db(const fs::path& path, bool readonly)
{
    sqlite3* db_raw = nullptr;
    int      flags  = SQLITE_OPEN_FULLMUTEX;
    if(readonly)
        flags |= SQLITE_OPEN_READONLY;
    else
        flags |= SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
    if(path.empty())
    {
        // open in-memory
        flags |= SQLITE_OPEN_MEMORY;
    }
    if(sqlite3_open_v2(path.string().c_str(), &db_raw, flags, nullptr) != SQLITE_OK)
    {
        sqlite3_close(db_raw);
        return nullptr;
    }

    sqlite3_ptr db(db_raw);

    // we can potentially want to write a bunch of kernels in
    // parallel (when doing mass compilation).  set a bigger busy
    // timeout (5s) so that concurrent modifications will wait for one
    // another
    sqlite3_busy_timeout(db_raw, 5000);

    if(!readonly)
    {
        // create the default table
        auto create = prepare_stmt(db,
                                   "CREATE TABLE IF NOT EXISTS cache_v1 ("
                                   "  kernel_name TEXT NOT NULL,"
                                   "  arch TEXT NOT NULL,"
                                   "  hip_version INTEGER NOT NULL,"
                                   "  generator_sum BLOB NOT NULL,"
                                   "  code BLOB NOT NULL,"
                                   "  timestamp INTEGER NOT NULL,"
                                   "  PRIMARY KEY ("
                                   "      kernel_name, arch, hip_version, generator_sum"
                                   "      ))");
        if(sqlite3_step(create.get()) != SQLITE_DONE)
            return nullptr;
    }

    return db;
}

RTCCache::RTCCache()
{
    auto sys_paths = rtccache_db_sys_paths();
    for(const auto& p : sys_paths)
    {
        db_sys = connect_db(p, true);
        if(db_sys)
            break;
    }

    auto paths = rtccache_db_user_paths();
    for(const auto& p : paths)
    {
        db_user = connect_db(p, false);
        if(db_user)
            break;
    }

    static const char* get_stmt_text = "SELECT code "
                                       "FROM cache_v1 "
                                       "WHERE"
                                       "  kernel_name = :kernel_name "
                                       "  AND arch = :arch "
                                       "  AND hip_version = :hip_version "
                                       "  AND generator_sum = :generator_sum ";

    static const char* store_stmt_text = "INSERT OR REPLACE INTO cache_v1 ("
                                         "    kernel_name,"
                                         "    arch,"
                                         "    hip_version,"
                                         "    generator_sum,"
                                         "    code,"
                                         "    timestamp"
                                         ")"
                                         "VALUES ("
                                         "    :kernel_name,"
                                         "    :arch,"
                                         "    :hip_version,"
                                         "    :generator_sum,"
                                         "    :code,"
                                         "    CAST(STRFTIME('%s','now') AS INTEGER)"
                                         ")";

    // prepare get/store statements once so they can be called many
    // times
    if(db_sys)
    {
        // it's possible that the sys cache exists but is not usable.
        // so if we are unable to talk to it, just stop using it
        try
        {
            get_stmt_sys = prepare_stmt(db_sys, get_stmt_text);
        }
        catch(std::exception&)
        {
            db_sys.reset();
        }
    }
    if(db_user)
    {
        get_stmt_user   = prepare_stmt(db_user, get_stmt_text);
        store_stmt_user = prepare_stmt(db_user, store_stmt_text);
    }
}

static std::vector<char> get_code_object_impl(const std::string&          kernel_name,
                                              const std::string&          gpu_arch,
                                              const std::array<char, 32>& generator_sum,
                                              sqlite3_ptr&                db,
                                              sqlite3_stmt_ptr&           get_stmt,
                                              std::mutex&                 get_mutex)
{
    std::vector<char> code;

    // allow env variable to disable reads
    if(!rocfft_getenv("ROCFFT_RTC_CACHE_READ_DISABLE").empty())
        return code;

    std::lock_guard<std::mutex> lock(get_mutex);

    auto s = get_stmt.get();
    sqlite3_reset(s);

    // bind arguments to the query and execute
    if(sqlite3_bind_text(s, 1, kernel_name.c_str(), kernel_name.size(), SQLITE_TRANSIENT)
           != SQLITE_OK
       || sqlite3_bind_text(s, 2, gpu_arch.c_str(), gpu_arch.size(), SQLITE_TRANSIENT) != SQLITE_OK
       || sqlite3_bind_int64(s, 3, HIP_VERSION) != SQLITE_OK
       || sqlite3_bind_blob(s, 4, generator_sum.data(), generator_sum.size(), SQLITE_TRANSIENT)
              != SQLITE_OK)
    {
        throw std::runtime_error(std::string("get_code_object bind: ") + sqlite3_errmsg(db.get()));
    }
    if(sqlite3_step(s) == SQLITE_ROW)
    {
        // cache hit, get the value out
        int         nbytes = sqlite3_column_bytes(s, 0);
        const char* data   = static_cast<const char*>(sqlite3_column_blob(s, 0));
        std::copy(data, data + nbytes, std::back_inserter(code));
    }
    sqlite3_reset(s);
    return code;
}

std::vector<char> RTCCache::get_code_object(const std::string&          kernel_name,
                                            const std::string&          gpu_arch,
                                            const std::array<char, 32>& generator_sum)
{
    std::vector<char> code;
    // try user cache first
    if(get_stmt_user)
        code = get_code_object_impl(
            kernel_name, gpu_arch, generator_sum, db_user, get_stmt_user, get_mutex_user);
    // fall back to system cache
    if(code.empty() && get_stmt_sys)
        code = get_code_object_impl(
            kernel_name, gpu_arch, generator_sum, db_sys, get_stmt_sys, get_mutex_sys);
    return code;
}

void RTCCache::store_code_object(const std::string&          kernel_name,
                                 const std::string&          gpu_arch,
                                 const std::array<char, 32>& generator_sum,
                                 const std::vector<char>&    code)
{
    // allow env variable to disable writes
    if(!rocfft_getenv("ROCFFT_RTC_CACHE_WRITE_DISABLE").empty())
        return;

    std::lock_guard<std::mutex> lock(store_mutex_user);

    auto s = store_stmt_user.get();
    sqlite3_reset(s);

    // bind arguments to the query and execute
    if(sqlite3_bind_text(s, 1, kernel_name.c_str(), kernel_name.size(), SQLITE_TRANSIENT)
           != SQLITE_OK
       || sqlite3_bind_text(s, 2, gpu_arch.c_str(), gpu_arch.size(), SQLITE_TRANSIENT) != SQLITE_OK
       || sqlite3_bind_int64(s, 3, HIP_VERSION) != SQLITE_OK
       || sqlite3_bind_blob(s, 4, generator_sum.data(), generator_sum.size(), SQLITE_TRANSIENT)
              != SQLITE_OK
       || sqlite3_bind_blob(s, 5, code.data(), code.size(), SQLITE_TRANSIENT))
    {
        throw std::runtime_error(std::string("store_code_object bind: ")
                                 + sqlite3_errmsg(db_user.get()));
    }
    if(sqlite3_step(s) != SQLITE_DONE)
    {
        std::cerr << "Error: failed to store code object for " << kernel_name << ": "
                  << sqlite3_errmsg(db_user.get()) << std::endl;
        // some kind of problem storing the row?  log it
        if(LOG_RTC_ENABLED())
            (*LogSingleton::GetInstance().GetRTCOS())
                << "Error: failed to store code object for " << kernel_name << ": "
                << sqlite3_errmsg(db_user.get()) << std::flush;
    }
    sqlite3_reset(s);
}

rocfft_status RTCCache::serialize(void** buffer, size_t* buffer_len_bytes)
{
    sqlite3_int64 db_size = 0;
    auto          ptr     = sqlite3_serialize(db_user.get(), "main", &db_size, 0);
    if(ptr)
    {
        *buffer           = ptr;
        *buffer_len_bytes = db_size;
        return rocfft_status_success;
    }
    return rocfft_status_failure;
}

void RTCCache::serialize_free(void* buffer)
{
    sqlite3_free(buffer);
}

rocfft_status RTCCache::deserialize(const void* buffer, size_t buffer_len_bytes)
{
    std::lock_guard<std::mutex> lock(deserialize_mutex);

    // attach an empty database named "deserialized"
    sqlite3_exec(
        db_user.get(), "ATTACH DATABASE ':memory:' AS deserialized", nullptr, nullptr, nullptr);
    // the attach might fail if somehow this is our second
    // deserialize and the db already existed.  later steps will
    // notice this, so we can skip this error check

    // sqlite's API is prepared to write to the pointer, but we tell
    // it to be read-only
    auto buffer_mut = const_cast<unsigned char*>(static_cast<const unsigned char*>(buffer));

    int sql_err = sqlite3_deserialize(db_user.get(),
                                      "deserialized",
                                      buffer_mut,
                                      buffer_len_bytes,
                                      buffer_len_bytes,
                                      SQLITE_DESERIALIZE_READONLY);
    if(sql_err != SQLITE_OK)
        return rocfft_status_failure;

    // now the deserialized db is in memory.  run an additive query to
    // update the real db with the temp contents.
    sql_err           = sqlite3_exec(db_user.get(),
                           "INSERT OR REPLACE INTO cache_v1 ("
                           "    kernel_name,"
                           "    arch,"
                           "    hip_version,"
                           "    generator_sum,"
                           "    timestamp,"
                           "    code"
                           ")"
                           "SELECT"
                           "    kernel_name,"
                           "    arch,"
                           "    hip_version,"
                           "    generator_sum,"
                           "    timestamp,"
                           "    code "
                           "FROM deserialized.cache_v1",
                           nullptr,
                           nullptr,
                           nullptr);
    rocfft_status ret = sql_err == SQLITE_OK ? rocfft_status_success : rocfft_status_failure;

    // detach the temp db
    sqlite3_exec(db_user.get(), "DETACH DATABASE deserialized", nullptr, nullptr, nullptr);

    return ret;
}

// allow user control of whether RTC is done in-process or out-of-process
enum class RTCProcessType
{
    // allow one in-process compile, fallback to out-of-process if
    // one is already in progress.  fall back further to waiting for
    // the lock if subprocess failed.
    DEFAULT,
    // only try in-process, waiting for lock if necessary
    FORCE_IN_PROCESS,
    // only try out-of-process, never needs lock
    FORCE_OUT_PROCESS,
};

static RTCProcessType get_rtc_process_type()
{
    auto var = rocfft_getenv("ROCFFT_RTC_PROCESS");
    // defined and equal to 0 means force in-process
    if(var == "0")
        return RTCProcessType::FORCE_IN_PROCESS;
    // defined and equal to 1 means force out-process
    if(var == "1")
        return RTCProcessType::FORCE_OUT_PROCESS;
    // ideal default behaviour - try in-process first and use
    // out-process if necessary
    if(var == "2")
        return RTCProcessType::DEFAULT;

    return RTCProcessType::DEFAULT;
}

static std::string gpu_arch_strip_flags(const std::string gpu_arch_with_flags)
{
    return gpu_arch_with_flags.substr(0, gpu_arch_with_flags.find(':'));
}

static std::vector<char> cached_compile_impl(const std::string&          kernel_name,
                                             const std::string&          gpu_arch,
                                             kernel_src_gen_t            generate_src,
                                             const std::array<char, 32>& generator_sum)
{
    // check cache first
    std::vector<char> code;
    if(RTCCache::single)
    {
        code = RTCCache::single->get_code_object(kernel_name, gpu_arch, generator_sum);
    }

    if(!code.empty())
    {
        // cache hit
        try
        {
            if(LOG_RTC_ENABLED())
            {
                (*LogSingleton::GetInstance().GetRTCOS())
                    << "// cache hit for " << kernel_name << std::endl;
            }
            return code;
        }
        catch(std::exception&)
        {
            // if for some reason the cached object was not
            // usable, fall through to generating the source and
            // recompiling
            if(LOG_RTC_ENABLED())
            {
                (*LogSingleton::GetInstance().GetRTCOS())
                    << "// cache unusable for " << kernel_name << std::endl;
            }
        }
    }

    // callbacks are always potentially enabled, and activated by
    // checking the enable_callbacks variable later
    std::string kernel_src{"#define ROCFFT_CALLBACKS_ENABLED\n"};

    auto generate_begin = std::chrono::steady_clock::now();
    kernel_src += generate_src(kernel_name);
    auto generate_end = std::chrono::steady_clock::now();

    if(LOG_RTC_ENABLED())
    {
        std::chrono::duration<float, std::milli> generate_ms = generate_end - generate_begin;

        (*LogSingleton::GetInstance().GetRTCOS())
            << "// ROCFFT_RTC_BEGIN " << kernel_name << "\n"
            << kernel_src << "\n// ROCFFT_RTC_END " << kernel_name << "\n// " << kernel_name
            << " generate duration: " << static_cast<int>(generate_ms.count()) << " ms"
            << std::endl;
    }

    // try to set compile_begin time right when we're really
    // about to compile (i.e. after acquiring any locks)
    std::chrono::time_point<std::chrono::steady_clock> compile_begin;

    RTCProcessType process_type = get_rtc_process_type();
    switch(process_type)
    {
    case RTCProcessType::FORCE_OUT_PROCESS:
    {
        compile_begin = std::chrono::steady_clock::now();
        try
        {
            code = compile_subprocess(kernel_src, gpu_arch);
            break;
        }
        catch(std::exception&)
        {
            // if subprocess had a problem, ignore it and
            // fall through to forced-in-process compile
        }
    }
    case RTCProcessType::FORCE_IN_PROCESS:
    {
        std::lock_guard<std::mutex> lck(compile_lock);
        compile_begin = std::chrono::steady_clock::now();
        code          = compile_inprocess(kernel_src, gpu_arch);
        break;
    }
    default:
    {
        // do it in-process if possible
        std::unique_lock<std::mutex> lock(compile_lock, std::try_to_lock);
        if(lock.owns_lock())
        {
            compile_begin = std::chrono::steady_clock::now();
            code          = compile_inprocess(kernel_src, gpu_arch);
            lock.unlock();
        }
        else
        {
            // couldn't acquire lock, so try instead in a subprocess
            try
            {
                compile_begin = std::chrono::steady_clock::now();
                code          = compile_subprocess(kernel_src, gpu_arch);
            }
            catch(std::exception&)
            {
                // subprocess still didn't work, re-acquire lock
                // and fall back to in-process if something went
                // wrong
                std::lock_guard<std::mutex> lck(compile_lock);
                compile_begin = std::chrono::steady_clock::now();
                code          = compile_inprocess(kernel_src, gpu_arch);
            }
        }
    }
    }
    auto compile_end = std::chrono::steady_clock::now();

    if(LOG_RTC_ENABLED())
    {
        std::chrono::duration<float, std::milli> compile_ms = compile_end - compile_begin;

        (*LogSingleton::GetInstance().GetRTCOS())
            << "// " << kernel_name << " compile duration: " << static_cast<int>(compile_ms.count())
            << " ms\n"
            << std::endl;
    }

    if(RTCCache::single)
    {
        RTCCache::single->store_code_object(kernel_name, gpu_arch, generator_sum, code);
    }
    return code;
}

// RAII type to ensure a pending compile we insert is removed at
// scope exit
struct PendingCompileCleanup
{
    PendingCompileCleanup(const RTCCache::pending_key& key)
        : key(key)
    {
    }
    ~PendingCompileCleanup()
    {
        if(RTCCache::single)
        {
            std::lock_guard<std::mutex> lock(RTCCache::single->pending_compiles_mutex);
            RTCCache::single->pending_compiles.erase(key);
        }
    }
    const RTCCache::pending_key& key;
};

std::vector<char> RTCCache::cached_compile(const std::string&          kernel_name,
                                           const std::string&          gpu_arch_with_flags,
                                           kernel_src_gen_t            generate_src,
                                           const std::array<char, 32>& generator_sum)
{
#ifdef ADDRESS_SANITIZER
    // The address sanitizer is reported to work better when we include xnack+, so don't strip this
    // from the architecture string when building:
    const std::string gpu_arch = gpu_arch_with_flags;
#else
    // Supplied gpu arch may have extra flags on it
    // (e.g. gfx90a:sramecc+:xnack-), Strip those from the arch name
    // since omitting them will generate code that handles either
    // case.
    //
    // As of this writing, there are no known performance benefits to
    // including the flags.  If that changes, we may need to be more
    // selective about which flags to strip.
    const std::string gpu_arch = gpu_arch_strip_flags(gpu_arch_with_flags);
#endif
    std::shared_future<std::vector<char>> result;

    const pending_key                    key{kernel_name, gpu_arch};
    std::optional<PendingCompileCleanup> cleanup;
    if(RTCCache::single)
    {
        // check the map of pending work for this compile
        std::lock_guard<std::mutex> lock(RTCCache::single->pending_compiles_mutex);
        auto                        pc = RTCCache::single->pending_compiles.find(key);
        if(pc == RTCCache::single->pending_compiles.end())
        {
            // not in the pending map, so add a future and launch the
            // work to look up the kernel in the cache or compile it.
            auto compile = [=](std::promise<std::vector<char>> compile_promise) {
                try
                {
                    compile_promise.set_value(
                        cached_compile_impl(kernel_name, gpu_arch, generate_src, generator_sum));
                }
                catch(std::exception e)
                {
                    compile_promise.set_exception(std::current_exception());
                }
            };

            std::promise<std::vector<char>> compile_promise;
            pc = RTCCache::single->pending_compiles.emplace(key, compile_promise.get_future())
                     .first;
            std::thread compile_thread(compile, std::move(compile_promise));
            // we'll wait for the future so the thread can continue
            // without being managed by this object
            compile_thread.detach();

            cleanup.emplace(key);
        }
        result = pc->second;
    }
    else
    {
        // no cache?  just directly compile
        std::promise<std::vector<char>> p;
        p.set_value(cached_compile_impl(kernel_name, gpu_arch, generate_src, generator_sum));
        result = p.get_future();
    }
    return result.get();
}

void RTCCache::enable_write_mostly()
{
    // increase sqlite timeout as many processes may be contending over
    // this database
    sqlite3_busy_timeout(db_user.get(), 30000);

    // set the cache file to WAL mode, which allows for faster writes
    // that don't block with reads
    auto wal_stmt = prepare_stmt(db_user, "PRAGMA journal_mode=WAL");
    // ignore error here - WAL is helpful, but we can still proceed
    // even if enabling it fails
    sqlite3_step(wal_stmt.get());
}

void RTCCache::write_aot_cache(const std::string&              output_path,
                               const std::array<char, 32>&     generator_sum,
                               const std::vector<std::string>& gpu_archs)
{
    // remove the path if it already exists, since we want to output a
    // cleanly created file
    if(fs::exists(output_path))
        fs::remove(output_path);

    // connect to the output path but discard the returned pointer -
    // this will create the cache tables but close the connection so
    // we can reopen it from inside our existing db handle.
    connect_db(output_path, false);

    auto attach_stmt = prepare_stmt(db_user, "ATTACH DATABASE :db AS out_db");
    sqlite3_reset(attach_stmt.get());
    if(sqlite3_bind_text(
           attach_stmt.get(), 1, output_path.c_str(), output_path.size(), SQLITE_TRANSIENT)
       != SQLITE_OK)
        throw std::runtime_error(std::string("write_aot_cache attach bind: ")
                                 + sqlite3_errmsg(db_user.get()));
    if(sqlite3_step(attach_stmt.get()) != SQLITE_DONE)
        throw std::runtime_error(std::string("write_aot_cache attach step: ")
                                 + sqlite3_errmsg(db_user.get()));
    sqlite3_reset(attach_stmt.get());

    // copy only the required arches, in case more are present in the
    // cache than we need
    auto create_temp_stmt = prepare_stmt(db_user,
                                         "CREATE TABLE IF NOT EXISTS temp.aot_arch ("
                                         "  arch TEXT NOT NULL )");
    if(sqlite3_step(create_temp_stmt.get()) != SQLITE_DONE)
        throw std::runtime_error(std::string("write_aot_cache create temp table: ")
                                 + sqlite3_errmsg(db_user.get()));

    auto insert_temp_stmt = prepare_stmt(db_user, "INSERT INTO temp.aot_arch VALUES ( ? )");
    for(const auto& gpu_arch_with_flags : gpu_archs)
    {
        std::string gpu_arch = gpu_arch_strip_flags(gpu_arch_with_flags);

        if(sqlite3_bind_text(
               insert_temp_stmt.get(), 1, gpu_arch.c_str(), gpu_arch.size(), SQLITE_TRANSIENT)
           != SQLITE_OK)
            throw std::runtime_error(std::string("write_aot_cache temp bind: ")
                                     + sqlite3_errmsg(db_user.get()));
        if(sqlite3_step(insert_temp_stmt.get()) != SQLITE_DONE)
            throw std::runtime_error(std::string("write_aot_cache temp step: ")
                                     + sqlite3_errmsg(db_user.get()));
        sqlite3_reset(insert_temp_stmt.get());
    }

    // copy the kernels over in a consistent order and zero out the timestamps
    auto copy_stmt = prepare_stmt(db_user,
                                  "INSERT INTO out_db.cache_v1 ("
                                  "    kernel_name,"
                                  "    arch,"
                                  "    hip_version,"
                                  "    generator_sum,"
                                  "    code,"
                                  "    timestamp"
                                  ")"
                                  "SELECT kernel_name, arch, hip_version, generator_sum, code, 0 "
                                  "FROM cache_v1 "
                                  "WHERE "
                                  "  generator_sum = :generator_sum "
                                  "  AND hip_version = :hip_version "
                                  "  AND arch IN ("
                                  "    SELECT arch FROM temp.aot_arch "
                                  "  ) "
                                  "ORDER BY kernel_name, arch, hip_version");
    if(sqlite3_bind_blob(
           copy_stmt.get(), 1, generator_sum.data(), generator_sum.size(), SQLITE_TRANSIENT)
           != SQLITE_OK
       || sqlite3_bind_int64(copy_stmt.get(), 2, HIP_VERSION) != SQLITE_OK)
        throw std::runtime_error(std::string("write_aot_cache copy bind: ")
                                 + sqlite3_errmsg(db_user.get()));

    if(sqlite3_step(copy_stmt.get()) != SQLITE_DONE)
        throw std::runtime_error(std::string("write_aot_cache copy step: ")
                                 + sqlite3_errmsg(db_user.get()));
    sqlite3_reset(copy_stmt.get());
}

void RTCCache::cleanup_cache(sqlite3_int64 target_size_bytes)
{
    // delete any kernels that are older than the newest
    // target-size-worth of kernels
    auto delete_stmt = prepare_stmt(db_user,
                                    "DELETE "
                                    "FROM cache_v1 "
                                    "WHERE "
                                    "  ROWID NOT IN ( "
                                    "    SELECT "
                                    "      rid "
                                    "    FROM "
                                    "      ( "
                                    "      SELECT "
                                    "        ROWID AS rid, "
                                    "        kernel_name "
                                    "        timestamp, "
                                    "        SUM(LENGTH(code) + LENGTH(kernel_name)) "
                                    "          OVER "
                                    "          ( "
                                    "          ORDER BY "
                                    "            timestamp DESC, "
                                    "            kernel_name "
                                    "          ) AS total_code_length "
                                    "      FROM cache_v1 "
                                    "      ) totals "
                                    "    WHERE total_code_length < :target_size_bytes "
                                    "    ) ");
    if(sqlite3_bind_int64(delete_stmt.get(), 1, target_size_bytes) != SQLITE_OK)
        throw std::runtime_error(std::string("cleanup_cache delete bind: ")
                                 + sqlite3_errmsg(db_user.get()));
    if(sqlite3_step(delete_stmt.get()) != SQLITE_DONE)
        throw std::runtime_error(std::string("cleanup_cache delete step: ")
                                 + sqlite3_errmsg(db_user.get()));
    delete_stmt.reset();

    // check if we can reclaim 20% or more of the file's space by vacuuming
    auto          page_count_stmt = prepare_stmt(db_user, "PRAGMA page_count");
    sqlite3_int64 page_count      = 0;
    if(sqlite3_step(page_count_stmt.get()) == SQLITE_ROW)
        page_count = sqlite3_column_int64(page_count_stmt.get(), 0);
    page_count_stmt.reset();

    auto          freelist_count_stmt = prepare_stmt(db_user, "PRAGMA freelist_count");
    sqlite3_int64 freelist_count      = 0;
    if(sqlite3_step(freelist_count_stmt.get()) == SQLITE_ROW)
        freelist_count = sqlite3_column_int64(freelist_count_stmt.get(), 0);
    freelist_count_stmt.reset();

    if(freelist_count >= page_count * 0.2)
    {
        auto vacuum_stmt = prepare_stmt(db_user, "VACUUM");
        if(sqlite3_step(vacuum_stmt.get()) != SQLITE_DONE)
            throw std::runtime_error(std::string("cleanup_cache vacuum step: ")
                                     + sqlite3_errmsg(db_user.get()));
    }
}
