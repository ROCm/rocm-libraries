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

#ifndef ROCFFT_RTC_CACHE_H
#define ROCFFT_RTC_CACHE_H

#include "rocfft/rocfft.h"
#include "rtc_generator.h"
#include "sqlite3.h"
#include <array>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

// deleters for sqlite objects
struct sqlite3_deleter
{
    void operator()(sqlite3* db) const
    {
        sqlite3_close(db);
    }
};
struct sqlite3_stmt_deleter
{
    void operator()(sqlite3_stmt* stmt) const
    {
        sqlite3_finalize(stmt);
    }
};

// smart pointers for sqlite objects
typedef std::unique_ptr<sqlite3, sqlite3_deleter>           sqlite3_ptr;
typedef std::unique_ptr<sqlite3_stmt, sqlite3_stmt_deleter> sqlite3_stmt_ptr;

struct RTCCache
{
    // Get compiled code object for a kernel.  Checks the cache to
    // see if the kernel has already been compiled and returns the
    // cached kernel if present.
    //
    // Otherwise, calls "generate_src" to generate the source, compiles
    // the source, and updates the cache before returning the compiled
    // kernel.  Tries in-process compile first and falls back to
    // subprocess if necessary.
    static std::vector<char> cached_compile(const std::string&          kernel_name,
                                            const std::string&          gpu_arch_with_flags,
                                            kernel_src_gen_t            generate_src,
                                            const std::array<char, 32>& generator_sum);

    RTCCache();
    ~RTCCache() = default;

    // get bytes for a matching code object from the cache.
    // returns empty vector if a matching kernel was not found.
    std::vector<char> get_code_object(const std::string&          kernel_name,
                                      const std::string&          gpu_arch,
                                      const std::array<char, 32>& generator_sum);

    // store the code object into the cache.
    void store_code_object(const std::string&          kernel_name,
                           const std::string&          gpu_arch,
                           const std::array<char, 32>& generator_sum,
                           const std::vector<char>&    code);

    // allocates buffer, call serialize_free to free it
    rocfft_status serialize(void** buffer, size_t* buffer_len_bytes);
    static void   serialize_free(void* buffer);
    rocfft_status deserialize(const void* buffer, size_t buffer_len_bytes);

    // adjust the current cache file to be write-mostly, such as doing
    // many compilations in parallel when building the library
    void enable_write_mostly();

    // write out kernels in the current cache to the output path.
    // this copies the kernels in a consistent order and clears out
    // the timestamp fields so that the resulting file is a
    // reproducible build artifact, suitable for use as an AOT cache.
    void write_aot_cache(const std::string&              output_path,
                         const std::array<char, 32>&     generator_sum,
                         const std::vector<std::string>& gpu_archs);

    // remove kernels in the current cache to keep it roughly under a
    // target size - this counts just the kernel name and code
    // length, and ignores other overhead like indexes and other
    // metadata about the kernels
    void cleanup_cache(sqlite3_int64 target_size_bytes);

    // singleton allocated in rocfft_setup and freed in rocfft_cleanup
    static std::unique_ptr<RTCCache> single;

private:
    sqlite3_ptr connect_db(const std::filesystem::path& path, bool readonly);

    // database handles to system- and user-level caches.  either or
    // both may be a null pointer, if that particular cache could not
    // be located.
    sqlite3_ptr db_sys;
    sqlite3_ptr db_user;

    // query handles, with mutexes to prevent concurrent queries that
    // might stomp on one another's bound values
    sqlite3_stmt_ptr get_stmt_sys;
    std::mutex       get_mutex_sys;
    sqlite3_stmt_ptr get_stmt_user;
    std::mutex       get_mutex_user;
    sqlite3_stmt_ptr store_stmt_user;
    std::mutex       store_mutex_user;

    // lock around deserialization, since that attaches a fixed-name
    // schema to the db and we don't want a collision
    std::mutex deserialize_mutex;

    // keep track of compiles we've started but haven't finished, so
    // that if two identical kernel requests come at the same time, we
    // only compile once.
    struct pending_key
    {
        std::string kernel_name;
        std::string gpu_arch;
        bool        operator<(const pending_key& other) const
        {
            if(kernel_name < other.kernel_name)
                return true;
            if(kernel_name > other.kernel_name)
                return false;
            return gpu_arch < other.gpu_arch;
        };
    };
    // map kernel name + arch to a code object future
    std::map<pending_key, std::shared_future<std::vector<char>>> pending_compiles;
    std::mutex                                                   pending_compiles_mutex;
    friend struct PendingCompileCleanup;
};

#endif
