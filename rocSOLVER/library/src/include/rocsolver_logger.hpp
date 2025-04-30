/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <forward_list>
#include <fstream>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common_host_helpers.hpp"
#include "lib_host_helpers.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_datatype2string.hpp"
#include "rocsolver_logvalue.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/***************************************************************************
 * rocSOLVER logging macros
 ***************************************************************************/

#define ROCSOLVER_ENTER_TOP(name, ...)                                                      \
    std::unique_ptr<rocsolver_logger::scope_guard<T>> _log_token;                           \
    do                                                                                      \
    {                                                                                       \
        if(rocsolver_logger::is_logging_enabled())                                          \
        {                                                                                   \
            rocsolver_logger::instance()->log_enter_top_level<T>(handle, "rocsolver", name, \
                                                                 __VA_ARGS__);              \
            _log_token = std::make_unique<rocsolver_logger::scope_guard<T>>(true, handle);  \
        }                                                                                   \
    } while(0)
#define ROCSOLVER_ENTER(name, ...)                                                              \
    std::unique_ptr<rocsolver_logger::scope_guard<T>> _log_token;                               \
    do                                                                                          \
    {                                                                                           \
        if(rocsolver_logger::is_logging_enabled())                                              \
        {                                                                                       \
            rocsolver_logger::instance()->log_enter<T>(handle, "rocsolver", name, __VA_ARGS__); \
            _log_token = std::make_unique<rocsolver_logger::scope_guard<T>>(false, handle);     \
        }                                                                                       \
    } while(0)
#define ROCBLAS_ENTER(name, ...)                                                              \
    std::unique_ptr<rocsolver_logger::scope_guard<T>> _log_token;                             \
    do                                                                                        \
    {                                                                                         \
        if(rocsolver_logger::is_logging_enabled())                                            \
        {                                                                                     \
            rocsolver_logger::instance()->log_enter<T>(handle, "rocblas", name, __VA_ARGS__); \
            _log_token = std::make_unique<rocsolver_logger::scope_guard<T>>(false, handle);   \
        }                                                                                     \
    } while(0)
#define ROCSOLVER_LAUNCH_KERNEL(name, ...)                                                          \
    do                                                                                              \
    {                                                                                               \
        std::unique_ptr<rocsolver_logger::scope_guard<T>> _kernel_log_token;                        \
        if(rocsolver_logger::is_logging_enabled() && rocsolver_logger::is_kernel_logging_enabled()) \
        {                                                                                           \
            rocsolver_logger::instance()->log_enter<T>(handle, nullptr, #name);                     \
            _kernel_log_token = std::make_unique<rocsolver_logger::scope_guard<T>>(false, handle);  \
        }                                                                                           \
        hipLaunchKernelGGL((name), __VA_ARGS__);                                                    \
    } while(0)

/***************************************************************************
 * The rocsolver_log_entry struct records function data for trace and
 * profile logging purposes.
 ***************************************************************************/
struct rocsolver_log_entry
{
    std::vector<std::string> callers;
    std::string name;
    int level;
    double start_time;

    rocsolver_log_entry()
        : level(0)
        , start_time(0)
    {
    }

    // Move constructor
    rocsolver_log_entry(rocsolver_log_entry&&) = default;

    // Copy constructor
    rocsolver_log_entry(const rocsolver_log_entry&) = default;
};

/***************************************************************************
 * The rocsolver_profile_entry struct records function data for profile
 * logging purposes.
 ***************************************************************************/
struct rocsolver_profile_entry;
using rocsolver_profile_map = std::unordered_map<std::string, rocsolver_profile_entry>;

struct rocsolver_profile_entry
{
    std::string name;
    int level;
    int calls;
    double time;
    std::unique_ptr<rocsolver_profile_map> internal_calls;

    rocsolver_profile_entry()
        : level(0)
        , calls(0)
        , time(0)
    {
    }

    // Move constructor
    rocsolver_profile_entry(rocsolver_profile_entry&&) = default;

    // Copy constructor is deleted
    rocsolver_profile_entry(const rocsolver_profile_entry&) = delete;
};

/***************************************************************************
 * The rocsolver_logger class provides functions to be called upon entering
 * or exiting a function that will output multi-level logging information.
 ***************************************************************************/
class rocsolver_logger
{
private:
    // static singleton instance
    static rocsolver_logger* _instance;
    // static mutex for multithreading
    static std::mutex _mutex;
    // profile logging data keyed by function name
    rocsolver_profile_map profile;
    // function call stack keyed by handle
    std::unordered_map<rocblas_handle, std::vector<rocsolver_log_entry>> call_stack;
    // the maximum depth at which nested function calls will appear in the log
    int max_levels;
    // layer mode enum describing which logging facilities are enabled
    rocblas_layer_mode_flags layer_mode;
    // streams for different logging types
    std::ostream* trace_os;
    std::ostream* bench_os;
    std::ostream* profile_os;
    std::forward_list<std::ofstream> file_streams;
    std::string trace_str;

    // returns a unique_ptr to a file stream or a given default stream
    std::ostream* open_log_stream(const char* environment_variable);

    // returns a log entry on the call stack
    rocsolver_log_entry& push_log_entry(rocblas_handle handle, std::string&& name);
    rocsolver_log_entry& peek_log_entry(rocblas_handle handle);
    rocsolver_log_entry pop_log_entry(rocblas_handle handle);

    // prints the results of profile logging
    void append_profile(std::string& str,
                        rocsolver_profile_map::iterator start,
                        rocsolver_profile_map::iterator end);

    // combines a function prefix and name into an std::string
    template <typename T>
    std::string get_func_name(const char* func_prefix, const char* func_name)
    {
        if(func_prefix)
            return fmt::format("{}_{}{}", func_prefix, rocblas2char_precision<T>, func_name);
        else
            return std::string(func_name);
    }
    std::string get_template_name(const char* func_prefix, const char* func_name)
    {
        if(func_prefix)
            return fmt::format("{}_{}_template", func_prefix, func_name);
        else
            return std::string(func_name);
    }

    // outputs bench logging
    template <typename T, typename... Ts>
    void log_bench(int level, const char* func_prefix, const char* func_name, Ts... args)
    {
        fmt::print(*bench_os, "./rocsolver-bench -f {} -r {} {}\n", func_name,
                   rocblas2char_precision<T>, fmt::join(std::tie(args...), " "));
        bench_os->flush();
    }

    // outputs trace logging
    template <typename T, typename... Ts>
    void log_trace(int level, const char* func_prefix, const char* func_name, Ts... args)
    {
        constexpr int shift_width = 4;
        int indent_level = level - 1;
        int indent = shift_width * indent_level;

        if(sizeof...(Ts) > 0)
        {
            std::string pairs;
            pairs_to_string(pairs, ", ", args...);

            trace_str += fmt::format("{: <{}}{} ({})\n", "", indent,
                                     get_template_name(func_prefix, func_name), pairs);
        }
        else
        {
            trace_str
                += fmt::format("{: <{}}{}\n", "", indent, get_template_name(func_prefix, func_name));
        }
    }

    // populates profile logging data with information from call_stack
    template <typename T>
    void log_profile(rocblas_handle handle, rocsolver_log_entry& from_stack)
    {
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);
        double time = get_time_us_sync(stream) - from_stack.start_time;

        const std::lock_guard<std::mutex> lock(rocsolver_logger::_mutex);

        rocsolver_profile_map* map = &profile;
        for(const std::string& caller_name : from_stack.callers)
        {
            rocsolver_profile_entry& entry = (*map)[caller_name];
            if(!entry.internal_calls)
                entry.internal_calls = std::make_unique<rocsolver_profile_map>();
            map = entry.internal_calls.get();
        }

        rocsolver_profile_entry& from_profile = (*map)[from_stack.name];
        from_profile.name = from_stack.name;
        from_profile.level = from_stack.level;
        from_profile.calls++;
        from_profile.time += time;
    }

    static std::unique_lock<std::mutex> acquire_lock()
    {
        return std::unique_lock<std::mutex>(rocsolver_logger::_mutex);
    }

public:
    // return the singleton instance
    static rocsolver_logger* instance()
    {
        return rocsolver_logger::_instance;
    }

    // returns true if logging facilities are enabled
    static __forceinline__ bool is_logging_enabled()
    {
        return (rocsolver_logger::_instance != nullptr)
            && (rocsolver_logger::_instance->layer_mode
                & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                   | rocblas_layer_mode_log_profile));
    }

    // returns true if logging facilities are enabled for kernels
    static __forceinline__ bool is_kernel_logging_enabled()
    {
        return (rocsolver_logger::_instance != nullptr)
            && (rocsolver_logger::_instance->layer_mode & rocblas_layer_mode_ex_log_kernel);
    }

    // logging function to be called upon entering a top-level (i.e. impl) function
    template <typename T, typename... Ts>
    void log_enter_top_level(rocblas_handle handle,
                             const char* func_prefix,
                             const char* func_name,
                             Ts... args)
    {
        auto lock = acquire_lock();
        auto entry = push_log_entry(handle, get_func_name<T>(func_prefix, func_name));
        bool bench_enabled = layer_mode & rocblas_layer_mode_log_bench;
        bool trace_enabled = layer_mode & rocblas_layer_mode_log_trace;
        lock.unlock();
        ROCSOLVER_ASSUME(entry.level == 0);

        if(bench_enabled)
            log_bench<T>(entry.level, func_prefix, func_name, rocsolver_make_logvalue(args)...);

        if(trace_enabled)
            trace_str += fmt::format("------- ENTER {} trace tree -------\n", entry.name);
    }

    // logging function to be called before exiting a top-level (i.e. impl) function
    template <typename T>
    void log_exit_top_level(rocblas_handle handle)
    {
        auto lock = acquire_lock();
        auto entry = pop_log_entry(handle);
        bool trace_enabled = layer_mode & rocblas_layer_mode_log_trace;
        lock.unlock();
        ROCSOLVER_ASSUME(entry.level == 0);

        if(trace_enabled)
        {
            trace_str += fmt::format("------- EXIT {} trace tree -------\n\n", entry.name);
            *trace_os << trace_str;
            trace_str.clear();
            trace_os->flush();
        }
    }

    // logging function to be called upon entering a sub-level (i.e. template) function
    template <typename T, typename... Ts>
    void log_enter(rocblas_handle handle, const char* func_prefix, const char* func_name, Ts... args)
    {
        auto lock = acquire_lock();
        auto entry = push_log_entry(handle, get_template_name(func_prefix, func_name));
        bool trace_enabled = layer_mode & rocblas_layer_mode_log_trace && entry.level <= max_levels;
        lock.unlock();

        if(trace_enabled)
            log_trace<T>(entry.level, func_prefix, func_name, rocsolver_make_logvalue(args)...);
    }

    // logging function to be called before exiting a sub-level (i.e. template) function
    template <typename T>
    void log_exit(rocblas_handle handle)
    {
        auto lock = acquire_lock();
        auto entry = pop_log_entry(handle);
        bool profile_enabled = layer_mode & rocblas_layer_mode_log_profile;
        lock.unlock();

        if(profile_enabled)
            log_profile<T>(handle, entry);
    }

    /***************************************************************************
     * The scope_guard struct will call an appropriate logging exit function
     * upon the function losing scope.
     ***************************************************************************/
    template <typename T>
    struct scope_guard
    {
        bool top_level;
        rocblas_handle handle;

        // Constructor
        scope_guard(bool top_level, rocblas_handle handle)
            : top_level(top_level)
            , handle(handle)
        {
        }

        // Copy constructor is deleted
        scope_guard(const scope_guard&) = delete;

        // Destructor
        ~scope_guard()
        {
            if(top_level)
                rocsolver_logger::instance()->log_exit_top_level<T>(handle);
            else
                rocsolver_logger::instance()->log_exit<T>(handle);
        }

        // Assignment operator is deleted
        scope_guard& operator=(const scope_guard&) = delete;
    };

    friend rocblas_status rocsolver_log_begin_impl(void);
    friend rocblas_status rocsolver_log_end_impl(void);
    friend rocblas_status rocsolver_log_set_layer_mode_impl(const rocblas_layer_mode_flags layer_mode);
    friend rocblas_status rocsolver_log_set_max_levels_impl(const rocblas_int max_levels);
    friend rocblas_status rocsolver_log_restore_defaults_impl(void);
    friend rocblas_status rocsolver_log_write_profile_impl(void);
    friend rocblas_status rocsolver_log_flush_profile_impl(void);
};

ROCSOLVER_END_NAMESPACE
